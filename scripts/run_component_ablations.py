#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / 'fim_experiments'
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import main as experiment_main
from ablation_systems import AblatedFIMSystem, variant_switches


DEFAULT_VARIANTS = ['full', 'no_memory', 'no_retrieval', 'no_salience_gating']


@lru_cache(maxsize=1)
def source_identity() -> str:
    """Hash the executable source/protocol tree when no repository commit exists."""
    paths = [
        *sorted((ROOT / 'fim').rglob('*.py')),
        *sorted((ROOT / 'fim_experiments').glob('*.py')),
        *sorted((ROOT / 'scripts').glob('*.py')),
        *sorted((ROOT / 'scripts').glob('*.sh')),
        *sorted((ROOT / 'tests').rglob('*.py')),
        ROOT / 'research' / 'CURRENT_COMPONENT_ABLATION_PROTOCOL_20260829.md',
        ROOT / 'pyproject.toml',
        ROOT / 'requirements.txt',
    ]
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(ROOT).as_posix().encode('utf-8')
        digest.update(relative + b'\0' + path.read_bytes() + b'\0')
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return 'UNKNOWN'


def model_factory(switches: dict[str, bool]):
    def factory(**kwargs):
        return AblatedFIMSystem(**kwargs, **switches)
    return factory


def metric_scalar(metrics: dict[str, Any], key: str):
    value = metrics.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def run_one(args, benchmark: str, seed: int, variant: str) -> dict[str, Any]:
    switches = variant_switches(variant)
    experiment_main.FIMSystem = model_factory(switches)

    cfg = experiment_main.default_config()
    cfg['experiment']['name'] = f'current_ablation_{benchmark}_{variant}_s{seed}'
    cfg['runtime']['seed'] = seed
    cfg['runtime']['device'] = args.device
    cfg['runtime']['results_root'] = str(args.results_root)
    cfg['benchmark']['name'] = benchmark
    cfg['model']['name'] = 'fim_plus'
    cfg['train']['epochs'] = args.epochs
    cfg['train']['batch_size'] = args.batch_size
    cfg['train']['dataset_size'] = args.dataset_size
    cfg['train']['rollout_steps'] = args.rollout_steps
    cfg['train']['workers'] = 0
    cfg['eval']['rollout_steps'] = args.eval_steps

    started = datetime.now(timezone.utc).isoformat()
    metrics = experiment_main.run_experiment(cfg)
    ended = datetime.now(timezone.utc).isoformat()
    exp_dir = (
        args.results_root
        / cfg['experiment']['name']
        / f'{benchmark}_fim_plus'
    )

    artifact_paths = {
        'summary_file': exp_dir / 'metrics' / 'summary.json',
        'run_results_file': exp_dir / 'metrics' / 'run_results.json',
        'resolved_config_file': exp_dir / 'configs' / 'resolved_config.yaml',
        'checkpoint_file': exp_dir / 'checkpoints' / 'final_state.pt',
        'log_file': exp_dir / 'logs' / 'run.log',
    }
    missing = [str(path) for path in artifact_paths.values() if not path.is_file()]
    if missing:
        raise RuntimeError('experiment completed without required artifacts: ' + ', '.join(missing))
    row = {
        'benchmark': benchmark,
        'seed': seed,
        'variant': variant,
        **switches,
        'started_utc': started,
        'ended_utc': ended,
        'git_commit': git_commit(),
        'source_identity': source_identity(),
        'identity_kind': 'sha256_source_tree',
        'experiment_name': cfg['experiment']['name'],
        'results_root': str(args.results_root),
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'dataset_size': args.dataset_size,
        'train_rollout_steps': args.rollout_steps,
        'eval_rollout_steps': args.eval_steps,
        **{key: str(path.relative_to(ROOT)) for key, path in artifact_paths.items()},
        'artifact_sha256': {key: file_sha256(path) for key, path in artifact_paths.items()},
    }
    for key in ['rollout_mse', 'rollout_mae', 'mse', 'mae']:
        value = metric_scalar(metrics, key)
        if value is not None:
            row[key] = value
    return row


def _payload(args, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        'schema_version': 2,
        'updated_utc': datetime.now(timezone.utc).isoformat(),
        'git_commit': git_commit(),
        'source_identity': source_identity(),
        'identity_kind': 'sha256_source_tree',
        'scientific_boundary': (
            'Fresh current-code component ablations only. These results do not reproduce or replace historical '
            'paper-reference labels unless a separate equivalence audit establishes matching semantics.'
        ),
        'benchmarks': args.benchmarks,
        'seeds': args.seeds,
        'variants': args.variants,
        'protocol': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'dataset_size': args.dataset_size,
            'train_rollout_steps': args.rollout_steps,
            'eval_rollout_steps': args.eval_steps,
        },
        'runs': rows,
    }


def _write_manifest(args, rows: list[dict[str, Any]]) -> None:
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.manifest.with_suffix(args.manifest.suffix + '.tmp')
    temporary.write_text(json.dumps(_payload(args, rows), indent=2) + '\n', encoding='utf-8')
    temporary.replace(args.manifest)


def _load_reusable_rows(args) -> list[dict[str, Any]]:
    if not args.manifest.is_file():
        return []
    try:
        data = json.loads(args.manifest.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return []
    expected = _payload(args, [])
    keys = ('source_identity', 'identity_kind', 'benchmarks', 'seeds', 'variants', 'protocol')
    if any(data.get(key) != expected.get(key) for key in keys):
        raise RuntimeError('existing ablation manifest conflicts with the current source or frozen protocol')
    rows = data.get('runs', [])
    if not isinstance(rows, list):
        raise RuntimeError('existing ablation manifest runs must be a list')
    reusable = []
    for row in rows:
        artifact_keys = ('summary_file', 'run_results_file', 'resolved_config_file', 'checkpoint_file', 'log_file')
        hashes = row.get('artifact_sha256', {})
        valid = True
        for key in artifact_keys:
            value = row.get(key)
            path = ROOT / value if isinstance(value, str) else None
            if path is None or not path.is_file() or hashes.get(key) != file_sha256(path):
                valid = False
                break
        if valid:
            reusable.append(row)
    return reusable


def main() -> None:
    ap = argparse.ArgumentParser(
        description='Run explicit current-code FIM component ablations. Historical unsupported labels fail closed.'
    )
    ap.add_argument('--benchmarks', nargs='+', default=['lorenz96', 'delayed_recall'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[11, 23, 37])
    ap.add_argument('--variants', nargs='+', default=DEFAULT_VARIANTS)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--dataset-size', type=int, default=2048)
    ap.add_argument('--rollout-steps', type=int, default=4)
    ap.add_argument('--eval-steps', type=int, default=30)
    ap.add_argument('--results-root', type=Path, default=ROOT / 'results' / 'current_component_ablations')
    ap.add_argument('--manifest', type=Path, default=ROOT / 'results' / 'current_component_ablations' / 'manifest.json')
    args = ap.parse_args()
    if not args.results_root.is_absolute():
        args.results_root = (ROOT / args.results_root).resolve()
    if not args.manifest.is_absolute():
        args.manifest = (ROOT / args.manifest).resolve()

    # Resolve every label before any training so unsupported historical labels
    # fail before compute and cannot silently map to a different mechanism.
    for variant in args.variants:
        variant_switches(variant)

    args.results_root.mkdir(parents=True, exist_ok=True)
    rows = _load_reusable_rows(args)
    completed = {(row['benchmark'], int(row['seed']), row['variant']) for row in rows}
    for benchmark in args.benchmarks:
        for seed in args.seeds:
            for variant in args.variants:
                if (benchmark, seed, variant) in completed:
                    print(f'CACHE benchmark={benchmark} seed={seed} variant={variant}', flush=True)
                    continue
                print(f'RUN benchmark={benchmark} seed={seed} variant={variant}', flush=True)
                rows.append(run_one(args, benchmark, seed, variant))
                _write_manifest(args, rows)
    _write_manifest(args, rows)
    print(f'WROTE {args.manifest} runs={len(rows)}', flush=True)


if __name__ == '__main__':
    main()
