from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault('MPLBACKEND', 'Agg')


def _find_summary_files(root: Path) -> List[Path]:
    return sorted(root.glob('**/metrics/summary.json'))


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def _extract_metrics(item: Dict[str, Any]) -> Dict[str, Any]:
    metrics = item.get('metrics', {})
    eval_block = item.get('evaluation', {})
    train_history = item.get('train_history', {})
    train_losses = train_history.get('train_loss') or []
    val_losses = train_history.get('val_loss') or []
    model_field = item.get('model', 'unknown')
    benchmark_field = item.get('benchmark', 'unknown')
    model_name = model_field.get('name', model_field) if isinstance(model_field, dict) else model_field
    benchmark_name = benchmark_field.get('name', benchmark_field) if isinstance(benchmark_field, dict) else benchmark_field
    out = {
        'model': model_name,
        'benchmark': benchmark_name,
        'params': item.get('parameters', None),
        'train_last': train_losses[-1] if train_losses else None,
        'val_last': val_losses[-1] if val_losses else None,
        'rollout_mse': metrics.get('mse', eval_block.get('mse_mean', None)),
        'final_step_mse': metrics.get('final_step_mse', eval_block.get('mse_final', None)),
        'rollout_score': metrics.get('rollout_score', None),
    }
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--results_root', type=Path, default=Path('results') / 'compact_suite')
    p.add_argument('--output_dir', type=Path, default=Path('results') / 'processed')
    p.add_argument('--summary_name', type=str, default='suite_summary')
    return p.parse_args()


def _group_best(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        bench = str(row.get('benchmark'))
        score = row.get('rollout_mse')
        if score is None:
            continue
        if bench not in best or float(score) < float(best[bench]['rollout_mse']):
            best[bench] = row
    return [best[k] for k in sorted(best)]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for path in _find_summary_files(args.results_root):
        try:
            data = _load_json(path)
        except Exception:
            continue
        rows.append(_extract_metrics(data))

    rows.sort(key=lambda x: (str(x.get('benchmark')), str(x.get('model'))))
    best_rows = _group_best(rows)

    md_lines = [
        '| benchmark | model | params | train_last | val_last | rollout_mse | final_step_mse | rollout_score |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for r in rows:
        md_lines.append(
            f"| {r['benchmark']} | {r['model']} | {r['params']} | {r['train_last']} | {r['val_last']} | {r['rollout_mse']} | {r['final_step_mse']} | {r['rollout_score']} |"
        )

    (args.output_dir / f'{args.summary_name}.md').write_text('\n'.join(md_lines), encoding='utf-8')
    (args.output_dir / f'{args.summary_name}.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    (args.output_dir / f'{args.summary_name}_best.json').write_text(json.dumps(best_rows, indent=2), encoding='utf-8')

    if rows:
        import matplotlib
        matplotlib.use('Agg', force=True)
        import matplotlib.pyplot as plt

        labels = [f"{r['benchmark']}\n{r['model']}" for r in rows]
        values = [float(r['rollout_mse']) if r['rollout_mse'] is not None else 0.0 for r in rows]
        plt.figure(figsize=(max(8, len(rows) * 0.8), 4))
        plt.bar(labels, values)
        plt.xticks(rotation=30, ha='right')
        plt.ylabel('Rollout MSE')
        plt.title('Compact Suite Summary')
        plt.tight_layout()
        plt.savefig(args.output_dir / f'{args.summary_name}.png', dpi=200)
        plt.close()

        if best_rows:
            plt.figure(figsize=(max(8, len(best_rows) * 0.8), 4))
            plt.bar([r['benchmark'] for r in best_rows], [float(r['rollout_mse']) for r in best_rows])
            plt.ylabel('Best Rollout MSE')
            plt.title('Best Model per Benchmark')
            plt.tight_layout()
            plt.savefig(args.output_dir / f'{args.summary_name}_best.png', dpi=200)
            plt.close()

    print((args.output_dir / f'{args.summary_name}.md').as_posix())


if __name__ == '__main__':
    main()
