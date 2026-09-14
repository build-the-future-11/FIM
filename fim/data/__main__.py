from __future__ import annotations

import argparse
import json
from pathlib import Path

from .registry import prepare_dataset, validate_dataset


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Prepare and validate deterministic FIM datasets.")
    sub = root.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="List supported generated datasets.")
    download = sub.add_parser("download", help="Report external download requirements.")
    download.add_argument("--name", default="all")
    prepare = sub.add_parser("prepare", help="Materialize a generated dataset and manifest.")
    prepare.add_argument("--name", default="lorenz96")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--size", type=int, default=128)
    prepare.add_argument("--steps", type=int, default=16)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--config-json", default="{}")
    validate = sub.add_parser("validate", help="Validate tensors, splits, and checksums.")
    validate.add_argument("--output", type=Path, required=True)
    inspect = sub.add_parser("inspect", help="Print a dataset manifest.")
    inspect.add_argument("--output", type=Path, required=True)
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "list":
        print(json.dumps({"generated": ["lorenz96", "burgers", "reaction_diffusion", "kuramoto", "fractional", "levy", "delayed_recall"], "external": []}, indent=2))
    elif args.command == "download":
        print(json.dumps({"downloaded": [], "message": "Current benchmarks are generated locally; no external dataset downloader is implemented."}, indent=2))
    elif args.command == "prepare":
        result = prepare_dataset(benchmark_name=args.name, benchmark_config=json.loads(args.config_json), output_dir=args.output, size=args.size, steps=args.steps, seed=args.seed)
        print(json.dumps(result, indent=2))
    elif args.command == "validate":
        print(json.dumps(validate_dataset(args.output), indent=2))
    else:
        print((args.output / "manifest.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
