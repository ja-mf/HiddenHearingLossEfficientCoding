#!/usr/bin/env python
import argparse
from lib.processing import build_rlfs
from lib.raw_registry import validate_registry

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build RLF parquet files from raw .mat recordings.")
    parser.add_argument("cohort", choices=["HHL", "CHL"])
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    problems = validate_registry(require_files=True)
    if problems:
        raise SystemExit("\n".join(problems))
    if args.check_only:
        print("raw registry OK")
    else:
        df = build_rlfs(args.cohort)
        print(f"{args.cohort}: {len(df)} rows, {df['neuron_id'].nunique()} neurons")
