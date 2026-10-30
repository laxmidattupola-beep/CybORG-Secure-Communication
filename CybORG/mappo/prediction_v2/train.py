from __future__ import annotations
import argparse
from pathlib import Path
from .config import DATA_DIR, MODEL_DIR, load_config
from .models import train_dataset

def main():
    p = argparse.ArgumentParser(description="Train zone-risk models with episode-grouped held-out evaluation")
    p.add_argument("--dataset", type=Path, default=DATA_DIR / "zone_risk_dataset.csv")
    p.add_argument("--output-dir", type=Path, default=MODEL_DIR)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args(); cfg = load_config()
    if not args.dataset.exists(): p.error(f"Dataset not found: {args.dataset}. Run collect_data first after simulator execution is approved.")
    report = train_dataset(args.dataset, args.output_dir, seed=args.seed if args.seed is not None else cfg.random_seed, windows=cfg.windows)
    print(report.to_string(index=False))
    print(f"\nSaved models and metrics under {args.output_dir.resolve()}")
if __name__ == "__main__": main()
