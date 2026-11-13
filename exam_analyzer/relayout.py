"""Recompute the knowledge-space layout for one knowledge database.

Usage::

    python relayout.py --db intermediate/9618_knowledge.db --dry-run
    python relayout.py --db intermediate/9618_knowledge.db --epochs 20

Dry-run prints the evaluation metrics without writing anything; without
--dry-run the pass rewrites vector_positions, knowledge_points,
qa_kp_membership and the dynamic_topics mirror.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from src.knowledge._encoding import EmbeddingEncoder  # noqa: E402
from src.knowledge._geometry import LayoutConfig  # noqa: E402
from src.knowledge._layout import run_layout  # noqa: E402
from src.knowledge_base import QADatabase  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recompute the knowledge-space layout")
    parser.add_argument("--db", required=True, help="path to the knowledge database")
    parser.add_argument("--epochs", type=int, default=20, help="relaxation sweeps")
    parser.add_argument("--anchor", type=float, default=0.20, help="anchor strength")
    parser.add_argument("--radius", type=float, default=0.40, help="cluster radius")
    parser.add_argument("--seed", type=int, default=0, help="deterministic seed")
    parser.add_argument("--encoder-version", default="", help="stamped on positions")
    parser.add_argument("--dry-run", action="store_true", help="compute metrics only")
    parser.add_argument("--no-split", action="store_true",
                        help="do not derive fragments from MS text when empty")
    parser.add_argument("--json", default="", help="optional path for the metrics report")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not os.path.exists(args.db):
        print(f"ERROR: database not found: {args.db}")
        return 2

    config = LayoutConfig(
        epochs=args.epochs,
        anchor_strength=args.anchor,
        radius=args.radius,
        seed=args.seed,
    )
    db = QADatabase(args.db)
    started = time.time()
    try:
        result = run_layout(
            db,
            EmbeddingEncoder(),
            config,
            persist=not args.dry_run,
            split_missing=not args.no_split,
            encoder_version=args.encoder_version,
            progress=lambda msg: print(f"[layout] {msg}"),
        )
    finally:
        db.close()

    payload = {
        "db": args.db,
        "dry_run": args.dry_run,
        "elapsed_seconds": round(time.time() - started, 2),
        "kp_count": result.kp_count,
        "fragment_count": result.fragment_count,
        "qa_count": result.qa_count,
        "edge_count": result.edge_count,
        "collapsed": result.collapsed,
        "persisted": result.persisted,
        "metrics": result.metrics,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        print(f"[layout] report written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
