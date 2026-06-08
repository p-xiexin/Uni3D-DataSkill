from __future__ import annotations

import argparse
import json
from pathlib import Path

from .inspect import BlendedMVSInspector
from .reports import write_json_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="unidata-skill")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Inspect a source dataset before conversion.")
    inspect_parser.add_argument("--dataset", default="blendedmvs", choices=["blendedmvs"])
    inspect_parser.add_argument("--input", required=True, help="Source dataset root.")
    inspect_parser.add_argument("--output", help="Path to inspection_report.json.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "inspect":
        report = BlendedMVSInspector().inspect(args.input)
        if args.output:
            write_json_report(report, args.output)
        else:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["status"] == "ok" else 2

    parser.error(f"unknown command: {args.command}")
    return 2
