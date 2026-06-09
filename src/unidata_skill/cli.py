from __future__ import annotations

import argparse
import json

from .datasets import validate_kitti360_pi3x_dataloader


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="unidata-skill")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser(
        "validate-kitti360-pi3x",
        help="Validate direct KITTI-360 loading through a Pi3X-compatible dataset.",
    )
    validate_parser.add_argument("--kitti360-root", required=True, help="KITTI-360 dataset root.")
    validate_parser.add_argument("--pi3-root", help="Local yyfz/Pi3 training-branch checkout.")
    validate_parser.add_argument("--sequence", action="append", dest="sequences", help="Sequence to include. Can be repeated.")
    validate_parser.add_argument("--camera", action="append", dest="cameras", choices=["image_00", "image_01"], help="Camera to include. Can be repeated.")
    validate_parser.add_argument("--frame-num", type=int, default=8)
    validate_parser.add_argument("--stride", type=int, default=5)
    validate_parser.add_argument("--resolution", default="512x384", help="Width x height, for example 512x384.")
    validate_parser.add_argument("--max-samples", type=int, default=4)
    validate_parser.add_argument("--batch-size", type=int, default=1)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "validate-kitti360-pi3x":
        width, height = (int(item) for item in args.resolution.lower().split("x", 1))
        result = validate_kitti360_pi3x_dataloader(
            kitti360_root=args.kitti360_root,
            pi3_root=args.pi3_root,
            sequences=args.sequences,
            cameras=tuple(args.cameras or ["image_00"]),
            frame_num=args.frame_num,
            stride=args.stride,
            resolution=(width, height),
            max_samples=args.max_samples,
            batch_size=args.batch_size,
        )
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return 0 if result.status == "ok" else 2

    parser.error(f"unknown command: {args.command}")
    return 2
