#!/usr/bin/env python3
"""Generate a single-camera parameter file for one uncalibrated image.

This writes a camera file compatible with this repository's loader format:
- 1 camera block
- Intrinsics estimated from image size
- Identity extrinsics (R=I, t=0)
"""

import argparse
import os
import os.path as osp

import cv2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate single_cam_params.txt from one image"
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to input image (used to read width/height)",
    )
    parser.add_argument(
        "--output",
        default="data/single_cam_params.txt",
        help="Output camera parameter text file",
    )
    parser.add_argument(
        "--focal_multiplier",
        type=float,
        default=1.2,
        help="If fx/fy are not provided, use fx=fy=focal_multiplier*image_width",
    )
    parser.add_argument(
        "--camera_id",
        type=int,
        default=0,
        help="Camera index to write in the file",
    )
    parser.add_argument("--fx", type=float, default=None,
                        help="Override focal length fx")
    parser.add_argument("--fy", type=float, default=None,
                        help="Override focal length fy")
    parser.add_argument("--cx", type=float, default=None,
                        help="Override principal point cx")
    parser.add_argument("--cy", type=float, default=None,
                        help="Override principal point cy")
    return parser.parse_args()


def format_num(x):
    # Keep output readable and stable while preserving precision.
    return f"{x:.6f}".rstrip("0").rstrip(".")


def main():
    args = parse_args()

    if not osp.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")

    img = cv2.imread(args.image)
    if img is None:
        raise RuntimeError(f"Could not read image: {args.image}")

    height, width = img.shape[:2]

    cx = args.cx if args.cx is not None else width / 2.0
    cy = args.cy if args.cy is not None else height / 2.0

    default_f = args.focal_multiplier * float(width)
    fx = args.fx if args.fx is not None else default_f
    fy = args.fy if args.fy is not None else default_f

    out_dir = osp.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    lines = [
        str(args.camera_id),
        f"{format_num(fx)} 0 {format_num(cx)}",
        f"0 {format_num(fy)} {format_num(cy)}",
        "0 0 1",
        "0 0",
        "1 0 0 0",
        "0 1 0 0",
        "0 0 1 0",
        "",
    ]

    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Wrote camera parameter file: {args.output}")
    print(f"Image size: {width}x{height}")
    print(f"Intrinsics: fx={fx:.4f}, fy={fy:.4f}, cx={cx:.4f}, cy={cy:.4f}")
    print("Extrinsics: identity (R=I, t=0)")


if __name__ == "__main__":
    main()
