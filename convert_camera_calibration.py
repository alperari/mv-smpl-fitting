#!/usr/bin/env python3
"""Convert Skeletool-style camera.calibration into MvSMPLfitting cam.txt format.

Input format example block:
  name          0
    intrinsic   1497.693 0 1024.704 0 ... 0 0 0 1
    extrinsic   0.965... 0.004... ... 0 0 0 1

Output format (one block per selected camera):
  <new_index>
  fx 0 cx
  0 fy cy
  0 0 1
  0 0
  r11 r12 r13 t1
  r21 r22 r23 t2
  r31 r32 r33 t3
"""

import argparse
import os
import os.path as osp
import re
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert camera.calibration to MvSMPLfitting cam.txt"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to source camera.calibration file",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output cam.txt",
    )
    parser.add_argument(
        "--camera_ids",
        default="",
        help=(
            "Comma-separated source camera IDs to export, e.g. '0,2,7,8'. "
            "If empty, exports all cameras in source order."
        ),
    )
    parser.add_argument(
        "--scene_dir",
        default="",
        help=(
            "Optional scene folder (e.g. data_x/images/scene_1). "
            "If set and --camera_ids is empty, IDs are inferred from cam* folders."
        ),
    )
    return parser.parse_args()


def _to_float_list(parts: List[str]) -> List[float]:
    return [float(x) for x in parts]


def parse_calibration_file(path: str) -> Dict[int, Dict[str, List[float]]]:
    cameras: Dict[int, Dict[str, List[float]]] = {}
    current_id = None

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            if line.startswith("name"):
                # Supports: "name          0"
                parts = line.split()
                if len(parts) < 2:
                    raise ValueError(f"Malformed camera name line: {line}")
                current_id = int(parts[-1])
                cameras[current_id] = {}
                continue

            if current_id is None:
                continue

            if line.startswith("intrinsic"):
                vals = _to_float_list(line.split()[1:])
                if len(vals) != 16:
                    raise ValueError(
                        f"Camera {current_id}: expected 16 intrinsic values, got {len(vals)}"
                    )
                cameras[current_id]["intrinsic"] = vals
            elif line.startswith("extrinsic"):
                vals = _to_float_list(line.split()[1:])
                if len(vals) != 16:
                    raise ValueError(
                        f"Camera {current_id}: expected 16 extrinsic values, got {len(vals)}"
                    )
                cameras[current_id]["extrinsic"] = vals

    missing = [
        cid for cid, data in cameras.items() if "intrinsic" not in data or "extrinsic" not in data
    ]
    if missing:
        raise ValueError(f"Missing intrinsic/extrinsic in cameras: {missing}")

    return cameras


def infer_ids_from_scene_dir(scene_dir: str) -> List[int]:
    if not osp.isdir(scene_dir):
        raise FileNotFoundError(f"scene_dir does not exist: {scene_dir}")

    ids = []
    for name in sorted(os.listdir(scene_dir)):
        # Expected folder names like cam0, cam2, Camera00, etc.
        match = re.search(r"(\d+)$", name)
        if match:
            ids.append(int(match.group(1)))

    if not ids:
        raise ValueError(
            f"Could not infer camera IDs from scene_dir: {scene_dir}"
        )
    return ids


def format_num(x: float) -> str:
    return f"{x:.12g}"


def make_output_block(new_idx: int, intrinsic16: List[float], extrinsic16: List[float]) -> str:
    # 4x4 row-major matrices
    K = [intrinsic16[0:4], intrinsic16[4:8],
         intrinsic16[8:12], intrinsic16[12:16]]
    E = [extrinsic16[0:4], extrinsic16[4:8],
         extrinsic16[8:12], extrinsic16[12:16]]

    lines = [
        str(new_idx),
        f"{format_num(K[0][0])} {format_num(K[0][1])} {format_num(K[0][2])}",
        f"{format_num(K[1][0])} {format_num(K[1][1])} {format_num(K[1][2])}",
        f"{format_num(K[2][0])} {format_num(K[2][1])} {format_num(K[2][2])}",
        "0 0",
        f"{format_num(E[0][0])} {format_num(E[0][1])} {format_num(E[0][2])} {format_num(E[0][3])}",
        f"{format_num(E[1][0])} {format_num(E[1][1])} {format_num(E[1][2])} {format_num(E[1][3])}",
        f"{format_num(E[2][0])} {format_num(E[2][1])} {format_num(E[2][2])} {format_num(E[2][3])}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    cameras = parse_calibration_file(args.input)

    selected_ids: List[int]
    if args.camera_ids.strip():
        selected_ids = [int(x.strip())
                        for x in args.camera_ids.split(",") if x.strip()]
    elif args.scene_dir.strip():
        selected_ids = infer_ids_from_scene_dir(args.scene_dir)
    else:
        selected_ids = sorted(cameras.keys())

    for cid in selected_ids:
        if cid not in cameras:
            raise KeyError(
                f"Requested camera id {cid} not found in input file")

    out_blocks = []
    for new_idx, src_id in enumerate(selected_ids):
        cam = cameras[src_id]
        out_blocks.append(
            make_output_block(new_idx, cam["intrinsic"], cam["extrinsic"])
        )

    out_dir = osp.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(out_blocks))

    print(f"Wrote {len(selected_ids)} cameras to: {args.output}")
    print(f"Source camera IDs: {selected_ids}")


if __name__ == "__main__":
    main()
