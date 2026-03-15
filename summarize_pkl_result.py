#!/usr/bin/env python3
"""Summarize a MvSMPLfitting result pkl file.

Usage:
  python summarize_pkl_result.py --pkl output/results/scene1/1/000.pkl
"""

import argparse
import os
import pickle
import pprint
from typing import Any


def _fmt_scalar(x: Any) -> str:
    try:
        return f"{float(x):.6f}"
    except Exception:
        return str(x)


def _print_array_stats(name: str, arr: Any) -> None:
    if not hasattr(arr, "shape"):
        return
    shape = getattr(arr, "shape", None)
    print(f"{name:20s} shape={shape}")

    # Print min/mean/max only for numeric arrays
    try:
        import numpy as np

        a = np.asarray(arr)
        if a.size == 0:
            return
        if a.dtype.kind in {"f", "i", "u"}:
            print(
                f"{'':20s} min={a.min():.6f} mean={a.mean():.6f} max={a.max():.6f}"
            )
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize a fitting result pkl")
    parser.add_argument("--pkl", required=True,
                        help="Path to result .pkl file")
    args = parser.parse_args()

    pkl_path = args.pkl
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"PKL file not found: {pkl_path}")

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    print("=== PKL Summary ===")
    print(f"file: {pkl_path}")
    print(f"type: {type(data).__name__}")

    if not isinstance(data, dict):
        print("This file does not contain a dict-like result object.")
        print(data)
        return

    keys = sorted(data.keys())
    print(f"keys ({len(keys)}): {keys}")

    if "loss" in data:
        print(f"loss: {_fmt_scalar(data['loss'])}")

    for key in ["transl", "global_orient", "scale", "betas", "body_pose", "pose", "pose_embedding"]:
        if key in data:
            _print_array_stats(key, data[key])

    print("\n=== All Raw Entries ===")
    for k in keys:
        v = data[k]
        print(f"\n--- {k} ---")
        if hasattr(v, "shape"):
            print(f"shape: {v.shape}")
        try:
            import numpy as np

            if isinstance(v, np.ndarray):
                with np.printoptions(threshold=np.inf, linewidth=200):
                    print(v)
            else:
                pprint.pprint(v, width=200, compact=False)
        except Exception:
            pprint.pprint(v, width=200, compact=False)


if __name__ == "__main__":
    main()
