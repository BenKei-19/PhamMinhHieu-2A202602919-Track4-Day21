"""Đo latency p50/p95 của các bước QA calibration (bỏ lần chạy đầu, lặp N lần).

    python -m src.latency_bench
    python -m src.latency_bench --repeats 50 --out results/latency.csv
"""
from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

from src.calib_qa import (edge_score, estimate_offset, fov_ratio, image_edge_proximity, lidar_depth_edges,
                          object_box_hits, object_refs, ring_ids)
from src.calib_sweep import write_csv
from starter.datasets import dataset_type, load_frame
from starter.projection import project_velo_to_image


def cpu_name() -> str:
    try:
        import winreg
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    except (ImportError, OSError):
        return platform.processor() or platform.machine()


def bench(fn, repeats: int) -> np.ndarray:
    fn()                                   # lần đầu (khởi tạo, cache) bị bỏ
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t) * 1000)
    return np.array(times)


def main() -> None:
    ap = argparse.ArgumentParser(description="Đo latency p50/p95 (ms) của projection, box-hit, edge score, phép thử đỉnh")
    ap.add_argument("--frames", nargs="*", default=["data/kitti_mini:000011", "data/nuscenes_mini_subset:scene-0103_010"],
                    help="danh sách <data-root>:<frame-id>")
    ap.add_argument("--repeats", type=int, default=30, help="số lần đo (sau khi bỏ lần đầu), >= 20")
    ap.add_argument("--out", default="results/latency.csv")
    args = ap.parse_args()

    hw = f"{cpu_name()}, {os.cpu_count()} luồng, Python {platform.python_version()}, numpy {np.__version__}, CPU only"
    rows = []
    grid = np.round(np.arange(-4, 4.001, 0.25), 2)
    for item in args.frames:
        root, fid = item.split(":", 1)
        ds = dataset_type(root)
        fr = load_frame(root, fid)
        pts, calib, img = fr["points"], fr["calib"], fr["image"]
        refs = object_refs(pts, calib, fr["labels"], img.shape)
        edge_pts = pts[lidar_depth_edges(pts, ring_ids(root, fid, pts)), :3]
        prox, _ = image_edge_proximity(img)
        steps = {
            "projection (toàn bộ point cloud)": lambda: project_velo_to_image(pts, calib, img.shape),
            "fov_ratio": lambda: fov_ratio(pts, calib, img.shape),
            "box_hit (mọi object)": lambda: object_box_hits(refs, calib, img.shape),
            "depth-edge LiDAR + Canny ảnh": lambda: (lidar_depth_edges(pts, ring_ids(root, fid, pts)),
                                                     image_edge_proximity(img)),
            "edge_score (1 calib)": lambda: edge_score(edge_pts, calib, prox),
            f"phép thử đỉnh yaw ({len(grid)} calib)": lambda: estimate_offset(edge_pts, calib, prox, ds, "yaw", grid),
        }
        for name, fn in steps.items():
            t = bench(fn, args.repeats)
            rows.append({"dataset": ds, "frame": fid, "n_points": len(pts), "step": name, "repeats": args.repeats,
                         "p50_ms": round(float(np.percentile(t, 50)), 3), "p95_ms": round(float(np.percentile(t, 95)), 3),
                         "hardware": hw})
            print(f"{ds:8s} {fid:16s} {name:38s} p50={rows[-1]['p50_ms']:8.2f} ms  p95={rows[-1]['p95_ms']:8.2f} ms")
    write_csv(Path(args.out), rows)
    print(f"Phần cứng: {hw}\n-> {args.out}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
