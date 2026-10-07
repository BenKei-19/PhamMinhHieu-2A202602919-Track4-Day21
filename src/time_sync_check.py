"""Lớp lỗi Time: đo ảnh hưởng của việc KHÔNG bù chuyển động xe giữa thời điểm chụp LiDAR và camera (nuScenes).

Với mỗi frame: chiếu điểm của object bằng calib có bù ego-motion (đúng) và không bù (giả định 2 sensor chụp
cùng lúc), so box-hit ratio. Ghi kèm độ lệch thời gian và quãng xe đi được trong khoảng lệch đó.

    python -m src.time_sync_check
    python -m src.time_sync_check --data-root data/nuscenes_mini_subset --out results/time_sync_nusc.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from src.calib_qa import object_box_hits, object_refs
from src.calib_sweep import write_csv
from starter.datasets import dataset_type, list_frames, load_frame


def ego_motion_between(calib_ego, calib_noego) -> tuple[float, float]:
    """Dịch chuyển (m) và góc xoay (độ) của xe giữa 2 thời điểm, suy từ hiệu hai ma trận extrinsic."""
    Ta, Tb = np.eye(4), np.eye(4)
    Ta[:3], Tb[:3] = calib_ego.Tr_velo_to_cam, calib_noego.Tr_velo_to_cam
    D = Ta @ np.linalg.inv(Tb)
    angle = np.degrees(np.arccos(np.clip((np.trace(D[:3, :3]) - 1) / 2, -1, 1)))
    return float(np.linalg.norm(D[:3, 3])), float(angle)


def main() -> None:
    ap = argparse.ArgumentParser(description="So box-hit ratio khi có / không bù ego-motion giữa LiDAR và camera")
    ap.add_argument("--data-root", default="data/nuscenes_mini_subset", help="chỉ dùng cho nuScenes")
    ap.add_argument("--out", default="results/time_sync_nusc.csv", help="file CSV kết quả")
    args = ap.parse_args()
    if dataset_type(args.data_root) != "nuscenes":
        raise SystemExit("KITTI không phát hành timestamp camera/LiDAR riêng, chỉ chạy được với nuScenes.")

    rows = []
    for fid in list_frames(args.data_root):
        fr = load_frame(args.data_root, fid)
        noego = load_frame(args.data_root, fid, use_ego_motion=False)["calib"]
        shape = fr["image"].shape
        refs = object_refs(fr["points"], fr["calib"], fr["labels"], shape)
        hit_ego = object_box_hits(refs, fr["calib"], shape)
        hit_noego = object_box_hits(refs, noego, shape)
        disp, rot = ego_motion_between(fr["calib"], noego)
        rows.append({
            "frame": fid,
            "dt_cam_minus_lidar_ms": (fr["timestamp_camera_us"] - fr["timestamp_lidar_us"]) / 1000,
            "ego_disp_m": round(disp, 4), "ego_rot_deg": round(rot, 4), "n_objects": len(refs),
            "hit_with_ego": round(float(np.mean([h["hit_ratio"] for h in hit_ego])), 4) if refs else np.nan,
            "hit_without_ego": round(float(np.mean([h["hit_ratio"] for h in hit_noego])), 4) if refs else np.nan,
            "min_hit_without_ego": round(min(h["hit_ratio"] for h in hit_noego), 4) if refs else np.nan,
            "shift_px_median": round(float(np.median([h["shift_px"] for h in hit_noego])), 2) if refs else np.nan,
        })
    write_csv(Path(args.out), rows)

    a = np.array([[r["dt_cam_minus_lidar_ms"], r["ego_disp_m"], r["hit_with_ego"], r["hit_without_ego"],
                   r["shift_px_median"]] for r in rows], dtype=float)
    print(f"{len(rows)} frame | lệch thời gian camera - LiDAR: trung vị {np.median(a[:, 0]):.1f} ms | "
          f"xe đi được: trung vị {np.median(a[:, 1]):.2f} m, max {a[:, 1].max():.2f} m")
    print(f"box-hit trung bình: có bù {np.nanmean(a[:, 2]):.1%}  ->  không bù {np.nanmean(a[:, 3]):.1%} | "
          f"độ dời pixel trung vị {np.nanmedian(a[:, 4]):.1f} px, max {np.nanmax(a[:, 4]):.1f} px")
    worst = min((r for r in rows if r["n_objects"]), key=lambda r: r["hit_without_ego"])
    print(f"frame tệ nhất: {worst['frame']} hit {worst['hit_with_ego']:.0%} -> {worst['hit_without_ego']:.0%} "
          f"(xe đi {worst['ego_disp_m']:.2f} m trong {abs(worst['dt_cam_minus_lidar_ms']):.0f} ms)")
    print(f"-> {args.out}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
