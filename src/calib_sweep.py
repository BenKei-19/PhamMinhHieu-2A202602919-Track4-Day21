"""Sweep calibration drift: làm lệch extrinsic từng trục một, đo 3 metric trên mọi frame, ghi CSV.

Ví dụ:
    python -m src.calib_sweep --data-root data/kitti_mini --tag kitti
    python -m src.calib_sweep --data-root data/nuscenes_mini_subset --tag nusc
    python -m src.calib_sweep --data-root data/kitti_mini --tag kitti_yaw --axes yaw --rot-levels 0.5 1 2

Mỗi lần chạy chỉ thay đổi MỘT yếu tố (một trục, một mức), giữ nguyên frame, label, ngưỡng.
Không có phép ngẫu nhiên nào, nên chạy lại luôn ra đúng cùng số.

Đầu ra (với --tag T):
    results/calib_sweep_T.csv          tổng hợp: mỗi dòng một (trục, mức)
    results/calib_sweep_T_objects.csv  chi tiết từng object: hit ratio, độ dời pixel, khoảng cách
    results/calib_sweep_T_frames.csv   chi tiết từng frame: FOV ratio, edge score, offset ước lượng
    results/calib_sweep_T_windows.csv  phép thử đỉnh trên cửa sổ W frame liên tiếp (cộng dồn score)
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

from src.calib_qa import (DIST_BINS, dist_bin, edge_score, estimate_offset, fov_ratio, image_edge_proximity,
                          lidar_depth_edges, object_box_hits, object_refs, perturb_physical, ring_ids)
from starter.datasets import dataset_type, list_frames, load_frame

ROT_AXES = ("yaw", "pitch", "roll")
TRANS_AXES = ("t_fwd", "t_left", "t_up")


def build_configs(axes: list[str], rot_levels: list[float], trans_levels: list[float]) -> list[tuple[str, float]]:
    configs = [("none", 0.0)]
    for ax in axes:
        levels = rot_levels if ax in ROT_AXES else trans_levels
        configs += [(ax, float(v)) for v in levels if v != 0]
    return configs


def perturb_kwargs(axis: str, level: float) -> dict:
    if axis == "none":
        return {}
    return {f"{axis}_deg": level} if axis in ROT_AXES else {axis: level}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def flag_drift(est: np.ndarray, prom: np.ndarray, tau: float, min_prom: float) -> np.ndarray:
    """Báo drift khi đỉnh score đủ nhô (tin cậy được) VÀ nằm cách 0 ít nhất tau độ.
    Frame có đường cong score phẳng (đỉnh không nhô) thì "không kết luận", không báo drift."""
    ok = np.isfinite(est) & np.isfinite(prom) & (prom >= min_prom)
    return ok & (np.abs(np.nan_to_num(est)) >= tau)


def window_groups(frames: list[str], size: int) -> list[list[int]]:
    """Chia frame thành các cửa sổ `size` frame liên tiếp, không vượt qua ranh giới scene (nuScenes: scene-XXXX_NNN)."""
    by_scene: dict[str, list[int]] = {}
    for i, fid in enumerate(frames):
        by_scene.setdefault(fid.rsplit("_", 1)[0] if "_" in fid else "all", []).append(i)
    return [idx[k:k + size] for idx in by_scene.values() for k in range(0, len(idx) - size + 1, size)]


def window_rows(configs, frames, curves, grid, size: int) -> list[dict]:
    """Cộng dồn đường cong score (đã chuẩn hoá theo median của từng frame) trong mỗi cửa sổ rồi mới tìm đỉnh."""
    rows = []
    for axis, level in configs:
        for w, idx in enumerate(window_groups(frames, size)):
            row = {"axis": axis, "level": level, "window": w, "frames": " ".join(frames[i] for i in idx)}
            for test in ("yaw", "pitch"):
                mean_curve = np.nanmean([curves[(axis, level, test)][i] for i in idx], axis=0)
                row[f"est_{test}_deg"] = float(grid[int(np.nanargmax(mean_curve))])
                row[f"{test}_peak_prom"] = float(np.nanmax(mean_curve) / np.nanmedian(mean_curve))
            rows.append(row)
    return rows


def summarize(configs, obj_rows, frame_rows, tau: float, min_prom: float,
              win_rows: list[dict] | None = None) -> list[dict]:
    """Bảng tổng hợp. Dòng axis=none: cột detect_* chính là tỉ lệ BÁO ĐỘNG GIẢ (frame không lệch mà vẫn báo)."""
    def mean(xs):
        return round(float(np.mean(xs)), 4) if len(xs) else float("nan")

    base_fov = {r["frame"]: r["fov_ratio"] for r in frame_rows if r["axis"] == "none"}
    sources = {"test": frame_rows, "window": win_rows or []}

    summary = []
    for axis, level in configs:
        objs = [r for r in obj_rows if r["axis"] == axis and r["level"] == level]
        frs = [r for r in frame_rows if r["axis"] == axis and r["level"] == level]
        row = {"axis": axis, "level": level, "unit": "deg" if axis in ROT_AXES else "m" if axis != "none" else "",
               "n_frames": len(frs), "n_objects": len(objs),
               "fov_ratio": mean([r["fov_ratio"] for r in frs]),
               "fov_change_pp": mean([100 * (r["fov_ratio"] - base_fov[r["frame"]]) for r in frs]),
               "hit_all": mean([r["hit_ratio"] for r in objs])}
        for _, _, name in DIST_BINS:
            row[f"hit_{name}"] = mean([r["hit_ratio"] for r in objs if r["dist_bin"] == name])
        for grp in ("vehicle", "vru"):
            row[f"hit_{grp}"] = mean([r["hit_ratio"] for r in objs if r["group"] == grp])
        row["objects_below_50pct"] = mean([r["hit_ratio"] < 0.5 for r in objs])
        row["box_w_deg_median"] = round(float(np.median([r["box_w_deg"] for r in objs])), 2) if objs else float("nan")
        row["shift_px_median"] = round(float(np.nanmedian([r["shift_px"] for r in objs])), 2) if objs else float("nan")
        norm = np.array([r["edge_score_norm"] for r in frs], dtype=float)
        row["edge_norm_mean"] = round(float(np.nanmean(norm)), 4)
        row["edge_norm_std"] = round(float(np.nanstd(norm)), 4)
        for test in ("yaw", "pitch"):
            for kind, src in sources.items():
                rs = [r for r in src if r["axis"] == axis and r["level"] == level]
                if not rs:
                    continue
                est = np.array([r[f"est_{test}_deg"] for r in rs], dtype=float)
                prom = np.array([r[f"{test}_peak_prom"] for r in rs], dtype=float)
                row[f"confident_{test}_{kind}"] = round(float(np.mean(prom >= min_prom)), 4)
                row[f"detect_{test}_{kind}"] = round(float(np.mean(flag_drift(est, prom, tau, min_prom))), 4)
        summary.append(row)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Sweep calibration drift LiDAR-camera: FOV ratio, box-hit ratio, "
                                             "edge alignment score. Ghi kết quả ra CSV trong results/.")
    ap.add_argument("--data-root", default="data/kitti_mini", help="data/kitti_mini, data/nuscenes_mini_subset, ...")
    ap.add_argument("--tag", default=None, help="hậu tố tên file CSV (mặc định: tên thư mục dữ liệu)")
    ap.add_argument("--frames", nargs="*", default=None, help="danh sách frame id (mặc định: toàn bộ)")
    ap.add_argument("--axes", nargs="*", default=list(ROT_AXES + TRANS_AXES),
                    choices=list(ROT_AXES + TRANS_AXES), help="các trục cần làm lệch")
    ap.add_argument("--rot-levels", nargs="*", type=float, default=[0.25, 0.5, 1, 2, 3], help="mức xoay (độ)")
    ap.add_argument("--trans-levels", nargs="*", type=float, default=[0.02, 0.05, 0.10, 0.20], help="mức dịch (mét)")
    ap.add_argument("--grid-max", type=float, default=4.0, help="phép thử đỉnh: tìm offset trong [-max, max] độ")
    ap.add_argument("--grid-step", type=float, default=0.25, help="bước lưới của phép thử đỉnh (độ)")
    ap.add_argument("--tau", type=float, default=0.5,
                    help="báo drift khi |offset ước lượng| >= tau độ (điểm vận hành cố định, = 2 bước lưới)")
    ap.add_argument("--min-prominence", type=float, default=1.15,
                    help="phép thử đỉnh chỉ kết luận khi max(score)/median(score) >= giá trị này")
    ap.add_argument("--window", type=int, default=5,
                    help="số frame liên tiếp cộng dồn score cho phép thử đỉnh theo cửa sổ")
    ap.add_argument("--no-peak-test", action="store_true", help="bỏ phép thử đỉnh (chạy nhanh hơn ~5 lần)")
    ap.add_argument("--out-dir", default="results", help="thư mục ghi CSV")
    args = ap.parse_args()

    ds = dataset_type(args.data_root)
    tag = args.tag or Path(args.data_root).name
    frames = args.frames or list_frames(args.data_root)
    configs = build_configs(args.axes, args.rot_levels, args.trans_levels)
    grid = np.round(np.arange(-args.grid_max, args.grid_max + 1e-9, args.grid_step), 4)
    print(f"{args.data_root} ({ds}): {len(frames)} frame x {len(configs)} cấu hình")

    obj_rows, frame_rows, curves = [], [], {}
    t0 = time.perf_counter()
    for i, fid in enumerate(frames):
        fr = load_frame(args.data_root, fid)
        pts, calib, shape = fr["points"], fr["calib"], fr["image"].shape
        refs = object_refs(pts, calib, fr["labels"], shape)
        edge_pts = pts[lidar_depth_edges(pts, ring_ids(args.data_root, fid, pts)), :3]
        prox, _ = image_edge_proximity(fr["image"])
        s0 = edge_score(edge_pts, calib, prox)
        for axis, level in configs:
            c = perturb_physical(calib, ds, **perturb_kwargs(axis, level))
            for h in object_box_hits(refs, c, shape):
                obj_rows.append({"axis": axis, "level": level, "frame": fid, **h, "dist_bin": dist_bin(h["distance_m"])})
            s = edge_score(edge_pts, c, prox)
            row = {"axis": axis, "level": level, "frame": fid, "fov_ratio": fov_ratio(pts, c, shape),
                   "n_edge_points": len(edge_pts), "edge_score": s, "edge_score_norm": s / s0 if s0 else np.nan,
                   "est_yaw_deg": np.nan, "yaw_peak_prom": np.nan, "est_pitch_deg": np.nan, "pitch_peak_prom": np.nan}
            if not args.no_peak_test:
                for test in ("yaw", "pitch"):
                    est, scores = estimate_offset(edge_pts, c, prox, ds, test, grid)
                    curves.setdefault((axis, level, test), []).append(scores / np.nanmedian(scores))
                    row[f"est_{test}_deg"] = est
                    row[f"{test}_peak_prom"] = float(np.nanmax(scores) / np.nanmedian(scores)) if np.isfinite(est) else np.nan
            frame_rows.append(row)
        print(f"  [{i + 1}/{len(frames)}] {fid}: {len(refs)} object, {len(edge_pts)} điểm depth-edge "
              f"({time.perf_counter() - t0:.0f}s)")

    win_rows = [] if args.no_peak_test else window_rows(configs, frames, curves, grid, args.window)
    summary = summarize(configs, obj_rows, frame_rows, args.tau, args.min_prominence, win_rows)
    out = Path(args.out_dir)
    if win_rows:
        write_csv(out / f"calib_sweep_{tag}_windows.csv", win_rows)
    write_csv(out / f"calib_sweep_{tag}.csv", summary)
    write_csv(out / f"calib_sweep_{tag}_objects.csv", obj_rows)
    write_csv(out / f"calib_sweep_{tag}_frames.csv", frame_rows)
    print(f"Phép thử đỉnh: báo drift khi |offset| >= {args.tau}° và max/median score >= {args.min_prominence}; "
          f"cửa sổ {args.window} frame. Dòng 'none' = tỉ lệ báo động giả.")
    print(f"{'axis':>7} {'level':>6} {'fov':>6} {'hit':>6} {'<15m':>6} {'15-30':>6} {'>30m':>6} "
          f"{'shift':>6} {'edge':>6} {'detY':>5} {'detP':>5} {'wY':>5} {'wP':>5}")
    for r in summary:
        print(f"{r['axis']:>7} {r['level']:>6} {r['fov_ratio']:>6.3f} {r['hit_all']:>6.3f} {r['hit_<15m']:>6.3f} "
              f"{r['hit_15-30m']:>6.3f} {r['hit_>30m']:>6.3f} {r['shift_px_median']:>6.1f} {r['edge_norm_mean']:>6.3f} "
              f"{r.get('detect_yaw_test', np.nan):>5.2f} {r.get('detect_pitch_test', np.nan):>5.2f} "
              f"{r.get('detect_yaw_window', np.nan):>5.2f} {r.get('detect_pitch_window', np.nan):>5.2f}")
    print(f"-> {out / f'calib_sweep_{tag}.csv'} (+ _objects.csv, _frames.csv)")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
