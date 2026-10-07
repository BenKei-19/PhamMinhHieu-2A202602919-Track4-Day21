"""Tạo ảnh demo (overlay) và ảnh failure case cho REPORT.

    python -m src.make_demos

Đầu ra trong results/figures/:
    demo_01_three_distances.png            overlay KITTI ở 3 khoảng cách (gần / trung bình / xa)
    demo_02_yaw_drift_000011.png           cùng một frame ở yaw 0° / 1° / 3°
    demo_03_nuscenes_day_night.png         overlay nuScenes ban ngày và ban đêm
    fail_01_yaw2deg_far_car_fov_blind.png  lệch yaw 2°: xe xa mất điểm trong box, FOV ratio không đổi
    fail_02_edge_score_weak_on_pitch.png   edge score có đỉnh nhọn với yaw nhưng đỉnh bẹt với pitch
    fail_03_time_sync_noego_0103_008.png   không bù ego-motion 36 ms: điểm trượt khỏi xe gần
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.calib_qa import (estimate_offset, fov_ratio, image_edge_proximity, lidar_depth_edges,  # noqa: E402
                          object_box_hits, object_refs, perturb_physical, ring_ids)
from src.make_figures import CAT, INK2, MUTED  # noqa: E402  (dùng chung style biểu đồ)
from starter.datasets import list_frames, load_frame  # noqa: E402
from starter.projection import draw_box2d, overlay_points, project_velo_to_image  # noqa: E402

KITTI, NUSC = "data/kitti_mini", "data/nuscenes_mini_subset"
OUT = Path("results/figures")
OBJ_COLOR = (255, 0, 255)   # BGR magenta: điểm LiDAR thuộc object (nằm trong 3D box)
BOX_COLOR = (0, 255, 0)


def header(img: np.ndarray, text: str, height: int = 34) -> np.ndarray:
    bar = np.full((height, img.shape[1], 3), 30, np.uint8)
    cv2.putText(bar, text, (10, height - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return np.vstack([bar, img])


def full_overlay(fr: dict, calib) -> np.ndarray:
    uv, depth, _ = project_velo_to_image(fr["points"], calib, fr["image"].shape)
    vis = overlay_points(fr["image"], uv, depth)
    for obj in fr["labels"]:
        vis = draw_box2d(vis, obj.bbox, color=BOX_COLOR, label=obj.type)
    return vis


def object_panel(fr: dict, calib, refs, crop, scale: int = 3) -> tuple[np.ndarray, list[dict]]:
    """Crop phóng to: điểm nền màu theo depth (nhỏ), điểm của object màu tím (to), 2D box xanh lá."""
    x1, y1, x2, y2 = crop
    img = cv2.resize(fr["image"][y1:y2, x1:x2], None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    img = (img * 0.75).astype(np.uint8)
    uv, depth, _ = project_velo_to_image(fr["points"], calib, fr["image"].shape)
    keep = (uv[:, 0] >= x1) & (uv[:, 0] < x2) & (uv[:, 1] >= y1) & (uv[:, 1] < y2)
    img = overlay_points(img, (uv[keep] - [x1, y1]) * scale, depth[keep], radius=1)
    hits = object_box_hits(refs, calib, fr["image"].shape)
    for ref, h in zip(refs, hits):
        bx1, by1, bx2, by2 = ref.obj.bbox
        img = draw_box2d(img, ((bx1 - x1) * scale, (by1 - y1) * scale, (bx2 - x1) * scale, (by2 - y1) * scale),
                         color=BOX_COLOR)
        cv2.putText(img, f"{ref.obj.type} {ref.distance_m:.0f}m: {h['hit_ratio']:.0%} in box",
                    (int((bx1 - x1) * scale), max(18, int((by1 - y1) * scale) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        ouv, _, _ = project_velo_to_image(ref.points, calib, fr["image"].shape)
        for u, v in ((ouv - [x1, y1]) * scale).astype(int):
            cv2.circle(img, (int(u), int(v)), 3, OBJ_COLOR, -1)
    return img, hits


def crop_around(refs, image_shape, pad_x: int = 70, pad_y: int = 40) -> tuple[int, int, int, int]:
    b = np.array([r.obj.bbox for r in refs])
    h, w = image_shape[:2]
    return (int(max(0, b[:, 0].min() - pad_x)), int(max(0, b[:, 1].min() - pad_y)),
            int(min(w, b[:, 2].max() + pad_x)), int(min(h, b[:, 3].max() + pad_y)))


# ----------------------------------------------------------------------------- demo
def demo_three_distances() -> None:
    panels = []
    for fid, tag in [("000019", "GAN"), ("000011", "TRUNG BINH"), ("000004", "XA")]:
        fr = load_frame(KITTI, fid)
        d = sorted(np.hypot(o.location[0], o.location[2]) for o in fr["labels"] if o.type != "DontCare")
        panels.append(header(full_overlay(fr, fr["calib"]),
                             f"KITTI {fid} ({tag}): object o {d[0]:.0f}-{d[-1]:.0f} m | mau diem: do = gan, xanh = xa"))
    cv2.imwrite(str(OUT / "demo_01_three_distances.png"), np.vstack(panels))


def demo_yaw_drift(fid: str = "000011") -> None:
    fr = load_frame(KITTI, fid)
    refs = object_refs(fr["points"], fr["calib"], fr["labels"], fr["image"].shape)
    panels = []
    for yaw in (0.0, 1.0, 3.0):
        c = perturb_physical(fr["calib"], "kitti", yaw_deg=yaw)
        hits = object_box_hits(refs, c, fr["image"].shape)
        fov = fov_ratio(fr["points"], c, fr["image"].shape)
        panels.append(header(full_overlay(fr, c), f"KITTI {fid}, yaw +{yaw:.0f} deg: box-hit trung binh "
                                                  f"{np.mean([h['hit_ratio'] for h in hits]):.0%}, FOV ratio {fov:.1%}"))
    cv2.imwrite(str(OUT / f"demo_02_yaw_drift_{fid}.png"), np.vstack(panels))


def demo_nuscenes() -> None:
    panels = []
    for fid, tag in [("scene-0103_010", "ban ngay"), ("scene-1094_020", "ban dem, sau mua")]:
        fr = load_frame(NUSC, fid)
        vis = cv2.resize(full_overlay(fr, fr["calib"]), None, fx=0.6, fy=0.6)
        panels.append(header(vis, f"nuScenes {fid} ({tag}), LiDAR 32 beam"))
    cv2.imwrite(str(OUT / "demo_03_nuscenes_day_night.png"), np.hstack(panels))


# ----------------------------------------------------------------------------- failure
def fail_far_car(fid: str = "000004", yaw: float = 2.0) -> None:
    fr = load_frame(KITTI, fid)
    refs = object_refs(fr["points"], fr["calib"], fr["labels"], fr["image"].shape)
    crop = crop_around(refs, fr["image"].shape)
    panels = []
    for y in (0.0, yaw):
        c = perturb_physical(fr["calib"], "kitti", yaw_deg=y)
        img, _ = object_panel(fr, c, refs, crop)
        fov = fov_ratio(fr["points"], c, fr["image"].shape)
        panels.append(header(img, f"yaw +{y:.0f} deg | FOV ratio = {fov:.2%}"))
    gap = np.full((panels[0].shape[0], 12, 3), 255, np.uint8)
    out = header(np.hstack([panels[0], gap, panels[1]]),
                 f"FAIL 01 - KITTI {fid}: lech yaw {yaw:.0f} deg day diem cua xe xa ra khoi 2D box, "
                 f"nhung FOV ratio KHONG doi (diem tim = diem trong 3D box)", 40)
    cv2.imwrite(str(OUT / "fail_01_yaw2deg_far_car_fov_blind.png"), out)


def peak_width(grid: np.ndarray, curve: np.ndarray) -> float:
    """Độ rộng (độ) của vùng quanh đỉnh mà score còn cao hơn nửa độ nhô (max + median) / 2."""
    half = (np.nanmax(curve) + np.nanmedian(curve)) / 2
    i = j = int(np.nanargmax(curve))
    while i > 0 and curve[i - 1] >= half:
        i -= 1
    while j < len(curve) - 1 and curve[j + 1] >= half:
        j += 1
    return float(grid[j] - grid[i])


def fail_edge_pitch(grid=np.round(np.arange(-4, 4.001, 0.25), 2), drift: float = 1.0) -> None:
    """Đường cong edge score trung bình 20 frame KITTI: tìm yaw khi calib lệch yaw vs tìm pitch khi lệch pitch."""
    curves = {"yaw": [], "pitch": []}
    for fid in list_frames(KITTI):
        fr = load_frame(KITTI, fid)
        pts = fr["points"]
        edge_pts = pts[lidar_depth_edges(pts, ring_ids(KITTI, fid, pts)), :3]
        prox, _ = image_edge_proximity(fr["image"])
        for axis in curves:
            c = perturb_physical(fr["calib"], "kitti", **{f"{axis}_deg": drift})
            _, s = estimate_offset(edge_pts, c, prox, "kitti", axis, grid)
            curves[axis].append(s / np.nanmedian(s))

    fr = load_frame(KITTI, "000011")
    pts = fr["points"]
    edge_pts = pts[lidar_depth_edges(pts, ring_ids(KITTI, "000011", pts)), :3]
    _, canny = image_edge_proximity(fr["image"])
    vis = (fr["image"] * 0.45).astype(np.uint8)
    vis[canny > 0] = (200, 200, 200)
    uv, _, _ = project_velo_to_image(edge_pts, fr["calib"], fr["image"].shape)
    for u, v in uv.astype(int):
        cv2.circle(vis, (int(u), int(v)), 2, (0, 200, 255), -1)

    fig = plt.figure(figsize=(12, 6.6))
    ax0 = fig.add_axes([0.04, 0.47, 0.92, 0.45])
    ax0.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    ax0.set_axis_off()
    ax0.set_title("Điểm depth-edge của LiDAR (vàng) trên biên Canny (xám), KITTI 000011: chúng xếp thành CỘT DỌC "
                  "trên biên đứng của cột, cây, người", loc="left", fontsize=10)
    ax1 = fig.add_axes([0.07, 0.08, 0.86, 0.30])
    widths = {}
    for color, axis in zip(CAT, ("yaw", "pitch")):
        m = np.nanmean(curves[axis], axis=0)
        widths[axis] = peak_width(grid, m)
        ax1.plot(grid, m, color=color, marker="o", ms=4,
                 label=f"calib lệch {axis} +{drift:.0f}°, tìm theo {axis}: đỉnh ở {grid[np.nanargmax(m)]:+.2f}°, "
                       f"cao {np.nanmax(m):.2f}, rộng {widths[axis]:.2f}°")
    ax1.axvline(-drift, color=MUTED, ls="--", lw=1)
    ax1.annotate(f"đỉnh đúng phải ở {-drift:+.0f}°", (-drift, ax1.get_ylim()[1]), xytext=(4, -12),
                 textcoords="offset points", fontsize=9, color=INK2)
    ax1.set_xlabel("Offset thử thêm (độ)")
    ax1.set_ylabel("score / median (TB 20 frame)")
    ax1.legend(loc="lower left", fontsize=9)
    fig.suptitle("FAIL 02 - Edge score: đỉnh yaw nhọn, đỉnh pitch thấp và bẹt -> trên 1 frame, pitch bị nhiễu lấn át",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.savefig(OUT / "fail_02_edge_score_weak_on_pitch.png", dpi=120)
    plt.close(fig)
    print(f"fail_02: độ rộng đỉnh yaw {widths['yaw']:.2f}°, pitch {widths['pitch']:.2f}°")


def fail_time_sync(fid: str = "scene-0103_008") -> None:
    fr = load_frame(NUSC, fid)
    noego = load_frame(NUSC, fid, use_ego_motion=False)["calib"]
    refs = object_refs(fr["points"], fr["calib"], fr["labels"], fr["image"].shape)
    worst = sorted(zip(object_box_hits(refs, noego, fr["image"].shape), refs), key=lambda t: t[0]["hit_ratio"])[0][1]
    crop = crop_around([worst], fr["image"].shape, pad_x=120, pad_y=60)
    dt = (fr["timestamp_camera_us"] - fr["timestamp_lidar_us"]) / 1000
    panels = []
    for calib, tag in [(fr["calib"], "CO bu ego-motion"), (noego, "KHONG bu ego-motion")]:
        img, _ = object_panel(fr, calib, [worst], crop, scale=2)
        panels.append(header(img, tag))
    gap = np.full((panels[0].shape[0], 12, 3), 255, np.uint8)
    out = header(np.hstack([panels[0], gap, panels[1]]),
                 f"FAIL 03 - nuScenes {fid}: camera chup lech LiDAR {abs(dt):.0f} ms, xe di ~0.33 m -> "
                 f"diem cua vat gan truot khoi box", 40)
    cv2.imwrite(str(OUT / "fail_03_time_sync_noego_0103_008.png"), out)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    demo_three_distances()
    demo_yaw_drift()
    demo_nuscenes()
    fail_far_car()
    fail_edge_pitch()
    fail_time_sync()
    print(f"-> {OUT}/demo_0*.png, fail_0*.png")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
