"""Vẽ biểu đồ từ CSV của src.calib_sweep (chạy sau khi đã có results/calib_sweep_kitti*.csv và _nusc*.csv).

    python -m src.make_figures
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Bảng màu: categorical theo thứ tự cố định, khoảng cách dùng thang 1 màu xanh nhạt -> đậm (gần -> xa).
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
CAT = ["#2a78d6", "#eb6834", "#1baf7a"]
DIST_RAMP = {"<15m": "#86b6ef", "15-30m": "#2a78d6", ">30m": "#104281"}
DATASET = {"kitti": ("KITTI (64 beam)", CAT[0]), "nusc": ("nuScenes (32 beam)", CAT[1])}

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2, "lines.markersize": 6,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
})


def read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def series(rows: list[dict], axis: str, col: str, scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """(level, giá trị) của một trục, có điểm gốc level=0 lấy từ dòng axis=none."""
    base = [r for r in rows if r["axis"] == "none"]
    rs = base + [r for r in rows if r["axis"] == axis]
    x = np.array([float(r["level"]) for r in rs]) * scale
    y = np.array([float(r[col]) for r in rs])
    return x, y


def fig_hit_vs_drift(kitti: list[dict], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    fov0 = float(next(r for r in kitti if r["axis"] == "none")["fov_ratio"])
    for ax, axis, scale, xlabel, title in [
        (axes[0], "yaw", 1.0, "Lệch yaw (độ)", "Xoay (yaw): vật càng XA càng hỏng"),
        (axes[1], "t_left", 100.0, "Lệch ngang t_left (cm)", "Dịch ngang: gần như không phụ thuộc khoảng cách"),
    ]:
        for name, color in DIST_RAMP.items():
            x, y = series(kitti, axis, f"hit_{name}", scale)
            ax.plot(x, 100 * y, marker="o", color=color, label=f"object {name}")
            if axis == "yaw":
                ax.annotate(f"{100 * y[-1]:.0f}%", (x[-1], 100 * y[-1]), xytext=(6, 0), textcoords="offset points",
                            va="center", fontsize=9, color=INK2)
        x, y = series(kitti, axis, "fov_ratio", scale)
        ax.plot(x, 100 * y / fov0, ls="--", color=MUTED, label="FOV ratio (so với lúc chưa lệch)")
        ax.set_xlabel(xlabel)
        ax.set_title(title, loc="left")
        ax.set_ylim(0, 108)
    axes[0].set_ylabel("% điểm LiDAR của object nằm trong 2D box")
    axes[0].legend(loc="lower left", fontsize=9)
    base = next(r for r in kitti if r["axis"] == "none")
    fig.suptitle(f"KITTI, {base['n_frames']} frame, {base['n_objects']} object: box-hit ratio theo mức lệch calibration",
                 x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_edge_score(data: dict[str, list[dict]], out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=True)
    groups = [(("yaw", "pitch", "roll"), 1.0, "Mức xoay (độ)"), (("t_fwd", "t_left", "t_up"), 100.0, "Mức dịch (cm)")]
    for row, (tag, rows) in enumerate(data.items()):
        for col, (axes_names, scale, xlabel) in enumerate(groups):
            ax = axes[row, col]
            for color, axis in zip(CAT, axes_names):
                x, y = series(rows, axis, "edge_norm_mean", scale)
                ax.plot(x, y, marker="o", color=color, label=axis)
            ax.axhline(1.0, color=MUTED, lw=1, ls=":")
            ax.set_title(f"{DATASET[tag][0]}", loc="left")
            ax.set_xlabel(xlabel)
            ax.legend(loc="lower left", fontsize=9)
        axes[row, 0].set_ylabel("Edge score / score khi chưa lệch")
    fig.suptitle("Edge alignment score (không cần label): giảm rõ với yaw, gần như mù với dịch t_fwd / t_up",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_detection(data: dict[str, list[dict]], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, axis in zip(axes, ("yaw", "pitch")):
        for tag, rows in data.items():
            label, color = DATASET[tag]
            x, y = series(rows, axis, f"detect_{axis}_test")
            ax.plot(x, 100 * y, marker="o", ls="--", color=color, label=f"{label}, 1 frame")
            x, y = series(rows, axis, f"detect_{axis}_window")
            ax.plot(x, 100 * y, marker="s", color=color, label=f"{label}, cửa sổ 5 frame")
        ax.set_xlabel(f"Lệch {axis} thật sự (độ)")
        ax.set_title(f"Phép thử đỉnh theo {axis}", loc="left")
        ax.set_ylim(-3, 105)
    axes[0].set_ylabel("% frame / cửa sổ bị báo drift\n(tại 0° = báo động giả)")
    axes[0].legend(loc="lower right", fontsize=9)
    axes[1].set_title("Phép thử đỉnh theo pitch (báo động giả cao)", loc="left")
    fig.suptitle("Phát hiện drift không cần label: báo khi |offset ước lượng| ≥ 0.5° và đỉnh score đủ nhô",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_datasets(data: dict[str, list[dict]], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for tag, rows in data.items():
        label, color = DATASET[tag]
        x, y = series(rows, "yaw", "hit_all")
        axes[0].plot(x, 100 * y, marker="o", color=color, label=label)
        x, y = series(rows, "yaw", "shift_px_median")
        axes[1].plot(x, y, marker="o", color=color, label=label)
        axes[1].annotate(f"{y[-1]:.0f} px", (x[-1], y[-1]), xytext=(6, 0), textcoords="offset points",
                         va="center", fontsize=9, color=INK2)
    axes[0].set_ylabel("% điểm của object trong 2D box")
    axes[0].set_ylim(0, 105)
    axes[0].set_title("Box-hit ratio", loc="left")
    axes[1].set_ylabel("Độ dời pixel trung vị của điểm object")
    axes[1].set_title("Độ dời pixel", loc="left")
    for ax in axes:
        ax.set_xlabel("Lệch yaw (độ)")
        ax.legend(loc="best", fontsize=9)
    fig.suptitle("Cùng một lệch yaw: nuScenes dời pixel gấp đôi (f lớn hơn) nhưng box-hit giảm chậm hơn",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Vẽ biểu đồ từ results/calib_sweep_<tag>.csv")
    ap.add_argument("--results", default="results", help="thư mục chứa CSV")
    args = ap.parse_args()
    res = Path(args.results)
    data = {tag: read(res / f"calib_sweep_{tag}.csv") for tag in ("kitti", "nusc")}
    figs = res / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    fig_hit_vs_drift(data["kitti"], figs / "sweep_01_box_hit_vs_drift_kitti.png")
    fig_edge_score(data, figs / "sweep_02_edge_score_vs_drift.png")
    fig_detection(data, figs / "sweep_03_drift_detection_rate.png")
    fig_datasets(data, figs / "sweep_04_kitti_vs_nuscenes_yaw.png")
    print(f"-> {figs}/sweep_0[1-4]_*.png")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
