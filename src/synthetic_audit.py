"""Tự động tìm lỗi cài sẵn trong một thư mục định dạng KITTI (mặc định data/synthetic) bằng các quy tắc + ngưỡng.

    python -m src.synthetic_audit
    python -m src.synthetic_audit --data-root data/kitti_mini --out results/audit_kitti.csv

Mỗi quy tắc so một frame với chính nó hoặc với TRUNG VỊ của các frame còn lại, nên không cần biết trước
"giá trị đúng". Kết quả: results/synthetic_audit.csv (mỗi dòng một phát hiện, kèm cách phát hiện).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from src.calib_qa import object_box_hits, object_refs
from src.calib_sweep import write_csv
from starter.datasets import list_frames, load_frame

AZ_BIN_DEG = 5


def az_hist(points: np.ndarray) -> np.ndarray:
    az = np.degrees(np.arctan2(points[:, 1], points[:, 0]))
    return np.histogram(az, bins=360 // AZ_BIN_DEG, range=(-180, 180))[0]


def sectors(bins: np.ndarray) -> list[tuple[int, int]]:
    """Gộp các bin liên tiếp thành dải góc [start, end] độ."""
    out = []
    for b in bins:
        lo = int(b * AZ_BIN_DEG - 180)
        if out and lo == out[-1][1]:
            out[-1] = (out[-1][0], lo + AZ_BIN_DEG)
        else:
            out.append((lo, lo + AZ_BIN_DEG))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Phát hiện frame lỗi: NaN, thiếu điểm, mất sector, nhảy timestamp, "
                                             "calib khác nhau, label lệch điểm")
    ap.add_argument("--data-root", default="data/synthetic")
    ap.add_argument("--out", default="results/synthetic_audit.csv")
    ap.add_argument("--min-count-ratio", type=float, default=0.97, help="báo nếu số điểm < tỉ lệ này x trung vị")
    ap.add_argument("--sector-ratio", type=float, default=0.5, help="báo bin azimuth < tỉ lệ này x trung vị bin đó")
    ap.add_argument("--gap-ratio", type=float, default=1.5, help="báo nếu khoảng cách timestamp > tỉ lệ này x trung vị")
    args = ap.parse_args()

    frames = list_frames(args.data_root)
    data = {f: load_frame(args.data_root, f) for f in frames}
    findings = []

    def add(check, frame, value, rule, verdict):
        findings.append({"check": check, "frame": frame, "value": value, "rule": rule, "verdict": verdict})

    # 1. Điểm không hợp lệ (NaN / Inf)
    for f, fr in data.items():
        bad = ~np.isfinite(fr["points"]).all(axis=1)
        if bad.any():
            add("NaN/Inf trong point cloud", f, f"{bad.sum()} điểm ({bad.mean():.2%}), cột: "
                f"{[c for c, n in zip('xyzi', (~np.isfinite(fr['points'])).sum(0)) if n]}",
                "số điểm không hữu hạn > 0", "LỖI")

    # 2. Số điểm tụt so với trung vị các frame
    counts = {f: int(np.isfinite(fr["points"]).all(axis=1).sum()) for f, fr in data.items()}
    med = np.median(list(counts.values()))
    for f, n in counts.items():
        if n < args.min_count_ratio * med:
            add("Thiếu điểm", f, f"{n} điểm = {n / med:.1%} trung vị ({med:.0f})",
                f"< {args.min_count_ratio:.0%} trung vị", "LỖI")

    # 3. Mất một dải azimuth (sector dropout / bị che): so từng bin 5° với trung vị bin đó qua các frame
    hists = {f: az_hist(fr["points"][np.isfinite(fr["points"]).all(axis=1)]) for f, fr in data.items()}
    med_hist = np.median(np.stack(list(hists.values())), axis=0)
    for f, h in hists.items():
        low = np.flatnonzero((med_hist > 20) & (h < args.sector_ratio * med_hist))
        for lo, hi in sectors(low):
            lost = int(sum(med_hist[(lo + 180) // AZ_BIN_DEG:(hi + 180) // AZ_BIN_DEG]
                           - h[(lo + 180) // AZ_BIN_DEG:(hi + 180) // AZ_BIN_DEG]))
            add("Mất sector azimuth", f, f"[{lo}°, {hi}°] (0° = phía trước, dương = bên trái), mất ~{lost} điểm",
                f"bin {AZ_BIN_DEG}° < {args.sector_ratio:.0%} trung vị bin đó", "LỖI")

    # 4. Timestamp: nhảy cóc hoặc đi lùi
    ts_path = Path(args.data_root) / "training" / "timestamps.txt"
    if ts_path.exists():
        ts = np.array([float(x) for x in ts_path.read_text().split()])
        dts = np.diff(ts)
        med_dt = np.median(dts)
        for i, dt in enumerate(dts):
            if dt <= 0 or dt > args.gap_ratio * med_dt:
                add("Timestamp nhảy cóc / đi lùi", f"{frames[i]}->{frames[i + 1]}",
                    f"dt = {dt:.3f} s (trung vị {med_dt:.3f} s) -> ~{dt / med_dt - 1:.0f} frame bị rơi",
                    f"dt <= 0 hoặc > {args.gap_ratio}x trung vị", "LỖI")

    # 5. Calibration có đổi giữa các frame không (cùng một xe thì không được đổi)
    ref = data[frames[0]]["calib"]
    for f, fr in data.items():
        c = fr["calib"]
        diff = max(np.abs(c.P2 - ref.P2).max(), np.abs(c.R0_rect - ref.R0_rect).max(),
                   np.abs(c.Tr_velo_to_cam - ref.Tr_velo_to_cam).max())
        add("Calib giống frame đầu", f, f"chênh lớn nhất {diff:.2e}", "chênh > 1e-6 là lỗi",
            "LỖI" if diff > 1e-6 else "OK")

    # 6. Label có khớp point cloud không: điểm trong 3D box phải chiếu vào 2D box
    for f, fr in data.items():
        refs = object_refs(fr["points"], fr["calib"], fr["labels"], fr["image"].shape)
        hits = object_box_hits(refs, fr["calib"], fr["image"].shape)
        worst = min((h["hit_ratio"] for h in hits), default=np.nan)
        add("Label khớp point cloud", f, f"{len(hits)}/{len(fr['labels'])} object có >= 10 điểm, box-hit thấp nhất "
            f"{worst:.1%}", "box-hit < 90% là lỗi", "LỖI" if (np.isnan(worst) or worst < 0.9) else "OK")

    write_csv(Path(args.out), findings)
    for r in findings:
        print(f"[{r['verdict']:3s}] {r['check']:28s} {r['frame']:16s} {r['value']}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
