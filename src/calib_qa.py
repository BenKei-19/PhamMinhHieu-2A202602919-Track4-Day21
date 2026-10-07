"""Thư viện đo chất lượng calibration LiDAR-camera (topic A).

Ba nhóm metric, đều tính trên cùng một frame và cùng một calibration bị làm lệch:
  1. fov_ratio        : % điểm LiDAR (hữu hạn) chiếu được vào khung ảnh. Không cần label.
  2. object_box_hits  : với mỗi object có label, lấy các điểm LiDAR nằm trong 3D box (xác định bằng
                        calib GỐC), chiếu bằng calib LỆCH, đo % điểm còn rơi vào 2D box. Cần label.
  3. edge_score       : alignment score không cần label: điểm LiDAR ở biên độ sâu (depth edge) có
                        rơi gần biên ảnh (Canny) không. Ý tưởng theo Levinson & Thrun,
                        "Automatic Online Calibration of Cameras and Lasers", RSS 2013.

Mọi phép chiếu đều đi qua 2 hàm TODO(CP2) trong starter/projection.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from starter import nuscenes_io
from starter.datasets import dataset_type
from starter.kitti_io import KittiCalib, KittiObject
from starter.projection import cam_to_image, perturb_extrinsic, project_velo_to_image, velo_to_cam

VEHICLE = {"Car", "Van", "Truck", "Bus", "Trailer", "ConstructionVehicle", "Tram"}
VRU = {"Pedestrian", "Cyclist", "Person_sitting", "Bicycle", "Motorcycle"}  # người tham gia giao thông dễ bị tổn thương
DIST_BINS = [(0.0, 15.0, "<15m"), (15.0, 30.0, "15-30m"), (30.0, np.inf, ">30m")]


# ----------------------------------------------------------------------------- perturb theo trục vật lý
def perturb_physical(calib: KittiCalib, dataset: str, yaw_deg: float = 0.0, pitch_deg: float = 0.0,
                     roll_deg: float = 0.0, t_fwd: float = 0.0, t_left: float = 0.0, t_up: float = 0.0) -> KittiCalib:
    """Làm lệch extrinsic theo trục VẬT LÝ của xe (trước / trái / lên), dùng được cho cả KITTI và nuScenes.

    `perturb_extrinsic` của starter xoay quanh trục của chính LiDAR frame. KITTI LiDAR là x-trước, y-trái,
    nhưng nuScenes LiDAR là x-PHẢI, y-TRƯỚC. Nếu truyền thẳng `pitch_deg` cho nuScenes thì thực chất là
    xoay quanh trục y = trục dọc xe, tức là ROLL. Hàm này đổi trục để hai dataset so sánh được với nhau.
    Quy ước dấu: yaw dương = quay sang trái (quanh trục lên), theo quy tắc bàn tay phải.
    """
    if dataset == "nuscenes":
        # trục trước = +y_lidar, trục trái = -x_lidar, trục lên = +z_lidar
        return perturb_extrinsic(calib, roll_deg=-pitch_deg, pitch_deg=roll_deg, yaw_deg=yaw_deg,
                                 t_xyz_m=(-t_left, t_fwd, t_up))
    return perturb_extrinsic(calib, roll_deg=roll_deg, pitch_deg=pitch_deg, yaw_deg=yaw_deg,
                             t_xyz_m=(t_fwd, t_left, t_up))


# ----------------------------------------------------------------------------- metric 1: FOV ratio
def fov_ratio(points: np.ndarray, calib: KittiCalib, image_shape) -> float:
    """% điểm hữu hạn rơi vào khung ảnh (đã lọc depth <= 0 và NaN)."""
    n_finite = int(np.isfinite(points[:, :3]).all(axis=1).sum())
    _, _, mask = project_velo_to_image(points, calib, image_shape)
    return float(mask.sum() / max(n_finite, 1))


# ----------------------------------------------------------------------------- metric 2: box hit ratio
def points_in_box3d(points_cam: np.ndarray, obj: KittiObject, margin_m: float = 0.1,
                    ground_clear_m: float = 0.15) -> np.ndarray:
    """Mask (N,) các điểm (camera frame) nằm trong 3D box của label.

    Box KITTI: `location` là tâm ĐÁY, trục y camera hướng xuống, nên y_obj chạy từ -h (nóc) tới 0 (đáy).
    Bỏ lớp `ground_clear_m` sát đáy để không đếm nhầm điểm mặt đường dưới gầm xe.
    """
    h, w, l = obj.dimensions
    c, s = np.cos(obj.rotation_y), np.sin(obj.rotation_y)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])      # object -> camera
    local = (points_cam - obj.location) @ R                # camera -> object (nhân R^T theo hàng)
    return ((np.abs(local[:, 0]) <= l / 2 + margin_m) & (np.abs(local[:, 2]) <= w / 2 + margin_m)
            & (local[:, 1] <= -ground_clear_m) & (local[:, 1] >= -h - margin_m))


@dataclass
class ObjectRef:
    """Các điểm LiDAR thuộc một object, xác định một lần bằng calibration gốc."""
    obj: KittiObject
    points: np.ndarray       # (K, 3) velodyne frame
    uv_true: np.ndarray      # (K, 2) pixel khi chiếu bằng calib gốc
    distance_m: float


def object_refs(points: np.ndarray, calib_true: KittiCalib, labels: list[KittiObject], image_shape,
                min_points: int = 10) -> list[ObjectRef]:
    """Chọn điểm LiDAR thuộc từng object: nằm trong 3D box VÀ chiếu được vào ảnh với calib gốc."""
    pts = points[np.isfinite(points[:, :3]).all(axis=1), :3]
    pts_cam = velo_to_cam(pts, calib_true)
    refs = []
    for obj in labels:
        if obj.type in {"DontCare", "Misc"}:
            continue
        inside = points_in_box3d(pts_cam, obj)
        if inside.sum() < min_points:
            continue
        uv, _, mask = cam_to_image(pts_cam[inside], calib_true.P2, image_shape)
        if mask.sum() < min_points:
            continue
        refs.append(ObjectRef(obj, pts[inside][mask], uv, float(np.hypot(obj.location[0], obj.location[2]))))
    return refs


def object_box_hits(refs: list[ObjectRef], calib: KittiCalib, image_shape) -> list[dict]:
    """Với mỗi object: % điểm của nó rơi vào 2D box khi chiếu bằng `calib`, và độ dời pixel trung vị."""
    rows = []
    for ref in refs:
        uv, _, mask = project_velo_to_image(ref.points, calib, image_shape)
        x1, y1, x2, y2 = ref.obj.bbox
        in_box = (uv[:, 0] >= x1) & (uv[:, 0] <= x2) & (uv[:, 1] >= y1) & (uv[:, 1] <= y2)
        shift = np.linalg.norm(uv - ref.uv_true[mask], axis=1) if mask.any() else np.array([np.nan])
        rows.append({
            "type": ref.obj.type,
            "group": "vehicle" if ref.obj.type in VEHICLE else "vru" if ref.obj.type in VRU else "other",
            "distance_m": ref.distance_m,
            "n_points": len(ref.points),
            "hit_ratio": float(in_box.sum() / len(ref.points)),   # điểm bị đẩy ra ngoài ảnh cũng tính là trượt
            "shift_px": float(np.median(shift)),
            "box_w_deg": float(np.degrees((x2 - x1) / calib.P2[0, 0])),   # bề rộng góc của 2D box
        })
    return rows


def dist_bin(d: float) -> str:
    return next(name for lo, hi, name in DIST_BINS if lo <= d < hi)


# ----------------------------------------------------------------------------- metric 3: edge alignment score
def ring_ids(data_root: str, frame_id: str, points: np.ndarray) -> np.ndarray:
    """Chỉ số beam (ring) của từng điểm, để tìm điểm kề nhau trên cùng một tia quét.

    - nuScenes: file .pcd.bin có sẵn cột thứ 5 là ring index (starter bỏ cột này nên đọc lại ở đây).
    - KITTI: điểm được lưu lần lượt từng beam, azimuth tăng dần; mỗi lần azimuth nhảy ngược ~360° là sang beam mới.
    - Dữ liệu bị xáo thứ tự (data/synthetic): gom theo góc ngẩng 0.4° làm beam xấp xỉ.
    """
    if dataset_type(data_root) == "nuscenes":
        raw = np.fromfile(nuscenes_io.lidar_path(data_root, frame_id), dtype=np.float32).reshape(-1, 5)
        return raw[:, 4].astype(int)
    az = np.degrees(np.arctan2(points[:, 1], points[:, 0]))
    wraps = np.diff(az) < -180
    if 16 <= wraps.sum() <= 256:
        return np.r_[0, np.cumsum(wraps)]
    elev = np.degrees(np.arctan2(points[:, 2], np.linalg.norm(points[:, :2], axis=1)))
    return np.floor(np.nan_to_num(elev, nan=-999) / 0.4).astype(int)


def lidar_depth_edges(points: np.ndarray, rings: np.ndarray, min_jump_m: float = 0.5, rel_jump: float = 0.1,
                      max_gap_deg: float = 1.0, min_range_m: float = 2.0) -> np.ndarray:
    """Mask (N,) điểm ở biên tiền cảnh: điểm kề bên trên cùng beam ở xa hơn nó ít nhất max(0.5 m, 10%).

    Đây là biên vật thể nhìn từ LiDAR (xe, người, cột so với nền phía sau), không phụ thuộc calibration.
    """
    finite = np.isfinite(points[:, :3]).all(axis=1)
    rng = np.linalg.norm(points[:, :3], axis=1)
    az = np.degrees(np.arctan2(points[:, 1], points[:, 0]))
    edge = np.zeros(len(points), dtype=bool)
    valid = np.flatnonzero(finite & (rng > min_range_m))
    order = valid[np.lexsort((az[valid], rings[valid]))]   # sắp theo beam rồi theo azimuth
    a, b = order[:-1], order[1:]
    same = (rings[a] == rings[b]) & (np.abs(az[b] - az[a]) < max_gap_deg)
    jump = rng[b] - rng[a]
    thr_a = np.maximum(min_jump_m, rel_jump * rng[a])
    thr_b = np.maximum(min_jump_m, rel_jump * rng[b])
    edge[a[same & (jump > thr_a)]] = True    # điểm a gần hơn điểm kế tiếp -> a là biên tiền cảnh
    edge[b[same & (-jump > thr_b)]] = True
    return edge


def image_edge_proximity(image: np.ndarray, sigma_px: float = 3.0, canny_lo: int = 50,
                         canny_hi: int = 150) -> tuple[np.ndarray, np.ndarray]:
    """Bản đồ (H, W) trong [0, 1]: bằng 1 trên biên Canny, giảm theo exp(-d/sigma) khi xa biên."""
    gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges = cv2.Canny(gray, canny_lo, canny_hi)
    dist = cv2.distanceTransform((edges == 0).astype(np.uint8), cv2.DIST_L2, 3)
    return np.exp(-dist / sigma_px).astype(np.float32), edges


def edge_score(edge_points: np.ndarray, calib: KittiCalib, proximity: np.ndarray, min_points: int = 30) -> float:
    """Trung bình độ gần biên ảnh tại vị trí chiếu của các điểm depth-edge. Càng cao càng khớp."""
    uv, _, _ = project_velo_to_image(edge_points, calib, proximity.shape)
    if len(uv) < min_points:
        return float("nan")
    ui, vi = uv[:, 0].astype(int), uv[:, 1].astype(int)
    return float(proximity[vi, ui].mean())


def estimate_offset(edge_points: np.ndarray, calib: KittiCalib, proximity: np.ndarray, dataset: str,
                    axis: str, grid_deg: np.ndarray) -> tuple[float, np.ndarray]:
    """Phép thử "đỉnh" không cần calib tham chiếu: thử xoay thêm delta quanh `axis`, lấy delta cho score cao nhất.

    Calibration đúng thì score đạt cực đại tại delta = 0. Nếu calib đã lệch d độ thì cực đại ở delta ~ -d.
    """
    scores = np.array([edge_score(edge_points, perturb_physical(calib, dataset, **{f"{axis}_deg": float(d)}),
                                  proximity) for d in grid_deg])
    if np.all(np.isnan(scores)):
        return float("nan"), scores
    return float(grid_deg[int(np.nanargmax(scores))]), scores
