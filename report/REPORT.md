# Báo cáo Day 6: Kiểm tra calibration LiDAR-camera bằng projection

- **Họ tên:** Phạm Minh Hiếu
- **MSSV:** 2A202602919
- **Lớp:** VinUni AI20K — Track 4
- **Link repo:** https://github.com/BenKei-19/PhamMinhHieu-2A202602919-Track4-Day21
- **Topic:** A — LiDAR-camera projection QA
- **Dataset:** data/kitti_mini (thí nghiệm chính), data/nuscenes_mini_subset (so sánh), data/synthetic (debug)
- **Các frame đã dùng:** KITTI: toàn bộ 20 frame; nuScenes: toàn bộ 80 keyframe

## 1. Claim

(Nháp CP1) Lệch yaw 1° làm hơn 10% điểm LiDAR của object rơi ra ngoài 2D box ở xe xa > 30 m, trong khi % điểm nằm trong FOV gần như không đổi.

## 2. Evidence

(đang làm)

## 3. Failure case

(đang làm)

## 4. Khuyến nghị nếu triển khai thật

(đang làm)

## 5. Cách chạy lại

```bash
python -m starter.data_health --data-root data/synthetic
```

## 6. Khai báo sử dụng AI

(đang làm)
