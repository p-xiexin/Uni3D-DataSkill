#!/usr/bin/env bash
set -e

PYTHONPATH=src python tools/compare_kitti_raw_pykitti.py \
  --raw-root /home/huawei/pxx/data/kitti_raw \
  --depth-root /home/huawei/pxx/data/kitti_depth/data_depth_annotated \
  --sequence 2011_09_26_drive_0001_sync \
  --camera image_02 \
  --split train \
  --max-frames 10
