"""Shared utilities for VIBE demo scripts."""

import cv2
import numpy as np

from vibe_model import SMPL_SKELETON, detect_persons


def draw_skeleton(img, joints_2d, color=(0, 255, 0), thickness=2):
    """Draw SMPL skeleton on an image."""
    for i, j in SMPL_SKELETON:
        pt1 = (int(joints_2d[i, 0]), int(joints_2d[i, 1]))
        pt2 = (int(joints_2d[j, 0]), int(joints_2d[j, 1]))
        cv2.line(img, pt1, pt2, color, thickness)
    for idx in range(joints_2d.shape[0]):
        pt = (int(joints_2d[idx, 0]), int(joints_2d[idx, 1]))
        cv2.circle(img, pt, 3, (0, 0, 255), -1)
    return img


def detect_best_person(frame, detector):
    """Detect persons and return the highest-confidence bounding box, or None."""
    detections = detect_persons(frame, detector)
    if detections:
        best = max(detections, key=lambda d: d[4])
        return best[:4]
    return None


def crop_to_image_coords(j2d_crop, bbox, img_h, img_w, crop_img_size=224):
    """Transform 2D joint coordinates from crop space to original image space.

    Args:
        j2d_crop: (N, 2) array of joint coordinates in crop space.
        bbox: Bounding box [x1, y1, x2, y2] or None for center crop.
        img_h: Original image height.
        img_w: Original image width.
        crop_img_size: Size of the crop used for preprocessing.

    Returns:
        (N, 2) array of joint coordinates in original image space.
    """
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        crop_size = max(x2 - x1, y2 - y1) * 1.2
    else:
        crop_size = min(img_h, img_w)
        cx, cy = img_w / 2, img_h / 2

    scale = crop_size / crop_img_size
    j2d_orig = j2d_crop * scale
    j2d_orig[:, 0] += cx - crop_size / 2
    j2d_orig[:, 1] += cy - crop_size / 2
    return j2d_orig
