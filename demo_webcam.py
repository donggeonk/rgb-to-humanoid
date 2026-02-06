#!/usr/bin/env python3
"""
VIBE Demo: Webcam → SMPL Parameters (Real-time)

Captures frames from a webcam, detects the primary person, and runs
VIBE temporal inference to estimate SMPL body model parameters.

Controls:
  q     - Quit and save results
  s     - Toggle SMPL vertex output
  space - Pause/resume

Usage:
  python demo_webcam.py --checkpoint resnet50_vibe_pw3d-2e05a122_20211201.pth
  python demo_webcam.py --checkpoint ckpt.pth --camera 0 --output webcam_output/
"""

import argparse
import os
import time
from collections import deque

import cv2
import numpy as np
import torch

from vibe_model import (
    VIBE, SMPL_SKELETON,
    preprocess_image, project_joints_2d, detect_persons, load_detector,
)


def draw_skeleton(img, joints_2d, color=(0, 255, 0), thickness=2):
    """Draw SMPL skeleton on image."""
    for i, j in SMPL_SKELETON:
        pt1 = (int(joints_2d[i, 0]), int(joints_2d[i, 1]))
        pt2 = (int(joints_2d[j, 0]), int(joints_2d[j, 1]))
        cv2.line(img, pt1, pt2, color, thickness)
    for idx in range(joints_2d.shape[0]):
        pt = (int(joints_2d[idx, 0]), int(joints_2d[idx, 1]))
        cv2.circle(img, pt, 4, (0, 0, 255), -1)


def run_webcam(args):
    device = torch.device(args.device)

    # Load model
    print("Loading VIBE model...")
    model = VIBE.from_pretrained(args.checkpoint, device=device)
    seq_len = args.seq_len

    # Load detector
    detector = load_detector(device=args.device)

    # Open webcam
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {args.camera}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[Webcam] Resolution: {actual_w}x{actual_h}")

    # Rolling feature buffer for temporal inference
    feature_buffer = deque(maxlen=seq_len)
    bbox_buffer = deque(maxlen=seq_len)

    # Results storage
    all_results = []
    frame_count = 0
    paused = False
    show_info = True

    # For FPS computation
    fps_counter = deque(maxlen=30)

    print("\n[Controls] q=quit  space=pause  i=toggle info")
    print("[Status] Warming up temporal buffer...\n")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break

            t_start = time.time()

            # Detect person
            detections = detect_persons(frame, detector)
            if detections:
                best = max(detections, key=lambda d: d[4])
                bbox = best[:4]
            else:
                bbox = None

            # Preprocess and extract features
            img_tensor = preprocess_image(frame, bbox).unsqueeze(0).to(device)
            feat = model.extract_features(img_tensor).squeeze(0)  # (2048,)

            feature_buffer.append(feat)
            bbox_buffer.append(bbox)
            frame_count += 1

            # Run temporal inference when buffer has enough frames
            vis_frame = frame.copy()
            result = None

            if len(feature_buffer) >= seq_len:
                feats = torch.stack(list(feature_buffer)).unsqueeze(0)  # (1,T,2048)
                output = model(features=feats, run_smpl=True)

                # Use last frame's prediction
                t_idx = -1
                result = {
                    'rotmat': output['pred_rotmat'][0, t_idx].cpu().numpy(),
                    'shape': output['pred_shape'][0, t_idx].cpu().numpy(),
                    'cam': output['pred_cam'][0, t_idx].cpu().numpy(),
                    'joints': output['joints'][0, t_idx].cpu().numpy(),
                    'pose_aa': output['pose_axis_angle'][0, t_idx].cpu().numpy(),
                }
                all_results.append(result)

                # Project and draw skeleton
                joints_3d = output['joints'][0, t_idx]
                cam = output['pred_cam'][0, t_idx]
                j2d_crop = project_joints_2d(joints_3d, cam, img_size=224)
                j2d_np = j2d_crop.cpu().numpy()

                # Transform crop coords → original image coords
                h, w = frame.shape[:2]
                if bbox is not None:
                    x1, y1, x2, y2 = bbox
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    crop_size = max(x2 - x1, y2 - y1) * 1.2
                else:
                    crop_size = min(h, w)
                    cx, cy = w / 2, h / 2

                scale = crop_size / 224.0
                j2d_orig = j2d_np * scale
                j2d_orig[:, 0] += cx - crop_size / 2
                j2d_orig[:, 1] += cy - crop_size / 2

                draw_skeleton(vis_frame, j2d_orig)

                # Draw bounding box
                if bbox is not None:
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2),
                                  (255, 255, 0), 2)

            t_end = time.time()
            fps_counter.append(1.0 / max(t_end - t_start, 1e-6))

            # HUD
            if show_info:
                avg_fps = sum(fps_counter) / len(fps_counter)
                buf_status = f"Buffer: {len(feature_buffer)}/{seq_len}"
                cv2.putText(vis_frame, f'FPS: {avg_fps:.1f}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(vis_frame, buf_status, (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(vis_frame, f'Frame: {frame_count}', (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                if len(feature_buffer) < seq_len:
                    cv2.putText(vis_frame, 'Warming up...', (10, 130),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (0, 165, 255), 2)

            cv2.imshow('VIBE Webcam', vis_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
            print(f"[{'Paused' if paused else 'Resumed'}]")
        elif key == ord('i'):
            show_info = not show_info

    cap.release()
    cv2.destroyAllWindows()

    # Save results
    if all_results and args.output:
        os.makedirs(args.output, exist_ok=True)
        npz_path = os.path.join(args.output, 'webcam_smpl.npz')
        np.savez(
            npz_path,
            pred_rotmat=np.stack([r['rotmat'] for r in all_results]),
            pred_shape=np.stack([r['shape'] for r in all_results]),
            pred_cam=np.stack([r['cam'] for r in all_results]),
            joints_3d=np.stack([r['joints'] for r in all_results]),
            pose_axis_angle=np.stack([r['pose_aa'] for r in all_results]),
        )
        print(f"\n[Saved] {len(all_results)} frames → {npz_path}")

    print(f"Total frames captured: {frame_count}")
    print(f"Frames with predictions: {len(all_results)}")


def main():
    parser = argparse.ArgumentParser(
        description='VIBE: Webcam → SMPL Parameters (Real-time)')
    parser.add_argument('--checkpoint', required=True,
                        help='mmhuman3d VIBE checkpoint path')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera index (default: 0)')
    parser.add_argument('--output', default='webcam_output',
                        help='Output directory for saved results')
    parser.add_argument('--device',
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device (cuda/cpu)')
    parser.add_argument('--seq_len', type=int, default=16,
                        help='Temporal sequence length (default: 16)')
    args = parser.parse_args()
    run_webcam(args)


if __name__ == '__main__':
    main()
