#!/usr/bin/env python3
"""
VIBE Demo: MP4 Video → SMPL Parameters

Processes a video file, detects persons per frame, and runs VIBE to
produce per-frame SMPL parameters (pose, shape, camera).

Outputs:
  - smpl_output.npz: SMPL parameters for each frame
  - overlay.mp4 (optional): visualization with skeleton overlay

Usage:
  python demo_video.py --video input.mp4 --checkpoint resnet50_vibe_pw3d-2e05a122_20211201.pth
  python demo_video.py --video input.mp4 --checkpoint ckpt.pth --output out/ --visualize
"""

import argparse
import os
import time

import cv2
import numpy as np
import torch

from vibe_model import (
    VIBE, SMPL_JOINT_NAMES,
    preprocess_image, project_joints_2d, load_detector,
)
from demo_utils import draw_skeleton, detect_best_person, crop_to_image_coords


def extract_frames(video_path):
    """Read all frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    print(f"[Video] Loaded {len(frames)} frames at {fps:.1f} FPS")
    return frames, fps


def process_video(args):
    device = torch.device(args.device)

    # Load model
    print("Loading VIBE model...")
    model = VIBE.from_pretrained(args.checkpoint, device=device)

    # Load detector
    detector = load_detector(device=args.device)

    # Read video
    frames, fps = extract_frames(args.video)
    num_frames = len(frames)
    seq_len = args.seq_len

    # ---- Phase 1: Detect persons and extract features ----
    print("Extracting features...")
    all_features = []
    all_bboxes = []
    t0 = time.time()

    batch_size = args.batch_size
    for start in range(0, num_frames, batch_size):
        end = min(start + batch_size, num_frames)
        batch_imgs = []
        batch_bboxes = []

        for i in range(start, end):
            bbox = detect_best_person(frames[i], detector)
            batch_bboxes.append(bbox)
            img_tensor = preprocess_image(frames[i], bbox)
            batch_imgs.append(img_tensor)

        batch = torch.stack(batch_imgs).to(device)
        feats = model.extract_features(batch)  # (B, 2048)
        all_features.append(feats.cpu())
        all_bboxes.extend(batch_bboxes)

    all_features = torch.cat(all_features, dim=0)  # (N, 2048)
    feat_time = time.time() - t0
    print(f"Feature extraction: {feat_time:.1f}s "
          f"({num_frames / feat_time:.1f} FPS)")

    # ---- Phase 2: Temporal inference ----
    print("Running temporal inference...")
    t0 = time.time()

    all_rotmats = []
    all_shapes = []
    all_cams = []
    all_joints = []
    all_verts = []
    all_pose_aa = []

    stride = args.stride
    # Process overlapping windows
    window_starts = list(range(0, max(1, num_frames - seq_len + 1), stride))
    if not window_starts or window_starts[-1] + seq_len < num_frames:
        window_starts.append(max(0, num_frames - seq_len))

    # Track which frames have predictions (use center frame of each window)
    frame_predictions = [None] * num_frames

    for ws in window_starts:
        we = min(ws + seq_len, num_frames)
        feats_window = all_features[ws:we]

        # Pad if sequence is shorter than seq_len
        if feats_window.shape[0] < seq_len:
            pad_len = seq_len - feats_window.shape[0]
            feats_window = torch.cat([
                feats_window,
                feats_window[-1:].expand(pad_len, -1)
            ], dim=0)

        feats_window = feats_window.unsqueeze(0).to(device)  # (1, T, 2048)
        output = model(features=feats_window, run_smpl=True)

        actual_len = we - ws
        for t in range(actual_len):
            frame_idx = ws + t
            if frame_predictions[frame_idx] is None:
                frame_predictions[frame_idx] = {
                    'rotmat': output['pred_rotmat'][0, t].cpu(),
                    'shape': output['pred_shape'][0, t].cpu(),
                    'cam': output['pred_cam'][0, t].cpu(),
                    'joints': output['joints'][0, t].cpu(),
                    'vertices': output['vertices'][0, t].cpu(),
                    'pose_aa': output['pose_axis_angle'][0, t].cpu(),
                }

    # Collect results
    for i in range(num_frames):
        pred = frame_predictions[i]
        if pred is None:
            pred = frame_predictions[max(0, i - 1)]  # fallback
        all_rotmats.append(pred['rotmat'])
        all_shapes.append(pred['shape'])
        all_cams.append(pred['cam'])
        all_joints.append(pred['joints'])
        all_verts.append(pred['vertices'])
        all_pose_aa.append(pred['pose_aa'])

    all_rotmats = torch.stack(all_rotmats).numpy()
    all_shapes = torch.stack(all_shapes).numpy()
    all_cams = torch.stack(all_cams).numpy()
    all_joints = torch.stack(all_joints).numpy()
    all_verts = torch.stack(all_verts).numpy()
    all_pose_aa = torch.stack(all_pose_aa).numpy()

    infer_time = time.time() - t0
    print(f"Temporal inference: {infer_time:.1f}s "
          f"({num_frames / infer_time:.1f} FPS)")

    # ---- Phase 3: Save results ----
    os.makedirs(args.output, exist_ok=True)
    npz_path = os.path.join(args.output, 'smpl_output.npz')
    np.savez(
        npz_path,
        pred_rotmat=all_rotmats,       # (N, 24, 3, 3)
        pred_shape=all_shapes,          # (N, 10)
        pred_cam=all_cams,              # (N, 3)
        joints_3d=all_joints,           # (N, 24, 3)
        vertices=all_verts,             # (N, 6890, 3)
        pose_axis_angle=all_pose_aa,    # (N, 24, 3)
        bboxes=np.array([b if b is not None else [0, 0, 0, 0]
                         for b in all_bboxes]),
        fps=fps,
    )
    print(f"[Saved] SMPL parameters → {npz_path}")

    # ---- Phase 4: Visualization ----
    if args.visualize:
        print("Rendering visualization...")
        out_video = os.path.join(args.output, 'overlay.mp4')
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(out_video, fourcc, fps, (w, h))

        for i in range(num_frames):
            vis = frames[i].copy()
            bbox = all_bboxes[i]
            joints_3d = torch.from_numpy(all_joints[i])
            cam = torch.from_numpy(all_cams[i])

            # Project joints to 2D in crop space
            j2d_crop = project_joints_2d(joints_3d, cam, img_size=224)
            j2d_np = j2d_crop.numpy()

            # Transform from crop space to original image space
            j2d_orig = crop_to_image_coords(j2d_np, bbox, h, w)

            draw_skeleton(vis, j2d_orig)

            # Frame info
            cv2.putText(vis, f'Frame {i}/{num_frames}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            writer.write(vis)

        writer.release()
        print(f"[Saved] Visualization → {out_video}")

    print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description='VIBE: Video → SMPL Parameters')
    parser.add_argument('--video', required=True, help='Input video path')
    parser.add_argument('--checkpoint', required=True,
                        help='mmhuman3d VIBE checkpoint path')
    parser.add_argument('--output', default='output',
                        help='Output directory (default: output)')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device (cuda/cpu)')
    parser.add_argument('--seq_len', type=int, default=16,
                        help='Temporal sequence length (default: 16)')
    parser.add_argument('--stride', type=int, default=16,
                        help='Sliding window stride (default: 16, use 1 for best quality)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for feature extraction')
    parser.add_argument('--visualize', action='store_true',
                        help='Render skeleton overlay video')
    args = parser.parse_args()
    process_video(args)


if __name__ == '__main__':
    main()
