#!/usr/bin/env python3
"""
VIBE Demo: Webcam → SMPL Parameters (Real-time)

Captures frames from a webcam, detects the primary person, and runs
VIBE temporal inference to estimate SMPL body model parameters.

Controls:
  q     - Quit and save results
  m     - Toggle mesh rendering
  space - Pause/resume

Usage:
  python demo_webcam.py --checkpoint resnet50_vibe_pw3d-2e05a122_20211201.pth
  python demo_webcam.py --checkpoint ckpt.pth --camera 0 --output webcam_output/ --render_mesh
"""

import argparse
import os
import time
from collections import deque

import cv2
import numpy as np
import torch

from vibe_model import (
    VIBE,
    preprocess_image, project_joints_2d, load_detector,
)
from demo_utils import (
    draw_skeleton, detect_best_person, crop_to_image_coords,
    SMPLRenderer, convert_vibe_to_smplx, OneEuroFilter,
)


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
    
    # Initialize mesh renderer if requested
    renderer = None
    render_mesh = args.render_mesh
    if render_mesh:
        faces = model.smpl.faces.cpu().numpy()
        renderer = SMPLRenderer(faces)
        print("[Renderer] Initialized SMPL mesh renderer")

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
    session_start = time.time()

    # Real-time smoothing filters (One-Euro)
    cam_filter = OneEuroFilter(dim=3, fps=30.0, min_cutoff=0.8, beta=0.01)
    bbox_filter = OneEuroFilter(dim=4, fps=30.0, min_cutoff=0.5, beta=0.005)

    print("\n[Controls] q=quit  space=pause  m=toggle mesh  i=toggle info")
    print("[Status] Warming up temporal buffer...\n")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break

            t_start = time.time()

            # Detect person
            bbox_raw = detect_best_person(frame, detector)

            # Smooth bounding box
            if bbox_raw is not None:
                bbox = bbox_filter(bbox_raw).tolist()
            else:
                bbox = bbox_raw

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
                    'vertices': output['vertices'][0, t_idx].cpu().numpy(),
                    'pose_aa': output['pose_axis_angle'][0, t_idx].cpu().numpy(),
                }
                all_results.append(result)

                # Smooth camera params for rendering
                cam_smooth = cam_filter(result['cam'])

                # Render mesh or skeleton
                if render_mesh and renderer is not None:
                    h, w = frame.shape[:2]
                    vis_frame = renderer.render_on_image(
                        vis_frame, result['vertices'], cam_smooth,
                        bbox, h, w)
                else:
                    # Project and draw skeleton
                    joints_3d = output['joints'][0, t_idx]
                    cam_t = torch.from_numpy(cam_smooth).float()
                    j2d_crop = project_joints_2d(joints_3d, cam_t, img_size=224)
                    j2d_np = j2d_crop.cpu().numpy()

                    # Transform crop coords → original image coords
                    h, w = frame.shape[:2]
                    j2d_orig = crop_to_image_coords(j2d_np, bbox, h, w)

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
                
                # Show render mode
                mode_text = 'Mesh' if render_mesh else 'Skeleton'
                cv2.putText(vis_frame, f'Mode: {mode_text}', (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                if len(feature_buffer) < seq_len:
                    cv2.putText(vis_frame, 'Warming up...', (10, 160),
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
        elif key == ord('m'):
            if renderer is not None:
                render_mesh = not render_mesh
                print(f"[Render mode: {'Mesh' if render_mesh else 'Skeleton'}]")
            else:
                print("[Warning] Renderer not initialised (start with --render_mesh)")

    cap.release()
    cv2.destroyAllWindows()

    # Save results
    if all_results and args.output:
        os.makedirs(args.output, exist_ok=True)
        # Estimate fps from capture
        elapsed = time.time() - session_start
        est_fps = frame_count / max(elapsed, 1e-6)

        pose_aa = np.stack([r['pose_aa'] for r in all_results])
        shapes  = np.stack([r['shape']   for r in all_results])
        cams    = np.stack([r['cam']     for r in all_results])

        smplx_dict = convert_vibe_to_smplx(
            pose_aa, shapes, cams, est_fps, gender=args.gender)
        npz_path = os.path.join(args.output, 'smplx_output.npz')
        np.savez(npz_path, **smplx_dict)
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
    parser.add_argument('--render_mesh', action='store_true',
                        help='Render SMPL mesh instead of skeleton')
    parser.add_argument('--gender', default='neutral',
                        choices=['neutral', 'male', 'female'],
                        help='Gender for SMPL-X output (default: neutral)')
    args = parser.parse_args()
    run_webcam(args)


if __name__ == '__main__':
    main()
