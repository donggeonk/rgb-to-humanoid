"""GVHMR inference demo – video/webcam → SMPL parameters.

Self-contained pipeline. No hmr4d dependencies.
Requires: torch, cv2, numpy, ultralytics, tqdm

Usage:
    python gvhmr_demo.py --video path/to/video.mp4 --out results.npz
    python gvhmr_demo.py --webcam --out results.npz
"""

import os
import sys
import argparse
import math
import numpy as np
import torch
import cv2
from tqdm import tqdm

# Add this file's directory to path for sibling imports
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from hmr2_model import load_hmr2, crop_and_resize, preprocess_image
from vitpose_model import build_vitpose
from gvhmr_model import (
    GVHMRModel,
    estimate_K,
    compute_bbox_info_bedlam,
    normalize_kp2d,
    compute_cam_angvel,
    static_cam_angvel,
    gaussian_smooth,
)

# ============================================================
# Keypoints from ViTPose heatmaps (mmpose-style)
# ============================================================


def _get_max_preds(heatmaps):
    N, K, H, W = heatmaps.shape
    flat = heatmaps.reshape(N, K, -1)
    idx = np.argmax(flat, 2).reshape(N, K, 1)
    maxvals = np.amax(flat, 2).reshape(N, K, 1)
    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
    preds[..., 0] = preds[..., 0] % W
    preds[..., 1] = preds[..., 1] // W
    preds = np.where(np.tile(maxvals, (1, 1, 2)) > 0.0, preds, -1)
    return preds, maxvals


def _post_dark_udp(coords, batch_heatmaps, kernel=3):
    B, K, H, W = batch_heatmaps.shape
    N = coords.shape[0]
    for heatmaps in batch_heatmaps:
        for heatmap in heatmaps:
            cv2.GaussianBlur(heatmap, (kernel, kernel), 0, heatmap)
    np.clip(batch_heatmaps, 0.001, 50, batch_heatmaps)
    np.log(batch_heatmaps, batch_heatmaps)
    pad = np.pad(
        batch_heatmaps, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="edge"
    ).flatten()
    index = coords[..., 0] + 1 + (coords[..., 1] + 1) * (W + 2)
    index += (W + 2) * (H + 2) * np.arange(0, B * K).reshape(-1, K)
    index = index.astype(int).reshape(-1, 1)
    i_ = pad[index]
    ix1 = pad[index + 1]
    iy1 = pad[index + W + 2]
    ix1y1 = pad[index + W + 3]
    ix1_y1_ = pad[index - W - 3]
    ix1_ = pad[index - 1]
    iy1_ = pad[index - 2 - W]
    dx = 0.5 * (ix1 - ix1_)
    dy = 0.5 * (iy1 - iy1_)
    derivative = np.concatenate([dx, dy], axis=1).reshape(N, K, 2, 1)
    dxx = ix1 - 2 * i_ + ix1_
    dyy = iy1 - 2 * i_ + iy1_
    dxy = 0.5 * (ix1y1 - ix1 - iy1 + i_ + i_ - ix1_ - iy1_ + ix1_y1_)
    hessian = np.concatenate([dxx, dxy, dxy, dyy], axis=1).reshape(N, K, 2, 2)
    hessian = np.linalg.inv(hessian + np.finfo(np.float32).eps * np.eye(2))
    coords -= np.einsum("ijmn,ijnk->ijmk", hessian, derivative).squeeze()
    return coords


def keypoints_from_heatmaps(heatmaps, center, scale, use_udp=True):
    heatmaps = heatmaps.copy()
    N, K, H, W = heatmaps.shape
    preds, maxvals = _get_max_preds(heatmaps)
    if use_udp:
        preds = _post_dark_udp(preds, heatmaps)
    else:
        for n in range(N):
            for k in range(K):
                px, py = int(preds[n, k, 0]), int(preds[n, k, 1])
                if 1 < px < W - 1 and 1 < py < H - 1:
                    diff = np.array(
                        [
                            heatmaps[n, k, py, px + 1] - heatmaps[n, k, py, px - 1],
                            heatmaps[n, k, py + 1, px] - heatmaps[n, k, py - 1, px],
                        ]
                    )
                    preds[n, k] += np.sign(diff) * 0.25
    scale_px = scale * 200.0
    sx = scale_px[:, 0:1] / (W - 1.0 if use_udp else W)
    sy = scale_px[:, 1:2] / (H - 1.0 if use_udp else H)
    target = np.zeros_like(preds)
    target[..., 0] = (
        preds[..., 0] * sx[:, None, 0] + center[:, None, 0] - scale_px[:, None, 0] * 0.5
    )
    target[..., 1] = (
        preds[..., 1] * sy[:, None, 0] + center[:, None, 1] - scale_px[:, None, 1] * 0.5
    )
    return target, maxvals


# ============================================================
# YOLO person detector
# ============================================================


class PersonDetector:
    def __init__(self, ckpt_path, device="cuda", conf_thr=0.5):
        from ultralytics import YOLO

        self.model = YOLO(ckpt_path)
        self.model.to(device)
        self.device = device
        self.conf_thr = conf_thr

    @torch.no_grad()
    def detect_frame(self, frame_bgr: np.ndarray):
        results = self.model(frame_bgr, classes=[0], verbose=False, conf=self.conf_thr)
        boxes = []
        for r in results:
            if r.boxes is None:
                continue
            for xyxy in r.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = xyxy
                boxes.append(
                    np.array(
                        [(x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1)],
                        dtype=np.float32,
                    )
                )
        return boxes


def select_primary_person(dets_per_frame: list):
    F = len(dets_per_frame)
    track = np.zeros((F, 3), dtype=np.float32)
    first = next((i for i, d in enumerate(dets_per_frame) if len(d) > 0), None)
    if first is None:
        return track
    track[first] = max(dets_per_frame[first], key=lambda b: b[2])
    for i in range(first + 1, F):
        dets = dets_per_frame[i]
        if not dets:
            track[i] = track[i - 1]
        else:
            prev = track[i - 1, :2]
            track[i] = dets[np.argmin([np.linalg.norm(d[:2] - prev) for d in dets])]
    for i in range(first - 1, -1, -1):
        track[i] = track[i + 1]
    return track


# ============================================================
# Visual Odometry (SimpleVO)
# ============================================================


class SimpleVO:
    def compute(self, frames: np.ndarray, step: int = 8) -> np.ndarray:
        F, H, W = frames.shape[:3]
        fl = math.sqrt(W**2 + H**2)
        K = np.array([[fl, 0, W / 2], [0, fl, H / 2], [0, 0, 1]], dtype=np.float64)
        sift = cv2.SIFT_create()
        flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
        idxs = np.arange(0, F, step)
        if idxs[-1] != F - 1:
            idxs = np.append(idxs, F - 1)
        gray0 = cv2.cvtColor(frames[idxs[0]], cv2.COLOR_BGR2GRAY)
        kp0, des0 = sift.detectAndCompute(gray0, None)
        R_list = {idxs[0]: np.eye(3)}
        for t_idx in range(1, len(idxs)):
            i = idxs[t_idx]
            gray1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            kp1, des1 = sift.detectAndCompute(gray1, None)
            R_prev = R_list[idxs[t_idx - 1]]
            if (
                des0 is not None
                and des1 is not None
                and len(kp0) >= 8
                and len(kp1) >= 8
            ):
                matches = flann.knnMatch(
                    des0.astype(np.float32), des1.astype(np.float32), k=2
                )
                good = [m for m, n in matches if m.distance < 0.7 * n.distance]
                if len(good) >= 8:
                    pts0, pts1 = (
                        np.float32([kp0[m.queryIdx].pt for m in good]),
                        np.float32([kp1[m.trainIdx].pt for m in good]),
                    )
                    E, mask = cv2.findEssentialMat(
                        pts0, pts1, K, method=cv2.RANSAC, prob=0.999, threshold=1.0
                    )
                    if E is not None and E.shape == (3, 3):
                        _, R_delta, _, _ = cv2.recoverPose(E, pts0, pts1, K, mask=mask)
                        R_prev = R_delta @ R_prev
            R_list[i], kp0, des0 = R_prev, kp1, des1
        R_w2c = np.zeros((F, 3, 3), dtype=np.float32)
        sampled = sorted(R_list.keys())
        for j in range(len(sampled) - 1):
            i0, i1 = sampled[j], sampled[j + 1]
            R0, R1 = R_list[i0], R_list[i1]
            for k in range(i0, i1 + 1):
                alpha = (k - i0) / (i1 - i0) if i1 > i0 else 0
                R_interp = (1 - alpha) * R0 + alpha * R1
                U, _, Vt = np.linalg.svd(R_interp)
                R_w2c[k] = U @ Vt
        return R_w2c


# ============================================================
# Feature extraction helpers
# ============================================================


@torch.no_grad()
def extract_hmr2_features(
    frames_bgr, bbx_xys, hmr2_model, device="cuda", batch_size=16
):
    F = len(frames_bgr)
    all_feats = []
    imgs = []
    for i in range(F):
        crop, _ = crop_and_resize(frames_bgr[i], bbx_xys[i, :2], bbx_xys[i, 2])
        imgs.append(preprocess_image(crop, device=device))
    for start in range(0, F, batch_size):
        batch = torch.cat(imgs[start : start + batch_size], dim=0)
        all_feats.append(hmr2_model({"img": batch}).cpu())
    return torch.cat(all_feats, dim=0)


@torch.no_grad()
def extract_vitpose_keypoints(
    frames_bgr,
    bbx_xys,
    vitpose_model_inst,
    device="cuda",
    batch_size=16,
    flip_test=False,
):
    F = len(frames_bgr)
    all_kps = []
    imgs = []
    for i in range(F):
        crop, _ = crop_and_resize(frames_bgr[i], bbx_xys[i, :2], bbx_xys[i, 2])
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
            [0.229, 0.224, 0.225], dtype=np.float32
        )
        imgs.append(torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0))
    for start in tqdm(range(0, F, batch_size), desc="ViTPose", leave=False):
        batch = torch.cat(imgs[start : start + batch_size], dim=0).to(device)[
            :, :, :, 32:224
        ]
        if flip_test:
            hm_orig, hm_flip = vitpose_model_inst(
                torch.cat([batch, batch.flip(3)], 0)
            ).chunk(2)
            hm = (
                hm_orig
                + hm_flip[
                    :, [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15], :, :
                ].flip(3)
            ) * 0.5
        else:
            hm = vitpose_model_inst(batch)
        bxys_batch = bbx_xys[start : start + batch_size]
        scale_np = (
            np.stack([bxys_batch[:, 2] * 24 / 32, bxys_batch[:, 2]], axis=1) / 200.0
        )
        preds, maxvals = keypoints_from_heatmaps(
            hm.cpu().numpy(), bxys_batch[:, :2], scale_np
        )
        all_kps.append(
            torch.from_numpy(np.concatenate([preds, maxvals], -1).astype(np.float32))
        )
    return torch.cat(all_kps, dim=0)


# ============================================================
# Main inference pipeline
# ============================================================


def run_gvhmr(
    source,
    output_path: str = "gvhmr_output.npz",
    static_cam: bool = True,
    device: str = "cuda",
    batch_size: int = 16,
    det_conf: float = 0.5,
    smooth: bool = True,
):
    ckpt_base = os.path.join(_THIS_DIR, "checkpoints")
    gvhmr_path = os.path.join(ckpt_base, "gvhmr/gvhmr_siga24_release.ckpt")
    hmr2_path = os.path.join(ckpt_base, "hmr2/epoch=10-step=25000-001.ckpt")
    vit_path = os.path.join(ckpt_base, "vitpose/vitpose-s.pth")
    yolo_path = os.path.join(_THIS_DIR, "yolo26n.pt")

    print(f"[GVHMR] Reading: {source}")
    cap = cv2.VideoCapture(source if isinstance(source, int) else str(source))
    fps, frames = cap.get(cv2.CAP_PROP_FPS) or 30.0, []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError("No frames read")
    frames_np = np.stack(frames)
    F, H, W = frames_np.shape[:3]
    K = estimate_K(W, H)

    detector = PersonDetector(yolo_path, device=device, conf_thr=det_conf)
    dets = [
        detector.detect_frame(frames_np[f])
        for f in tqdm(range(F), desc="YOLO", leave=False)
    ]
    bbx_np = select_primary_person(dets)
    bbx_np[:, 2] = np.clip(bbx_np[:, 2], 32, max(W, H) * 2)

    hmr2 = load_hmr2(hmr2_path, device=device)
    f_imgseq = extract_hmr2_features(
        frames_np, bbx_np, hmr2, device=device, batch_size=batch_size
    )
    del hmr2

    vitpose = build_vitpose(vit_path, device=device)
    kp2d = extract_vitpose_keypoints(
        frames_np, bbx_np, vitpose, device=device, batch_size=batch_size
    )
    del vitpose

    if static_cam:
        cam_angvel = static_cam_angvel(F, torch.device("cpu"))
    else:
        print("[GVHMR] Running SimpleVO ...")
        R_w2c = SimpleVO().compute(frames_np)
        cam_angvel = compute_cam_angvel(torch.from_numpy(R_w2c))

    model = GVHMRModel.from_pretrained(gvhmr_path, device=device)
    with torch.no_grad():
        out = model(
            obs=normalize_kp2d(
                kp2d.unsqueeze(0), torch.from_numpy(bbx_np).unsqueeze(0)
            ).to(device),
            f_cliffcam=compute_bbox_info_bedlam(
                torch.from_numpy(bbx_np).unsqueeze(0),
                K.unsqueeze(0).unsqueeze(1).expand(1, F, -1, -1),
            ).to(device),
            f_cam_angvel=cam_angvel.unsqueeze(0).to(device),
            f_imgseq=f_imgseq.unsqueeze(0).to(device),
            bbx_xys=torch.from_numpy(bbx_np).unsqueeze(0).to(device),
            K_fullimg=K.unsqueeze(0).to(device),
        )

    pi = {k: v[0].cpu() for k, v in out["smpl_params_incam"].items()}
    pg = {k: v[0].cpu() for k, v in out["smpl_params_global"].items()}

    if smooth:
        for p in pi:
            if p != "betas":
                pi[p] = gaussian_smooth(pi[p], sigma=2.0, dim=0)
        for p in pg:
            if p != "betas":
                pg[p] = gaussian_smooth(pg[p], sigma=2.0, dim=0)

    betas_10 = pg["betas"][0].numpy()
    betas_16 = np.concatenate([betas_10, np.zeros(6, dtype=betas_10.dtype)])

    # Convert GVHMR coords (X-right, Y-fwd, Z-up) to SMPLX convention
    # Same rotation as GMR's get_gvhmr_data_offline_fast
    _R_corr = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)

    def _rotate_rotvec(rv, R):
        """Apply correction rotation to axis-angle: R_new = R_corr @ R_orig."""
        from scipy.spatial.transform import Rotation
        mats = Rotation.from_rotvec(rv).as_matrix()          # (F, 3, 3)
        return Rotation.from_matrix(R @ mats).as_rotvec().astype(np.float32)

    g_orient = _rotate_rotvec(pg["global_orient"].numpy(), _R_corr)
    g_trans = (pg["transl"].numpy() @ _R_corr.T)

    res = {
        "root_orient": g_orient,                              # (F, 3)
        "pose_body": pg["body_pose"].numpy(),                 # (F, 63)
        "trans": g_trans,                                     # (F, 3)
        "betas": betas_16,                                    # (16,)
        "gender": np.array("neutral"),
        "mocap_frame_rate": np.array(fps),
        "cam_root_orient": pi["global_orient"].numpy(),
        "cam_pose_body": pi["body_pose"].numpy(),
        "cam_trans": pi["transl"].numpy(),
        "bbx_xys": bbx_np,
        "kp2d": kp2d.numpy(),
        "img_wh": np.array([W, H]),
        "K_fullimg": K.numpy(),
    }
    np.savez(output_path, **res)
    print(f"[GVHMR] Saved → {output_path}")
    return res


# ============================================================
# Real-time webcam mode
# ============================================================

# COCO-17 skeleton edges for drawing
_COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),          # head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # arms
    (5, 11), (6, 12), (11, 12),                # torso
    (11, 13), (13, 15), (12, 14), (14, 16),    # legs
]


def _draw_kp2d(frame, kp2d, bbx, conf_thr=0.3):
    """Draw 2D keypoints and bbox on frame (in-place)."""
    cx, cy, s = bbx
    half = s / 2
    cv2.rectangle(frame, (int(cx - half), int(cy - half)),
                  (int(cx + half), int(cy + half)), (0, 255, 0), 2)
    for i, j in _COCO_SKELETON:
        if kp2d[i, 2] > conf_thr and kp2d[j, 2] > conf_thr:
            p1 = (int(kp2d[i, 0]), int(kp2d[i, 1]))
            p2 = (int(kp2d[j, 0]), int(kp2d[j, 1]))
            cv2.line(frame, p1, p2, (0, 200, 255), 2)
    for k in range(17):
        if kp2d[k, 2] > conf_thr:
            cv2.circle(frame, (int(kp2d[k, 0]), int(kp2d[k, 1])), 3, (0, 0, 255), -1)


@torch.no_grad()
def _extract_single_frame(frame_bgr, bbx, hmr2_model, vitpose_model, device):
    """Extract HMR2 features and ViTPose keypoints for a single frame."""
    crop, _ = crop_and_resize(frame_bgr, bbx[:2], bbx[2])
    # HMR2
    img_t = preprocess_image(crop, device=device)
    f_img = hmr2_model({"img": img_t}).cpu()
    # ViTPose
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - np.float32([0.485, 0.456, 0.406])) / np.float32([0.229, 0.224, 0.225])
    vit_in = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)[:, :, :, 32:224]
    hm = vitpose_model(vit_in)
    scale_np = np.array([[bbx[2] * 24 / 32, bbx[2]]], dtype=np.float32) / 200.0
    preds, maxvals = keypoints_from_heatmaps(
        hm.cpu().numpy(), bbx[:2].reshape(1, 2), scale_np
    )
    kp = torch.from_numpy(np.concatenate([preds, maxvals], -1).astype(np.float32))[0]
    return f_img[0], kp  # (1024,), (17, 3)


def run_webcam(device="cuda", det_conf=0.5, buf_len=120):
    """Real-time webcam loop with GVHMR inference and overlay."""
    import time

    ckpt_base = os.path.join(_THIS_DIR, "checkpoints")
    print("[Webcam] Loading models ...")
    detector = PersonDetector(
        os.path.join(_THIS_DIR, "yolo26n.pt"), device=device, conf_thr=det_conf
    )
    hmr2 = load_hmr2(
        os.path.join(ckpt_base, "hmr2/epoch=10-step=25000-001.ckpt"), device=device
    )
    vitpose = build_vitpose(
        os.path.join(ckpt_base, "vitpose/vitpose-s.pth"), device=device
    )
    gvhmr = GVHMRModel.from_pretrained(
        os.path.join(ckpt_base, "gvhmr/gvhmr_siga24_release.ckpt"), device=device
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")
    _, probe = cap.read()
    H, W = probe.shape[:2]
    K = estimate_K(W, H)

    # Rolling buffers
    bbx_buf = []      # list of (3,) np arrays
    feat_buf = []     # list of (1024,) tensors
    kp_buf = []       # list of (17, 3) tensors
    prev_bbx = None

    print("[Webcam] Running. Press 'q' to quit.")
    while True:
        t0 = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break

        # Detect
        dets = detector.detect_frame(frame)
        if dets:
            if prev_bbx is None:
                bbx = max(dets, key=lambda b: b[2])
            else:
                bbx = dets[np.argmin([np.linalg.norm(d[:2] - prev_bbx[:2]) for d in dets])]
            bbx[2] = np.clip(bbx[2], 32, max(W, H) * 2)
        elif prev_bbx is not None:
            bbx = prev_bbx.copy()
        else:
            fps = 1.0 / max(time.perf_counter() - t0, 1e-6)
            cv2.putText(frame, f"FPS: {fps:.0f} | No person", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("GVHMR Webcam", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue
        prev_bbx = bbx

        # Extract features
        f_img, kp = _extract_single_frame(frame, bbx, hmr2, vitpose, device)
        bbx_buf.append(bbx)
        feat_buf.append(f_img)
        kp_buf.append(kp)

        # Keep rolling buffer
        if len(bbx_buf) > buf_len:
            bbx_buf.pop(0)
            feat_buf.pop(0)
            kp_buf.pop(0)

        L = len(bbx_buf)

        # GVHMR needs >=2 frames for cam_angvel computation
        if L >= 2:
            bbx_t = torch.from_numpy(np.stack(bbx_buf)).unsqueeze(0)        # (1, L, 3)
            kp_t = torch.stack(kp_buf).unsqueeze(0)                          # (1, L, 17, 3)
            feat_t = torch.stack(feat_buf).unsqueeze(0)                       # (1, L, 1024)
            cam_angvel = static_cam_angvel(L, torch.device("cpu"))            # (L, 6)

            out = gvhmr(
                obs=normalize_kp2d(kp_t, bbx_t).to(device),
                f_cliffcam=compute_bbox_info_bedlam(
                    bbx_t, K.unsqueeze(0).unsqueeze(1).expand(1, L, -1, -1)
                ).to(device),
                f_cam_angvel=cam_angvel.unsqueeze(0).to(device),
                f_imgseq=feat_t.to(device),
                bbx_xys=bbx_t.to(device),
                K_fullimg=K.unsqueeze(0).to(device),
            )

        # Draw latest keypoints + bbox
        _draw_kp2d(frame, kp_buf[-1].numpy(), bbx_buf[-1])

        fps = 1.0 / max(time.perf_counter() - t0, 1e-6)
        cv2.putText(frame, f"FPS: {fps:.1f} | buf: {L}/{buf_len}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("GVHMR Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[Webcam] Done.")


# ============================================================
# CLI
# ============================================================


def main():
    parser = argparse.ArgumentParser(description="GVHMR inference demo")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", type=str)
    group.add_argument("--webcam", action="store_true")
    parser.add_argument("--out", type=str, default="gvhmr_output.npz")
    parser.add_argument("--static_cam", action="store_true", default=True)
    parser.add_argument("--no_static_cam", action="store_false", dest="static_cam")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--det_conf", type=float, default=0.5)
    parser.add_argument("--no_smooth", action="store_false", dest="smooth")
    parser.add_argument(
        "--visualize",
        type=str,
        default=None,
        metavar="OUT_VIDEO",
        help="Render SMPL-X overlay video to this path (requires pyrender, smplx)",
    )
    parser.add_argument(
        "--vis_mode", default="incam", choices=["incam", "global", "side_by_side"]
    )
    args = parser.parse_args()

    if args.webcam:
        run_webcam(device=args.device, det_conf=args.det_conf)
    else:
        run_gvhmr(
            args.video,
            args.out,
            args.static_cam,
            args.device,
            args.batch_size,
            args.det_conf,
            args.smooth,
        )
        if args.visualize:
            from gvhmr_visualize import visualize

            visualize(
                args.video, args.out, args.visualize,
                mode=args.vis_mode, device=args.device,
            )


if __name__ == "__main__":
    main()
