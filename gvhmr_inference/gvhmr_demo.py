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
from scipy.spatial.transform import Rotation
from tqdm import tqdm
from hmr2_model import load_hmr2, crop_and_resize
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
from ultralytics import YOLO

# Add this file's directory to path for sibling imports
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)


# ImageNet normalization constants
_IMAGENET_MEAN_NP = np.float32([0.485, 0.456, 0.406])
_IMAGENET_STD_NP = np.float32([0.229, 0.224, 0.225])
_IMAGENET_MEAN_T = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(
    3, 1, 1
)
_IMAGENET_STD_T = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

# GVHMR→SMPLX coordinate correction: (x,y,z) → (x,-z,y)
_R_CORR = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)

# Inverse of the internal [0,0,π] any→y-up rotation (self-inverse: R² = I)
_R_G2TG_INV = torch.tensor([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])


def _rotate_rotvec(rv: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Left-multiply rotation-vector array rv (..., 3) by matrix R (3, 3)."""
    return (
        Rotation.from_matrix(R @ Rotation.from_rotvec(rv).as_matrix())
        .as_rotvec()
        .astype(np.float32)
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
        self.model = YOLO(ckpt_path)
        self.model.to(device)
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


def select_primary_person(dets_per_frame: list) -> np.ndarray:
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
                    pts0 = np.float32([kp0[m.queryIdx].pt for m in good])
                    pts1 = np.float32([kp1[m.trainIdx].pt for m in good])
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
# Preprocessing helpers
# ============================================================


def _crop_to_hmr2_tensor(crop_bgr):
    """256×256 BGR crop → (3,256,256) normalized float32 tensor."""
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).float().div_(255.0).permute(2, 0, 1)
    return t.sub_(_IMAGENET_MEAN_T).div_(_IMAGENET_STD_T)


def _crop_to_vitpose_tensor(crop_bgr):
    """256×256 BGR crop → (3,256,192) normalized float32 tensor."""
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    rgb = (rgb / 255.0 - _IMAGENET_MEAN_NP) / _IMAGENET_STD_NP
    return torch.from_numpy(rgb.transpose(2, 0, 1))[:, :, 32:224]


def _kp2d_from_vitpose(hm_np, bbx_xys_np):
    """Convert a batch of ViTPose heatmaps + bounding boxes to (N,17,3) keypoints."""
    scale_np = np.stack([bbx_xys_np[:, 2] * 24 / 32, bbx_xys_np[:, 2]], axis=1) / 200.0
    preds, maxvals = keypoints_from_heatmaps(hm_np, bbx_xys_np[:, :2], scale_np)
    return torch.from_numpy(np.concatenate([preds, maxvals], -1).astype(np.float32))


# ============================================================
# Feature extraction
# ============================================================


@torch.no_grad()
def _extract_all_features(
    frames_bgr, bbx_xys, hmr2_model, vitpose_model, device="cuda", batch_size=16
):
    """Extract HMR2 image features and ViTPose keypoints for all frames."""
    F = len(frames_bgr)
    hmr2_tensors = []
    vit_tensors = []
    for i in range(F):
        crop, _ = crop_and_resize(frames_bgr[i], bbx_xys[i, :2], bbx_xys[i, 2])
        hmr2_tensors.append(_crop_to_hmr2_tensor(crop))
        vit_tensors.append(_crop_to_vitpose_tensor(crop))

    all_feats = []
    for start in tqdm(range(0, F, batch_size), desc="HMR2", leave=False):
        batch = torch.stack(hmr2_tensors[start : start + batch_size]).to(device)
        all_feats.append(hmr2_model({"img": batch}).cpu())
    f_imgseq = torch.cat(all_feats, 0)

    all_kps = []
    for start in tqdm(range(0, F, batch_size), desc="ViTPose", leave=False):
        batch = torch.stack(vit_tensors[start : start + batch_size]).to(device)
        hm = vitpose_model(batch).cpu().numpy()
        all_kps.append(_kp2d_from_vitpose(hm, bbx_xys[start : start + batch_size]))
    kp2d = torch.cat(all_kps, 0)

    return f_imgseq, kp2d


@torch.no_grad()
def _extract_single_frame(frame_bgr, bbx, hmr2_model, vitpose_model, device):
    """Extract HMR2 features and ViTPose keypoints for one frame."""
    crop, _ = crop_and_resize(frame_bgr, bbx[:2], bbx[2])
    f_img = hmr2_model(
        {"img": _crop_to_hmr2_tensor(crop).unsqueeze(0).to(device)}
    ).cpu()
    hm = (
        vitpose_model(_crop_to_vitpose_tensor(crop).unsqueeze(0).to(device))
        .cpu()
        .numpy()
    )
    kp = _kp2d_from_vitpose(hm, bbx[None])[0]
    return f_img[0], kp


def _resolve_joint_name(joint_names, candidates):
    for name in candidates:
        if name in joint_names:
            return name
    raise KeyError(f"None of {candidates} found in SMPL-X joints")


def _save_and_plot_foot_traj(
    pg, output_path, fps, body_model_dir=None, static_conf_logits=None
):
    """Save and plot torso-relative left/right foot trajectories for video mode.

    Args:
        pg: global SMPL parameters dict (tensors).
        output_path: base path for output files.
        fps: video frame rate.
        body_model_dir: path to SMPL-X body models directory.
        static_conf_logits: (F, 6) tensor of raw foot-contact logits from GVHMR
            (channels 0-2 = left foot, 3-5 = right foot).  Optional; if provided,
            per-foot stationary probability curves are added to the plot.
    """
    import smplx as smplx_lib
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from smplx.joint_names import JOINT_NAMES

    if body_model_dir is None:
        body_model_dir = os.path.join(_THIS_DIR, "..", "GMR", "assets", "body_models")

    if not os.path.isdir(body_model_dir):
        raise FileNotFoundError(f"SMPL-X body model dir not found: {body_model_dir}")

    body_model = smplx_lib.create(
        body_model_dir,
        "smplx",
        gender="neutral",
        use_pca=False,
    ).eval()

    F = pg["transl"].shape[0]
    nb = getattr(body_model, "num_betas", int(pg["betas"].shape[-1]))
    betas_one = torch.zeros(nb, dtype=pg["transl"].dtype)
    ncopy = min(nb, int(pg["betas"].shape[-1]))
    betas_one[:ncopy] = pg["betas"][0, :ncopy]
    betas_f = betas_one.unsqueeze(0).expand(F, -1)
    zeros_45 = torch.zeros(F, 45, dtype=pg["transl"].dtype)
    zeros_3 = torch.zeros(F, 3, dtype=pg["transl"].dtype)
    num_expr = getattr(body_model, "num_expression_coeffs", 10)
    zeros_expr = torch.zeros(F, num_expr, dtype=pg["transl"].dtype)
    smplx_out = body_model(
        global_orient=pg["global_orient"],
        body_pose=pg["body_pose"],
        betas=betas_f,
        transl=pg["transl"],
        expression=zeros_expr,
        left_hand_pose=zeros_45,
        right_hand_pose=zeros_45,
        jaw_pose=zeros_3,
        leye_pose=zeros_3,
        reye_pose=zeros_3,
        return_verts=False,
    )

    joint_names = JOINT_NAMES[: len(body_model.parents)]
    joints = smplx_out.joints[:, : len(joint_names), :].detach().cpu().numpy()
    joints = joints @ _R_CORR.T

    torso_name = _resolve_joint_name(
        joint_names,
        ["spine3", "spine2", "spine1", "pelvis"],
    )
    lfoot_name = _resolve_joint_name(joint_names, ["left_foot", "left_ankle"])
    rfoot_name = _resolve_joint_name(joint_names, ["right_foot", "right_ankle"])

    torso = joints[:, joint_names.index(torso_name)]
    left_foot = joints[:, joint_names.index(lfoot_name)]
    right_foot = joints[:, joint_names.index(rfoot_name)]
    left_rel = left_foot
    right_rel = right_foot

    # Global root XY trajectory (after coordinate correction)
    global_xy = pg["transl"].detach().cpu().numpy() @ _R_CORR.T  # (F, 3)

    # Per-foot stationary probabilities from model logits
    left_contact_prob = right_contact_prob = None
    if static_conf_logits is not None:
        logits_np = static_conf_logits.detach().cpu().float().numpy()  # (F, 6)
        probs = 1.0 / (1.0 + np.exp(-logits_np))  # sigmoid
        left_contact_prob = probs[:, :3].mean(axis=-1)  # (F,)
        right_contact_prob = probs[:, 3:].mean(axis=-1)  # (F,)

    traj_path = os.path.splitext(output_path)[0] + "_foot_traj.npz"
    save_dict = dict(
        fps=np.array(fps, dtype=np.float32),
        torso_joint=np.array(torso_name),
        left_foot_joint=np.array(lfoot_name),
        right_foot_joint=np.array(rfoot_name),
        torso_xyz=torso,
        left_foot_xyz=left_foot,
        right_foot_xyz=right_foot,
        left_foot_rel_xyz=left_rel,
        right_foot_rel_xyz=right_rel,
        global_xy=global_xy,
    )
    if left_contact_prob is not None:
        save_dict["left_contact_prob"] = left_contact_prob
        save_dict["right_contact_prob"] = right_contact_prob
    np.savez(traj_path, **save_dict)

    t = np.arange(F, dtype=np.float32) / max(float(fps), 1e-6)

    plot_path = os.path.splitext(output_path)[0] + "_foot_traj.png"

    # Layout: 2 rows × 2 columns
    #   [0,0] Z vs Time   [0,1] X-Y foot plane
    #   [1,0] Contact prob  [1,1] Global X-Y root
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax_top, ax_xy, ax_contact, ax_global = (
        axes[0, 0],
        axes[0, 1],
        axes[1, 0],
        axes[1, 1],
    )

    ax_top.plot(t, left_rel[:, 2], label=f"L ({lfoot_name})", lw=1.7)
    ax_top.plot(t, right_rel[:, 2], label=f"R ({rfoot_name})", lw=1.7)
    ax_top.set_title("Foot Z vs Time (global)")
    ax_top.set_xlabel("Time (s)")
    ax_top.set_ylabel("Z (m)")
    ax_top.grid(alpha=0.3)
    ax_top.legend()

    ax_xy.plot(left_rel[:, 0], left_rel[:, 1], label=f"L ({lfoot_name})", lw=1.7)
    ax_xy.plot(right_rel[:, 0], right_rel[:, 1], label=f"R ({rfoot_name})", lw=1.7)
    ax_xy.set_title("Foot X-Y plane (global)")
    ax_xy.set_xlabel("X (m)")
    ax_xy.set_ylabel("Y (m)")
    ax_xy.axis("equal")
    ax_xy.grid(alpha=0.3)
    ax_xy.legend()

    if left_contact_prob is not None:
        ax_contact.plot(t, left_contact_prob, label="L stationary prob", lw=1.7)
        ax_contact.plot(t, right_contact_prob, label="R stationary prob", lw=1.7)
        ax_contact.axhline(0.3, color="gray", lw=1.0, ls="--", label="threshold 0.3")
        ax_contact.set_ylim(0.0, 1.05)
    else:
        ax_contact.text(
            0.5,
            0.5,
            "No contact logits available",
            ha="center",
            va="center",
            transform=ax_contact.transAxes,
        )
    ax_contact.set_title("Foot stationary probability vs Time")
    ax_contact.set_xlabel("Time (s)")
    ax_contact.set_ylabel("Probability")
    ax_contact.grid(alpha=0.3)
    ax_contact.legend()

    ax_global.plot(global_xy[:, 0], global_xy[:, 1], lw=1.7, color="tab:purple")
    ax_global.scatter(
        global_xy[0, 0],
        global_xy[0, 1],
        marker="o",
        s=60,
        color="green",
        zorder=5,
        label="start",
    )
    ax_global.scatter(
        global_xy[-1, 0],
        global_xy[-1, 1],
        marker="X",
        s=60,
        color="red",
        zorder=5,
        label="end",
    )
    ax_global.set_title("Global root X-Y trajectory")
    ax_global.set_xlabel("X (m)")
    ax_global.set_ylabel("Y (m)")
    ax_global.axis("equal")
    ax_global.grid(alpha=0.3)
    ax_global.legend()

    fig.tight_layout()
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    return traj_path, plot_path


# ============================================================
# Main inference pipeline (video)
# ============================================================


def run_gvhmr(
    source,
    output_path: str = "gvhmr_output.npz",
    static_cam: bool = True,
    device: str = "cuda",
    batch_size: int = 16,
    det_conf: float = 0.5,
    smooth: bool = True,
    body_model_dir: str = None,
):
    ckpt_base = os.path.join(_THIS_DIR, "checkpoints")
    gvhmr_path = os.path.join(ckpt_base, "gvhmr/gvhmr_siga24_release.ckpt")
    hmr2_path = os.path.join(ckpt_base, "hmr2/epoch=10-step=25000-001.ckpt")
    vit_path = os.path.join(ckpt_base, "vitpose/vitpose-s.pth")
    yolo_path = os.path.join(ckpt_base, "yolo/yolo26n.pt")

    print(f"[GVHMR] Reading: {source}")
    cap = cv2.VideoCapture(source if isinstance(source, int) else str(source))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError("No frames read")
    frames_np = np.stack(frames)
    del frames
    F, H, W = frames_np.shape[:3]
    K = estimate_K(W, H)

    # Detection
    detector = PersonDetector(yolo_path, device=device, conf_thr=det_conf)
    dets = [
        detector.detect_frame(frames_np[f])
        for f in tqdm(range(F), desc="YOLO", leave=False)
    ]
    del detector
    bbx_np = select_primary_person(dets)
    bbx_np[:, 2] = np.clip(bbx_np[:, 2], 32, max(W, H) * 2)

    # Smooth the bbox track so noisy YOLO detections don't propagate into
    # keypoint normalization and depth estimation (same fix as webcam mode).
    bbx_t_smooth = torch.from_numpy(bbx_np).T.unsqueeze(0)  # (1, 3, F)
    bbx_t_smooth = gaussian_smooth(bbx_t_smooth, sigma=2.0, dim=-1)
    bbx_np = bbx_t_smooth[0].T.numpy()

    # Feature extraction
    hmr2 = load_hmr2(hmr2_path, device=device)
    vitpose = build_vitpose(vit_path, device=device)
    f_imgseq, kp2d = _extract_all_features(
        frames_np, bbx_np, hmr2, vitpose, device=device, batch_size=batch_size
    )
    del hmr2, vitpose

    if not static_cam:
        print("[GVHMR] Running SimpleVO ...")
        R_w2c = SimpleVO().compute(frames_np)
    del frames_np

    cam_angvel = (
        static_cam_angvel(F, torch.device("cpu"))
        if static_cam
        else compute_cam_angvel(torch.from_numpy(R_w2c))
    )

    # GVHMR inference
    model = GVHMRModel.from_pretrained(gvhmr_path, device=device)
    bbx_t = torch.from_numpy(bbx_np).unsqueeze(0)
    with torch.no_grad():
        out = model(
            obs=normalize_kp2d(kp2d.unsqueeze(0), bbx_t).to(device),
            f_cliffcam=compute_bbox_info_bedlam(
                bbx_t, K.unsqueeze(0).unsqueeze(1).expand(1, F, -1, -1)
            ).to(device),
            f_cam_angvel=cam_angvel.unsqueeze(0).to(device),
            f_imgseq=f_imgseq.unsqueeze(0).to(device),
            bbx_xys=bbx_t.to(device),
            K_fullimg=K.unsqueeze(0).to(device),
            foot_contact_threshold=0.3,
        )
    del model

    pi = {k: v[0].cpu() for k, v in out["smpl_params_incam"].items()}
    pg = {k: v[0].cpu() for k, v in out["smpl_params_global"].items()}
    static_conf_logits = out["static_conf_logits"][0].cpu()  # (F, 6)

    if smooth:
        for p in pi:
            if p != "betas":
                pi[p] = gaussian_smooth(pi[p], sigma=2.0, dim=0)
        for p in pg:
            if p != "betas":
                pg[p] = gaussian_smooth(pg[p], sigma=2.0, dim=0)

    betas_10 = pg["betas"][0].numpy()
    betas_16 = np.concatenate([betas_10, np.zeros(6, dtype=betas_10.dtype)])

    res = {
        "root_orient": _rotate_rotvec(pg["global_orient"].numpy(), _R_CORR),
        "pose_body": pg["body_pose"].numpy(),
        "trans": pg["transl"].numpy() @ _R_CORR.T,
        "betas": betas_16,
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

    traj_path, plot_path = _save_and_plot_foot_traj(
        pg,
        output_path=output_path,
        fps=fps,
        body_model_dir=body_model_dir,
        static_conf_logits=static_conf_logits,
    )
    print(f"[GVHMR] Saved foot trajectories → {traj_path}")
    print(f"[GVHMR] Saved foot trajectory plot → {plot_path}")
    return res


# ============================================================
# Real-time webcam mode
# ============================================================

# COCO-17 skeleton edges for drawing
_COCO_SKELETON = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]


def _draw_kp2d(frame, kp2d, bbx, conf_thr=0.3):
    cx, cy, s = bbx
    half = s / 2
    cv2.rectangle(
        frame,
        (int(cx - half), int(cy - half)),
        (int(cx + half), int(cy + half)),
        (0, 255, 0),
        2,
    )
    for i, j in _COCO_SKELETON:
        if kp2d[i, 2] > conf_thr and kp2d[j, 2] > conf_thr:
            cv2.line(
                frame,
                (int(kp2d[i, 0]), int(kp2d[i, 1])),
                (int(kp2d[j, 0]), int(kp2d[j, 1])),
                (0, 200, 255),
                2,
            )
    for k in range(17):
        if kp2d[k, 2] > conf_thr:
            cv2.circle(frame, (int(kp2d[k, 0]), int(kp2d[k, 1])), 3, (0, 0, 255), -1)


def _quat_mul_batch(q1, q2):
    """Hamilton product q1 (4,) × q2 (N,4) → (N,4), scalar-first."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


def _aa_ema_slerp(
    raw_aa: np.ndarray, prev_aa: np.ndarray, alpha: float = 0.5
) -> np.ndarray:
    """EMA-style smoothing for axis-angle orientations via quaternion SLERP.

    Naive linear blending of axis-angle vectors causes sudden 180° flips
    because axis-angle is not a smooth manifold (e.g. (axis, θ) and
    (-axis, 2π−θ) represent near-identical rotations but have wildly
    different vectors).  This helper converts to quaternions, ensures the
    short-arc hemisphere, and SLERPs with factor *alpha* for the new
    observation.
    """
    q_raw = Rotation.from_rotvec(raw_aa).as_quat(scalar_first=True)  # (4,)
    q_prev = Rotation.from_rotvec(prev_aa).as_quat(scalar_first=True)  # (4,)
    # Ensure shortest-arc: flip q_raw if it's in the opposite hemisphere
    if np.dot(q_raw, q_prev) < 0:
        q_raw = -q_raw
    # SLERP: Rotation.slerp is not available for single pairs; use scipy key_times
    from scipy.spatial.transform import Slerp

    slerp = Slerp(
        [0.0, 1.0],
        Rotation.from_quat(
            np.stack([q_prev, q_raw])[:, [1, 2, 3, 0]]  # scipy uses scalar-last
        ),
    )
    blended = slerp(alpha)  # at alpha → new observation
    return blended.as_rotvec().astype(raw_aa.dtype)


def _smplx_frame_to_joint_dict(
    body_model, global_orient, body_pose, betas, transl, _cache={}
):
    """Run SMPLX forward for one frame; return {joint_name: (pos, quat)} dict."""
    if "corr_quat" not in _cache:
        from smplx.joint_names import JOINT_NAMES

        R_corr = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
        _cache["R_corr"] = R_corr
        _cache["corr_quat"] = Rotation.from_matrix(R_corr).as_quat(scalar_first=True)
        _cache["JOINT_NAMES"] = JOINT_NAMES
        _cache["zeros_45"] = torch.zeros(1, 45)
        _cache["zeros_3"] = torch.zeros(1, 3)

    R_corr = _cache["R_corr"]
    corr_quat = _cache["corr_quat"]
    JOINT_NAMES = _cache["JOINT_NAMES"]

    nb = body_model.num_betas
    if betas.shape[0] != nb:
        b = torch.zeros(nb)
        b[: min(betas.shape[0], nb)] = betas[:nb]
        betas = b

    out = body_model(
        global_orient=global_orient.unsqueeze(0),
        body_pose=body_pose.unsqueeze(0),
        betas=betas.unsqueeze(0),
        transl=transl.unsqueeze(0),
        left_hand_pose=_cache["zeros_45"],
        right_hand_pose=_cache["zeros_45"],
        jaw_pose=_cache["zeros_3"],
        leye_pose=_cache["zeros_3"],
        reye_pose=_cache["zeros_3"],
        return_full_pose=True,
    )
    joints = out.joints[0].detach().numpy()
    full_pose = out.full_pose[0].reshape(-1, 3).detach().numpy()
    go = out.global_orient[0].squeeze().detach().numpy()
    joint_names = JOINT_NAMES[: len(body_model.parents)]
    parents = body_model.parents

    joint_oris = []
    for i in range(len(joint_names)):
        rot = (
            Rotation.from_rotvec(go)
            if i == 0
            else joint_oris[parents[i]] * Rotation.from_rotvec(full_pose[i])
        )
        joint_oris.append(rot)

    all_quats = np.array([r.as_quat(scalar_first=True) for r in joint_oris])
    return {
        name: (joints[i] @ R_corr.T, _quat_mul_batch(corr_quat, all_quats)[i])
        for i, name in enumerate(joint_names)
    }


def run_webcam(device="cuda", det_conf=0.5, buf_len=120, robot=None):
    """Real-time webcam loop: GVHMR inference with optional GMR robot retargeting."""
    import time
    import smplx as smplx_lib

    ckpt_base = os.path.join(_THIS_DIR, "checkpoints")
    gmr_base = os.path.join(_THIS_DIR, "..", "GMR")

    print("[Webcam] Loading models ...")
    detector = PersonDetector(
        os.path.join(ckpt_base, "yolo/yolo26n.pt"), device=device, conf_thr=det_conf
    )
    hmr2 = load_hmr2(
        os.path.join(ckpt_base, "hmr2/epoch=10-step=25000-001.ckpt"), device=device
    )
    vitpose = build_vitpose(
        os.path.join(ckpt_base, "vitpose/vitpose-s.pth"), device=device
    )
    gvhmr_model = GVHMRModel.from_pretrained(
        os.path.join(ckpt_base, "gvhmr/gvhmr_siga24_release.ckpt"), device=device
    )
    smplx_model = smplx_lib.create(
        os.path.join(gmr_base, "assets/body_models"),
        "smplx",
        gender="neutral",
        use_pca=False,
    ).eval()

    retarget, viewer = None, None
    if robot:
        from general_motion_retargeting import (
            GeneralMotionRetargeting,
            RobotMotionViewer,
        )

        retarget = GeneralMotionRetargeting(
            actual_human_height=1.66, src_human="smplx", tgt_robot=robot, verbose=False
        )
        viewer = RobotMotionViewer(robot_type=robot, motion_fps=30, transparent_robot=0)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")
    _, probe = cap.read()
    H, W = probe.shape[:2]
    K = estimate_K(W, H)

    max_cam_angvel = static_cam_angvel(buf_len, torch.device("cpu"))

    # Rolling buffers
    bbx_buf, feat_buf, kp_buf = [], [], []
    prev_bbx = None
    bbx_ema = None  # smoothed bbox (EMA, α=0.4)

    # Translation anchor carried across window slides (Fix 1)
    transl_anchor = None

    # Per-frame output EMA for streaming smoothness (Fix 3, α=0.5)
    ema_transl = ema_orient = ema_bpose = None

    # Latency tracking (EMA, α=0.3)
    lat = {
        "yolo": 0.0,
        "feat": 0.0,
        "gvhmr": 0.0,
        "smplx": 0.0,
        "gmr": 0.0,
        "viewer": 0.0,
    }

    def _tick(key, t1):
        lat[key] = 0.3 * (time.perf_counter() - t1) * 1000 + 0.7 * lat[key]

    print("[Webcam] Running. Press 'q' to quit.")
    while True:
        t0 = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO detection
        t1 = time.perf_counter()
        dets = detector.detect_frame(frame)
        _tick("yolo", t1)

        if dets:
            bbx = (
                max(dets, key=lambda b: b[2])
                if prev_bbx is None
                else dets[
                    np.argmin([np.linalg.norm(d[:2] - prev_bbx[:2]) for d in dets])
                ]
            )
            bbx[2] = np.clip(bbx[2], 32, max(W, H) * 2)
            bbx_ema = bbx.copy() if bbx_ema is None else 0.4 * bbx + 0.6 * bbx_ema
            bbx = bbx_ema.copy()
        elif prev_bbx is not None:
            bbx = prev_bbx.copy()
        else:
            cv2.putText(
                frame,
                "No person",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.imshow("GVHMR Webcam", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue
        prev_bbx = bbx

        # Feature extraction
        t1 = time.perf_counter()
        f_img, kp = _extract_single_frame(frame, bbx, hmr2, vitpose, device)
        _tick("feat", t1)

        bbx_buf.append(bbx)
        feat_buf.append(f_img)
        kp_buf.append(kp)

        window_slid = len(bbx_buf) > buf_len
        if window_slid:
            bbx_buf.pop(0)
            feat_buf.pop(0)
            kp_buf.pop(0)

        L = len(bbx_buf)
        gvhmr_out = None

        if L >= 2:
            t1 = time.perf_counter()
            bbx_t = torch.from_numpy(np.stack(bbx_buf)).unsqueeze(0)
            gvhmr_out = gvhmr_model(
                obs=normalize_kp2d(torch.stack(kp_buf).unsqueeze(0), bbx_t).to(device),
                f_cliffcam=compute_bbox_info_bedlam(
                    bbx_t, K.unsqueeze(0).unsqueeze(1).expand(1, L, -1, -1)
                ).to(device),
                f_cam_angvel=max_cam_angvel[:L].unsqueeze(0).to(device),
                f_imgseq=torch.stack(feat_buf).unsqueeze(0).to(device),
                bbx_xys=bbx_t.to(device),
                K_fullimg=K.unsqueeze(0).to(device),
                transl_0=transl_anchor,
                foot_contact_threshold=0.3,
            )
            _tick("gvhmr", t1)

            # Update translation anchor for next slide.
            # pg["transl"] is post-R_g2tg([0,0,π]); invert via _R_G2TG_INV (self-inverse)
            # before passing back into rollout_local_transl_vel next call.
            if window_slid:
                t1_post = gvhmr_out["smpl_params_global"]["transl"][:, 1:2, :]
                transl_anchor = torch.einsum(
                    "ij,...j->...i", _R_G2TG_INV.to(t1_post), t1_post
                )
            else:
                transl_anchor = None

        if gvhmr_out is not None and retarget is not None:
            pg = gvhmr_out["smpl_params_global"]
            raw_t = pg["transl"][0, -1].cpu().float().numpy()
            raw_o = pg["global_orient"][0, -1].cpu().float().numpy()
            raw_b = pg["body_pose"][0, -1].cpu().float().numpy()
            if ema_transl is None:
                ema_transl, ema_orient, ema_bpose = (
                    raw_t.copy(),
                    raw_o.copy(),
                    raw_b.copy(),
                )
            else:
                ema_transl = 0.5 * raw_t + 0.5 * ema_transl
                ema_orient = _aa_ema_slerp(raw_o, ema_orient, alpha=0.5)
                ema_bpose = 0.5 * raw_b + 0.5 * ema_bpose

            t1 = time.perf_counter()
            joint_dict = _smplx_frame_to_joint_dict(
                smplx_model,
                torch.from_numpy(ema_orient),
                torch.from_numpy(ema_bpose),
                pg["betas"][0, -1].cpu().float(),
                torch.from_numpy(ema_transl),
            )
            _tick("smplx", t1)

            t1 = time.perf_counter()
            qpos = retarget.retarget(joint_dict, offset_to_ground=True)
            _tick("gmr", t1)

            t1 = time.perf_counter()
            viewer.step(
                root_pos=qpos[:3],
                root_rot=qpos[3:7],
                dof_pos=qpos[7:],
                human_motion_data=retarget.scaled_human_data,
                human_pos_offset=np.array([0.0, 0.0, 0.0]),
                rate_limit=False,
                follow_camera=True,
            )
            _tick("viewer", t1)

        _draw_kp2d(frame, kp_buf[-1].numpy(), bbx_buf[-1])

        total_ms = (time.perf_counter() - t0) * 1000
        fps_disp = 1000.0 / max(total_ms, 0.1)
        cv2.putText(
            frame,
            f"FPS: {fps_disp:.1f}  buf: {L}/{buf_len}  total: {total_ms:.0f}ms",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            f"yolo:{lat['yolo']:.0f} feat:{lat['feat']:.0f} gvhmr:{lat['gvhmr']:.0f} "
            f"smplx:{lat['smplx']:.0f} gmr:{lat['gmr']:.0f} view:{lat['viewer']:.0f}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        cv2.imshow("GVHMR Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    if viewer:
        viewer.close()
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
        "--body_model_dir",
        type=str,
        default=None,
        help="Path to SMPL-X body models (default: ../GMR/assets/body_models)",
    )
    parser.add_argument(
        "--robot",
        type=str,
        default="unitree_g1",
        help="Robot type for GMR retargeting in webcam mode",
    )
    parser.add_argument(
        "--visualize",
        type=str,
        default=None,
        metavar="OUT_VIDEO",
        help="Render SMPL-X overlay video (requires pyrender, smplx)",
    )
    parser.add_argument(
        "--vis_mode", default="incam", choices=["incam", "global", "side_by_side"]
    )
    args = parser.parse_args()

    if args.webcam:
        run_webcam(device=args.device, det_conf=args.det_conf, robot=args.robot)
    else:
        run_gvhmr(
            args.video,
            args.out,
            args.static_cam,
            args.device,
            args.batch_size,
            args.det_conf,
            args.smooth,
            args.body_model_dir,
        )
        if args.visualize:
            from gvhmr_visualize import visualize

            visualize(
                args.video,
                args.out,
                args.visualize,
                mode=args.vis_mode,
                device=args.device,
            )


if __name__ == "__main__":
    main()
