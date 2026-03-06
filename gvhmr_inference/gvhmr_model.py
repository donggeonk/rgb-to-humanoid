"""Self-contained GVHMR inference model.

No dependencies on hmr4d or pytorch3d. Requires: torch, einops, timm, scipy.
Implements the GVHMR denoiser (NetworkEncoderRoPE) and all geometric utilities.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange, repeat
from timm.models.vision_transformer import Mlp
from scipy.ndimage._filters import _gaussian_kernel1d


# ============================================================
# Rotation Utilities (pytorch3d-compatible conventions)
# ============================================================


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """6D rotation → 3×3 matrix (pytorch3d convention: rows of R).
    d6: (..., 6)  →  R: (..., 3, 3)
    """
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def matrix_to_rotation_6d(R: torch.Tensor) -> torch.Tensor:
    """3×3 matrix → 6D rotation (first two rows flattened).
    R: (..., 3, 3)  →  (..., 6)
    """
    return R[..., :2, :].clone().reshape(*R.shape[:-2], 6)


def axis_angle_to_matrix(aa: torch.Tensor) -> torch.Tensor:
    """Axis-angle (Rodrigues) → rotation matrix.
    aa: (..., 3)  →  R: (..., 3, 3)
    """
    angle = aa.norm(dim=-1, keepdim=True)
    # safe axis (avoid zero-division)
    axis = aa / angle.clamp(min=1e-8)
    cos = torch.cos(angle).squeeze(-1)
    sin = torch.sin(angle).squeeze(-1)
    nx, ny, nz = axis[..., 0], axis[..., 1], axis[..., 2]
    one_minus_cos = 1 - cos
    R = torch.stack(
        [
            cos + nx * nx * one_minus_cos,
            nx * ny * one_minus_cos - nz * sin,
            nx * nz * one_minus_cos + ny * sin,
            ny * nx * one_minus_cos + nz * sin,
            cos + ny * ny * one_minus_cos,
            ny * nz * one_minus_cos - nx * sin,
            nz * nx * one_minus_cos - ny * sin,
            nz * ny * one_minus_cos + nx * sin,
            cos + nz * nz * one_minus_cos,
        ],
        dim=-1,
    ).reshape(*aa.shape[:-1], 3, 3)
    # identity for zero rotation
    identity = torch.eye(3, device=aa.device, dtype=aa.dtype).expand(
        *aa.shape[:-1], -1, -1
    )
    return torch.where(
        (angle.squeeze(-1) < 1e-8).unsqueeze(-1).unsqueeze(-1), identity, R
    )


def matrix_to_axis_angle(R: torch.Tensor) -> torch.Tensor:
    """Rotation matrix → axis-angle.
    R: (..., 3, 3)  →  (..., 3)
    """
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos = ((trace - 1) / 2).clamp(-1 + 1e-7, 1 - 1e-7)
    angle = torch.acos(cos)
    skew = torch.stack(
        [
            R[..., 2, 1] - R[..., 1, 2],
            R[..., 0, 2] - R[..., 2, 0],
            R[..., 1, 0] - R[..., 0, 1],
        ],
        dim=-1,
    )
    # axis-angle = skew / (2 sin θ) * θ
    axis_angle = (
        skew / (2 * torch.sin(angle).unsqueeze(-1) + 1e-8) * angle.unsqueeze(-1)
    )
    # near-zero case → zero vector
    return torch.where(
        (angle < 1e-6).unsqueeze(-1), torch.zeros_like(axis_angle), axis_angle
    )


# ============================================================
# Gaussian Smooth (matches hmr4d/utils/net_utils.py)
# ============================================================


def gaussian_smooth(x: torch.Tensor, sigma: float = 3, dim: int = -1) -> torch.Tensor:
    """1-D Gaussian smoothing along `dim`."""
    kernel_np = _gaussian_kernel1d(sigma=sigma, order=0, radius=int(4 * sigma + 0.5))
    kernel = torch.from_numpy(kernel_np).float()[None, None].to(x)  # (1, 1, K)
    rad = kernel.size(-1) // 2
    x = x.transpose(dim, -1)
    x_shape = x.shape[:-1]
    x = rearrange(x, "... f -> (...) 1 f")
    x = F.pad(x[None], (rad, rad, 0, 0), mode="replicate")[0]
    x = F.conv1d(x, kernel)
    x = x.squeeze(1).reshape(*x_shape, -1)
    return x.transpose(-1, dim)


# ============================================================
# Camera / Geometry Utilities
# ============================================================


def estimate_K(img_w: int, img_h: int) -> torch.Tensor:
    """Estimate a pinhole camera matrix from image dimensions (diagonal FOV ≈ 53°)."""
    fl = (img_w**2 + img_h**2) ** 0.5
    K = torch.eye(3, dtype=torch.float32)
    K[0, 0] = fl
    K[1, 1] = fl
    K[0, 2] = img_w / 2.0
    K[1, 2] = img_h / 2.0
    return K


def compute_bbox_info_bedlam(
    bbx_xys: torch.Tensor, K_fullimg: torch.Tensor
) -> torch.Tensor:
    """BEDLAM-style bounding-box encoding (CLIFF-cam input).
    bbx_xys: (..., 3) [cx, cy, s]; K_fullimg: (..., 3, 3)
    Returns: (..., 3)
    """
    fl = K_fullimg[..., 0, 0].unsqueeze(-1)
    icx = K_fullimg[..., 0, 2]
    icy = K_fullimg[..., 1, 2]
    cx, cy, b = bbx_xys[..., 0], bbx_xys[..., 1], bbx_xys[..., 2]
    return torch.stack([cx - icx, cy - icy, b], dim=-1) / fl


def normalize_kp2d(obs_kp2d: torch.Tensor, bbx_xys: torch.Tensor) -> torch.Tensor:
    """Normalize 2-D keypoints relative to bounding box, to [-1, 1].
    obs_kp2d: (..., J, 3) [x, y, conf]; bbx_xys: (..., 3)
    Returns: (..., J, 3)
    """
    obs_xy = obs_kp2d[..., :2]
    obs_conf = obs_kp2d[..., 2]
    center = bbx_xys[..., :2]
    scale = bbx_xys[..., [2]]
    xy_max = center + scale / 2
    xy_min = center - scale / 2
    invisible = (
        (obs_xy[..., 0] < xy_min[..., None, 0])
        | (obs_xy[..., 0] > xy_max[..., None, 0])
        | (obs_xy[..., 1] < xy_min[..., None, 1])
        | (obs_xy[..., 1] > xy_max[..., None, 1])
    )
    obs_conf = obs_conf * ~invisible
    normalized_xy = 2 * (obs_xy - center.unsqueeze(-2)) / scale.unsqueeze(-2)
    return torch.cat([normalized_xy, obs_conf[..., None]], dim=-1)


def compute_transl_incam(
    pred_cam: torch.Tensor, bbx_xys: torch.Tensor, K_fullimg: torch.Tensor
) -> torch.Tensor:
    """Weak-perspective cam params → 3-D translation in camera frame.
    pred_cam: (..., 3) [s, tx, ty]; bbx_xys: (..., 3); K_fullimg: (..., 3, 3)
    Returns: (..., 3)
    """
    s, tx, ty = pred_cam[..., 0], pred_cam[..., 1], pred_cam[..., 2]
    fl = K_fullimg[..., 0, 0]
    icx = K_fullimg[..., 0, 2]
    icy = K_fullimg[..., 1, 2]
    sb = s * bbx_xys[..., 2]
    cx = 2 * (bbx_xys[..., 0] - icx) / (sb + 1e-9)
    cy = 2 * (bbx_xys[..., 1] - icy) / (sb + 1e-9)
    tz = 2 * fl / (sb + 1e-9)
    return torch.stack([tx + cx, ty + cy, tz], dim=-1)


def compute_cam_angvel(R_w2c: torch.Tensor) -> torch.Tensor:
    """Camera angular velocity as 6D rotation.
    R_w2c: (F, 3, 3)  →  cam_angvel: (F, 6)
    """
    rel = R_w2c[1:] @ R_w2c[:-1].transpose(-1, -2)  # (F-1, 3, 3)
    cam_angvel = matrix_to_rotation_6d(rel)  # (F-1, 6)
    return torch.cat([cam_angvel, cam_angvel[-1:]], dim=0)  # pad last → (F, 6)


def static_cam_angvel(length: int, device: torch.device) -> torch.Tensor:
    """Identity cam_angvel for a static camera (F, 6)."""
    I = torch.eye(3, device=device).unsqueeze(0).expand(length, -1, -1)
    return compute_cam_angvel(I)


# ============================================================
# Global SMPL params rollout
# ============================================================


def rollout_local_transl_vel(
    local_transl_vel: torch.Tensor,
    global_orient: torch.Tensor,
    transl_0: torch.Tensor = None,
) -> torch.Tensor:
    """Integrate local translation velocity to global translation.
    local_transl_vel: (*, L, 3); global_orient: (*, L, 3) axis-angle
    Returns transl: (*, L, 3)
    """
    R = axis_angle_to_matrix(global_orient)  # (*, L, 3, 3)
    vel = torch.einsum("...lij,...lj->...li", R, local_transl_vel)
    if transl_0 is None:
        transl_0 = torch.zeros_like(vel[..., :1, :])
    transl_ = torch.cat([transl_0, vel[..., :-1, :]], dim=-2)
    return torch.cumsum(transl_, dim=-2)


# rot by [0, 0, π] to convert "any" gravity-dir to "ay" (y-up)
_TSF_ANY_TO_AY = [0.0, 0.0, math.pi]


def get_tgtcoord_rootparam(global_orient: torch.Tensor, transl: torch.Tensor):
    """Rotate params by [0,0,π] (any→ay coordinate transform).
    Returns (global_orient_new, transl_new, R_g2tg)
    """
    device = global_orient.device
    aa = torch.tensor(_TSF_ANY_TO_AY, device=device, dtype=global_orient.dtype)
    R_g2tg = axis_angle_to_matrix(aa)  # (3, 3)
    R = axis_angle_to_matrix(global_orient)
    new_orient = matrix_to_axis_angle(R_g2tg @ R)
    new_transl = torch.einsum("ij,...j->...i", R_g2tg, transl)
    return new_orient, new_transl, R_g2tg


def apply_foot_contact_constraints(
    local_transl_vel: torch.Tensor,
    static_conf_logits: torch.Tensor,
    conf_threshold: float = 0.3,
) -> torch.Tensor:
    """Suppress root translation velocity when either foot is in static contact.

    The model outputs ``static_conf_logits`` with shape (B, L, 6).  The 6
    channels correspond to contact probabilities for 3 left-foot vertices
    (channels 0-2) and 3 right-foot vertices (channels 3-5).

    Instead of averaging all 6 channels (which dilutes single-foot contact
    during walking), we compute per-foot contact as the mean of each foot's
    3 vertices and take the **max** across left/right — so *either* foot
    being planted triggers the constraint.

    We also use **soft damping** instead of hard binary zeroing: the xz
    velocity is scaled by ``(1 - damping_factor)`` where the damping factor
    ramps smoothly from 0 to 1 over the range [conf_threshold, 1.0].  This
    prevents discontinuous velocity jumps at the contact boundary that cause
    visible foot sliding.

    Args:
        local_transl_vel:   (B, L, 3) local root velocity
        static_conf_logits: (B, L, 6) raw logits from static_conf_head
        conf_threshold:     probability above which damping begins (default 0.3)
    Returns:
        local_transl_vel with xz damped for contact frames  (B, L, 3)
    """
    probs = torch.sigmoid(static_conf_logits)  # (B, L, 6)

    # Per-foot contact: mean over 3 vertices each, then max across feet
    left_prob = probs[..., :3].mean(dim=-1)  # (B, L)
    right_prob = probs[..., 3:].mean(dim=-1)  # (B, L)
    contact_prob = torch.max(left_prob, right_prob)  # (B, L)

    # Soft damping: linearly ramp from 0→1 over [threshold, 1.0]
    # damping = 0 when contact_prob <= threshold (no suppression)
    # damping = 1 when contact_prob >= 1.0 (full suppression)
    damping = ((contact_prob - conf_threshold) / (1.0 - conf_threshold)).clamp(0.0, 1.0)

    vel = local_transl_vel.clone()
    scale = 1.0 - damping  # (B, L)
    vel[..., 0] = vel[..., 0] * scale
    vel[..., 2] = vel[..., 2] * scale
    return vel


def get_smpl_params_global(
    global_orient_gv: torch.Tensor,
    local_transl_vel: torch.Tensor,
    global_orient_c: torch.Tensor,
    cam_angvel: torch.Tensor,
    transl_0: torch.Tensor = None,
) -> dict:
    """Compute global (world) SMPL params from network outputs.

    Args:
        global_orient_gv: (B, L, 3) axis-angle in gravity-aligned view
        local_transl_vel: (B, L, 3) local translation velocity
        global_orient_c: (B, L, 3) axis-angle in camera frame
        cam_angvel:       (B, L, 6) camera angular velocity (6D rotation)
        transl_0:         (B, 1, 3) optional absolute position of first frame
    Returns:
        dict with keys: "global_orient" (B, L, 3), "transl" (B, L, 3)
    """

    def as_identity(R):
        is_I = matrix_to_axis_angle(R).norm(dim=-1) < 1e-5
        R = R.clone()
        R[is_I] = torch.eye(3, device=R.device, dtype=R.dtype).expand(
            is_I.sum(), -1, -1
        )
        return R

    B = cam_angvel.shape[0]
    R_t_to_tp1 = rotation_6d_to_matrix(cam_angvel)  # (B, L, 3, 3)
    R_t_to_tp1 = as_identity(R_t_to_tp1)

    R_gv = axis_angle_to_matrix(global_orient_gv)  # (B, L, 3, 3)
    R_c = axis_angle_to_matrix(global_orient_c)  # (B, L, 3, 3)
    R_c2gv = R_gv @ R_c.mT  # (B, L, 3, 3)

    # Project view axis into xz-plane and compute frame-to-frame yaw
    R_cnext2gv = R_c2gv @ R_t_to_tp1.mT
    view1 = R_c2gv[..., 2].clone()
    view1[..., 1] = 0
    view1 = F.normalize(view1, dim=-1)
    view2 = R_cnext2gv[..., 2].clone()
    view2[..., 1] = 0
    view2 = F.normalize(view2, dim=-1)

    aa_tp1_to_t = view2.cross(view1, dim=-1)
    angle = torch.acos((view1 * view2).sum(-1, keepdim=True).clamp(-1.0, 1.0))
    aa_tp1_to_t = F.normalize(aa_tp1_to_t, dim=-1) * angle

    aa_tp1_to_t = gaussian_smooth(aa_tp1_to_t, dim=-2)
    R_tp1_to_t = axis_angle_to_matrix(aa_tp1_to_t).mT  # (B, L, 3, 3)

    # Cumulative rotation R_{t→0}
    R_t_to_0 = [
        torch.eye(3, device=cam_angvel.device, dtype=cam_angvel.dtype)
        .unsqueeze(0)
        .expand(B, -1, -1)
    ]
    for i in range(1, R_t_to_tp1.shape[1]):
        R_t_to_0.append(R_t_to_0[-1] @ R_tp1_to_t[:, i])
    R_t_to_0 = torch.stack(R_t_to_0, dim=1)  # (B, L, 3, 3)
    R_t_to_0 = as_identity(R_t_to_0)

    global_orient = matrix_to_axis_angle(R_t_to_0 @ R_gv)
    transl = rollout_local_transl_vel(
        local_transl_vel, global_orient, transl_0=transl_0
    )
    global_orient, transl, _ = get_tgtcoord_rootparam(global_orient, transl)
    return {"global_orient": global_orient, "transl": transl}


# ============================================================
# MM_V1_AMASS_LOCAL_BEDLAM_CAM Normalization Statistics
# ============================================================

_MM_STATS_MEAN = [
    0.96969,
    -0.059719,
    -0.0377,
    0.058256,
    0.908,
    0.10972,
    0.97636,
    0.043401,
    0.004311,
    -0.043032,
    0.90261,
    0.14478,
    0.99288,
    0.0035673,
    0.016264,
    -0.002226,
    0.9347,
    -0.23495,
    0.97147,
    0.052553,
    -0.093666,
    -0.05455,
    0.83321,
    -0.24246,
    0.97971,
    -0.038429,
    0.0053575,
    0.015537,
    0.81449,
    -0.30926,
    0.99532,
    -0.0094398,
    -0.038328,
    0.0085141,
    0.9888,
    0.00019976,
    0.95602,
    -0.039528,
    0.20017,
    0.010363,
    0.95965,
    0.1377,
    0.96223,
    -0.046278,
    -0.15177,
    0.066705,
    0.95545,
    0.12519,
    0.99767,
    -0.012616,
    -0.00025442,
    0.011661,
    0.99376,
    -0.036222,
    0.99511,
    -0.010583,
    0.01213,
    0.0076461,
    0.99137,
    0.020029,
    0.99295,
    0.0072917,
    0.0049454,
    -0.0080286,
    0.99137,
    0.0023707,
    0.97698,
    0.019943,
    0.0013808,
    -0.022006,
    0.97375,
    -0.067936,
    0.92804,
    0.25005,
    -0.057167,
    -0.24047,
    0.94246,
    0.025863,
    0.92957,
    -0.21329,
    0.11112,
    0.20741,
    0.94876,
    0.029901,
    0.97683,
    -0.04121,
    0.0023248,
    0.040967,
    0.97365,
    0.0057309,
    0.64513,
    0.61999,
    -0.25469,
    -0.62342,
    0.68177,
    0.035524,
    0.66192,
    -0.59341,
    0.27136,
    0.59269,
    0.68966,
    0.031309,
    0.68946,
    -0.11676,
    -0.49859,
    0.040969,
    0.93656,
    -0.14875,
    0.62787,
    0.13793,
    0.54289,
    -0.091946,
    0.92868,
    -0.11927,
    0.93012,
    -0.08381,
    -0.11951,
    0.097211,
    0.89118,
    0.059887,
    0.93033,
    0.071047,
    0.075264,
    -0.080679,
    0.88562,
    0.04896,
    0.231,
    0.175,
    0.2931,
    -0.1859,
    -1.1163,
    -1.1028,
    -0.2573,
    0.3555,
    0.3732,
    0.2852,
    -0.0049862,
    -0.00087136,
    -0.0014187,
    0.014825,
    -0.94419,
    -0.051653,
    0.00036018,
    -0.00022327,
    0.0022316,
    -0.044879,
    -0.97435,
    0.10021,
    -0.0002,
    -0.0006,
    0.0069,
]

_MM_STATS_STD = [
    0.0612,
    0.139,
    0.1779,
    0.1415,
    0.1826,
    0.3268,
    0.044,
    0.1382,
    0.1542,
    0.1348,
    0.193,
    0.3272,
    0.0132,
    0.0801,
    0.0855,
    0.0729,
    0.1255,
    0.2238,
    0.0554,
    0.1088,
    0.1727,
    0.0939,
    0.3294,
    0.3559,
    0.0532,
    0.1082,
    0.1554,
    0.0768,
    0.3446,
    0.3407,
    0.012,
    0.065,
    0.0584,
    0.0632,
    0.0198,
    0.1335,
    0.0631,
    0.125,
    0.1574,
    0.1047,
    0.073,
    0.2091,
    0.0759,
    0.1241,
    0.1667,
    0.1112,
    0.0831,
    0.2185,
    0.006,
    0.0441,
    0.0502,
    0.0441,
    0.0102,
    0.0946,
    0.0237,
    0.0722,
    0.061,
    0.0738,
    0.0479,
    0.0949,
    0.0369,
    0.0943,
    0.061,
    0.0966,
    0.0498,
    0.0729,
    0.0425,
    0.1001,
    0.1824,
    0.0972,
    0.0408,
    0.1887,
    0.0594,
    0.1842,
    0.1884,
    0.202,
    0.0457,
    0.1018,
    0.064,
    0.199,
    0.1854,
    0.2133,
    0.0467,
    0.091,
    0.0392,
    0.1049,
    0.1776,
    0.1037,
    0.0413,
    0.1945,
    0.1733,
    0.2612,
    0.1905,
    0.2963,
    0.1512,
    0.1861,
    0.171,
    0.2663,
    0.1896,
    0.3135,
    0.1568,
    0.2219,
    0.3976,
    0.1594,
    0.281,
    0.1855,
    0.0845,
    0.2398,
    0.4398,
    0.1629,
    0.2685,
    0.199,
    0.0998,
    0.2556,
    0.1137,
    0.2837,
    0.1419,
    0.2761,
    0.1678,
    0.2973,
    0.1172,
    0.301,
    0.1394,
    0.291,
    0.1724,
    0.3039,
    0.8831,
    0.7965,
    1.0899,
    1.1788,
    1.2128,
    1.1081,
    0.978,
    1.1434,
    0.8498,
    1.1462,
    0.7048,
    0.1713,
    0.6884,
    0.1548,
    0.1546,
    0.2403,
    0.607,
    0.5355,
    0.5873,
    0.6285,
    0.2336,
    0.7675,
    0.0064,
    0.007,
    0.0138,
]

assert len(_MM_STATS_MEAN) == 151 and len(_MM_STATS_STD) == 151


def decode_smpl_params(
    pred_x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> dict:
    """Decode normalized 151-dim network output to SMPL params.
    pred_x: (B, L, 151)
    Returns dict with body_pose, betas, global_orient, global_orient_gv, local_transl_vel.
    """
    B, L, _ = pred_x.shape
    x = pred_x * std + mean  # denormalize

    body_pose_r6d = x[..., :126]
    betas = x[..., 126:136]
    orient_c_r6d = x[..., 136:142]
    orient_gv_r6d = x[..., 142:148]
    local_transl_vel = x[..., 148:151]

    body_pose = matrix_to_axis_angle(
        rotation_6d_to_matrix(body_pose_r6d.reshape(B, L, -1, 6))
    ).flatten(-2)
    global_orient = matrix_to_axis_angle(rotation_6d_to_matrix(orient_c_r6d))
    global_orient_gv = matrix_to_axis_angle(rotation_6d_to_matrix(orient_gv_r6d))

    return {
        "body_pose": body_pose,  # (B, L, 63)
        "betas": betas,  # (B, L, 10)
        "global_orient": global_orient,  # (B, L, 3)  in camera
        "global_orient_gv": global_orient_gv,  # (B, L, 3)  in gravity-view
        "local_transl_vel": local_transl_vel,  # (B, L, 3)
    }


# ============================================================
# RoPE (Rotary Positional Embedding)
# ============================================================


def _rope_get_encoding(d_model: int, max_seq_len: int = 4096) -> torch.Tensor:
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
    freqs = torch.einsum("i,j->ij", t, freqs)
    return repeat(freqs, "i j -> i (j r)", r=2)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    return rearrange(torch.stack((-x2, x1), dim=-1), "... d r -> ... (d r)")


def _apply_rotary_emb(
    freqs: torch.Tensor, t: torch.Tensor, seq_dim: int = -2
) -> torch.Tensor:
    if t.ndim == 3:
        freqs = freqs[-t.shape[seq_dim] :].to(t)
    rot_dim = freqs.shape[-1]
    t_rot = t[..., :rot_dim]
    t_pass = t[..., rot_dim:]
    t_rot = (t_rot * freqs.cos()) + (_rotate_half(t_rot) * freqs.sin())
    return torch.cat((t_rot, t_pass), dim=-1)


class ROPE(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int = 4096):
        super().__init__()
        self.d_model = d_model
        encoding = _rope_get_encoding(d_model, max_seq_len)
        self.register_buffer("encoding", encoding, persistent=False)

    def rotate_queries_or_keys(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, H, L, D)"""
        seq_len, d_model = x.shape[-2:]
        if seq_len > self.encoding.shape[0]:
            enc = _rope_get_encoding(d_model, seq_len).to(x)
        else:
            enc = self.encoding[:seq_len]
        return _apply_rotary_emb(enc, x, seq_dim=-2)


# ============================================================
# Transformer Blocks
# ============================================================


def _zero_module(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        p.detach().zero_()
    return module


class RoPEAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.rope = ROPE(self.head_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        B, L, _ = x.shape
        xq = self.query(x).reshape(B, L, self.num_heads, -1).transpose(1, 2)
        xk = self.key(x).reshape(B, L, self.num_heads, -1).transpose(1, 2)
        xv = self.value(x).reshape(B, L, self.num_heads, -1).transpose(1, 2)
        xq = self.rope.rotate_queries_or_keys(xq)
        xk = self.rope.rotate_queries_or_keys(xk)
        score = einsum(xq, xk, "b n i c, b n j c -> b n i j") / math.sqrt(self.head_dim)
        if attn_mask is not None:
            score = score.masked_fill(attn_mask.reshape(1, 1, L, L), float("-inf"))
        if key_padding_mask is not None:
            score = score.masked_fill(
                key_padding_mask.reshape(B, 1, 1, L), float("-inf")
            )
        score = self.dropout(torch.softmax(score, dim=-1))
        out = einsum(score, xv, "b n i j, b n j c -> b n i c")
        return self.proj(out.transpose(1, 2).reshape(B, L, -1))


class EncoderRoPEBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn = RoPEAttention(hidden_size, num_heads, dropout)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            hidden_size,
            int(hidden_size * mlp_ratio),
            act_layer=approx_gelu,
            drop=dropout,
        )
        self.gate_msa = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.gate_mlp = nn.Parameter(torch.zeros(1, 1, hidden_size))

    def forward(self, x, attn_mask=None, tgt_key_padding_mask=None):
        x = x + self.gate_msa * self.attn(
            self.norm1(x), attn_mask=attn_mask, key_padding_mask=tgt_key_padding_mask
        )
        x = x + self.gate_mlp * self.mlp(self.norm2(x))
        return x


# ============================================================
# GVHMR Denoiser: NetworkEncoderRoPE
# ============================================================


def _length_to_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    return torch.arange(max_len, device=lengths.device).expand(
        len(lengths), max_len
    ) < lengths.unsqueeze(1)


class NetworkEncoderRoPE(nn.Module):
    """GVHMR transformer denoiser. Architecture matches gvhmr_siga24_release.ckpt."""

    def __init__(
        self,
        output_dim: int = 151,
        max_len: int = 120,
        cliffcam_dim: int = 3,
        cam_angvel_dim: int = 6,
        imgseq_dim: int = 1024,
        latent_dim: int = 512,
        num_layers: int = 12,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        pred_cam_dim: int = 3,
        static_conf_dim: int = 6,
        dropout: float = 0.1,
        avgbeta: bool = True,
    ):
        super().__init__()
        self.max_len = max_len
        self.avgbeta = avgbeta

        # 2-D keypoint embedding (17 joints × 32 features)
        self.learned_pos_linear = nn.Linear(2, 32)
        self.learned_pos_params = nn.Parameter(torch.randn(17, 32))
        self.embed_noisyobs = Mlp(
            17 * 32,
            hidden_features=latent_dim * 2,
            out_features=latent_dim,
            drop=dropout,
        )

        # Condition embedders
        self.cliffcam_embedder = nn.Sequential(
            nn.Linear(cliffcam_dim, latent_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            _zero_module(nn.Linear(latent_dim, latent_dim)),
        )
        self.cam_angvel_embedder = nn.Sequential(
            nn.Linear(cam_angvel_dim, latent_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            _zero_module(nn.Linear(latent_dim, latent_dim)),
        )
        self.imgseq_embedder = nn.Sequential(
            nn.LayerNorm(imgseq_dim),
            _zero_module(nn.Linear(imgseq_dim, latent_dim)),
        )

        # Transformer
        self.blocks = nn.ModuleList(
            [
                EncoderRoPEBlock(latent_dim, num_heads, mlp_ratio, dropout)
                for _ in range(num_layers)
            ]
        )

        # Output heads
        self.final_layer = Mlp(latent_dim, out_features=output_dim)

        self.pred_cam_head = Mlp(latent_dim, out_features=pred_cam_dim)
        self.register_buffer(
            "pred_cam_mean", torch.tensor([1.0606, -0.0027, 0.2702]), persistent=False
        )
        self.register_buffer(
            "pred_cam_std", torch.tensor([0.1784, 0.0956, 0.0764]), persistent=False
        )

        self.static_conf_head = Mlp(latent_dim, out_features=static_conf_dim)

    def forward(self, length, obs, f_cliffcam, f_cam_angvel, f_imgseq):
        """
        length:       (B,)  valid frame count per sample
        obs:          (B, L, 17, 3)  normalized 2D keypoints [x,y,conf]
        f_cliffcam:   (B, L, 3)
        f_cam_angvel: (B, L, 6)
        f_imgseq:     (B, L, 1024)
        """
        B, L, J, C = obs.shape
        obs = obs.clone()
        visible = obs[..., [2]] > 0.5
        obs[~visible[..., 0]] = 0.0

        f_obs = self.learned_pos_linear(obs[..., :2])  # (B, L, J, 32)
        f_obs = (
            f_obs * visible + self.learned_pos_params.expand(B, L, -1, -1) * ~visible
        )
        x = self.embed_noisyobs(f_obs.reshape(B, L, -1))  # (B, L, latent)

        x = x + self.cliffcam_embedder(f_cliffcam)
        x = x + self.cam_angvel_embedder(f_cam_angvel)
        x = x + self.imgseq_embedder(f_imgseq)

        pmask = ~_length_to_mask(length, L)  # (B, L) padding mask

        attnmask = None
        if L > self.max_len:
            attnmask = torch.ones(L, L, device=x.device, dtype=torch.bool)
            for i in range(L):
                lo = max(0, min(L - self.max_len, i - self.max_len // 2))
                hi = lo + self.max_len
                attnmask[i, lo:hi] = False

        for block in self.blocks:
            x = block(x, attn_mask=attnmask, tgt_key_padding_mask=pmask)

        sample = self.final_layer(x)  # (B, L, 151)
        if self.avgbeta:
            valid_mask = ~pmask  # (B, L)
            betas_avg = (sample[..., 126:136] * valid_mask[..., None]).sum(1) / length[
                :, None
            ]
            sample = torch.cat(
                [
                    sample[..., :126],
                    repeat(betas_avg, "b c -> b l c", l=L),
                    sample[..., 136:],
                ],
                dim=-1,
            )

        pred_cam = self.pred_cam_head(x) * self.pred_cam_std + self.pred_cam_mean
        pred_cam[..., 0].clamp_(min=0.25)

        static_conf_logits = self.static_conf_head(x)

        return {
            "pred_x": sample,
            "pred_cam": pred_cam,
            "static_conf_logits": static_conf_logits,
            "pred_context": x,
        }


# ============================================================
# GVHMRModel – weight loading and full inference
# ============================================================


class GVHMRModel(nn.Module):
    """Complete GVHMR inference wrapper."""

    def __init__(self, normalize_cam_angvel: bool = True):
        super().__init__()
        self.denoiser = NetworkEncoderRoPE()
        mean = torch.tensor(_MM_STATS_MEAN, dtype=torch.float32)
        std = torch.tensor(_MM_STATS_STD, dtype=torch.float32)
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std", std, persistent=False)

        self.normalize_cam_angvel = normalize_cam_angvel
        if normalize_cam_angvel:
            # Stats from stats_compose.py: cam_angvel["manual"]
            cam_mean = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=torch.float32)
            cam_std = torch.tensor(
                [0.001, 0.1, 0.1, 0.1, 0.001, 0.1], dtype=torch.float32
            )
            self.register_buffer("cam_angvel_mean", cam_mean, persistent=False)
            self.register_buffer("cam_angvel_std", cam_std, persistent=False)

    @classmethod
    def from_pretrained(cls, ckpt_path: str, device: str = "cuda") -> "GVHMRModel":
        model = cls()
        ckpt = torch.load(ckpt_path, map_location="cpu")
        # Weights may be inside 'state_dict' key
        raw = ckpt.get("state_dict", ckpt)
        # Strip 'pipeline.denoiser3d.' prefix
        prefix = "pipeline.denoiser3d."
        state = {k[len(prefix) :]: v for k, v in raw.items() if k.startswith(prefix)}
        model.denoiser.load_state_dict(state, strict=True)
        return model.to(device).eval()

    @torch.no_grad()
    def forward(
        self,
        obs: torch.Tensor,
        f_cliffcam: torch.Tensor,
        f_cam_angvel: torch.Tensor,
        f_imgseq: torch.Tensor,
        bbx_xys: torch.Tensor,
        K_fullimg: torch.Tensor,
        length: torch.Tensor = None,
        transl_0: torch.Tensor = None,
        foot_contact_threshold: float = None,
    ) -> dict:
        """Full GVHMR inference for one person track.

        All tensors should be on the same device as the model.

        Args:
            obs:                    (B, L, 17, 3)  normalized 2D kps
            f_cliffcam:             (B, L, 3)
            f_cam_angvel:           (B, L, 6)
            f_imgseq:               (B, L, 1024)
            bbx_xys:                (B, L, 3)      raw bbox [cx, cy, s]
            K_fullimg:              (B, 3, 3) or (B, L, 3, 3)
            length:                 (B,) int, defaults to full sequence
            transl_0:               (B, 1, 3) world-space position of the first
                                    frame (carry-over anchor for rolling window).
                                    None → start at origin.
            foot_contact_threshold: sigmoid probability above which a frame is
                                    treated as foot-in-contact and its horizontal
                                    root velocity is zeroed before integration.
                                    None → no foot contact constraint applied.

        Returns:
            dict with:
                smpl_params_incam:  body_pose, betas, global_orient, transl (camera frame)
                smpl_params_global: body_pose, betas, global_orient, transl (world frame)
                static_conf_logits: (B, L, 6) raw foot contact logits
        """
        B, L = obs.shape[:2]
        if length is None:
            length = torch.full((B,), L, dtype=torch.long, device=obs.device)

        # Optional normalization of camera angular velocity
        f_cam_angvel_in = f_cam_angvel
        if self.normalize_cam_angvel:
            f_cam_angvel_in = (
                f_cam_angvel - self.cam_angvel_mean
            ) / self.cam_angvel_std

        out = self.denoiser(
            length=length,
            obs=obs,
            f_cliffcam=f_cliffcam,
            f_cam_angvel=f_cam_angvel_in,
            f_imgseq=f_imgseq,
        )

        # Expand K to (B, L, 3, 3) if needed
        if K_fullimg.dim() == 3:
            K_fullimg = K_fullimg.unsqueeze(1).expand(-1, L, -1, -1)

        decode = decode_smpl_params(out["pred_x"], self.mean, self.std)

        # In-camera params
        transl_incam = compute_transl_incam(out["pred_cam"], bbx_xys, K_fullimg)
        smpl_params_incam = {
            "body_pose": decode["body_pose"],
            "betas": decode["betas"],
            "global_orient": decode["global_orient"],
            "transl": transl_incam,
        }

        # Optionally apply foot-contact constraint before velocity integration
        local_transl_vel = decode["local_transl_vel"]
        if foot_contact_threshold is not None:
            local_transl_vel = apply_foot_contact_constraints(
                local_transl_vel,
                out["static_conf_logits"],
                conf_threshold=foot_contact_threshold,
            )

        # Global (world) params — transl_0 carries the anchor across window slides
        global_rt = get_smpl_params_global(
            global_orient_gv=decode["global_orient_gv"],
            local_transl_vel=local_transl_vel,
            global_orient_c=decode["global_orient"],
            cam_angvel=f_cam_angvel,
            transl_0=transl_0,
        )
        smpl_params_global = {
            "body_pose": decode["body_pose"],
            "betas": decode["betas"],
            **global_rt,
        }

        return {
            "smpl_params_incam": smpl_params_incam,
            "smpl_params_global": smpl_params_global,
            "static_conf_logits": out["static_conf_logits"],
        }
