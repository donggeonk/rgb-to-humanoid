"""
Standalone VIBE model for inference.
Uses pretrained weights from mmhuman3d (resnet50_vibe_pw3d).

Architecture:
  ResNet50 (backbone) → TemporalGRUEncoder (neck) → HMRHead (head) → SMPL (body model)

The checkpoint contains neck, head, and SMPL body model weights.
The backbone uses torchvision ImageNet-pretrained ResNet50.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np

SPIN_URL = (
    'https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/'
    'vibe/spin.pth?versionId=CAEQKhiBgMD9hdub9xciIDIxMGI4NmMxMGIy'
    'NzQxOGM5YzYxZjMyYTVmMjllOTAy'
)


# ============================================================
# Rotation Utilities
# ============================================================

def rot6d_to_rotmat(x):
    """Convert 6D rotation representation to 3x3 rotation matrices.
    Reference: Zhou et al., "On the Continuity of Rotation Representations
    in Neural Networks", CVPR 2019.
    """
    batch_size = x.shape[0]
    x = x.reshape(-1, 3, 2)
    a1, a2 = x[:, :, 0], x[:, :, 1]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


def rotmat_to_angle_axis(R):
    """Convert rotation matrices to axis-angle. R: (..., 3, 3) → (..., 3)."""
    batch_shape = R.shape[:-2]
    R = R.reshape(-1, 3, 3)

    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    cos_angle = ((trace - 1.0) / 2.0).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    angle = torch.acos(cos_angle)

    r = torch.stack([
        R[:, 2, 1] - R[:, 1, 2],
        R[:, 0, 2] - R[:, 2, 0],
        R[:, 1, 0] - R[:, 0, 1],
    ], dim=-1)

    sin_angle = torch.sin(angle)
    axis = r / (2.0 * sin_angle.unsqueeze(-1) + 1e-8)
    axis_angle = axis * angle.unsqueeze(-1)

    # Zero out near-identity rotations
    small = (angle.abs() < 1e-6).unsqueeze(-1)
    axis_angle = torch.where(small, torch.zeros_like(axis_angle), axis_angle)
    return axis_angle.reshape(*batch_shape, 3)


# ============================================================
# Minimal SMPL Layer (Linear Blend Skinning)
# ============================================================

class SMPLLayer(nn.Module):
    """Minimal SMPL forward pass. Parameters loaded from mmhuman3d checkpoint."""

    def __init__(self):
        super().__init__()
        self.register_buffer('v_template', torch.zeros(6890, 3))
        self.register_buffer('shapedirs', torch.zeros(6890, 3, 10))
        self.register_buffer('posedirs', torch.zeros(207, 20670))
        self.register_buffer('J_regressor', torch.zeros(24, 6890))
        self.register_buffer('lbs_weights', torch.zeros(6890, 24))
        self.register_buffer('parents', torch.zeros(24, dtype=torch.long))
        self.register_buffer('faces', torch.zeros(13776, 3, dtype=torch.long))

    def forward(self, rotmats, betas):
        """
        Args:
            rotmats: (B, 24, 3, 3) per-joint rotation matrices
            betas:   (B, 10) shape parameters
        Returns:
            vertices: (B, 6890, 3)
            joints:   (B, 24, 3)
        """
        B = rotmats.shape[0]
        device = rotmats.device

        # 1. Shape blend shapes
        v_shaped = self.v_template.unsqueeze(0) + torch.einsum(
            'vci,bi->bvc', self.shapedirs, betas)

        # 2. Joint locations
        J = torch.einsum('jv,bvc->bjc', self.J_regressor, v_shaped)

        # 3. Pose blend shapes (exclude root joint)
        ident = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0)
        pose_feature = (rotmats[:, 1:] - ident).reshape(B, -1)  # (B, 207)
        pose_offsets = torch.matmul(
            pose_feature, self.posedirs).reshape(B, 6890, 3)
        v_posed = v_shaped + pose_offsets

        # 4. Forward kinematics
        posed_joints, A = self._forward_kinematics(rotmats, J)

        # 5. Linear blend skinning
        W = self.lbs_weights.unsqueeze(0).expand(B, -1, -1)
        T = torch.matmul(W, A.reshape(B, 24, 16)).reshape(B, 6890, 4, 4)
        v_homo = F.pad(v_posed, [0, 1], value=1.0)
        vertices = torch.matmul(T, v_homo.unsqueeze(-1)).squeeze(-1)[:, :, :3]

        return vertices, posed_joints

    def _forward_kinematics(self, rot_mats, joints):
        B = rot_mats.shape[0]

        # Relative joint positions
        rel_joints = joints.clone()
        rel_joints[:, 1:] -= joints[:, self.parents[1:]]

        # Local transforms [R | t; 0 | 1]
        transforms = torch.zeros(B, 24, 4, 4,
                                 device=rot_mats.device, dtype=rot_mats.dtype)
        transforms[:, :, :3, :3] = rot_mats
        transforms[:, :, :3, 3] = rel_joints
        transforms[:, :, 3, 3] = 1.0

        # Chain through kinematic tree
        chain = [transforms[:, 0]]
        for i in range(1, 24):
            chain.append(torch.bmm(chain[self.parents[i]], transforms[:, i]))
        global_tf = torch.stack(chain, dim=1)

        posed_joints = global_tf[:, :, :3, 3].clone()

        # Compute skinning transforms (subtract rest-pose joint offset)
        joints_homo = F.pad(joints.unsqueeze(-1), [0, 0, 0, 1])  # (B,24,4,1)
        init_bone = F.pad(
            torch.matmul(global_tf, joints_homo), [3, 0])  # (B,24,4,4)
        A = global_tf - init_bone

        return posed_joints, A


# ============================================================
# Backbone: ResNet50 Feature Extractor
# ============================================================

class ResNet50Backbone(nn.Module):
    """ResNet50 backbone outputting 2048-dim features per image.
    Initialized from SPIN pretrained weights for human mesh recovery."""

    def __init__(self):
        super().__init__()
        # Build ResNet50 without pretrained weights (will load SPIN weights)
        resnet = models.resnet50(weights=None)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool

    def load_spin_weights(self, spin_state_dict):
        """Load backbone layers from SPIN checkpoint (flat keys)."""
        backbone_keys = {}
        for k, v in spin_state_dict.items():
            # Only load backbone params (conv1, bn1, layer1-4)
            if k.startswith(('conv1.', 'bn1.', 'layer1.', 'layer2.',
                             'layer3.', 'layer4.')):
                backbone_keys[k] = v
        missing, unexpected = self.load_state_dict(backbone_keys, strict=False)
        # avgpool has no learnable params, so missing keys for it are fine
        loaded = len(backbone_keys) - len(unexpected)
        print(f"[Backbone] Loaded {loaded} SPIN params "
              f"(skipped {len(missing)} non-backbone keys)")

    def forward(self, x):
        """x: (B, 3, 224, 224) → (B, 2048)"""
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.avgpool(x).flatten(1)


# ============================================================
# Temporal GRU Encoder (neck)
# ============================================================

class TemporalGRUEncoder(nn.Module):
    """Bidirectional=False GRU with linear projection and residual connection."""

    def __init__(self, input_size=2048, num_layers=2, hidden_size=1024):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, bidirectional=False)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        """x: (B, T, 2048) → (B, T, 2048)"""
        N, T = x.shape[:2]
        x = x.permute(1, 0, 2)                           # (T, N, F)
        y, _ = self.gru(x)
        y = self.linear(self.relu(y).reshape(-1, self.hidden_size))
        y = y.reshape(T, N, self.input_size) + x          # residual
        return y.permute(1, 0, 2).contiguous()             # (N, T, F)


# ============================================================
# HMR Regression Head
# ============================================================

class HMRHead(nn.Module):
    """Iterative SMPL parameter regressor (3 iterations)."""

    def __init__(self, feat_dim=2048, npose=144, nbeta=10, ncam=3, hdim=1024):
        super().__init__()
        self.fc1 = nn.Linear(feat_dim + npose + nbeta + ncam, hdim)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(hdim, hdim)
        self.drop2 = nn.Dropout()
        self.decpose = nn.Linear(hdim, npose)
        self.decshape = nn.Linear(hdim, nbeta)
        self.deccam = nn.Linear(hdim, ncam)

        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        self.register_buffer('init_pose', torch.zeros(1, npose))
        self.register_buffer('init_shape', torch.zeros(1, nbeta))
        self.register_buffer('init_cam', torch.FloatTensor([[1, 0, 0]]))

    def forward(self, x, n_iter=3):
        """x: (B, T, 2048) or (B, 2048). Returns dict of predictions."""
        output_seq = False
        if x.dim() == 3:
            output_seq = True
            B, T, L = x.shape
            x = x.reshape(-1, L)

        batch_size = x.shape[0]
        pred_pose = self.init_pose.expand(batch_size, -1)
        pred_shape = self.init_shape.expand(batch_size, -1)
        pred_cam = self.init_cam.expand(batch_size, -1)

        for _ in range(n_iter):
            xc = torch.cat([x, pred_pose, pred_shape, pred_cam], 1)
            xc = self.drop1(self.fc1(xc))
            xc = self.drop2(self.fc2(xc))
            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape
            pred_cam = self.deccam(xc) + pred_cam

        pred_rotmat = rot6d_to_rotmat(pred_pose).reshape(batch_size, 24, 3, 3)

        if output_seq:
            pred_rotmat = pred_rotmat.reshape(B, T, 24, 3, 3)
            pred_shape = pred_shape.reshape(B, T, 10)
            pred_cam = pred_cam.reshape(B, T, 3)

        return {
            'pred_rotmat': pred_rotmat,
            'pred_shape': pred_shape,
            'pred_cam': pred_cam,
        }


# ============================================================
# Complete VIBE Model
# ============================================================

# SMPL skeleton connectivity for visualization
SMPL_SKELETON = [
    (0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6),
    (4, 7), (5, 8), (6, 9), (7, 10), (8, 11),
    (9, 12), (9, 13), (9, 14), (12, 15),
    (13, 16), (14, 17), (16, 18), (17, 19),
    (18, 20), (19, 21), (20, 22), (21, 23),
]

SMPL_JOINT_NAMES = [
    'Pelvis', 'L_Hip', 'R_Hip', 'Spine1', 'L_Knee', 'R_Knee',
    'Spine2', 'L_Ankle', 'R_Ankle', 'Spine3', 'L_Foot', 'R_Foot',
    'Neck', 'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder',
    'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand',
]


class VIBE(nn.Module):
    """
    Complete VIBE model for video-based human pose and shape estimation.

    Forward pass:
        images (B, T, 3, 224, 224)
        → ResNet50 features (B, T, 2048)
        → GRU temporal encoding (B, T, 2048)
        → HMR iterative regression → SMPL params
        → SMPL forward pass → vertices + joints
    """

    def __init__(self):
        super().__init__()
        self.backbone = ResNet50Backbone()
        self.temporal_encoder = TemporalGRUEncoder(
            input_size=2048, num_layers=2, hidden_size=1024)
        self.hmr_head = HMRHead(feat_dim=2048)
        self.smpl = SMPLLayer()

    @classmethod
    def from_pretrained(cls, checkpoint_path, spin_path=None, device='cpu'):
        """Load model from mmhuman3d checkpoint + SPIN backbone.

        Args:
            checkpoint_path: path to resnet50_vibe_pw3d-*.pth
            spin_path: path to spin.pth (auto-downloaded if None)
            device: target device
        """
        model = cls()
        ckpt = torch.load(checkpoint_path, map_location='cpu',
                          weights_only=False)
        sd = ckpt['state_dict']

        # --- Load SPIN backbone ---
        if spin_path is None:
            spin_path = os.path.join(os.path.dirname(checkpoint_path),
                                     'spin.pth')
        if not os.path.exists(spin_path):
            print(f"[VIBE] Downloading SPIN backbone → {spin_path}")
            torch.hub.download_url_to_file(SPIN_URL, spin_path)
        spin_ckpt = torch.load(spin_path, map_location='cpu',
                               weights_only=False)
        spin_sd = spin_ckpt['state_dict']
        model.backbone.load_spin_weights(spin_sd)

        # --- Load temporal encoder (neck.*) ---
        neck_sd = {k[5:]: v for k, v in sd.items() if k.startswith('neck.')}
        model.temporal_encoder.load_state_dict(neck_sd)

        # --- Load HMR head (head.*) ---
        head_sd = {k[5:]: v for k, v in sd.items() if k.startswith('head.')}
        model.hmr_head.load_state_dict(head_sd)

        # --- Load SMPL body model (body_model_test.*) ---
        prefix = 'body_model_test.'
        model.smpl.v_template.copy_(sd[f'{prefix}v_template'])
        model.smpl.shapedirs.copy_(sd[f'{prefix}shapedirs'])
        model.smpl.posedirs.copy_(sd[f'{prefix}posedirs'])
        model.smpl.J_regressor.copy_(sd[f'{prefix}J_regressor'])
        model.smpl.lbs_weights.copy_(sd[f'{prefix}lbs_weights'])
        model.smpl.parents.copy_(sd[f'{prefix}parents'].long())
        model.smpl.faces.copy_(sd[f'{prefix}faces_tensor'].long())

        model = model.to(device).eval()
        print(f"[VIBE] Loaded checkpoint from {checkpoint_path}")
        print(f"[VIBE] Backbone: SPIN-pretrained ResNet50 ({spin_path})")
        print(f"[VIBE] Neck/Head/SMPL: from mmhuman3d checkpoint")
        return model

    @torch.no_grad()
    def extract_features(self, images):
        """Extract backbone features. images: (B, 3, 224, 224) → (B, 2048)"""
        return self.backbone(images)

    @torch.no_grad()
    def forward(self, images=None, features=None, run_smpl=True):
        """
        Run full inference.

        Args:
            images:   (B, T, 3, 224, 224) preprocessed image sequences
            features: (B, T, 2048) pre-extracted features (alternative to images)
            run_smpl: whether to compute vertices/joints via SMPL

        Returns:
            dict with keys: pred_rotmat, pred_shape, pred_cam,
                            [vertices, joints, pose_axis_angle]
        """
        if features is None:
            B, T = images.shape[:2]
            features = self.backbone(images.reshape(B * T, *images.shape[2:]))
            features = features.reshape(B, T, -1)

        B, T = features.shape[:2]

        # Temporal encoding
        features = self.temporal_encoder(features)

        # HMR regression
        output = self.hmr_head(features)

        if run_smpl:
            rotmats = output['pred_rotmat'].reshape(B * T, 24, 3, 3)
            betas = output['pred_shape'].reshape(B * T, 10)
            verts, joints = self.smpl(rotmats, betas)
            output['vertices'] = verts.reshape(B, T, 6890, 3)
            output['joints'] = joints.reshape(B, T, 24, 3)

            # Also provide axis-angle representation
            aa = rotmat_to_angle_axis(rotmats.reshape(-1, 3, 3))
            output['pose_axis_angle'] = aa.reshape(B, T, 24, 3)

        return output


# ============================================================
# Preprocessing Utilities
# ============================================================

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])


def preprocess_image(img_bgr, bbox=None, target_size=224):
    """
    Preprocess a BGR image for VIBE.
    Returns: (3, 224, 224) float tensor, normalized.
    """
    import cv2

    if bbox is not None:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        size = max(x2 - x1, y2 - y1) * 1.2
        x1c = int(cx - size / 2)
        y1c = int(cy - size / 2)
        x2c = int(cx + size / 2)
        y2c = int(cy + size / 2)

        # Pad if needed
        h, w = img_bgr.shape[:2]
        pad_l = max(0, -x1c)
        pad_t = max(0, -y1c)
        pad_r = max(0, x2c - w)
        pad_b = max(0, y2c - h)
        if pad_l or pad_t or pad_r or pad_b:
            img_bgr = cv2.copyMakeBorder(
                img_bgr, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT)
            x1c += pad_l
            y1c += pad_t
            x2c += pad_l
            y2c += pad_t
        crop = img_bgr[y1c:y2c, x1c:x2c]
    else:
        # Center crop (square)
        h, w = img_bgr.shape[:2]
        size = min(h, w)
        y1c = (h - size) // 2
        x1c = (w - size) // 2
        crop = img_bgr[y1c:y1c + size, x1c:x1c + size]

    crop = cv2.resize(crop, (target_size, target_size))
    # BGR → RGB → float → normalize
    img = torch.from_numpy(crop[:, :, ::-1].copy()).float() / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return img.permute(2, 0, 1)  # (3, H, W)


def project_joints_2d(joints_3d, pred_cam, img_size=224):
    """
    Project 3D joints to 2D using weak-perspective camera.
    joints_3d: (N, 3), pred_cam: (3,) = [s, tx, ty]
    Returns: (N, 2) pixel coordinates in [0, img_size].
    """
    s, tx, ty = pred_cam[0], pred_cam[1], pred_cam[2]
    j2d = joints_3d[:, :2] * s.unsqueeze(-1) + torch.stack([tx, ty])
    # Normalized [-1, 1] → pixel [0, img_size]
    return (j2d + 1.0) * (img_size / 2.0)


def detect_persons(frame_bgr, detector=None):
    """
    Detect persons in a BGR frame.
    Returns list of [x1, y1, x2, y2, confidence] arrays.
    Falls back to center crop if no detector is available.
    """
    if detector is not None:
        results = detector(frame_bgr, verbose=False, classes=[0])  # class 0 = person
        boxes = results[0].boxes
        if len(boxes) == 0:
            return []
        return [
            [*box.xyxy[0].cpu().numpy(), box.conf[0].cpu().item()]
            for box in [boxes[i] for i in range(len(boxes))]
        ]
    else:
        # Fallback: use full frame as one detection
        h, w = frame_bgr.shape[:2]
        return [[0, 0, w, h, 1.0]]


def load_detector(device='cpu'):
    """Try to load YOLOv8 person detector."""
    try:
        from ultralytics import YOLO
        detector = YOLO('yolov8n.pt')
        detector.to(device)
        print("[Detector] YOLOv8n loaded")
        return detector
    except Exception as e:
        print(f"[Detector] YOLOv8 not available ({e}), using center crop")
        return None
