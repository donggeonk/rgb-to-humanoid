"""Shared utilities for VIBE demo scripts."""

import cv2
import numpy as np
from collections import deque

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


# ------------------------------------------------------------------ #
#  Temporal smoothing helpers
# ------------------------------------------------------------------ #

def smooth_sequence(arr, window=5):
    """Apply causal (one-sided) moving-average to a (N, D) array.

    Each output frame is the mean of the current and up to ``window-1``
    preceding frames, so there is no look-ahead and no latency increase.
    """
    if arr.ndim == 1:
        arr = arr[:, None]
        squeeze = True
    else:
        squeeze = False

    N = len(arr)
    out = np.empty_like(arr)
    cumsum = np.cumsum(arr, axis=0)

    for i in range(N):
        lo = max(0, i - window + 1)
        span = i - lo + 1
        if lo == 0:
            out[i] = cumsum[i] / span
        else:
            out[i] = (cumsum[i] - cumsum[lo - 1]) / span

    return out.squeeze(-1) if squeeze else out


def smooth_bboxes(bboxes, window=5):
    """Smooth a list of bounding boxes ([x1,y1,x2,y2] or None).

    Missing (None) entries are forward-filled before smoothing.
    Returns an (N, 4) float array.
    """
    N = len(bboxes)
    arr = np.zeros((N, 4), dtype=np.float64)

    last_valid = None
    for i in range(N):
        if bboxes[i] is not None:
            last_valid = bboxes[i]
        if last_valid is not None:
            arr[i] = last_valid

    return smooth_sequence(arr, window)


class OneEuroFilter:
    """One-Euro filter for real-time scalar / vector smoothing.

    Provides low latency at slow speeds and strong smoothing at fast speeds.
    Good default: min_cutoff=1.0, beta=0.007, d_cutoff=1.0.
    """

    def __init__(self, dim, fps=30.0, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.fps = fps
        self.x_prev = np.zeros(dim)
        self.dx_prev = np.zeros(dim)
        self.initialized = False

    def _alpha(self, cutoff):
        tau = 1.0 / (2.0 * np.pi * cutoff)
        te = 1.0 / self.fps
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x):
        x = np.asarray(x, dtype=np.float64)
        if not self.initialized:
            self.x_prev = x.copy()
            self.dx_prev = np.zeros_like(x)
            self.initialized = True
            return x.copy()

        # Derivative
        a_d = self._alpha(self.d_cutoff)
        dx = (x - self.x_prev) * self.fps
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev

        # Adaptive cutoff
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = self._alpha(cutoff)

        x_hat = a * x + (1.0 - a) * self.x_prev

        self.x_prev = x_hat.copy()
        self.dx_prev = dx_hat.copy()
        return x_hat


class SMPLRenderer:
    """Render SMPL mesh using the same weak-perspective projection as joints,
    with depth-sorted face rasterization via OpenCV.  No extra dependencies."""

    def __init__(self, faces):
        """
        Args:
            faces: (F, 3) int array – SMPL triangle indices.
        """
        self.faces = faces

    # ------------------------------------------------------------------ #
    #  public API
    # ------------------------------------------------------------------ #

    def render_on_image(self, img_bgr, vertices, cam, bbox,
                        img_h, img_w, color=(0.65, 0.75, 0.90), alpha=0.75):
        """Render the SMPL mesh on top of a BGR frame.

        Uses the *same* weak-perspective camera model as ``project_joints_2d``
        so the mesh aligns exactly with the skeleton overlay.

        Args:
            img_bgr:  (H, W, 3) uint8 BGR image.
            vertices: (6890, 3) SMPL vertices.
            cam:      (3,) weak-perspective camera [s, tx, ty].
            bbox:     [x1, y1, x2, y2] or None.
            img_h:    Original image height.
            img_w:    Original image width.
            color:    Base RGB color for the mesh (0-1 floats).
            alpha:    Overlay opacity in [0, 1].

        Returns:
            (H, W, 3) uint8 BGR image with mesh overlay.
        """
        # --- 1. Project vertices to 2D (crop space) ---
        s, tx, ty = cam[0], cam[1], cam[2]
        verts_2d = vertices[:, :2] * s + np.array([[tx, ty]])
        verts_2d = (verts_2d + 1.0) * 112.0          # 224 / 2

        # --- 2. Transform crop-space → original image coords ---
        verts_2d = crop_to_image_coords(verts_2d, bbox, img_h, img_w)

        # --- 3. Per-face depth (use mean Z for sorting) ---
        face_verts = vertices[self.faces]             # (F, 3, 3)
        face_depth = face_verts[:, :, 2].mean(axis=1) # (F,)

        # --- 4. Simple directional shading from face normals ---
        v0 = face_verts[:, 0]
        v1 = face_verts[:, 1]
        v2 = face_verts[:, 2]
        normals = np.cross(v1 - v0, v2 - v0)
        norms = np.linalg.norm(normals, axis=1, keepdims=True).clip(1e-8)
        normals = normals / norms

        light_dir = np.array([0.0, 0.0, 1.0])       # toward camera
        diffuse = np.abs(normals @ light_dir)         # (F,)
        ambient = 0.4
        intensity = np.clip(ambient + (1.0 - ambient) * diffuse, 0.0, 1.0)

        # --- 5. Sort faces back-to-front (painter's algorithm) ---
        order = np.argsort(-face_depth)

        # --- 6. Rasterise on a copy of the image ---
        overlay = img_bgr.copy()
        base_color = np.array(color)                  # RGB in 0-1

        tri_pts = verts_2d[self.faces]                # (F, 3, 2)

        for idx in order:
            tri = tri_pts[idx].astype(np.int32)
            c = (base_color * intensity[idx] * 255).astype(np.uint8)
            c_bgr = (int(c[2]), int(c[1]), int(c[0]))
            cv2.fillConvexPoly(overlay, tri, c_bgr, lineType=cv2.LINE_AA)

        # --- 7. Alpha-blend only where mesh pixels differ ---
        output = cv2.addWeighted(overlay, alpha, img_bgr, 1.0 - alpha, 0)
        return output


def convert_vibe_to_smplx(pose_aa, betas, cam, fps, gender='neutral'):
    """Convert VIBE output arrays to SMPL-X npz dict.

    Args:
        pose_aa: (N, 24, 3) axis-angle per joint.
        betas:   (N, 10) shape params per frame.
        cam:     (N, 3) weak-perspective camera [s, tx, ty].
        fps:     float, video frame rate.
        gender:  str, 'neutral' / 'male' / 'female'.

    Returns:
        dict ready to be saved with ``np.savez(path, **d)``.

    The output contains the keys expected by GMR's ``load_smplx_file``:
        root_orient      (N, 3)
        pose_body        (N, 63)   – 21 body joints × 3
        betas            (16,)     – padded from 10 to 16
        trans            (N, 3)    – derived from weak-perspective cam
        gender           str
        mocap_frame_rate scalar
    """
    N = pose_aa.shape[0]

    # Split axis-angle: joint 0 = root, joints 1-21 = body (ignore 22-23 hands)
    root_orient = pose_aa[:, 0, :]                          # (N, 3)
    pose_body   = pose_aa[:, 1:22, :].reshape(N, 63)       # (N, 63)

    # Average betas across frames (shape is constant per person)
    betas_mean = betas.mean(axis=0)                         # (10,)
    betas_padded = np.concatenate([betas_mean, np.zeros(6, dtype=betas_mean.dtype)])  # (16,)

    # Derive a plausible global translation from the weak-perspective camera.
    # VIBE cam = [s, tx, ty].  A common heuristic (used e.g. in SPIN/VIBE):
    #   tz = 2 * focal_length / (crop_size * s)
    # with focal_length ≈ 5000 and crop_size = 224 this gives tz ≈ 44.6/s.
    # We use the convention from the VIBE paper.
    focal_length = 5000.0
    crop_size = 224.0
    tz = 2.0 * focal_length / (crop_size * cam[:, 0])      # (N,)
    trans = np.stack([cam[:, 1], cam[:, 2], tz], axis=-1)   # (N, 3)

    return {
        'root_orient':      root_orient,
        'pose_body':        pose_body,
        'betas':            betas_padded,
        'trans':            trans,
        'gender':           np.array(gender),
        'mocap_frame_rate': np.array(fps),
    }
