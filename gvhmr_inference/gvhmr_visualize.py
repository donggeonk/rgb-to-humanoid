#!/usr/bin/env python3
"""
Render SMPL-X mesh overlay from gvhmr_demo.py output (.npz).

Usage:
  python gvhmr_visualize.py --video input.mp4 --npz results.npz --out overlay.mp4
"""

import argparse
import os
import warnings
import cv2
import numpy as np
import pyrender
import smplx
import torch
import trimesh
from tqdm import tqdm

warnings.filterwarnings("ignore")

SMPLX_MODEL_DEFAULT = os.path.join(
    os.path.dirname(__file__),
    "../GMR/assets/body_models/smplx/SMPLX_NEUTRAL.npz",
)


def _get_smplx_vertices(global_orient, body_pose, betas, transl, model_path, device="cpu", batch_size=64):
    model = smplx.create(model_path, model_type="smplx", num_betas=10, use_pca=False, flat_hand_mean=True).to(device)
    model.eval()
    if body_pose.shape[1] > 63:
        body_pose = body_pose[:, :63]
    F = body_pose.shape[0]
    if betas.ndim == 1:
        betas = betas.unsqueeze(0).expand(F, -1)
    if betas.shape[1] > 10:
        betas = betas[:, :10]

    all_verts = []
    with torch.no_grad():
        for s in tqdm(range(0, F, batch_size), desc="SMPL-X", leave=False):
            e = min(s + batch_size, F)
            B = e - s
            out = model(global_orient=global_orient[s:e].to(device),
                        body_pose=body_pose[s:e].to(device),
                        betas=betas[s:e].to(device),
                        transl=transl[s:e].to(device),
                        expression=torch.zeros(B, 10, device=device),
                        jaw_pose=torch.zeros(B, 3, device=device),
                        leye_pose=torch.zeros(B, 3, device=device),
                        reye_pose=torch.zeros(B, 3, device=device),
                        left_hand_pose=torch.zeros(B, 45, device=device),
                        right_hand_pose=torch.zeros(B, 45, device=device))
            all_verts.append(out.vertices.cpu().numpy())
    return np.concatenate(all_verts, 0)


class IncamRenderer:
    def __init__(self, faces, H, W, K):
        self.faces, self.H, self.W = faces, H, W
        self._scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.4] * 3)
        self._scene.add(pyrender.IntrinsicsCamera(K[0, 0], K[1, 1], K[0, 2], K[1, 2], 0.05, 100.0))
        self._scene.add(pyrender.DirectionalLight([1] * 3, 4.0))
        self._material = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.65, 0.75, 0.9, 1.0], roughnessFactor=0.8)
        self._mesh_node, self._renderer = None, pyrender.OffscreenRenderer(W, H)

    def overlay(self, img, verts, alpha=0.75):
        v_gl = verts.copy(); v_gl[:, 1:3] *= -1
        pr_mesh = pyrender.Mesh.from_trimesh(trimesh.Trimesh(v_gl, self.faces, process=False), material=self._material)
        if self._mesh_node:
            self._scene.remove_node(self._mesh_node)
        self._mesh_node = self._scene.add(pr_mesh)
        rgba, _ = self._renderer.render(self._scene, flags=pyrender.RenderFlags.RGBA)
        mask = rgba[:, :, 3] > 0
        out = img.copy().astype(np.float32)
        out[mask] = alpha * rgba[mask, :3][:, ::-1] + (1 - alpha) * out[mask]
        return out.astype(np.uint8)


class GlobalRenderer:
    def __init__(self, faces, H, W, transl):
        self.faces, self.H, self.W = faces, H, W
        center, span = transl.mean(0), np.linalg.norm(transl.max(0) - transl.min(0)) + 2.0
        cam_pos = center + np.array([0, span * 0.6, -span * 0.8])
        fwd = (center - cam_pos) / np.linalg.norm(center - cam_pos)
        right = np.cross(fwd, [0, 1, 0]); right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        pose = np.eye(4); pose[:3, 0], pose[:3, 1], pose[:3, 2], pose[:3, 3] = right, up, -fwd, cam_pos
        self._scene = pyrender.Scene(bg_color=[30, 30, 30, 255], ambient_light=[0.35] * 3)
        self._scene.add(pyrender.PerspectiveCamera(np.radians(55), W / H), pose=pose)
        self._scene.add(pyrender.DirectionalLight([1] * 3, 4.5), pose=pose)
        self._material = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.65, 0.75, 0.9, 1.0], roughnessFactor=0.8)
        self._mesh_node, self._renderer = None, pyrender.OffscreenRenderer(W, H)
        self._add_floor(center, span)

    def _add_floor(self, c, s):
        n, step = 20, s / 20
        v, f, clr = [], [], []
        for i in range(n):
            for j in range(n):
                x0, x1 = c[0] + (i - n / 2) * step, c[0] + (i - n / 2 + 1) * step
                z0, z1 = c[2] + (j - n / 2) * step, c[2] + (j - n / 2 + 1) * step
                v.append([[x0, c[1] - 1, z0], [x1, c[1] - 1, z0], [x1, c[1] - 1, z1], [x0, c[1] - 1, z1]])
                f += [[len(f) * 2, len(f) * 2 + 1, len(f) * 2 + 2], [len(f) * 2, len(f) * 2 + 2, len(f) * 2 + 3]]
                clr += [[0.6 if (i + j) % 2 == 0 else 0.5] * 3] * 4
        self._scene.add(pyrender.Mesh.from_trimesh(trimesh.Trimesh(
            np.concatenate(v, 0), np.array(f),
            vertex_colors=(np.array(clr) * 255).astype(np.uint8))))

    def render(self, verts):
        pr_mesh = pyrender.Mesh.from_trimesh(trimesh.Trimesh(verts, self.faces, process=False), material=self._material)
        if self._mesh_node:
            self._scene.remove_node(self._mesh_node)
        self._mesh_node = self._scene.add(pr_mesh)
        return self._renderer.render(self._scene)[0]


def _parse_npz(path):
    """Load .npz and return (global_params, incam_params, K, fps) dicts with torch tensors."""
    raw = np.load(path)
    glob = {
        "global_orient": torch.from_numpy(raw["root_orient"]),
        "body_pose": torch.from_numpy(raw["pose_body"]),
        "transl": torch.from_numpy(raw["trans"]),
        "betas": torch.from_numpy(raw["betas"]),
    }
    incam = {
        "global_orient": torch.from_numpy(raw["cam_root_orient"]),
        "body_pose": torch.from_numpy(raw["cam_pose_body"]),
        "transl": torch.from_numpy(raw["cam_trans"]),
        "betas": torch.from_numpy(raw["betas"]),
    }
    K = raw["K_fullimg"] if "K_fullimg" in raw else None
    fps = float(raw.get("mocap_frame_rate", 30.0))
    return glob, incam, K, fps


def visualize(video_path, npz_path, out_path="gvhmr_overlay.mp4",
              mode="incam", alpha=0.75, smplx_model=SMPLX_MODEL_DEFAULT, device="cpu"):
    """Render SMPL-X overlay video from a .npz file produced by gvhmr_demo.py."""
    glob, incam, K_np, fps = _parse_npz(npz_path)

    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    H, W = frames[0].shape[:2]

    if K_np is None:
        K_np = np.array([[max(W, H), 0, W / 2], [0, max(W, H), H / 2], [0, 0, 1]])

    faces = smplx.create(smplx_model, model_type="smplx", num_betas=10).faces

    verts_incam = _get_smplx_vertices(
        incam["global_orient"], incam["body_pose"], incam["betas"], incam["transl"],
        smplx_model, device) if mode != "global" else None
    verts_global = _get_smplx_vertices(
        glob["global_orient"], glob["body_pose"], glob["betas"], glob["transl"],
        smplx_model, device) if mode != "incam" else None

    out_w = W * 2 if mode == "side_by_side" else W
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, H))
    rend_i = IncamRenderer(faces, H, W, K_np) if verts_incam is not None else None
    rend_g = GlobalRenderer(faces, H, W, verts_global.mean(1)) if verts_global is not None else None

    for i in tqdm(range(len(frames)), desc="Rendering"):
        if mode == "incam":
            vis = rend_i.overlay(frames[i], verts_incam[i], alpha)
        elif mode == "global":
            vis = rend_g.render(verts_global[i])[:, :, ::-1].copy()
        else:
            l = rend_i.overlay(frames[i], verts_incam[i], alpha)
            r = rend_g.render(verts_global[i])[:, :, ::-1].copy()
            cv2.putText(l, "In-Camera", (10, H - 15), 1, 1, [220] * 3, 2)
            cv2.putText(r, "Global", (10, H - 15), 1, 1, [220] * 3, 2)
            vis = np.concatenate([l, r], 1)
        writer.write(vis)
    writer.release()
    print(f"[Visualize] Saved → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--npz", required=True)
    parser.add_argument("--out", default="gvhmr_overlay.mp4")
    parser.add_argument("--mode", default="incam", choices=["incam", "global", "side_by_side"])
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--smplx_model", default=SMPLX_MODEL_DEFAULT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    visualize(args.video, args.npz, args.out, args.mode, args.alpha, args.smplx_model, args.device)


if __name__ == "__main__":
    main()
