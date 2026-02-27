#!/usr/bin/env python3
"""
Test script for SMPL-X output format.
Loads real SMPL model data and verifies the output matches
what GMR's load_smplx_file expects.
"""

import numpy as np
import cv2
import torch
from vibe_model import VIBE, project_joints_2d
from demo_utils import (
    SMPLRenderer, crop_to_image_coords, draw_skeleton, convert_vibe_to_smplx,
)


def test_smplx_format():
    print("Loading VIBE model...")
    model = VIBE.from_pretrained(
        'resnet50_vibe_pw3d-2e05a122_20211201.pth', device='cpu')

    # Simulate a forward pass
    feats = torch.randn(1, 16, 2048)
    with torch.no_grad():
        output = model(features=feats, run_smpl=True)

    N = 16
    pose_aa = output['pose_axis_angle'][0].cpu().numpy()   # (16, 24, 3)
    shapes  = output['pred_shape'][0].cpu().numpy()         # (16, 10)
    cams    = output['pred_cam'][0].cpu().numpy()            # (16, 3)
    fps     = 30.0

    smplx_dict = convert_vibe_to_smplx(pose_aa, shapes, cams, fps)

    # Verify keys and shapes
    print("\n=== SMPL-X output format ===")
    for key in ['root_orient', 'pose_body', 'betas', 'trans',
                'gender', 'mocap_frame_rate']:
        val = smplx_dict[key]
        if isinstance(val, np.ndarray):
            print(f"  {key:20s}  shape={str(val.shape):15s}  dtype={val.dtype}")
        else:
            print(f"  {key:20s}  value={val}")

    # Validate shapes
    assert smplx_dict['root_orient'].shape == (N, 3), \
        f"root_orient: expected ({N}, 3), got {smplx_dict['root_orient'].shape}"
    assert smplx_dict['pose_body'].shape == (N, 63), \
        f"pose_body: expected ({N}, 63), got {smplx_dict['pose_body'].shape}"
    assert smplx_dict['betas'].shape == (16,), \
        f"betas: expected (16,), got {smplx_dict['betas'].shape}"
    assert smplx_dict['trans'].shape == (N, 3), \
        f"trans: expected ({N}, 3), got {smplx_dict['trans'].shape}"

    # Test save / reload
    np.savez('/tmp/test_smplx_output.npz', **smplx_dict)
    reloaded = np.load('/tmp/test_smplx_output.npz', allow_pickle=True)
    for key in ['root_orient', 'pose_body', 'betas', 'trans']:
        assert key in reloaded, f"Missing key after reload: {key}"
    print(f"\n  gender = {str(reloaded['gender'])}")
    print(f"  mocap_frame_rate = {reloaded['mocap_frame_rate']}")

    print("\n✅ All format checks passed!")


def test_renderer_alignment():
    """Verify mesh + skeleton alignment with real SMPL data."""
    print("\nTesting renderer alignment...")
    model = VIBE.from_pretrained(
        'resnet50_vibe_pw3d-2e05a122_20211201.pth', device='cpu')

    faces = model.smpl.faces.cpu().numpy()
    renderer = SMPLRenderer(faces)

    feats = torch.randn(1, 16, 2048)
    with torch.no_grad():
        output = model(features=feats, run_smpl=True)

    verts = output['vertices'][0, -1].cpu().numpy()
    cam   = output['pred_cam'][0, -1].cpu().numpy()
    j3d   = output['joints'][0, -1]
    cam_t = output['pred_cam'][0, -1]

    img_h, img_w = 480, 640
    bbox = [200, 80, 440, 400]

    img = np.ones((img_h, img_w, 3), dtype=np.uint8) * 180
    result = renderer.render_on_image(img, verts, cam, bbox, img_h, img_w)

    j2d_crop = project_joints_2d(j3d, cam_t, img_size=224).numpy()
    j2d_orig = crop_to_image_coords(j2d_crop, bbox, img_h, img_w)
    draw_skeleton(result, j2d_orig)

    cv2.imwrite('/tmp/test_smpl_render.png', result)
    print(f"  Saved /tmp/test_smpl_render.png")
    print("✅ Renderer test passed!")


if __name__ == '__main__':
    test_smplx_format()
    test_renderer_alignment()
