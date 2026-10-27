"""Convert a GMR robot motion pickle into a SONIC v1 reference bundle.

GMR (and ``smooth_robot_motion.py``) write ``root_pos``, ``root_rot`` (xyzw),
``dof_pos`` and ``fps``. NVIDIA's GEAR-SONIC tracking policy consumes a
per-frame reference of joint positions, joint velocities and a body
orientation quaternion. This script resamples the motion to SONIC's rate,
finite-differences the joint velocities and writes ``sonic_stream_v1.npz``,
which ``run_sonic_from_output.py`` validates and publishes over ZMQ.

Usage:
    python gmr_to_sonic.py --input ../GMR/motions/tennis_g1_smoothed.pkl --output outputs/tennis/sonic_stream_v1.npz

Joint order is the Unitree G1 29-DoF order used by GMR's MJCF (legs, waist,
arms). It is stored in the bundle as ``joint_names`` so it can be checked
against the SONIC deployment you run.
"""

import argparse
import os
import pickle

import numpy as np

G1_JOINT_NAMES = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)


def resample(x, src_fps, dst_fps):
    """Linearly resample (F, D) along the frame axis."""
    F = x.shape[0]
    t_src = np.arange(F) / src_fps
    t_dst = np.arange(0.0, t_src[-1] + 1e-9, 1.0 / dst_fps)
    return np.stack([np.interp(t_dst, t_src, x[:, d]) for d in range(x.shape[1])], axis=1)


def resample_quat(q_xyzw, src_fps, dst_fps):
    """Resample quaternions after making the track sign-continuous; renormalise."""
    q = q_xyzw.copy()
    for i in range(1, len(q)):
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    q = resample(q, src_fps, dst_fps)
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def main():
    p = argparse.ArgumentParser(description="GMR motion .pkl → SONIC v1 bundle")
    p.add_argument("--input", required=True, help="GMR motion .pkl (raw or smoothed)")
    p.add_argument("--output", required=True, help="Path to write sonic_stream_v1.npz")
    p.add_argument("--fps", type=float, default=50.0, help="SONIC reference rate (Hz)")
    args = p.parse_args()

    with open(args.input, "rb") as f:
        motion = pickle.load(f)
    src_fps = float(motion["fps"])
    dof_pos = np.asarray(motion["dof_pos"], dtype=np.float64)
    root_rot = np.asarray(motion["root_rot"], dtype=np.float64)  # xyzw
    if dof_pos.shape[1] != len(G1_JOINT_NAMES):
        raise ValueError(
            f"Expected {len(G1_JOINT_NAMES)} DoFs (Unitree G1 29-DoF), got {dof_pos.shape[1]}"
        )

    joint_pos = resample(dof_pos, src_fps, args.fps)
    joint_vel = np.gradient(joint_pos, 1.0 / args.fps, axis=0)
    body_quat = resample_quat(root_rot, src_fps, args.fps)[:, [3, 0, 1, 2]]  # → wxyz
    N = joint_pos.shape[0]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez(
        args.output,
        joint_pos=joint_pos.astype(np.float32),
        joint_vel=joint_vel.astype(np.float32),
        body_quat=body_quat.astype(np.float32),
        frame_index=np.arange(N, dtype=np.int64),
        fps=np.array([args.fps], dtype=np.float32),
        joint_names=np.array(G1_JOINT_NAMES),
        source=np.array(os.path.abspath(args.input)),
    )
    print(f"[SONIC] {dof_pos.shape[0]} frames @ {src_fps:.0f} fps → {N} frames @ {args.fps:.0f} fps")
    print(f"[SONIC] Saved → {args.output}")


if __name__ == "__main__":
    main()
