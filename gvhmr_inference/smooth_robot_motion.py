"""Smooth a GMR robot motion with a One Euro filter.

GMR retargets the GVHMR estimate frame by frame, so per-frame pose noise in the
human estimate turns into joint jitter on the robot. It is most visible in the
legs, where the IK chases a noisy foot target while the foot should be planted.

This script runs a One Euro filter (Casiez, Roussel & Vogel, CHI 2012) over the
robot motion pickle that GMR writes with ``--save_path``. The filter is causal
and keeps one frame of state, so the same code works on a live teleop stream:
low cutoff when the joint is slow (kills jitter), higher cutoff when the joint
is fast (keeps swings sharp, little lag).

Usage:
    python smooth_robot_motion.py --input tennis_g1.pkl --output tennis_g1_smoothed.pkl
    python smooth_robot_motion.py --input tennis_g1.pkl --output out.pkl --joints legs
    python smooth_robot_motion.py --input tennis_g1.pkl --output out.pkl --record_video out.mp4

``--record_video`` renders the smoothed motion with GMR's MuJoCo viewer and
needs the GMR package installed (it is, if you ran ``uv sync`` in this folder).
"""

import argparse
import pickle
import warnings

import numpy as np

_LEG_KEYWORDS = ("hip", "knee", "ankle")


# ============================================================
# One Euro filter
# ============================================================


class OneEuroFilter:
    """Vectorised One Euro filter over an array of independent signals."""

    def __init__(self, fps, min_cutoff=1.0, beta=0.05, d_cutoff=1.0):
        self.fps = float(fps)
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = None
        self.dx_prev = None

    def _alpha(self, cutoff):
        tau = 1.0 / (2.0 * np.pi * cutoff)
        te = 1.0 / self.fps
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x):
        x = np.asarray(x, dtype=np.float64)
        if self.x_prev is None:
            self.x_prev = x.copy()
            self.dx_prev = np.zeros_like(x)
            return x.copy()
        # Smoothed derivative
        a_d = self._alpha(self.d_cutoff)
        dx = (x - self.x_prev) * self.fps
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev
        # Speed-adaptive cutoff: slow → heavy smoothing, fast → light smoothing
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = self._alpha(cutoff)
        x_hat = a * x + (1.0 - a) * self.x_prev
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat


def one_euro_sequence(seq, fps, min_cutoff, beta, d_cutoff=1.0):
    """Filter a (F, D) sequence causally along the frame axis."""
    seq = np.asarray(seq, dtype=np.float64)
    f = OneEuroFilter(fps, min_cutoff, beta, d_cutoff)
    return np.stack([f(x) for x in seq], axis=0)


def smooth_quaternions(quat_xyzw, fps, min_cutoff, beta):
    """Filter a (F, 4) quaternion track.

    Quaternions q and -q are the same rotation, so the track is first made
    sign-continuous (no hemisphere flips between frames), then filtered
    component-wise and re-normalised.
    """
    q = np.asarray(quat_xyzw, dtype=np.float64).copy()
    for i in range(1, len(q)):
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    q = one_euro_sequence(q, fps, min_cutoff, beta)
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


# ============================================================
# Joint selection
# ============================================================


def robot_dof_names(robot):
    """Actuated DoF names of a GMR robot, in dof_pos order (root joint excluded)."""
    try:
        import mujoco as mj
        from general_motion_retargeting import ROBOT_XML_DICT

        model = mj.MjModel.from_xml_path(str(ROBOT_XML_DICT[robot]))
        names = [
            mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, model.dof_jntid[i])
            for i in range(model.nv)
        ]
        return names[6:]  # first 6 DoFs belong to the floating root
    except Exception as e:  # GMR/mujoco missing or unknown robot
        warnings.warn(f"Could not read DoF names for '{robot}': {e}")
        return None


def select_dofs(joints, n_dof, names):
    """Return the dof_pos column indices selected by ``--joints``."""
    if joints == "all":
        return np.arange(n_dof)
    if names is None:
        warnings.warn("No DoF names available; smoothing all joints instead.")
        return np.arange(n_dof)
    keys = _LEG_KEYWORDS if joints == "legs" else tuple(joints.split(","))
    idx = [i for i, n in enumerate(names) if any(k in n for k in keys)]
    if not idx:
        raise ValueError(f"No DoF matched {keys}. Available: {names}")
    return np.array(idx)


# ============================================================
# Main
# ============================================================


def jitter(x):
    """Mean |second difference|: a cheap frame-to-frame jitter score."""
    return float(np.abs(np.diff(x, n=2, axis=0)).mean())


def main():
    p = argparse.ArgumentParser(description="One Euro smoothing for GMR robot motion")
    p.add_argument("--input", required=True, help="GMR motion .pkl (from --save_path)")
    p.add_argument("--output", required=True, help="Where to write the smoothed .pkl")
    p.add_argument("--robot", default="unitree_g1")
    p.add_argument(
        "--joints",
        default="all",
        help="'all', 'legs' (hip/knee/ankle), or comma-separated name substrings",
    )
    p.add_argument("--min_cutoff", type=float, default=1.0, help="Hz; lower = smoother at rest")
    p.add_argument("--beta", type=float, default=0.05, help="speed coefficient; higher = less lag")
    p.add_argument("--d_cutoff", type=float, default=1.0, help="Hz; cutoff for the derivative")
    p.add_argument("--skip_root", action="store_true", help="Leave root pos/rot untouched")
    p.add_argument("--record_video", default=None, metavar="OUT_MP4", help="Render with GMR viewer")
    args = p.parse_args()

    with open(args.input, "rb") as f:
        motion = pickle.load(f)
    fps = float(motion["fps"])
    dof_pos = np.asarray(motion["dof_pos"], dtype=np.float64)
    root_pos = np.asarray(motion["root_pos"], dtype=np.float64)
    root_rot = np.asarray(motion["root_rot"], dtype=np.float64)  # xyzw
    F, D = dof_pos.shape

    names = robot_dof_names(args.robot) if args.joints != "all" else None
    cols = select_dofs(args.joints, D, names)

    before = jitter(dof_pos[:, cols])
    smoothed = dof_pos.copy()
    smoothed[:, cols] = one_euro_sequence(
        dof_pos[:, cols], fps, args.min_cutoff, args.beta, args.d_cutoff
    )
    after = jitter(smoothed[:, cols])

    out = dict(motion)
    out["dof_pos"] = smoothed.astype(np.float32)
    if not args.skip_root:
        out["root_pos"] = one_euro_sequence(
            root_pos, fps, args.min_cutoff, args.beta, args.d_cutoff
        ).astype(np.float32)
        out["root_rot"] = smooth_quaternions(
            root_rot, fps, args.min_cutoff, args.beta
        ).astype(np.float32)

    with open(args.output, "wb") as f:
        pickle.dump(out, f)

    sel = "all" if names is None else [names[i] for i in cols]
    print(f"[Smooth] {F} frames @ {fps:.0f} fps, {len(cols)}/{D} DoFs: {sel}")
    print(f"[Smooth] jitter {before:.5f} → {after:.5f} ({100 * (1 - after / before):.1f}% lower)")
    print(f"[Smooth] Saved → {args.output}")

    if args.record_video:
        from general_motion_retargeting import RobotMotionViewer

        viewer = RobotMotionViewer(
            robot_type=args.robot,
            motion_fps=fps,
            camera_follow=False,
            record_video=True,
            video_path=args.record_video,
        )
        root_rot_wxyz = out["root_rot"][:, [3, 0, 1, 2]]
        for i in range(F):
            viewer.step(out["root_pos"][i], root_rot_wxyz[i], out["dof_pos"][i], rate_limit=False)
        viewer.close()
        print(f"[Smooth] Video → {args.record_video}")


if __name__ == "__main__":
    main()
