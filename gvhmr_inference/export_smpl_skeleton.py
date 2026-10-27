"""Export smpl_neutral_skeleton.npz from the official SMPL neutral model.

``gvhmr_model.py`` runs its forward kinematics on a lightweight skeleton
instead of the full SMPL mesh. That skeleton is three arrays derived from
SMPL_NEUTRAL.pkl (download it from https://smpl.is.tue.mpg.de/, sign-up
required; the SMPL license does not allow redistributing derived files):

    J_template   (24, 3)      joint positions of the mean body
    J_shapedirs  (24, 3, 10)  how each joint moves with the 10 shape betas
    parents      (24,)        kinematic tree, parents[0] = -1

Usage:
    python export_smpl_skeleton.py --smpl_pkl /path/to/SMPL_NEUTRAL.pkl

The .npz is written next to this script, where GVHMRModel looks for it.
If loading the .pkl fails with a chumpy import error, strip chumpy from it
first with smplx's ``tools/clean_ch.py``.
"""

import argparse
import os

import numpy as np
import smplx
import torch


def main():
    p = argparse.ArgumentParser(description="Export SMPL neutral skeleton to .npz")
    p.add_argument("--smpl_pkl", required=True, help="Path to SMPL_NEUTRAL.pkl")
    p.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "smpl_neutral_skeleton.npz"),
    )
    args = p.parse_args()

    model = smplx.create(args.smpl_pkl, model_type="smpl", gender="neutral", num_betas=10)
    J_regressor = model.J_regressor.float()  # (24, 6890)
    v_template = model.v_template.float()  # (6890, 3)
    shapedirs = model.shapedirs.float()[..., :10]  # (6890, 3, 10)
    parents = model.parents.clone().long()
    parents[0] = -1

    J_template = J_regressor @ v_template  # (24, 3)
    J_shapedirs = torch.einsum("jv,vcd->jcd", J_regressor, shapedirs)  # (24, 3, 10)

    np.savez(
        args.out,
        J_template=J_template.numpy(),
        J_shapedirs=J_shapedirs.numpy(),
        parents=parents.numpy(),
    )
    print(f"[Skeleton] J_template {tuple(J_template.shape)}, J_shapedirs {tuple(J_shapedirs.shape)}")
    print(f"[Skeleton] Saved → {args.out}")


if __name__ == "__main__":
    main()
