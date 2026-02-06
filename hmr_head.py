import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from einops.einops import rearrange


def rot6d_to_rotmat(x):
    """Convert 6D rotation representation to 3x3 rotation matrix.

    Based on Zhou et al., "On the Continuity of Rotation
    Representations in Neural Networks", CVPR 2019
    Input:
        (B,6) Batch of 6-D rotation representations
    Output:
        (B,3,3) Batch of corresponding rotation matrices
    """
    if x.shape[-1] == 6:
        batch_size = x.shape[0]
        if len(x.shape) == 3:
            num = x.shape[1]
            x = rearrange(x, "b n d -> (b n) d", d=6)
        else:
            num = 1
        x = rearrange(x, "b (k l) -> b k l", k=3, l=2)
        # x = x.view(-1,3,2)
        a1 = x[:, :, 0]
        a2 = x[:, :, 1]
        b1 = F.normalize(a1)
        b2 = F.normalize(a2 - torch.einsum("bi,bi->b", b1, a2).unsqueeze(-1) * b1)
        b3 = torch.cross(b1, b2, dim=-1)

        mat = torch.stack((b1, b2, b3), dim=-1)
        if num > 1:
            mat = rearrange(mat, "(b n) h w-> b n h w", b=batch_size, n=num, h=3, w=3)
    else:
        if isinstance(x, torch.Tensor):
            x = x.view(-1, 3, 2)
        elif isinstance(x, np.ndarray):
            x = x.reshape(-1, 3, 2)
        a1 = x[:, :, 0]
        a2 = x[:, :, 1]
        b1 = F.normalize(a1)
        b2 = F.normalize(a2 - torch.einsum("bi,bi->b", b1, a2).unsqueeze(-1) * b1)
        b3 = torch.cross(b1, b2)
        mat = torch.stack((b1, b2, b3), dim=-1)
    return mat


class HMRHead(nn.Module):
    """HMR head."""

    def __init__(
        self,
        feat_dim,
        smpl_mean_params=None,
        npose=144,
        nbeta=10,
        ncam=3,
        hdim=1024,
        # init_cfg=None,
    ):
        super(HMRHead, self).__init__()
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

        if smpl_mean_params is None:
            init_pose = torch.zeros([1, npose])
            init_shape = torch.zeros([1, nbeta])
            init_cam = torch.FloatTensor([[1, 0, 0]])
        else:
            mean_params = np.load(smpl_mean_params)
            init_pose = torch.from_numpy(mean_params["pose"][:]).unsqueeze(0)
            init_shape = torch.from_numpy(
                mean_params["shape"][:].astype("float32")
            ).unsqueeze(0)
            init_cam = torch.from_numpy(mean_params["cam"]).unsqueeze(0)
        self.register_buffer("init_pose", init_pose)
        self.register_buffer("init_shape", init_shape)
        self.register_buffer("init_cam", init_cam)

    def forward(self, x, init_pose=None, init_shape=None, init_cam=None, n_iter=3):
        """Forward function."""
        # hmr head only support one layer feature
        if isinstance(x, list) or isinstance(x, tuple):
            x = x[-1]

        output_seq = False
        if len(x.shape) == 4:
            # use feature from the last layer of the backbone
            # apply global average pooling on the feature map
            x = x.mean(dim=-1).mean(dim=-1)
        elif len(x.shape) == 3:
            # temporal feature
            output_seq = True
            B, T, L = x.shape
            x = x.view(-1, L)

        batch_size = x.shape[0]
        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1)

        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam
        for i in range(n_iter):
            xc = torch.cat([x, pred_pose, pred_shape, pred_cam], 1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)
            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape
            pred_cam = self.deccam(xc) + pred_cam

        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)

        if output_seq:
            pred_rotmat = pred_rotmat.view(B, T, 24, 3, 3)
            pred_shape = pred_shape.view(B, T, 10)
            pred_cam = pred_cam.view(B, T, 3)
        output = {
            "pred_pose": pred_rotmat,
            "pred_shape": pred_shape,
            "pred_cam": pred_cam,
        }
        return output


if __name__ == "__main__":
    m = HMRHead(feat_dim=2048)
    o = m(torch.randn(1, 2048))

    print(o)
