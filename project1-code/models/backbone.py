"""PointNet++-style hierarchical encoder (self-contained, no sparse-conv deps).

Optionally upgradeable to pointcept PTv3 — the config exposes the same
interface (points -> per-point features); see docs/framework_design.md.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def farthest_point_sample(xyz, npoint):
    """FPS on (B, N, 3) -> (B, npoint) indices."""
    B, N, _ = xyz.shape
    idx = torch.zeros(B, npoint, dtype=torch.long, device=xyz.device)
    dist = torch.full((B, N), 1e10, device=xyz.device)
    farthest = torch.randint(0, N, (B,), device=xyz.device)
    for i in range(npoint):
        idx[:, i] = farthest
        centroid = xyz[torch.arange(B), farthest][:, None, :]  # (B,1,3)
        d = ((xyz - centroid) ** 2).sum(-1)
        dist = torch.minimum(dist, d)
        farthest = dist.argmax(-1)
    return idx


def ball_query(center_xyz, xyz, radius, nsample):
    """Ball query with chunking: (B,M,3),(B,N,3) -> (B,M,nsample) indices."""
    B, M, _ = center_xyz.shape
    N = xyz.shape[1]
    idx = torch.zeros(B, M, nsample, dtype=torch.long, device=xyz.device)
    chunk = 256
    for c0 in range(0, M, chunk):
        c1 = min(c0 + chunk, M)
        d = torch.cdist(center_xyz[:, c0:c1], xyz)  # (B, mc, N)
        d[d > radius] = float("inf")
        topk = d.topk(nsample, dim=-1, largest=False).indices
        idx[:, c0:c1] = topk
    return idx


def index_points(points, idx):
    """Gather points/features by index tensor (B, M, nsample) -> (B, M, nsample, C)."""
    B = points.shape[0]
    view = points.shape[1:]
    batch_idx = torch.arange(B, device=points.device).view(B, 1, 1)
    return points[batch_idx, idx]


class SetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, mlp_dims):
        super().__init__()
        self.npoint, self.radius, self.nsample = npoint, radius, nsample
        layers = []
        for i in range(len(mlp_dims) - 1):
            layers += [nn.Conv2d(mlp_dims[i], mlp_dims[i + 1], 1), nn.BatchNorm2d(mlp_dims[i + 1]), nn.ReLU()]
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz, feats):
        B, N, _ = xyz.shape
        idx = farthest_point_sample(xyz, self.npoint)
        new_xyz = xyz[torch.arange(B, device=xyz.device).view(-1, 1), idx]  # (B, npoint, 3)
        gidx = ball_query(new_xyz, xyz, self.radius, self.nsample)  # (B, npoint, nsample)
        if feats is not None:
            # subsequent layers operate on features only (standard PointNet++ convention)
            grouped = index_points(feats, gidx)
        else:
            grouped_xyz = index_points(xyz, gidx) - new_xyz[:, :, None, :]  # (B, npoint, nsample, 3)
            grouped = grouped_xyz
        grouped = grouped.permute(0, 3, 1, 2)  # (B, C, npoint, nsample)
        out = self.mlp(grouped)  # (B, C', npoint, nsample)
        out = out.max(dim=-1).values.permute(0, 2, 1)  # (B, npoint, C')
        return new_xyz, out


class FeaturePropagation(nn.Module):
    """Interpolate features from M points back to N points (3-NN inverse distance)."""

    def __init__(self, in_ch, mlp_dims):
        super().__init__()
        layers = []
        for i in range(len(mlp_dims) - 1):
            layers += [nn.Conv1d(mlp_dims[i], mlp_dims[i + 1], 1), nn.BatchNorm1d(mlp_dims[i + 1]), nn.ReLU()]
        self.mlp = nn.Sequential(*layers)
        self.in_ch = in_ch

    def forward(self, xyz_target, xyz_src, feat_src, feat_target=None):
        B, N = xyz_target.shape[0], xyz_target.shape[1]
        d = torch.cdist(xyz_target, xyz_src)  # (B, N, M)
        dist, idx = d.topk(3, dim=-1, largest=False)
        w = 1.0 / (dist + 1e-8)
        w = w / w.sum(-1, keepdim=True)
        src = feat_src[torch.arange(B, device=xyz_target.device).view(-1, 1, 1), idx]  # (B,N,3,C)
        interp = (src * w.unsqueeze(-1)).sum(2)  # (B, N, C)
        if feat_target is not None:
            interp = torch.cat([interp, feat_target], dim=-1)
        out = self.mlp(interp.permute(0, 2, 1)).permute(0, 2, 1)
        return out


class Backbone(nn.Module):
    """Three set-abstraction layers + feature propagation to full resolution.

    Output: per-point features of dim `feat_dim`.
    """

    def __init__(self, feat_dim=128, in_ch=3):
        super().__init__()
        self.sa1 = SetAbstraction(1024, 0.3, 32, [in_ch, 64, 64, 128])
        self.sa2 = SetAbstraction(256, 0.6, 32, [128, 128, 128, 256])
        self.sa3 = SetAbstraction(64, 1.2, 32, [256, 256, 512])
        self.fp3 = FeaturePropagation(512 + 256, [768, 256, 256])
        self.fp2 = FeaturePropagation(256 + 128, [384, 128, 128])
        self.fp1 = FeaturePropagation(128 + in_ch, [128 + in_ch, 128, feat_dim])
        self.feat_dim = feat_dim

    def forward(self, points, feats=None):
        # center for numerical stability (measurements are translation-invariant)
        centroid = points.mean(dim=1, keepdim=True)
        xyz = points - centroid

        l1_xyz, l1_feat = self.sa1(xyz, feats)
        l2_xyz, l2_feat = self.sa2(l1_xyz, l1_feat)
        l3_xyz, l3_feat = self.sa3(l2_xyz, l2_feat)

        f3 = self.fp3(l2_xyz, l3_xyz, l3_feat, l2_feat)
        f2 = self.fp2(l1_xyz, l2_xyz, f3, l1_feat)
        f1 = self.fp1(xyz, l1_xyz, f2, xyz)
        return f1
