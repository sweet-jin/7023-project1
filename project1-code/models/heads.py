"""Task-aware router and the two heads of the TADH-MiC framework."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from geometry.measures import SLOT_DIMS

N_CLASSES = 8  # background, wall, floor, ceiling, door, window, hole, embedded
CLASS_INDEX = {"door": 4, "window": 5, "hole": 6, "embedded": 7}


class TaskRouter(nn.Module):
    """Predicts the primary inspection task and produces a task embedding.

    Soft routing: task embedding = softmax(logits) @ embedding table, so the
    mixture stays differentiable and supports multi-task samples at inference.
    """

    def __init__(self, feat_dim=128, n_tasks=6, emb_dim=64, hidden=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.ReLU(), nn.Linear(hidden, n_tasks)
        )
        self.task_emb = nn.Embedding(n_tasks, emb_dim)
        nn.init.normal_(self.task_emb.weight, std=0.02)

    def forward(self, point_feats):
        g = point_feats.max(dim=1).values  # (B, D)
        logits = self.mlp(g)  # (B, n_tasks)
        w = torch.softmax(logits, dim=-1)
        emb = w @ self.task_emb.weight  # (B, emb_dim)
        return logits, emb


class FiLM(nn.Module):
    """Feature-wise affine modulation conditioned on the task embedding."""

    def __init__(self, emb_dim, feat_dim):
        super().__init__()
        self.gamma = nn.Linear(emb_dim, feat_dim)
        self.beta = nn.Linear(emb_dim, feat_dim)

    def forward(self, x, emb):
        g = 1.0 + self.gamma(emb).unsqueeze(1)
        b = self.beta(emb).unsqueeze(1)
        return g * x + b


class SegHead(nn.Module):
    """Head 1: per-point semantic segmentation, conditioned on the task."""

    def __init__(self, feat_dim=128, emb_dim=64, n_classes=N_CLASSES, hidden=128):
        super().__init__()
        self.mlp1 = nn.Sequential(nn.Linear(feat_dim, hidden), nn.ReLU())
        self.film = FiLM(emb_dim, hidden)
        self.mlp2 = nn.Linear(hidden, n_classes)

    def forward(self, feats, task_emb):
        x = self.mlp1(feats)
        x = self.film(x, task_emb)
        return self.mlp2(x)  # (B, N, C) logits


class MeasureHead(nn.Module):
    """Head 2: geometry regression from soft-mask pooled region features.

    Outputs per-group slot vectors (meters), same layout as geometry.measures:
      module:  [length, width, height, flatness_max, perpendicularity_deg]
      door:    [width, height]
      window:  [width, height]
      hole:    [diameter_max, diameter_mean]
      embedded:[width, height, protrusion]
    """

    def __init__(self, feat_dim=128, emb_dim=64, hidden=128):
        super().__init__()
        self.feat_dim = feat_dim
        self.hidden = hidden
        self.module_net = nn.Sequential(
            nn.Linear(feat_dim + emb_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, SLOT_DIMS["module"]),
        )
        self.group_nets = nn.ModuleDict()
        for name in ("door", "window", "hole", "embedded"):
            self.group_nets[name] = nn.Sequential(
                nn.Linear(feat_dim + emb_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, SLOT_DIMS[name]),
            )

    def forward(self, point_feats, seg_logits, task_emb):
        probs = torch.softmax(seg_logits, dim=-1)  # (B, N, C)
        g = point_feats.max(dim=1).values
        module = self.module_net(torch.cat([g, task_emb], dim=-1))
        groups = {}
        for name, net in self.group_nets.items():
            ci = CLASS_INDEX[name]
            w = probs[:, :, ci]  # (B, N)
            denom = w.sum(dim=1, keepdim=True).clamp(min=1e-6)
            pooled = (w.unsqueeze(-1) * point_feats).sum(dim=1) / denom  # (B, D)
            groups[name] = net(torch.cat([pooled, task_emb], dim=-1))
        return {"module": module, **groups}


def dice_loss(logits, labels, n_classes, eps=1e-6):
    """Soft Dice loss for the segmentation head."""
    B = logits.shape[0]
    probs = torch.softmax(logits, dim=-1)
    onehot = F.one_hot(labels, num_classes=n_classes).float()  # (B, N, C)
    dice = 0.0
    for c in range(n_classes):
        p = probs[:, :, c]
        t = onehot[:, :, c]
        inter = (p * t).sum(dim=1)
        dice = dice + (2 * inter + eps) / (p.sum(dim=1) + t.sum(dim=1) + eps)
    return 1.0 - dice.mean() / n_classes


def seg_loss(logits, labels, n_classes=N_CLASSES):
    ce = F.cross_entropy(logits.permute(0, 2, 1), labels)
    dc = dice_loss(logits, labels, n_classes)
    return ce + dc


def task_loss(logits, task_ids):
    return F.cross_entropy(logits, task_ids)


def measure_loss(pred, gt, weight=1.0):
    """Smooth-L1 over present groups (absent groups have gt None)."""
    total = None
    for name in ("module", "door", "window", "hole", "embedded"):
        g = gt.get(name)
        if g is None:
            continue
        err = F.smooth_l1_loss(pred[name], g)
        total = err if total is None else total + err
    if total is None:
        return torch.zeros((), device=next(iter(pred.values())).device)
    return total * weight


def total_loss(out, labels, task_ids, gt, weights):
    l_seg = seg_loss(out["seg_logits"], labels)
    l_task = task_loss(out["task_logits"], task_ids)
    l_meas = measure_loss(out["meas"], gt)
    return (
        weights.get("seg", 1.0) * l_seg
        + weights.get("task", 0.5) * l_task
        + weights.get("meas", 1.0) * l_meas
    ), {"seg": l_seg.item(), "task": l_task.item(), "meas": l_meas.item()}
