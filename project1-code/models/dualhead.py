"""TADH-MiC: Task-Aware Dual-Head network for MiC geometric measurement."""
import torch
import torch.nn as nn

from .backbone import Backbone
from .heads import MeasureHead, SegHead, TaskRouter
from .vision_prior import build_prior


class TADHMiC(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        m = cfg["model"]
        in_ch = 3 + (1 if cfg["model"].get("use_vision_prior", False) else 0)
        self.backbone = Backbone(feat_dim=m["backbone"]["feat_dim"], in_ch=in_ch)
        self.router = TaskRouter(
            feat_dim=m["backbone"]["feat_dim"],
            n_tasks=m["router"]["n_tasks"],
            emb_dim=m["router"]["emb_dim"],
            hidden=m["router"]["hidden"],
        )
        self.seg_head = SegHead(
            feat_dim=m["backbone"]["feat_dim"],
            emb_dim=m["router"]["emb_dim"],
            hidden=m["seg_head"]["hidden"],
        )
        self.measure_head = MeasureHead(
            feat_dim=m["backbone"]["feat_dim"],
            emb_dim=m["router"]["emb_dim"],
            hidden=m["measure_head"]["hidden"],
        )
        self.vision_prior = build_prior(cfg["model"])

    def forward(self, points, feats=None, images=None, masks2d=None):
        prior = self.vision_prior(points, images, masks2d)
        if prior is not None:
            feats = torch.cat([feats, prior], dim=-1) if feats is not None else prior
        f = self.backbone(points, feats)  # (B, N, D)
        task_logits, task_emb = self.router(f)
        seg = self.seg_head(f, task_emb)
        meas = self.measure_head(f, seg, task_emb)
        return {"task_logits": task_logits, "seg_logits": seg, "meas": meas}
