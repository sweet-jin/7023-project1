"""Optional 2D foundation-model prior branch (SAM / Grounding DINO).

The synthetic pipeline does not produce RGB images, so the default prior is
NullPrior (no-op). SAMPrior documents the integration point for the CI3LAB
dataset, which may ship imagery alongside scans: run SAM automatic masks on
module photos, project them onto the point cloud via camera extrinsics, and
feed the resulting per-point prior as additional backbone input channels.

Dependency: `segment-anything` + weights (not installed by default).
"""
import torch
import torch.nn as nn


class VisionPrior(nn.Module):
    """Interface for 2D foundation-model priors."""

    prior_dim = 0

    def forward(self, points, images=None, masks2d=None):
        raise NotImplementedError


class NullPrior(VisionPrior):
    """Default: no 2D prior (framework runs without heavy dependencies)."""

    prior_dim = 0

    def forward(self, points, images=None, masks2d=None):
        return None


class SAMPrior(VisionPrior):
    """SAM automatic-mask prior projected onto the point cloud.

    Args:
        projector: callable(masks2d, uv) -> per-point binary/soft prior.
        The camera model (intrinsics/extrinsics) is dataset-specific; this
        class only wires the feature channel. Implement `projector` for the
        CI3LAB sensor setup when images become available.
    """

    prior_dim = 1

    def __init__(self, projector):
        super().__init__()
        self.projector = projector

    def forward(self, points, images=None, masks2d=None):
        if masks2d is None:
            return None
        prior = self.projector(masks2d, points)  # (B, N, 1)
        return prior


def build_prior(cfg):
    if cfg.get("use_vision_prior", False):
        raise NotImplementedError(
            "SAMPrior requires per-sample RGB images and a camera model; "
            "the synthetic generator has neither. Keep use_vision_prior=false."
        )
    return NullPrior()
