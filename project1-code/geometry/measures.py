"""Measurement slot definitions and deterministic measurement routines.

Slot layout is shared by:
- the synthetic dataset generator (ground-truth builder, datasets/synthetic.py)
- the learned measure head (models/measure_head.py)
- the deterministic geometry baseline (this module)

All values are in METERS; evaluation reports millimetres (x 1000).
"""
import numpy as np

from .fitting import (
    angle_deg,
    cluster_points,
    fit_circle_2d,
    fit_plane_lsq,
    fit_plane_ransac,
    pca_bbox,
    plane_residuals,
    split_walls,
)

# Semantic labels (index must match datasets.synthetic.LABEL_NAMES)
LABEL_BACKGROUND = 0
LABEL_WALL = 1
LABEL_FLOOR = 2
LABEL_CEILING = 3
LABEL_DOOR = 4
LABEL_WINDOW = 5
LABEL_HOLE = 6
LABEL_EMBEDDED = 7

# Measurement slot names per group
MODULE_SLOTS = ["length", "width", "height", "flatness_max", "perpendicularity_deg"]
DOOR_SLOTS = ["width", "height"]
WINDOW_SLOTS = ["width", "height"]
HOLE_SLOTS = ["diameter_max", "diameter_mean"]
EMBEDDED_SLOTS = ["width", "height", "protrusion"]

SLOT_NAMES = {
    "module": MODULE_SLOTS,
    "door": DOOR_SLOTS,
    "window": WINDOW_SLOTS,
    "hole": HOLE_SLOTS,
    "embedded": EMBEDDED_SLOTS,
}

SLOT_DIMS = {k: len(v) for k, v in SLOT_NAMES.items()}


def _module_from_points(points):
    """Module-level slots from the full (non-background) point cloud."""
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    extents = hi - lo  # (x, y, z) == (length, width, height)
    half = extents / 2.0

    wall_pts = None  # computed by caller if needed
    # flatness / perpendicularity need wall vs floor separation -> handled in caller
    return extents, half


def measure_from_labels(points, labels):
    """Deterministic measurement from a labeled point cloud (geometry baseline).

    Returns dict: {group: np.ndarray of slot values} — same layout as model output.
    Groups absent in the label set are returned as None.
    """
    valid = labels > LABEL_BACKGROUND
    pts = points[valid]
    if len(pts) < 3:
        return {"module": None, "door": None, "window": None, "hole": None, "embedded": None}

    # module-level extents (axis-aligned bounding box; rooms are near-axis-aligned)
    extents, half = _module_from_points(pts)

    # wall flatness: per-face plane fit on wall points
    wall_mask = labels == LABEL_WALL
    flatness_max = 0.0
    perp_deg = 0.0
    if wall_mask.sum() >= 3:
        walls = points[wall_mask]
        face_masks = split_walls(walls, half)
        wall_normals = []
        for fm in face_masks:
            if fm.sum() < 3:
                continue
            nrm, c, _ = fit_plane_ransac(walls[fm], dist_thresh=0.01, seed=0)
            res = plane_residuals(walls[fm], nrm, c)
            flatness_max = max(flatness_max, float(res.max()))
            wall_normals.append(nrm)
        # perpendicularity = deviation between floor normal and wall normals
        floor_mask = labels == LABEL_FLOOR
        if floor_mask.sum() >= 3 and len(wall_normals) >= 2:
            floor_n, _ = fit_plane_lsq(points[floor_mask])
            devs = []
            for n in wall_normals:
                # wall normal should be horizontal -> angle with vertical floor normal ~90
                devs.append(abs(90.0 - angle_deg(n, floor_n)))
            perp_deg = float(np.max(devs))

    module = np.array(
        [extents[0], extents[1], extents[2], flatness_max, perp_deg], dtype=np.float64
    )

    def measure_rect_group(lbl):
        """Axis-aligned dims: width = max(x,y extent), height = z extent.

        Rooms are near-axis-aligned (tilt <= ~1 deg), so the AABB is far more
        robust than PCA here (PCA axes are unstable when extents are similar).
        """
        m = labels == lbl
        if m.sum() < 3:
            return None
        ext = points[m].max(axis=0) - points[m].min(axis=0)
        width = max(ext[0], ext[1])
        height = ext[2]
        return np.array([width, height], dtype=np.float64)

    door = measure_rect_group(LABEL_DOOR)
    window = measure_rect_group(LABEL_WINDOW)

    # embedded parts: measure each cluster separately, report the largest box
    embedded = None
    emb_mask = labels == LABEL_EMBEDDED
    if emb_mask.sum() >= 3:
        clusters = cluster_points(points[emb_mask], eps=0.12)
        best = None
        for cl in clusters:
            if len(cl) < 5:
                continue
            p = points[emb_mask][cl]
            ext = p.max(axis=0) - p.min(axis=0)
            w = max(ext[0], ext[1])
            h = ext[2]
            prot = _embedded_protrusion(p, points, labels)
            if best is None or w > best[0]:
                best = (w, h, prot)
        if best is not None:
            embedded = np.array(best, dtype=np.float64)

    hole = None
    hole_mask = labels == LABEL_HOLE
    if hole_mask.sum() >= 3:
        hole = _hole_diameters(points[hole_mask])

    return {"module": module, "door": door, "window": window, "hole": hole, "embedded": embedded}


def _embedded_protrusion(emb_pts, all_points, labels):
    """Protrusion = max distance of embedded-part points from the wall plane behind it.

    Robust to corner-adjacent boxes: nearby wall points are projected onto the
    box's thinnest PCA axis (its protrusion direction) and the wall band is
    selected by density peak, so a perpendicular wall cannot hijack the fit.
    """
    wall_pts = all_points[labels == LABEL_WALL]
    if len(wall_pts) < 3:
        return 0.0
    _, _, axes = pca_bbox(emb_pts)
    n = axes[-1]  # thinnest direction == wall normal
    c = emb_pts.mean(axis=0)
    near = wall_pts[np.linalg.norm(wall_pts - c, axis=1) < 0.5]
    if len(near) < 3:
        near = wall_pts
    proj = (near - c) @ n
    # density peak of projections == the wall plane behind the box
    hist, edges = np.histogram(proj, bins=max(5, len(proj) // 10),
                               range=(proj.min(), proj.max()))
    center = 0.5 * (edges[hist.argmax()] + edges[hist.argmax() + 1])
    band = near[np.abs(proj - center) < 0.05]
    if len(band) < 3:
        band = near
    nrm, ctr, _ = fit_plane_ransac(band, dist_thresh=0.01)
    if nrm @ n < 0:
        nrm = -nrm
    return float(np.max(np.abs((emb_pts - ctr) @ nrm)))


def _hole_diameters(hole_pts):
    """Circle-fit diameters for hole clusters. Returns [diameter_max, diameter_mean].

    Clusters in 3D (eps=0.03), then for each cluster tries projecting out each
    of the three global axes (walls are near-axis-aligned; hole cylinders pierce
    along one of them) and keeps the fit with the best inlier ratio. Only
    plausible diameters (0.02-0.30 m) are accepted, making the routine robust
    to stray/merged clusters and to radius-dominant vs depth-dominant cylinders.
    """
    clusters = cluster_points(hole_pts, eps=0.03)
    diams = []
    for cl in clusters:
        if len(cl) < 5:
            continue
        p = hole_pts[cl]
        best = None  # (inlier_ratio, diameter)
        for ax in range(3):
            p2 = p[:, [a for a in range(3) if a != ax]]
            center, radius = fit_circle_2d(p2)
            d = 2.0 * radius
            if not (0.02 <= d <= 0.30):
                continue
            resid = np.abs(np.linalg.norm(p2 - center, axis=1) - radius)
            inl = float((resid < max(0.005, 0.05 * radius)).mean())
            if best is None or inl > best[0]:
                best = (inl, d)
        if best is not None:
            diams.append(best[1])
    if not diams:
        return None
    return np.array([float(np.max(diams)), float(np.mean(diams))], dtype=np.float64)


def gt_from_config(cfg):
    """Analytic ground-truth slots (meters) from a generator config.

    Mirrors the measure-head layout so losses can be computed directly.
    Returns dict group -> np.ndarray or None.
    """
    L, W, H = cfg["L"], cfg["W"], cfg["H"]
    flatness = cfg.get("waviness_amp", 0.0)
    perp = cfg.get("tilt_deg", 0.0)
    module = np.array([L, W, H, flatness, perp], dtype=np.float64)

    door = None
    if cfg.get("door") is not None:
        d = cfg["door"]
        door = np.array([d["w"], d["h"]], dtype=np.float64)
    window = None
    if cfg.get("window") is not None:
        w = cfg["window"]
        window = np.array([w["w"], w["h"]], dtype=np.float64)
    hole = None
    if cfg.get("holes"):
        rs = [h["r"] for h in cfg["holes"]]
        hole = np.array([max(rs) * 2.0, np.mean(rs) * 2.0], dtype=np.float64)
    embedded = None
    if cfg.get("embedded"):
        e = max(cfg["embedded"], key=lambda x: x["w"])  # report the largest box
        embedded = np.array([e["w"], e["h"], e["prot"]], dtype=np.float64)
    return {"module": module, "door": door, "window": window, "hole": hole, "embedded": embedded}
