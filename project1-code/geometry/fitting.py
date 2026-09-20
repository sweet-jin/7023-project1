"""Deterministic geometric fitting utilities (numpy).

Used by:
- the non-learning geometry baseline (eval.py --mode geom)
- inference-time measurement from predicted segmentation masks
"""
import numpy as np


def fit_plane_lsq(points):
    """Least-squares plane fit. Returns (normal, centroid)."""
    c = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - c, full_matrices=False)
    n = vh[-1]
    if n[2] < 0:
        n = -n
    return n, c


def plane_residuals(points, normal, centroid):
    return np.abs((points - centroid) @ normal)


def fit_plane_ransac(points, dist_thresh=0.01, max_iter=200, seed=0):
    """RANSAC plane fit. Returns (normal, centroid, inlier_mask)."""
    rng = np.random.default_rng(seed)
    n = len(points)
    if n < 3:
        raise ValueError("need at least 3 points to fit a plane")
    best = None
    for _ in range(max_iter):
        idx = rng.choice(n, 3, replace=False)
        nrm, c = fit_plane_lsq(points[idx])
        d = plane_residuals(points, nrm, c)
        inl = d < dist_thresh
        if best is None or inl.sum() > best[0]:
            best = (inl.sum(), inl, nrm, c)
    _, inl, nrm, c = best
    if inl.sum() >= 3:
        nrm, c = fit_plane_lsq(points[inl])
    return nrm, c, inl


def pca_bbox(points):
    """PCA-oriented bounding box. Returns (center, half_extents, axes_rows)."""
    c = points.mean(axis=0)
    p = points - c
    _, _, vh = np.linalg.svd(p, full_matrices=False)
    axes = vh  # rows are principal directions
    coords = p @ axes.T
    lo, hi = coords.min(axis=0), coords.max(axis=0)
    return c, (hi - lo) / 2.0, axes


def fit_circle_2d(points2d, max_iter=200, seed=0):
    """RANSAC circle fit in 2D. Returns (center2d, radius)."""
    rng = np.random.default_rng(seed)
    n = len(points2d)
    if n < 3:
        c = points2d.mean(axis=0)
        return c, float(np.linalg.norm(points2d - c, axis=1).mean())
    best = None
    for _ in range(max_iter):
        idx = rng.choice(n, 3, replace=False)
        p = points2d[idx]
        a = 2 * (p[1:] - p[0])
        b = (p[1:] ** 2).sum(axis=1) - (p[0] ** 2).sum()
        if abs(np.linalg.det(a)) < 1e-12:
            continue
        center = np.linalg.solve(a, b)
        r = float(np.linalg.norm(p[0] - center))
        d = np.abs(np.linalg.norm(points2d - center, axis=1) - r)
        inl = d < max(0.005, 0.05 * r)
        if best is None or inl.sum() > best[0]:
            best = (inl.sum(), center, r, inl)
    if best is None:
        c = points2d.mean(axis=0)
        return c, float(np.linalg.norm(points2d - c, axis=1).mean())
    return best[1], best[2]


def angle_deg(n1, n2):
    """Angle between two normals in degrees."""
    d = abs(np.clip(n1 @ n2, -1.0, 1.0))
    return float(np.degrees(np.arccos(d)))


def split_walls(points, half_extents):
    """Assign wall points to one of 4 vertical faces (+-x, +-y).

    half_extents: (hx, hy, hz) of the room, derived from the full point cloud.
    Returns list of 4 boolean masks in order (+x, -x, +y, -y).
    """
    specs = [
        (0, half_extents[0]),
        (0, -half_extents[0]),
        (1, half_extents[1]),
        (1, -half_extents[1]),
    ]
    d = np.stack([np.abs(points[:, ax] - target) for ax, target in specs], axis=1)
    assign = d.argmin(axis=1)
    return [assign == i for i in range(4)]


def cluster_points(points, eps=0.05):
    """Greedy proximity clustering (connected components). Returns list of index arrays."""
    if len(points) == 0:
        return []
    visited = np.zeros(len(points), dtype=bool)
    clusters = []
    for i in range(len(points)):
        if visited[i]:
            continue
        cluster = [i]
        visited[i] = True
        frontier = [i]
        while frontier:
            f = frontier.pop()
            d = np.linalg.norm(points - points[f], axis=1)
            nb = np.where((d < eps) & ~visited)[0]
            visited[nb] = True
            cluster.extend(nb.tolist())
            frontier.extend(nb.tolist())
        clusters.append(np.array(cluster))
    return clusters
