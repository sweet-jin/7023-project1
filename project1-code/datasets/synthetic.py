"""Synthetic MiC module point-cloud generator with analytic ground truth.

Simulates an as-built concrete MiC module (box room) scanned by a single-station
sensor: walls/floor/ceiling with openings, door leaf, window pane, cylindrical
through-holes and protruding embedded parts (switch panels / pipe boxes).
Sensor effects: gaussian noise, wall waviness, wall tilt, viewpoint occlusion,
random dropout.

The generator guarantees analytic ground truth for every measurement slot
(see geometry.measures), enabling fully supervised training and mm-level
evaluation without waiting for the CI3LAB dataset release.
"""
import numpy as np

LABEL_NAMES = [
    "background", "wall", "floor", "ceiling", "door", "window", "hole", "embedded",
]

# primary-task names (index == router task id)
TASK_NAMES = ["dimension", "flatness", "perpendicularity", "opening", "hole", "embedded"]

MODULE_TYPES = [
    "living", "kitchen", "bathroom",
    "living_bathroom", "living_kitchen", "bathroom_kitchen",
]

WALL_NAMES = ["x-", "x+", "y-", "y+"]
WALL_THICKNESS = 0.15  # m


def _u(a, b, rng):
    return float(rng.uniform(a, b))


def sample_config(rng, R, module_type, task):
    """Sample one module configuration dict (analytic geometry, meters)."""
    cfg = {
        "type": module_type,
        "task": task,
        "L": _u(*R["room"]["L"], rng),
        "W": _u(*R["room"]["W"], rng),
        "H": _u(*R["room"]["H"], rng),
        "door": None,
        "window": None,
        "holes": [],
        "embedded": [],
        "waviness_amp": _u(*R["waviness_amp"], rng),
        "waviness_freq": _u(*R["waviness_freq"], rng),
        "tilt_deg": _u(*R["tilt_deg"], rng),
        "noise_sigma": R["noise_sigma"],
        "dropout": R["dropout"],
        "max_range": R.get("max_range", None),
    }

    has = {
        "living": dict(door=True, window=True, holes=0, embedded=1),
        "kitchen": dict(door=False, window=True, holes=(1, 3), embedded=(1, 2)),
        "bathroom": dict(door=True, window=False, holes=(1, 2), embedded=(0, 1)),
        "living_bathroom": dict(door=True, window=True, holes=(1, 2), embedded=1),
        "living_kitchen": dict(door=True, window=True, holes=(1, 3), embedded=(1, 2)),
        "bathroom_kitchen": dict(door=True, window=False, holes=(1, 3), embedded=(1, 2)),
    }[module_type]

    def pick(rng, spec):
        if isinstance(spec, tuple):
            return int(rng.integers(spec[0], spec[1] + 1))
        return int(spec)

    door, window = has["door"], has["window"]
    n_holes = pick(rng, has["holes"])
    n_emb = pick(rng, has["embedded"])

    # task constraints: every task must have its measurement object present
    if task == "opening" and not door and not window:
        door = True
    if task == "hole" and n_holes == 0:
        n_holes = 1
    if task == "embedded" and n_emb == 0:
        n_emb = 1
    if task == "flatness":
        cfg["waviness_amp"] = _u(max(R["waviness_amp"][0], 0.001), R["waviness_amp"][1], rng)
    if task == "perpendicularity":
        cfg["tilt_deg"] = _u(max(R["tilt_deg"][0], 0.2), R["tilt_deg"][1], rng)

    if door:
        dw = _u(*R["door"]["w"], rng)
        dh = _u(*R["door"]["h"], rng)
        # corner-based in-plane coords: t along y in [0, W]; keep leaf inside the wall
        ct = _u(dw / 2 + 0.05, cfg["W"] - dw / 2 - 0.05, rng)
        cfg["door"] = {"wall": "x-", "w": dw, "h": dh, "ct": ct}
    if window:
        ww = _u(*R["window"]["w"], rng)
        wh = min(_u(*R["window"]["h"], rng), cfg["H"] - 1.6)  # keep zc range valid
        z_lo = 0.8 + wh / 2
        z_hi = cfg["H"] - 0.8 - wh / 2
        zc = _u(z_lo, max(z_lo + 1e-9, z_hi), rng)
        ct = _u(ww / 2 + 0.05, cfg["L"] - ww / 2 - 0.05, rng)
        cfg["window"] = {"wall": "y+", "w": ww, "h": wh, "ct": ct, "zc": zc}

    def tangent_len(wall):
        return cfg["W"] if wall in ("x-", "x+") else cfg["L"]

    def overlaps_openings(wall, lo_t, hi_t, lo_v, hi_v, margin):
        """Reject placements overlapping door/window openings (in-plane coords)."""
        for name, rect in (("door", cfg["door"]), ("window", cfg["window"])):
            if rect is None or rect["wall"] != wall:
                continue
            r0, r1 = rect["ct"] - rect["w"] / 2 - margin, rect["ct"] + rect["w"] / 2 + margin
            if name == "door":
                v0, v1 = 0.0 - margin, rect["h"] + margin
            else:
                v0, v1 = rect["zc"] - rect["h"] / 2 - margin, rect["zc"] + rect["h"] / 2 + margin
            if hi_t < r0 or lo_t > r1 or hi_v < v0 or lo_v > v1:
                continue
            return True
        return False

    for _ in range(n_holes):
        wall = rng.choice(WALL_NAMES)
        r = _u(*R["hole"]["r"], rng)
        T = tangent_len(wall)
        H = cfg["H"]
        # keep holes well separated (no circle overlap) and clear of openings
        for _try in range(20):
            ct = _u(r + 0.05, T - r - 0.05, rng)
            cv = _u(0.4, H - 0.4, rng)
            too_close = any(
                h["wall"] == wall
                and (h["ct"] - ct) ** 2 + (h["cv"] - cv) ** 2 < (h["r"] + r + 0.05) ** 2
                for h in cfg["holes"]
            )
            if too_close:
                continue
            if overlaps_openings(wall, ct - r, ct + r, cv - r, cv + r, 0.0):
                continue
            break
        cfg["holes"].append({"wall": wall, "ct": ct, "cv": cv, "r": r})
    for _ in range(n_emb):
        wall = rng.choice(WALL_NAMES)
        w = _u(*R["embedded"]["w"], rng)
        h = _u(*R["embedded"]["h"], rng)
        prot = _u(*R["embedded"]["prot"], rng)
        T = tangent_len(wall)
        H = cfg["H"]
        # keep boxes well separated, clear of openings and holes
        for _try in range(20):
            ct = _u(w / 2 + 0.05, T - w / 2 - 0.05, rng)
            cv = _u(0.5, H - 0.5, rng)
            too_close = any(
                e["wall"] == wall and (e["ct"] - ct) ** 2 + (e["cv"] - cv) ** 2 < 0.04
                for e in cfg["embedded"]
            )
            if too_close:
                continue
            if overlaps_openings(wall, ct - w / 2, ct + w / 2, cv - h / 2, cv + h / 2, 0.05):
                continue
            half_diag = 0.5 * (w ** 2 + h ** 2) ** 0.5
            if any(
                ho["wall"] == wall
                and (ho["ct"] - ct) ** 2 + (ho["cv"] - cv) ** 2 < (ho["r"] + half_diag + 0.05) ** 2
                for ho in cfg["holes"]
            ):
                continue
            break
        cfg["embedded"].append({"wall": wall, "ct": ct, "cv": cv, "w": w, "h": h, "prot": prot})
    return cfg


class MiCModuleGenerator:
    """Generates one labeled point cloud per config."""

    def __init__(self, cfg, seed):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

    # ---- geometry helpers -------------------------------------------------
    def _wall_pose(self, name):
        """(origin, e_tangent, e_vertical, e_normal) of a wall inner face.

        origin is at the face CORNER so in-plane coords span t in [0, T],
        v in [0, H]. e_normal points INTO the room (inward face normal).
        """
        L, W, H = self.cfg["L"], self.cfg["W"], self.cfg["H"]
        if name == "x-":
            return np.array([-L / 2, -W / 2, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])
        if name == "x+":
            return np.array([L / 2, -W / 2, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]), np.array([-1.0, 0.0, 0.0])
        if name == "y-":
            return np.array([-L / 2, -W / 2, 0.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0])
        if name == "y+":
            return np.array([-L / 2, W / 2, 0.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), np.array([0.0, -1.0, 0.0])
        raise ValueError(name)

    def _wall_inplane(self, name):
        """In-plane extent (tangent length, vertical length) of a wall."""
        L, W, H = self.cfg["L"], self.cfg["W"], self.cfg["H"]
        if name in ("x-", "x+"):
            return W, H
        return L, H

    def _rotate_wall(self, pts, name):
        """Rotate wall points about the horizontal tangent axis at the wall base.

        Simulates a LEANING wall panel (out-of-vertical tilt); GT
        perpendicularity deviation = tilt_deg.
        """
        th = np.radians(self.cfg["tilt_deg"])
        if th == 0.0:
            return pts
        L, W = self.cfg["L"], self.cfg["W"]
        a = {
            "x-": np.array([-L / 2, 0.0, 0.0]),
            "x+": np.array([L / 2, 0.0, 0.0]),
            "y-": np.array([0.0, -W / 2, 0.0]),
            "y+": np.array([0.0, W / 2, 0.0]),
        }[name]
        if name in ("x-", "x+"):  # rotate about the y-axis (wall tangent)
            R = np.array([
                [np.cos(th), 0.0, np.sin(th)],
                [0.0, 1.0, 0.0],
                [-np.sin(th), 0.0, np.cos(th)],
            ])
        else:  # rotate about the x-axis (wall tangent)
            R = np.array([
                [1.0, 0.0, 0.0],
                [0.0, np.cos(th), -np.sin(th)],
                [0.0, np.sin(th), np.cos(th)],
            ])
        return (pts - a) @ R.T + a

    # ---- element sampling -------------------------------------------------
    def _sample_wall_face(self, name, n, skip_rects=None, skip_circles=None):
        """Uniformly sample a wall face, rejecting openings."""
        origin, et, ev, en = self._wall_pose(name)
        T, V = self._wall_inplane(name)
        skip_rects = skip_rects or []
        skip_circles = skip_circles or []
        out = []
        tries = 0
        while len(out) < n and tries < n * 20:
            t = self.rng.uniform(0, T)
            v = self.rng.uniform(0, V)
            p = origin + t * et + v * ev
            if self._in_rects(p, name, skip_rects) or self._in_circles(p, name, skip_circles):
                tries += 1
                continue
            out.append(p)
        pts = np.asarray(out, dtype=np.float64) if out else np.zeros((0, 3))
        if len(pts):
            # waviness along the vertical direction
            amp = self.cfg["waviness_amp"]
            freq = self.cfg["waviness_freq"]
            phase = self.rng.uniform(0, 2 * np.pi)
            coord = (pts - origin) @ ev
            disp = amp * np.sin(2 * np.pi * freq * coord / V + phase)
            pts = pts + disp[:, None] * en
        return self._rotate_wall(pts, name)

    def _in_rects(self, p, wall, rects):
        for rc in rects:
            if rc["wall"] != wall:
                continue
            origin, et, ev, _ = self._wall_pose(wall)
            rel = p - origin
            t, v = rel @ et, rel @ ev
            if abs(t - rc["ct"]) <= rc["wt"] / 2 and rc["v0"] <= v <= rc["v1"]:
                return True
        return False

    def _in_circles(self, p, wall, circles):
        for cc in circles:
            if cc["wall"] != wall:
                continue
            origin, et, ev, _ = self._wall_pose(wall)
            rel = p - origin
            t, v = rel @ et, rel @ ev
            if (t - cc["ct"]) ** 2 + (v - cc["cv"]) ** 2 <= cc["r"] ** 2:
                return True
        return False

    def _sample_panel(self, wall, ct, wt, v0, vh, n):
        """Sample a vertical rectangle on a wall (door leaf / window pane)."""
        origin, et, ev, en = self._wall_pose(wall)
        t = self.rng.uniform(ct - wt / 2, ct + wt / 2, n)
        v = self.rng.uniform(v0, vh, n)
        pts = origin + t[:, None] * et + v[:, None] * ev
        return self._rotate_wall(pts, wall)

    def _sample_hole_cylinder(self, hc, n):
        """Side surface of a through-hole cylinder piercing the wall."""
        wall = hc["wall"]
        origin, et, ev, en = self._wall_pose(wall)
        ctr = origin + hc["ct"] * et + hc["cv"] * ev
        phi = self.rng.uniform(0, 2 * np.pi, n)
        depth = self.rng.uniform(0, WALL_THICKNESS, n)
        pts = ctr[None, :] + (hc["r"] * np.cos(phi))[:, None] * et + \
            (hc["r"] * np.sin(phi))[:, None] * ev + depth[:, None] * en
        return self._rotate_wall(pts, wall)

    def _sample_embedded_box(self, eb, n):
        """Exposed faces of a protruding box (switch panel / pipe box)."""
        wall = eb["wall"]
        origin, et, ev, en = self._wall_pose(wall)
        ctr = origin + eb["ct"] * et + eb["cv"] * ev + (eb["prot"] / 2) * en
        hw, hh, hp = eb["w"] / 2, eb["h"] / 2, eb["prot"] / 2
        faces = [
            # front face (parallel to wall), outward normal en
            lambda m: ctr + hp * en + self.rng.uniform(-hw, hw, m)[:, None] * et + self.rng.uniform(-hh, hh, m)[:, None] * ev,
            # 4 side faces
            lambda m: ctr + self.rng.uniform(-hp, hp, m)[:, None] * en + hw * et + self.rng.uniform(-hh, hh, m)[:, None] * ev,
            lambda m: ctr + self.rng.uniform(-hp, hp, m)[:, None] * en - hw * et + self.rng.uniform(-hh, hh, m)[:, None] * ev,
            lambda m: ctr + self.rng.uniform(-hp, hp, m)[:, None] * en + self.rng.uniform(-hw, hw, m)[:, None] * et + hh * ev,
            lambda m: ctr + self.rng.uniform(-hp, hp, m)[:, None] * en + self.rng.uniform(-hw, hw, m)[:, None] * et - hh * ev,
        ]
        per = max(1, n // 5)
        pts = np.concatenate([f(per) for f in faces], axis=0)[:n]
        return self._rotate_wall(pts, wall)

    # ---- occlusion --------------------------------------------------------
    def _occlusion_mask(self, points, boxes):
        """Single-station scan: mask of points occluded by embedded boxes."""
        if not boxes:
            return np.ones(len(points), dtype=bool)
        L, W, H = self.cfg["L"], self.cfg["W"], self.cfg["H"]
        vp = np.array([0.0, 0.0, H * 0.7])
        keep = np.ones(len(points), dtype=bool)
        for i, p in enumerate(points):
            for eb in boxes:
                origin, et, ev, en = self._wall_pose(eb["wall"])
                ctr = origin + eb["ct"] * et + eb["cv"] * ev + (eb["prot"] / 2) * en
                lo = ctr - np.array([eb["w"] / 2 + 0.02, eb["h"] / 2 + 0.02, eb["prot"] / 2 + 0.02])
                hi = ctr + np.array([eb["w"] / 2 + 0.02, eb["h"] / 2 + 0.02, eb["prot"] / 2 + 0.02])
                if self._segment_hits_aabb(vp, p, lo, hi):
                    keep[i] = False
                    break
        return keep

    @staticmethod
    def _segment_hits_aabb(o, p, lo, hi):
        d = p - o
        tmin, tmax = 0.0, 1.0
        for k in range(3):
            if abs(d[k]) < 1e-12:
                if o[k] < lo[k] or o[k] > hi[k]:
                    return False
            else:
                t1 = (lo[k] - o[k]) / d[k]
                t2 = (hi[k] - o[k]) / d[k]
                tmin = max(tmin, min(t1, t2))
                tmax = min(tmax, max(t1, t2))
                if tmin > tmax:
                    return False
        return True

    # ---- main -------------------------------------------------------------
    def generate(self, n_points=12000):
        cfg = self.cfg
        L, W, H = cfg["L"], cfg["W"], cfg["H"]
        rng = self.rng

        # rects / circles used for wall rejection (in-plane coords)
        rects = []
        if cfg["door"]:
            d = cfg["door"]
            rects.append({"wall": d["wall"], "ct": d["ct"], "wt": d["w"], "v0": 0.0, "v1": d["h"]})
        if cfg["window"]:
            w = cfg["window"]
            rects.append({"wall": w["wall"], "ct": w["ct"], "wt": w["w"], "v0": w["zc"] - w["h"] / 2, "v1": w["zc"] + w["h"] / 2})

        holes_c = []
        for h in cfg["holes"]:
            origin, et, ev, _ = self._wall_pose(h["wall"])
            holes_c.append({"wall": h["wall"], "ct": h["ct"], "cv": h["cv"], "r": h["r"]})

        # element point budgets
        n_door = 300 if cfg["door"] else 0
        n_win = 300 if cfg["window"] else 0
        n_hole = sum(250 for _ in cfg["holes"])
        n_emb = sum(180 for _ in cfg["embedded"])
        n_elem = n_door + n_win + n_hole + n_emb
        n_surface = max(3000, n_points - n_elem)

        # surface areas for allocation
        wall_area = 2 * (L * H + W * H)
        if cfg["door"]:
            wall_area -= cfg["door"]["w"] * cfg["door"]["h"]
        if cfg["window"]:
            wall_area -= cfg["window"]["w"] * cfg["window"]["h"]
        for h in cfg["holes"]:
            wall_area -= np.pi * h["r"] ** 2
        floor_area = L * W
        ceil_area = L * W
        total_area = wall_area + floor_area + ceil_area
        n_wall = int(n_surface * wall_area / total_area)
        n_floor = int(n_surface * floor_area / total_area)
        n_ceil = n_surface - n_wall - n_floor

        pts, labels = [], []
        for name in WALL_NAMES:
            n_face = max(0, n_wall // 4)
            p = self._sample_wall_face(name, n_face, rects, holes_c)
            pts.append(p)
            labels.append(np.full(len(p), 1, dtype=np.int64))

        rng_floor = self.rng
        fx = rng_floor.uniform(-L / 2, L / 2, n_floor)
        fy = rng_floor.uniform(-W / 2, W / 2, n_floor)
        pts.append(np.stack([fx, fy, np.zeros(n_floor)], axis=1))
        labels.append(np.full(n_floor, 2, dtype=np.int64))

        cx = rng.uniform(-L / 2, L / 2, n_ceil)
        cy = rng.uniform(-W / 2, W / 2, n_ceil)
        pts.append(np.stack([cx, cy, np.full(n_ceil, H)], axis=1))
        labels.append(np.full(n_ceil, 3, dtype=np.int64))

        if cfg["door"]:
            d = cfg["door"]
            p = self._sample_panel(d["wall"], d["ct"], d["w"], 0.0, d["h"], n_door)
            pts.append(p)
            labels.append(np.full(len(p), 4, dtype=np.int64))
        if cfg["window"]:
            w = cfg["window"]
            p = self._sample_panel(w["wall"], w["ct"], w["w"], w["zc"] - w["h"] / 2, w["zc"] + w["h"] / 2, n_win)
            pts.append(p)
            labels.append(np.full(len(p), 5, dtype=np.int64))
        for h in cfg["holes"]:
            p = self._sample_hole_cylinder(h, 250)
            pts.append(p)
            labels.append(np.full(len(p), 6, dtype=np.int64))
        for eb in cfg["embedded"]:
            p = self._sample_embedded_box(eb, 180)
            pts.append(p)
            labels.append(np.full(len(p), 7, dtype=np.int64))

        points = np.concatenate(pts, axis=0)
        labels = np.concatenate(labels, axis=0)

        # occlusion by protruding embedded boxes: boxes occlude everything EXCEPT
        # themselves (they are measurement targets and would self-occlude their
        # own front faces otherwise)
        keep = np.ones(len(points), dtype=bool)
        if cfg["embedded"]:
            is_embedded = labels == 7
            keep[~is_embedded] = self._occlusion_mask(points[~is_embedded], cfg["embedded"])
        points, labels = points[keep], labels[keep]

        # sensor noise
        if cfg["noise_sigma"] > 0:
            points = points + rng.normal(0, cfg["noise_sigma"], points.shape)

        # random dropout (registration gaps)
        if cfg["dropout"] > 0:
            keep = rng.random(len(points)) > cfg["dropout"]
            points, labels = points[keep], labels[keep]

        # range cutoff (single-station reach)
        if cfg["max_range"] is not None:
            vp = np.array([0.0, 0.0, H * 0.7])
            d = np.linalg.norm(points - vp, axis=1)
            keep = d <= cfg["max_range"]
            points, labels = points[keep], labels[keep]

        # shuffle
        idx = rng.permutation(len(points))
        points, labels = points[idx], labels[idx]

        return {
            "points": points.astype(np.float32),
            "labels": labels.astype(np.int64),
            "cfg": cfg,
        }


def build_dataset(cfg, split="train", start=0):
    """Build a deterministic dataset split as a list of sample dicts."""
    from geometry.measures import gt_from_config

    rng = np.random.default_rng(cfg["seed"] + (0 if split == "train" else 100000) + start)
    n = cfg["data"]["n_train"] if split == "train" else cfg["data"]["n_val"]
    samples = []
    for i in range(n):
        seed = int(rng.integers(0, 2**31 - 1))
        mtype = MODULE_TYPES[i % len(MODULE_TYPES)]
        task = TASK_NAMES[i % len(TASK_NAMES)]
        gcfg = sample_config(rng, cfg["data"], mtype, task)
        gen = MiCModuleGenerator(gcfg, seed)
        s = gen.generate(n_points=cfg["data"]["n_points"])
        s["task"] = TASK_NAMES.index(task)
        s["gt"] = gt_from_config(gcfg)
        samples.append(s)
    return samples
