"""Evaluation script: geometry baseline vs learned dual-head framework.

Modes:
  geom        deterministic geometry on GT labels (upper bound of the split-then-fit
              paradigm — the same one used by Shu 2023 / Wan 2025 style pipelines)
  geom-pred   deterministic geometry on model-predicted labels (full two-stage pipeline)
  learned     end-to-end TADH-MiC (soft-mask pooled regression, no geometric post-hoc)

Usage:
  python eval.py --config configs/default.yaml [--checkpoint out/default/best.pt]
                 [--mode geom|geom-pred|learned] [--split val] [--out out/eval.json]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from datasets.synthetic import build_dataset
from geometry.measures import measure_from_labels
from models.dualhead import TADHMiC

GROUP_ORDER = ["module", "door", "window", "hole", "embedded"]

# per-slot unit scales: meters -> mm for length slots, degrees kept as-is
SLOT_SCALES = {
    "module": [1000.0, 1000.0, 1000.0, 1000.0, 1.0],
    "door": [1000.0, 1000.0],
    "window": [1000.0, 1000.0],
    "hole": [1000.0, 1000.0],
    "embedded": [1000.0, 1000.0, 1000.0],
}


def scaled_err(pred, gt, group):
    scale = np.asarray(SLOT_SCALES[group], dtype=np.float64)
    k = min(len(gt), len(pred))
    return np.abs(pred[:k] - gt[:k]) * scale[:k]


def meas_to_mm(m):
    return {k: (None if v is None else v * 1000.0) for k, v in m.items()}


def report_errors(errs_by_group):
    """errs_by_group: group -> list of |pred-gt| vectors (mm)."""
    rows = {}
    for g, errs in errs_by_group.items():
        if not errs:
            continue
        arr = np.array(errs)
        rows[g] = {"n": int(arr.shape[0]), "mae_mm": float(arr.mean(axis=0).mean()),
                   "rmse_mm": float(np.sqrt((arr ** 2).mean()))}
    return rows


def eval_geometry(samples, use_gt_labels):
    errs_by_group = {g: [] for g in GROUP_ORDER}
    for s in samples:
        labels = s["labels"] if use_gt_labels else s.get("pred_labels")
        if labels is None:
            continue
        pred = measure_from_labels(s["points"], labels)
        for g in GROUP_ORDER:
            if s["gt"][g] is None or pred[g] is None:
                continue
            errs_by_group[g].append(scaled_err(pred[g], s["gt"][g], g))
    return report_errors(errs_by_group)


def eval_learned(model, samples, device):
    model.eval()
    errs_by_group = {g: [] for g in GROUP_ORDER}
    seg_accs, task_accs = [], []
    with torch.no_grad():
        for s in samples:
            points = torch.from_numpy(s["points"])[None].float().to(device)
            labels = torch.from_numpy(s["labels"])[None].long().to(device)
            task = torch.tensor([s["task"]], dtype=torch.long, device=device)
            out = model(points)
            pred = out["seg_logits"].argmax(-1)
            seg_accs.append((pred == labels).float().mean().item())
            task_accs.append((out["task_logits"].argmax(-1) == task).float().mean().item())
            # store predicted labels for geom-pred mode
            s["pred_labels"] = pred[0].cpu().numpy()
            for g in GROUP_ORDER:
                gt = s["gt"][g]
                if gt is None:
                    continue
                p = out["meas"][g][0].cpu().numpy()
                errs_by_group[g].append(scaled_err(p, gt, g))
    return {
        "measure": report_errors(errs_by_group),
        "seg_acc": float(np.mean(seg_accs)),
        "task_acc": float(np.mean(task_accs)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--mode", default="learned", choices=["geom", "geom-pred", "learned"])
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", default="out/eval.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-samples", type=int, default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.mode == "learned" and args.checkpoint is None:
        raise SystemExit("--checkpoint is required for mode=learned")
    if args.mode == "geom-pred" and args.checkpoint is None:
        raise SystemExit("--checkpoint is required for mode=geom-pred")

    samples = build_dataset(cfg, args.split)
    if args.n_samples:
        samples = samples[: args.n_samples]

    if args.mode in ("geom-pred", "learned"):
        device = torch.device(args.device)
        ckpt = torch.load(args.checkpoint, map_location=device)
        model = TADHMiC(cfg).to(device)
        model.load_state_dict(ckpt["model"])

    if args.mode == "geom":
        res = {"mode": "geom", "measure": eval_geometry(samples, use_gt_labels=True)}
    elif args.mode == "geom-pred":
        eval_learned(model, samples, device)  # fills pred_labels
        res = {"mode": "geom-pred", "measure": eval_geometry(samples, use_gt_labels=False)}
    else:
        res = {"mode": "learned", **eval_learned(model, samples, device)}

    # pass-rate summary vs config tolerances (m -> mm)
    tol = cfg.get("tolerances", {})
    tol_key = {"module": "length", "door": "door", "window": "window",
               "hole": "hole", "embedded": "embedded"}
    pass_rate = {}
    if res["measure"]:
        for g, r in res["measure"].items():
            t_mm = tol.get(tol_key[g], 5.0) * 1000.0
            pass_rate[g] = {"tolerance_mm": t_mm, "mae_mm": r["mae_mm"],
                            "mae_within_tol": bool(r["mae_mm"] <= t_mm)}
    res["pass_rate"] = pass_rate

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    print(f"[done] results written to {out_path}")


if __name__ == "__main__":
    main()
