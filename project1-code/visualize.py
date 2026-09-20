"""Visualization: export labeled point clouds as PLY (and optional PNG views).

Usage:
  python visualize.py --config configs/default.yaml --index 0 [--checkpoint out/default/best.pt]
                      [--out out/vis] [--split val] [--device cpu]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from datasets.synthetic import build_dataset, LABEL_NAMES
from models.dualhead import TADHMiC

PALETTE = np.array([
    [40, 40, 40],     # background
    [180, 180, 180],  # wall
    [120, 90, 60],    # floor
    [90, 120, 150],   # ceiling
    [200, 120, 60],   # door
    [90, 160, 220],   # window
    [220, 60, 60],    # hole
    [60, 200, 120],   # embedded
], dtype=np.uint8)


def write_ply(path, points, labels):
    rgb = PALETTE[labels]
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("property int label\nend_header\n")
        for p, c, l in zip(points, rgb, labels):
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {c[0]} {c[1]} {c[2]} {l}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", default="out/vis")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    samples = build_dataset(cfg, args.split)
    s = samples[args.index]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_ply(out_dir / f"{args.split}_{args.index}_gt.ply", s["points"], s["labels"])

    if args.checkpoint:
        device = torch.device(args.device)
        ckpt = torch.load(args.checkpoint, map_location=device)
        model = TADHMiC(cfg).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        with torch.no_grad():
            points = torch.from_numpy(s["points"])[None].float().to(device)
            out = model(points)
            pred = out["seg_logits"].argmax(-1)[0].cpu().numpy()
        write_ply(out_dir / f"{args.split}_{args.index}_pred.ply", s["points"], pred)
        print("predicted measurements (m):")
        for g, v in out["meas"].items():
            print(f"  {g:10s} {v[0].cpu().numpy()}")
        print("ground truth (m):")
        for g, v in s["gt"].items():
            print(f"  {g:10s} {None if v is None else v}")
    print(f"[done] PLY files in {out_dir}")


if __name__ == "__main__":
    main()
