"""Training script for the TADH-MiC dual-head framework.

Usage:
  python train.py --config configs/default.yaml [--epochs 40] [--out out/default]
                  [--device cpu|cuda] [--seed 42] [--n-train 80] [--n-val 16]
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
from eval import SLOT_SCALES
from models.dualhead import TADHMiC
from models.heads import total_loss


def to_tensor(sample, device):
    points = torch.from_numpy(sample["points"])[None].float().to(device)
    labels = torch.from_numpy(sample["labels"])[None].long().to(device)
    task = torch.tensor([sample["task"]], dtype=torch.long, device=device)
    gt = {k: (torch.from_numpy(v)[None].float().to(device) if v is not None else None)
          for k, v in sample["gt"].items()}
    return points, labels, task, gt


@torch.no_grad()
def evaluate(model, samples, device, weights):
    model.eval()
    seg_accs, task_accs, meas_errs = [], [], []
    for s in samples:
        points, labels, task, gt = to_tensor(s, device)
        out = model(points)
        pred = out["seg_logits"].argmax(-1)
        seg_accs.append((pred == labels).float().mean().item())
        task_accs.append((out["task_logits"].argmax(-1) == task).float().mean().item())
        for name, g in gt.items():
            if g is None:
                continue
            scale = np.asarray(SLOT_SCALES[name], dtype=np.float64)
            err = ((out["meas"][name] - g).abs().cpu().numpy() * scale).mean()
            meas_errs.append(float(err))
    model.train()
    return {
        "seg_acc": float(np.mean(seg_accs)),
        "task_acc": float(np.mean(task_accs)),
        "meas_mae": float(np.mean(meas_errs)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--out", default="out/default")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--n-val", type=int, default=None)
    ap.add_argument("--n-points", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.n_train is not None:
        cfg["data"]["n_train"] = args.n_train
    if args.n_val is not None:
        cfg["data"]["n_val"] = args.n_val
    if args.n_points is not None:
        cfg["data"]["n_points"] = args.n_points
    if args.warmup is not None:
        cfg["train"]["warmup_epochs"] = args.warmup
    epochs = args.epochs or cfg["train"]["epochs"]

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = torch.device(args.device)

    print(f"[data] building {cfg['data']['n_train']} train / {cfg['data']['n_val']} val samples ...")
    ds_train = build_dataset(cfg, "train")
    ds_val = build_dataset(cfg, "val")

    model = TADHMiC(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] TADH-MiC with {n_params:,} parameters on {device}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                            weight_decay=cfg["train"]["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_mae = float("inf")

    for epoch in range(1, epochs + 1):
        # curriculum: measure head joins after warmup
        weights = dict(cfg["train"]["loss_weights"])
        if epoch <= cfg["train"]["warmup_epochs"]:
            weights["meas"] = 0.0

        model.train()
        run_seg = run_task = run_meas = 0.0
        for i, s in enumerate(ds_train):
            points, labels, task, gt = to_tensor(s, device)
            out = model(points)
            loss, parts = total_loss(out, labels, task, gt, weights)
            opt.zero_grad()
            loss.backward()
            opt.step()
            run_seg += parts["seg"]
            run_task += parts["task"]
            run_meas += parts["meas"]
            if (i + 1) % cfg["train"]["log_every"] == 0:
                print(f"  ep{epoch} it{i+1}/{len(ds_train)} "
                      f"seg={parts['seg']:.3f} task={parts['task']:.3f} meas={parts['meas']:.3f}")
        sched.step()

        m = evaluate(model, ds_val, device, weights)
        m["epoch"] = epoch
        history.append(m)
        print(f"[epoch {epoch:3d}] val seg_acc={m['seg_acc']:.4f} task_acc={m['task_acc']:.4f} "
              f"meas_mae={m['meas_mae']:.2f}")

        if m["meas_mae"] < best_mae:
            best_mae = m["meas_mae"]
            torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch},
                       out_dir / "best.pt")
        if epoch % cfg["train"]["save_every"] == 0:
            torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch},
                       out_dir / f"ckpt_{epoch}.pt")

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"[done] best val meas MAE = {best_mae:.2f} (mm for length slots, deg for perpendicularity); artifacts in {out_dir}")


if __name__ == "__main__":
    main()
