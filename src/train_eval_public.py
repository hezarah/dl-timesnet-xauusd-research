from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from .windows import make_windows
from .labels import yraw_to_3class
from .model_toy_timesnet import ToyTimesNet
from .evaluation_public import compute_metrics


def time_split(n: int, train_ratio: float, val_ratio: float):
    tr_end = int(n * train_ratio)
    va_end = int(n * (train_ratio + val_ratio))
    idx = np.arange(n)
    return idx[:tr_end], idx[tr_end:va_end], idx[va_end:]


def normalize_windows(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True) + 1e-8
    return (x - mean) / std


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    seed = int(cfg["training"]["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)

    csv_path = Path(cfg["data"]["csv_path"])
    time_col = cfg["data"]["time_col"]
    close_col = cfg["data"]["close_col"]

    df = pd.read_csv(csv_path)
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.sort_values(time_col)

    close = df[close_col].astype(float).to_numpy()

    wl = int(cfg["windows"]["window_length"])
    horizon = int(cfg["windows"]["horizon"])
    stride = int(cfg["windows"]["stride"])

    X, y_raw = make_windows(close, wl, horizon, stride)
    X = normalize_windows(X)

    thr = float(cfg["labels"]["threshold"])
    y = yraw_to_3class(y_raw, thr)

    n = len(y)
    tr, va, te = time_split(n, float(cfg["split"]["train_ratio"]), float(cfg["split"]["val_ratio"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ToyTimesNet(
        d_model=int(cfg["model"]["d_model"]),
        num_layers=int(cfg["model"]["num_layers"]),
        dropout=float(cfg["model"]["dropout"]),
        num_classes=3,
    ).to(device)

    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["training"]["lr"]))

    bs = int(cfg["training"]["batch_size"])
    train_loader = DataLoader(TensorDataset(torch.tensor(X[tr]), torch.tensor(y[tr])), batch_size=bs, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X[va]), torch.tensor(y[va])), batch_size=bs, shuffle=False)
    test_loader = DataLoader(TensorDataset(torch.tensor(X[te]), torch.tensor(y[te])), batch_size=bs, shuffle=False)

    best_val = 1e9
    best_state = None

    for epoch in range(1, int(cfg["training"]["epochs"]) + 1):
        model.train()
        total = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(xb)

        model.eval()
        vtotal = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = loss_fn(logits, yb)
                vtotal += float(loss.item()) * len(xb)

        train_loss = total / len(tr)
        val_loss = vtotal / max(1, len(va))
        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            y_pred.append(pred)
            y_true.append(yb.numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    metrics = compute_metrics(y_true, y_pred)
    print("=== Research-only DL Demo Results ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
