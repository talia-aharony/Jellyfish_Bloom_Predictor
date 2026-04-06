"""
jellyfish/train.py
==================
Train jellyfish forecasting models and save weights to disk.

Two modes
─────────
Global training  (default):
    Trains on all beaches combined.  Produces one model file per architecture.
    Use as a baseline or as the starting point for per-beach fine-tuning.

Per-beach fine-tuning  (--finetune-per-beach):
    Loads a pre-trained global JellyfishNet checkpoint and fine-tunes a
    separate copy for each beach.  Sparse beaches inherit the global model's
    representations; well-observed beaches can specialise further.

Usage examples
──────────────
    # Global training (GRU + JellyfishNet)
    python -m jellyfish.train

    # Global training with integrated weather CSV
    python -m jellyfish.train --use-integrated-data

    # Per-beach fine-tuning from a global checkpoint
    python -m jellyfish.train --finetune-per-beach \\
        --global-checkpoint models/jellyfishnet_model.pth
"""

import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

warnings.filterwarnings("ignore")

if __package__ in (None, ""):
    ROOT = os.path.dirname(os.path.dirname(__file__))
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from jellyfish.data_loader import load_jellyfish_data
    from jellyfish.data_loader_forecasting import load_integrated_data
    from jellyfish.settings import (
        DEFAULT_LOOKBACK_DAYS,
        DEFAULT_WEATHER_CSV_PATH,
        DEFAULT_USE_INTEGRATED_DATA,
        DEFAULT_INCLUDE_LIVE_XML,
        DEFAULT_BATCH_SIZE,
        DEFAULT_LEARNING_RATE,
        DEFAULT_DROPOUT_PROB,
        DEFAULT_NUM_EPOCHS,
        DEFAULT_PATIENCE,
        DEFAULT_HYBRID_HIDDEN_DIM,
        DEFAULT_REPORT_PATH,
        DEFAULT_MODEL_NAMES,
        DEFAULT_OUTPUT_DIR,
        DEFAULT_FINETUNE_EPOCHS,
        DEFAULT_FINETUNE_LR,
        DEFAULT_MIN_SAMPLES_PER_BEACH,
        DEFAULT_PER_BEACH_PATIENCE,
        DEFAULT_LR_SCHEDULER_FACTOR,
        DEFAULT_LR_SCHEDULER_PATIENCE,
        DEFAULT_GRAD_CLIP_NORM,
        DEFAULT_THRESHOLD_MIN,
        DEFAULT_THRESHOLD_MAX,
        DEFAULT_THRESHOLD_STEPS,
    )
    from jellyfish.models import (
        BaselineLogisticRegression,
        FeedforwardNet,
        LSTMNet,
        GRUNet,
        Conv1DNet,
        HybridNet,
        JellyfishNet,
    )
else:
    from .data_loader import load_jellyfish_data
    from .data_loader_forecasting import load_integrated_data
    from .settings import (
        DEFAULT_LOOKBACK_DAYS,
        DEFAULT_WEATHER_CSV_PATH,
        DEFAULT_USE_INTEGRATED_DATA,
        DEFAULT_INCLUDE_LIVE_XML,
        DEFAULT_BATCH_SIZE,
        DEFAULT_LEARNING_RATE,
        DEFAULT_DROPOUT_PROB,
        DEFAULT_NUM_EPOCHS,
        DEFAULT_PATIENCE,
        DEFAULT_HYBRID_HIDDEN_DIM,
        DEFAULT_REPORT_PATH,
        DEFAULT_MODEL_NAMES,
        DEFAULT_OUTPUT_DIR,
        DEFAULT_FINETUNE_EPOCHS,
        DEFAULT_FINETUNE_LR,
        DEFAULT_MIN_SAMPLES_PER_BEACH,
        DEFAULT_PER_BEACH_PATIENCE,
        DEFAULT_LR_SCHEDULER_FACTOR,
        DEFAULT_LR_SCHEDULER_PATIENCE,
        DEFAULT_GRAD_CLIP_NORM,
        DEFAULT_THRESHOLD_MIN,
        DEFAULT_THRESHOLD_MAX,
        DEFAULT_THRESHOLD_STEPS,
    )
    from .models import (
        BaselineLogisticRegression,
        FeedforwardNet,
        LSTMNet,
        GRUNet,
        Conv1DNet,
        HybridNet,
        JellyfishNet,
    )

BATCH_SIZE    = DEFAULT_BATCH_SIZE
LEARNING_RATE = DEFAULT_LEARNING_RATE
DROPOUT_PROB  = DEFAULT_DROPOUT_PROB
NUM_EPOCHS    = DEFAULT_NUM_EPOCHS


# ============================================================================
# Feature engineering (for Baseline logistic regression)
# ============================================================================

def create_engineered_features_forecasting(X, lookback=DEFAULT_LOOKBACK_DAYS):
    """Flatten sequences into hand-crafted temporal statistics for the baseline."""
    n_samples, lb, n_features = X.shape
    out = []
    for i in range(n_samples):
        seq = X[i]
        row = []
        for f in range(n_features):
            ts    = seq[:, f]
            cur   = ts[-1]
            prev  = ts[-2] if lb >= 2 else ts[0]
            prev3 = ts[-4] if lb >= 4 else ts[0]
            trend = np.polyfit(np.arange(lb), ts, 1)[0]
            row.extend([
                cur, prev, prev3, trend,
                ts.mean(), ts.std(), ts.min(), ts.max(),
                cur - prev, cur - prev3,
            ])
        out.append(row)
    return np.array(out, dtype=np.float32)


# ============================================================================
# Trainer
# ============================================================================

class Trainer:
    """Training pipeline shared by all models."""

    def __init__(
        self,
        model,
        device="cpu",
        learning_rate=LEARNING_RATE,
        model_id="default",
        scheduler_factor=DEFAULT_LR_SCHEDULER_FACTOR,
        scheduler_patience=DEFAULT_LR_SCHEDULER_PATIENCE,
        grad_clip_norm=DEFAULT_GRAD_CLIP_NORM,
        threshold_min=DEFAULT_THRESHOLD_MIN,
        threshold_max=DEFAULT_THRESHOLD_MAX,
        threshold_steps=DEFAULT_THRESHOLD_STEPS,
    ):
        self.model      = model.to(device)
        self.device     = device
        self.model_id   = model_id
        self.grad_clip_norm = float(grad_clip_norm)
        self.threshold_min = float(threshold_min)
        self.threshold_max = float(threshold_max)
        self.threshold_steps = int(threshold_steps)
        self.best_state = None
        self.optimizer  = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion  = nn.BCELoss()
        self.scheduler  = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=float(scheduler_factor),
            patience=int(scheduler_patience),
        )
        self.train_losses = []
        self.val_losses   = []
        self.train_accs   = []
        self.val_accs     = []

    def train_epoch(self, loader):
        self.model.train()
        total_loss = correct = total = 0
        for X_b, y_b in loader:
            X_b = X_b.to(self.device)
            y_b = y_b.to(self.device).unsqueeze(1)
            out  = self.model(X_b)
            loss = self.criterion(out, y_b)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.optimizer.step()
            total_loss += loss.item()
            correct    += ((out > 0.5).float() == y_b).sum().item()
            total      += y_b.size(0)
        return total_loss / len(loader), correct / total

    def evaluate(self, loader):
        self.model.eval()
        total_loss = correct = total = 0
        with torch.no_grad():
            for X_b, y_b in loader:
                X_b = X_b.to(self.device)
                y_b = y_b.to(self.device).unsqueeze(1)
                out  = self.model(X_b)
                loss = self.criterion(out, y_b)
                total_loss += loss.item()
                correct    += ((out > 0.5).float() == y_b).sum().item()
                total      += y_b.size(0)
        return total_loss / len(loader), correct / total

    def fit(self, train_loader, val_loader, epochs=NUM_EPOCHS, patience=DEFAULT_PATIENCE):
        best_val, wait = float("inf"), 0
        for epoch in range(epochs):
            tr_loss, tr_acc = self.train_epoch(train_loader)
            vl_loss, vl_acc = self.evaluate(val_loader)
            self.scheduler.step(vl_loss)
            self.train_losses.append(tr_loss)
            self.val_losses.append(vl_loss)
            self.train_accs.append(tr_acc)
            self.val_accs.append(vl_acc)
            if (epoch + 1) % 20 == 0:
                print(
                    f"  Epoch {epoch+1:3d}/{epochs}  "
                    f"train_loss={tr_loss:.4f}  val_loss={vl_loss:.4f}  "
                    f"train_acc={tr_acc:.4f}  val_acc={vl_acc:.4f}"
                )
            if vl_loss < best_val:
                best_val, wait = vl_loss, 0
                self.best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                wait += 1
                if wait >= patience:
                    print(f"  Early stopping at epoch {epoch + 1}")
                    break
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)

    def _collect(self, loader):
        self.model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for X_b, y_b in loader:
                out = self.model(X_b.to(self.device))
                preds.extend(out.cpu().view(-1).tolist())
                labels.extend(y_b.view(-1).tolist())
        return np.array(preds), np.array(labels)

    @staticmethod
    def _metrics(labels, preds, threshold=0.5):
        binary = (preds >= threshold).astype(int)
        acc    = (binary == labels).mean()
        tp = ((binary == 1) & (labels == 1)).sum()
        fp = ((binary == 1) & (labels == 0)).sum()
        fn = ((binary == 0) & (labels == 1)).sum()
        tn = ((binary == 0) & (labels == 0)).sum()
        prec = tp / (tp + fp) if tp + fp else 0
        rec  = tp / (tp + fn) if tp + fn else 0
        f1   = 2 * prec * rec / (prec + rec) if prec + rec else 0
        return {
            "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
            "confusion_matrix": np.array([[tn, fp], [fn, tp]]),
            "predictions": preds, "labels": labels, "threshold": threshold,
        }

    def find_best_threshold(self, val_loader):
        preds, labels = self._collect(val_loader)
        best_thresh, best_f1 = 0.5, -1.0
        for t in np.linspace(self.threshold_min, self.threshold_max, self.threshold_steps):
            f1 = self._metrics(labels, preds, threshold=t)["f1"]
            if f1 > best_f1:
                best_f1, best_thresh = f1, float(t)
        return best_thresh, best_f1

    def test(self, test_loader, threshold=0.5):
        preds, labels = self._collect(test_loader)
        m = self._metrics(labels, preds, threshold)
        m["auc"] = self._auc(labels, preds)
        return m

    @staticmethod
    def _auc(y_true, y_pred):
        n_pos, n_neg = (y_true == 1).sum(), (y_true == 0).sum()
        if n_pos == 0 or n_neg == 0:
            return 0.5
        tprs, fprs = [], []
        for t in np.linspace(1, 0, 101):
            b  = (y_pred >= t).astype(int)
            tp = ((b == 1) & (y_true == 1)).sum()
            fp = ((b == 1) & (y_true == 0)).sum()
            tprs.append(tp / n_pos)
            fprs.append(fp / n_neg)
        return float(np.trapz(tprs, fprs))


# ============================================================================
# Reporting
# ============================================================================

def save_training_report(results, config, output_path):
    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": config,
        "results": {
            name: {
                "accuracy":         float(m["accuracy"]),
                "precision":        float(m["precision"]),
                "recall":           float(m["recall"]),
                "f1":               float(m["f1"]),
                "auc":              float(m["auc"]),
                "threshold":        float(m.get("threshold", 0.5)),
                "val_best_f1":      float(m.get("val_best_f1", 0.0)),
                "confusion_matrix": m["confusion_matrix"].tolist(),
            }
            for name, m in results.items()
        },
    }
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"✓ Saved training report: {output_path}")


# ============================================================================
# Global training
# ============================================================================

def train_all_models(
    lookback_days=DEFAULT_LOOKBACK_DAYS,
    use_integrated_data=DEFAULT_USE_INTEGRATED_DATA,
    weather_csv_path=DEFAULT_WEATHER_CSV_PATH,
    include_live_xml=DEFAULT_INCLUDE_LIVE_XML,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    dropout_prob=DROPOUT_PROB,
    num_epochs=NUM_EPOCHS,
    patience=DEFAULT_PATIENCE,
    hybrid_hidden_dim=DEFAULT_HYBRID_HIDDEN_DIM,
    model_names=None,
    report_path=DEFAULT_REPORT_PATH,
    output_dir=DEFAULT_OUTPUT_DIR,
    scheduler_factor=DEFAULT_LR_SCHEDULER_FACTOR,
    scheduler_patience=DEFAULT_LR_SCHEDULER_PATIENCE,
    grad_clip_norm=DEFAULT_GRAD_CLIP_NORM,
    threshold_min=DEFAULT_THRESHOLD_MIN,
    threshold_max=DEFAULT_THRESHOLD_MAX,
    threshold_steps=DEFAULT_THRESHOLD_STEPS,
):
    """Train selected models on the full (all-beach) dataset."""
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 100)
    print("JELLYFISH FORECASTING — GLOBAL TRAINING")
    print("=" * 100)

    config = {
        "lookback_days":              int(lookback_days),
        "use_integrated_data":        bool(use_integrated_data),
        "weather_csv_path":           str(weather_csv_path),
        "include_live_xml":           bool(include_live_xml),
        "batch_size":                 int(batch_size),
        "learning_rate":              float(learning_rate),
        "dropout_prob":               float(dropout_prob),
        "num_epochs":                 int(num_epochs),
        "patience":                   int(patience),
        "hybrid_hidden_dim":          int(hybrid_hidden_dim),
        "model_names":                list(model_names) if model_names else [m.strip() for m in DEFAULT_MODEL_NAMES.split(",") if m.strip()],
        "scheduler_factor":           float(scheduler_factor),
        "scheduler_patience":         int(scheduler_patience),
        "grad_clip_norm":             float(grad_clip_norm),
        "threshold_min":              float(threshold_min),
        "threshold_max":              float(threshold_max),
        "threshold_steps":            int(threshold_steps),
        "threshold_selection_metric": "f1_on_validation",
    }
    for k, v in config.items():
        print(f"  {k}: {v}")
    print()

    # ── Load data ────────────────────────────────────────────────────────────
    print("1. LOADING DATA")
    print("-" * 100)
    if use_integrated_data:
        integrated = load_integrated_data(
            weather_csv_path=weather_csv_path,
            lookback_days=lookback_days,
            forecast_days=1,
            include_live_xml=include_live_xml,
        )
        if integrated is None:
            raise RuntimeError("Failed to load integrated data.")
        X, y, metadata, feature_cols, *_ = integrated
    else:
        X, y, metadata = load_jellyfish_data(lookback_days=lookback_days, forecast_days=1)

    n_features = int(X.shape[2])
    config["n_features_per_day"] = n_features
    print()

    # ── Normalise ────────────────────────────────────────────────────────────
    print("2. DATA NORMALISATION")
    print("-" * 100)
    X_t  = torch.FloatTensor(X)
    mean = X_t.mean(dim=0)
    std  = X_t.std(dim=0)
    X_n  = (X_t - mean) / (std + 1e-8)
    y_t  = torch.FloatTensor(y)
    print("✓ Normalised")
    print()

    # ── Split ────────────────────────────────────────────────────────────────
    print("3. DATALOADERS")
    print("-" * 100)
    n    = len(X_n)
    n_tr = int(0.70 * n)
    n_val= int(0.15 * n)
    n_te = n - n_tr - n_val
    g    = torch.Generator().manual_seed(42)
    ds   = TensorDataset(X_n, y_t)
    tr_ds, val_ds, te_ds = random_split(ds, [n_tr, n_val, n_te], generator=g)
    tr_ld  = DataLoader(tr_ds,  batch_size=batch_size, shuffle=True)
    val_ld = DataLoader(val_ds, batch_size=batch_size)
    te_ld  = DataLoader(te_ds,  batch_size=batch_size)
    print(f"✓ Train: {n_tr}  Val: {n_val}  Test: {n_te}")
    print()

    # ── Model catalogue ──────────────────────────────────────────────────────
    print("4. TRAINING MODELS")
    print("-" * 100)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    requested = list(model_names) if model_names else [m.strip() for m in DEFAULT_MODEL_NAMES.split(",") if m.strip()]
    available = {
        "GRU": GRUNet(input_dim=n_features, dropout_prob=dropout_prob),
        "JellyfishNet": JellyfishNet(
            input_dim=n_features,
            hidden_dim=hybrid_hidden_dim,
            dropout_prob=dropout_prob,
            max_len=lookback_days,
        ),
        # Legacy models available on request
        "Hybrid":  HybridNet(input_dim=n_features, hidden_dim=hybrid_hidden_dim, dropout_prob=dropout_prob),
        "LSTM":    LSTMNet(input_dim=n_features, dropout_prob=dropout_prob),
        "Conv1D":  Conv1DNet(input_dim=n_features, dropout_prob=dropout_prob),
    }
    models = {}
    for name in requested:
        if name not in available:
            raise ValueError(f"Unknown model '{name}'. Choose from: {sorted(available)}")
        models[name] = available[name]

    results = {}
    for name, model in models.items():
        print(f"\n── {name} {'─'*(90 - len(name))}")
        t0      = time.time()
        trainer = Trainer(
            model,
            device=device,
            learning_rate=learning_rate,
            model_id=name.lower(),
            scheduler_factor=scheduler_factor,
            scheduler_patience=scheduler_patience,
            grad_clip_norm=grad_clip_norm,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            threshold_steps=threshold_steps,
        )
        trainer.fit(tr_ld, val_ld, epochs=num_epochs, patience=patience)
        thresh, val_f1 = trainer.find_best_threshold(val_ld)
        metrics        = trainer.test(te_ld, threshold=thresh)
        metrics["val_best_f1"] = val_f1
        elapsed = time.time() - t0

        save_path = os.path.join(output_dir, f"{name.lower()}_model.pth")
        torch.save(model.state_dict(), save_path)
        results[name] = metrics

        print(
            f"\n  Acc={metrics['accuracy']:.4f}  F1={metrics['f1']:.4f}  "
            f"AUC={metrics['auc']:.4f}  Thresh={thresh:.2f}  ({elapsed:.0f}s)"
        )
        print(f"  Saved → {save_path}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("TRAINING SUMMARY")
    print("=" * 100)
    print(f"{'Model':<20} {'Accuracy':>10} {'F1':>10} {'AUC':>10} {'Threshold':>10}")
    print("─" * 60)
    for name, m in sorted(results.items()):
        print(
            f"{name:<20} {m['accuracy']:>10.4f} {m['f1']:>10.4f} "
            f"{m['auc']:>10.4f} {m['threshold']:>10.2f}"
        )
    print("=" * 100)

    save_training_report(results, config, report_path)
    return results


# ============================================================================
# Per-beach fine-tuning
# ============================================================================

def finetune_per_beach(
    global_checkpoint,
    lookback_days=DEFAULT_LOOKBACK_DAYS,
    use_integrated_data=DEFAULT_USE_INTEGRATED_DATA,
    weather_csv_path=DEFAULT_WEATHER_CSV_PATH,
    include_live_xml=DEFAULT_INCLUDE_LIVE_XML,
    finetune_epochs=DEFAULT_FINETUNE_EPOCHS,
    finetune_lr=DEFAULT_FINETUNE_LR,
    dropout_prob=DROPOUT_PROB,
    hybrid_hidden_dim=DEFAULT_HYBRID_HIDDEN_DIM,
    min_samples=DEFAULT_MIN_SAMPLES_PER_BEACH,
    output_dir=DEFAULT_OUTPUT_DIR,
    report_path="training_report_per_beach.json",
    scheduler_factor=DEFAULT_LR_SCHEDULER_FACTOR,
    scheduler_patience=DEFAULT_LR_SCHEDULER_PATIENCE,
    grad_clip_norm=DEFAULT_GRAD_CLIP_NORM,
    threshold_min=DEFAULT_THRESHOLD_MIN,
    threshold_max=DEFAULT_THRESHOLD_MAX,
    threshold_steps=DEFAULT_THRESHOLD_STEPS,
):
    """
    Fine-tune a separate JellyfishNet for every beach.

    Beaches with fewer than `min_samples` sequences are skipped — the global
    model is used for them at prediction time (predictor falls back automatically).

    Parameters
    ──────────
    global_checkpoint : path to a pre-trained JellyfishNet .pth file
    min_samples       : minimum sequences a beach must have to be fine-tuned
    finetune_epochs   : max epochs per beach (default 20)
    finetune_lr       : learning rate for fine-tuning (default 1e-4, lower than
                        global training to avoid forgetting shared representations)
    output_dir        : where per-beach .pth files are written
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 100)
    print("JELLYFISH FORECASTING — PER-BEACH FINE-TUNING")
    print(f"Global checkpoint : {global_checkpoint}")
    print("=" * 100)

    # ── Load data ────────────────────────────────────────────────────────────
    if use_integrated_data:
        integrated = load_integrated_data(
            weather_csv_path=weather_csv_path,
            lookback_days=lookback_days,
            forecast_days=1,
            include_live_xml=include_live_xml,
        )
        if integrated is None:
            raise RuntimeError("Failed to load integrated data.")
        X, y, metadata, *_ = integrated
    else:
        X, y, metadata = load_jellyfish_data(lookback_days=lookback_days, forecast_days=1)

    n_features = int(X.shape[2])

    # Normalise using global statistics (must match global training)
    X_t  = torch.FloatTensor(X)
    mean = X_t.mean(dim=0)
    std  = X_t.std(dim=0)
    X_n  = (X_t - mean) / (std + 1e-8)
    y_t  = torch.FloatTensor(y)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    beach_ids = sorted(metadata["beach_id"].unique())
    results   = {}
    skipped   = []

    for beach_id in beach_ids:
        mask    = (metadata["beach_id"] == beach_id).values
        X_beach = X_n[mask]
        y_beach = y_t[mask]
        n       = len(X_beach)

        beach_name = metadata.loc[metadata["beach_id"] == beach_id, "beach_name"].iloc[0]

        if n < min_samples:
            print(f"\n  Beach {beach_id:2d} ({beach_name}) — skipped ({n} samples < {min_samples})")
            skipped.append(beach_id)
            continue

        print(f"\n── Beach {beach_id:2d} ({beach_name})  n={n} {'─'*60}")

        # Fresh copy of global weights for this beach
        model = JellyfishNet(
            input_dim=n_features,
            hidden_dim=hybrid_hidden_dim,
            dropout_prob=dropout_prob,
        )
        model.load_state_dict(torch.load(global_checkpoint, map_location=device))

        # 70 / 15 / 15 split for this beach
        n_tr  = max(1, int(0.70 * n))
        n_val = max(1, int(0.15 * n))
        n_te  = max(1, n - n_tr - n_val)
        while n_tr + n_val + n_te > n:
            n_te -= 1
        while n_tr + n_val + n_te < n:
            n_tr += 1

        g  = torch.Generator().manual_seed(beach_id)
        ds = TensorDataset(X_beach, y_beach)
        tr_ds, val_ds, te_ds = random_split(ds, [n_tr, n_val, n_te], generator=g)

        bs     = min(16, max(4, n_tr // 4))
        tr_ld  = DataLoader(tr_ds,  batch_size=bs, shuffle=True)
        val_ld = DataLoader(val_ds, batch_size=bs)
        te_ld  = DataLoader(te_ds,  batch_size=bs)

        trainer = Trainer(
            model, device=device,
            learning_rate=finetune_lr,
            model_id=f"beach_{beach_id}",
            scheduler_factor=scheduler_factor,
            scheduler_patience=scheduler_patience,
            grad_clip_norm=grad_clip_norm,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            threshold_steps=threshold_steps,
        )
        trainer.fit(tr_ld, val_ld, epochs=finetune_epochs, patience=DEFAULT_PER_BEACH_PATIENCE)

        thresh, val_f1 = trainer.find_best_threshold(val_ld)
        metrics        = trainer.test(te_ld, threshold=thresh)
        metrics["val_best_f1"] = val_f1
        metrics["n_samples"]   = n

        save_path = os.path.join(output_dir, f"beach_{beach_id}_model.pth")
        torch.save(model.state_dict(), save_path)
        results[str(beach_id)] = metrics

        print(
            f"  Acc={metrics['accuracy']:.4f}  F1={metrics['f1']:.4f}  "
            f"AUC={metrics['auc']:.4f}  Thresh={thresh:.2f}  → {save_path}"
        )

    print("\n" + "=" * 100)
    print(f"Fine-tuned : {len(results)} beaches")
    print(f"Skipped    : {len(skipped)} beaches (global model used for these)")
    if skipped:
        print(f"  Skipped IDs: {skipped}")
    print("=" * 100)

    config = {
        "mode":              "per_beach_finetune",
        "global_checkpoint": global_checkpoint,
        "lookback_days":     lookback_days,
        "finetune_epochs":   finetune_epochs,
        "finetune_lr":       finetune_lr,
        "min_samples":       min_samples,
        "skipped_beach_ids": skipped,
    }
    save_training_report(results, config, report_path)
    return results


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train jellyfish forecasting models")

    # Data
    parser.add_argument("--lookback-days",       type=int,   default=DEFAULT_LOOKBACK_DAYS)
    parser.add_argument("--use-integrated-data", action="store_true")
    parser.add_argument("--weather-csv-path",    type=str,   default=DEFAULT_WEATHER_CSV_PATH)
    parser.add_argument("--disable-live-xml",    action="store_true")

    # Global training
    parser.add_argument("--batch-size",          type=int,   default=BATCH_SIZE)
    parser.add_argument("--learning-rate",       type=float, default=LEARNING_RATE)
    parser.add_argument("--dropout-prob",        type=float, default=DROPOUT_PROB)
    parser.add_argument("--num-epochs",          type=int,   default=NUM_EPOCHS)
    parser.add_argument("--patience",            type=int,   default=DEFAULT_PATIENCE)
    parser.add_argument("--hybrid-hidden-dim",   type=int,   default=DEFAULT_HYBRID_HIDDEN_DIM)
    parser.add_argument("--models",              type=str,   default=DEFAULT_MODEL_NAMES,
                        help=f"Comma-separated model names (default: {DEFAULT_MODEL_NAMES})")
    parser.add_argument("--output-dir",          type=str,   default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-path",         type=str,   default=DEFAULT_REPORT_PATH)
    parser.add_argument("--scheduler-factor",    type=float, default=DEFAULT_LR_SCHEDULER_FACTOR)
    parser.add_argument("--scheduler-patience",  type=int,   default=DEFAULT_LR_SCHEDULER_PATIENCE)
    parser.add_argument("--grad-clip-norm",      type=float, default=DEFAULT_GRAD_CLIP_NORM)
    parser.add_argument("--threshold-min",       type=float, default=DEFAULT_THRESHOLD_MIN)
    parser.add_argument("--threshold-max",       type=float, default=DEFAULT_THRESHOLD_MAX)
    parser.add_argument("--threshold-steps",     type=int,   default=DEFAULT_THRESHOLD_STEPS)

    # Per-beach fine-tuning
    parser.add_argument("--finetune-per-beach",  action="store_true")
    parser.add_argument("--global-checkpoint",   type=str,   default=None,
                        help="Path to global JellyfishNet .pth for fine-tuning "
                             "(defaults to output-dir/jellyfishnet_model.pth)")
    parser.add_argument("--finetune-epochs",     type=int,   default=DEFAULT_FINETUNE_EPOCHS)
    parser.add_argument("--finetune-lr",         type=float, default=DEFAULT_FINETUNE_LR)
    parser.add_argument("--min-samples",         type=int,   default=DEFAULT_MIN_SAMPLES_PER_BEACH)

    args        = parser.parse_args()
    model_names = [m.strip() for m in args.models.split(",") if m.strip()]

    if args.finetune_per_beach:
        checkpoint = args.global_checkpoint or os.path.join(
            args.output_dir, "jellyfishnet_model.pth"
        )
        if not os.path.exists(checkpoint):
            print(f"\n✗ Global checkpoint not found: {checkpoint}")
            print("  Run global training first, or pass --global-checkpoint <path>")
            sys.exit(1)

        finetune_per_beach(
            global_checkpoint=checkpoint,
            lookback_days=args.lookback_days,
            use_integrated_data=args.use_integrated_data,
            weather_csv_path=args.weather_csv_path,
            include_live_xml=not args.disable_live_xml,
            finetune_epochs=args.finetune_epochs,
            finetune_lr=args.finetune_lr,
            dropout_prob=args.dropout_prob,
            hybrid_hidden_dim=args.hybrid_hidden_dim,
            min_samples=args.min_samples,
            output_dir=args.output_dir,
            report_path="training_report_per_beach.json",
            scheduler_factor=args.scheduler_factor,
            scheduler_patience=args.scheduler_patience,
            grad_clip_norm=args.grad_clip_norm,
            threshold_min=args.threshold_min,
            threshold_max=args.threshold_max,
            threshold_steps=args.threshold_steps,
        )
    else:
        train_all_models(
            lookback_days=args.lookback_days,
            use_integrated_data=args.use_integrated_data,
            weather_csv_path=args.weather_csv_path,
            include_live_xml=not args.disable_live_xml,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            dropout_prob=args.dropout_prob,
            num_epochs=args.num_epochs,
            patience=args.patience,
            hybrid_hidden_dim=args.hybrid_hidden_dim,
            model_names=model_names,
            output_dir=args.output_dir,
            report_path=args.report_path,
            scheduler_factor=args.scheduler_factor,
            scheduler_patience=args.scheduler_patience,
            grad_clip_norm=args.grad_clip_norm,
            threshold_min=args.threshold_min,
            threshold_max=args.threshold_max,
            threshold_steps=args.threshold_steps,
        )
