#!/usr/bin/env python3
"""
Evaluate model predictions against real jellyfish sightings from meduzot.co.il

Usage:
    python scripts/evaluate_real_sightings.py [--no-train] [--lookback 14]

This script:
1. Scrapes real jellyfish sightings from https://www.meduzot.co.il/overview-list/9/1
   (falls back to representative sample data when the site is unreachable)
2. Maps each sighting (beach name + date) to the model's beach IDs
3. Trains GRU (baseline) and JellyfishNet if no saved models are found
4. Runs GRU and JellyfishNet predictions for each sighting beach-date
5. Compares predictions against actual sightings
6. Generates graphs and saves results to reports/
"""

import argparse
import json
import os
import sys
import warnings
from datetime import date, datetime, timedelta
from difflib import SequenceMatcher

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from jellyfish.data_loader import load_jellyfish_data
from jellyfish.predictor import JellyfishPredictor
from jellyfish.settings import DEFAULT_LOOKBACK_DAYS

# ---------------------------------------------------------------------------
# Beach ID → name mapping (from citizen-science GBIF data)
# beach_id: (name, latitude, longitude)
# ---------------------------------------------------------------------------
MODEL_BEACHES = {
    1:  ("Nahariya-Rosh Hniqra",    33.014, 35.056),
    2:  ("Acco-Nahariyah",          32.908, 35.010),
    3:  ("Kiryat Yam-Acco",         32.855, 35.012),
    4:  ("Shiqmona-Kiryat yam",     32.837, 34.956),
    5:  ("Tira-Shiqmona",           32.820, 34.945),
    6:  ("Atlit-Tira",              32.799, 34.942),
    7:  ("Dor-Atlit",               32.708, 34.922),
    8:  ("Jisr a zarqa-Dor",        32.564, 34.871),
    9:  ("Hadera-Jisr a zarqa",     32.533, 34.868),
    10: ("Michmoret-Hadera",        32.383, 34.827),
    11: ("Natanya-Michmoret",       32.287, 34.819),
    12: ("Gaash-Natanya",           32.235, 34.790),
    13: ("Herzlia-Gaash",           32.233, 34.783),
    14: ("Tel Aviv-Herzlia",        32.053, 34.708),
    15: ("Jaffa-Tel Aviv",          32.007, 34.695),
    16: ("Rishon-Jaffa",            31.876, 34.621),
    17: ("Palmahim-Rishon",         31.845, 34.638),
    18: ("Ashdod-Palmahim",         31.810, 34.591),
    19: ("Ashkelon-Ashdod",         31.699, 34.549),
    20: ("Gaza-Ashkelon",           31.652, 34.525),
}

# ---------------------------------------------------------------------------
# Meduzot beach-name → model beach_id lookup
# Keys are lowercase substrings that appear in meduzot beach names.
# Covers both Hebrew-transliterated and English spellings.
# ---------------------------------------------------------------------------
MEDUZOT_TO_MODEL_ID = {
    # Northern coast
    "rosh hanikra":    1, "rosh hniqra": 1, "nahariya": 1, "nahariyah": 1,
    "acco": 2, "akko": 2, "acre": 2, "nahariya-acco": 2, "acco-nahariyah": 2,
    "kiryat yam": 3, "kiryat-yam": 3,
    "shiqmona": 4, "haifa bay": 4, "kiryat yam-acco": 4,
    "carmel": 5, "dado": 5, "tira": 5, "bat galim": 5,
    "atlit": 6, "atlit-tira": 6,
    "dor": 7, "dor beach": 7, "habonim": 7,
    # Central coast
    "jisr": 8, "jisr a zarqa": 8, "caesarea": 8,
    "hadera": 9, "hadera beach": 9,
    "michmoret": 10, "poleg": 10,
    "netanya": 11, "natanya": 11,
    "gaash": 12, "beit yanai": 12,
    "herzliya": 13, "herzlia": 13, "gaash-natanya": 13,
    "tel aviv": 14, "gordon beach": 14, "frishman": 14, "hilton": 14, "north tel aviv": 14,
    "jaffa": 15, "yafo": 15, "south tel aviv": 15,
    "rishon": 16, "bat yam": 16, "rishon lezion": 16,
    "palmahim": 17, "palmachim": 17,
    # Southern coast
    "ashdod": 18, "ashdod beach": 18,
    "ashkelon": 19, "afridar": 19,
    "gaza": 20,
}


# ---------------------------------------------------------------------------
# Scraper
# ---------------------------------------------------------------------------

def scrape_meduzot(pages: int = 5) -> list[dict]:
    """
    Try to scrape real sightings from meduzot.co.il/overview-list/9/{page}.

    Returns a list of dicts:
        {
            'date': date,
            'beach_name': str,
            'jellyfish_present': 1,   # sightings are always positive
            'species': str,
            'intensity': str,
            'source': 'meduzot'
        }

    Falls back to `_get_fallback_sightings()` when the site is unreachable.
    """
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        print("⚠  requests/BeautifulSoup not available – using fallback data")
        return _get_fallback_sightings()

    base_url = "https://www.meduzot.co.il/overview-list/9"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/123.0 Safari/537.36"
        ),
        "Accept-Language": "he-IL,he;q=0.9,en;q=0.8",
    }

    sightings = []
    for page in range(1, pages + 1):
        url = f"{base_url}/{page}"
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
        except Exception as exc:
            print(f"⚠  Could not reach {url}: {exc}")
            break

        soup = BeautifulSoup(resp.text, "html.parser")

        # ── Strategy 1: JSON embedded in <script type="application/json"> ──
        for tag in soup.find_all("script", type="application/json"):
            try:
                data = json.loads(tag.string or "")
                rows = _parse_json_blob(data)
                if rows:
                    sightings.extend(rows)
                    continue
            except Exception:
                pass

        # ── Strategy 2: <script> blocks containing array literals ──
        for tag in soup.find_all("script"):
            text = tag.string or ""
            if "sightings" in text.lower() or "meduzot" in text.lower():
                try:
                    start = text.find("[{")
                    if start >= 0:
                        end = text.rfind("}]") + 2
                        rows = _parse_json_blob(json.loads(text[start:end]))
                        sightings.extend(rows)
                except Exception:
                    pass

        # ── Strategy 3: HTML table/card rows ──
        rows = _parse_html_table(soup)
        sightings.extend(rows)

        if not sightings and page == 1:
            print("  Parsed HTML but found no structured sighting records")

    if sightings:
        print(f"✓ Scraped {len(sightings)} sightings from meduzot.co.il")
        return sightings

    print("⚠  No sightings scraped from website – using fallback data")
    return _get_fallback_sightings()


def _parse_json_blob(data) -> list[dict]:
    """Recursively search a JSON blob for sighting-like records."""
    results = []
    if isinstance(data, list):
        for item in data:
            results.extend(_parse_json_blob(item))
    elif isinstance(data, dict):
        date_val = (
            data.get("date") or data.get("event_date") or
            data.get("sighting_date") or data.get("Date") or ""
        )
        beach_val = (
            data.get("beach") or data.get("location") or
            data.get("beach_name") or data.get("Beach") or ""
        )
        if date_val and beach_val:
            try:
                parsed_date = _parse_date(str(date_val))
                if parsed_date:
                    results.append({
                        "date": parsed_date,
                        "beach_name": str(beach_val),
                        "jellyfish_present": 1,
                        "species": str(data.get("species", "unknown")),
                        "intensity": str(data.get("intensity", data.get("quantity", ""))),
                        "source": "meduzot",
                    })
            except Exception:
                pass
        for v in data.values():
            if isinstance(v, (dict, list)):
                results.extend(_parse_json_blob(v))
    return results


def _parse_html_table(soup) -> list[dict]:
    """Extract sightings from HTML table rows or card elements."""
    results = []

    # Look for table rows with date-like cells
    for table in soup.find_all("table"):
        headers = [th.get_text(strip=True).lower() for th in table.find_all("th")]
        for tr in table.find_all("tr")[1:]:
            cells = [td.get_text(strip=True) for td in tr.find_all("td")]
            if len(cells) < 2:
                continue
            # Try to find date and beach in cells
            parsed_date = None
            beach_name = None
            for cell in cells:
                if parsed_date is None:
                    parsed_date = _parse_date(cell)
                if beach_name is None and len(cell) > 2 and not _parse_date(cell):
                    beach_name = cell
            if parsed_date and beach_name:
                results.append({
                    "date": parsed_date,
                    "beach_name": beach_name,
                    "jellyfish_present": 1,
                    "species": "unknown",
                    "intensity": "",
                    "source": "meduzot",
                })

    # Look for div/li cards
    for card in soup.find_all(["div", "li", "article"],
                               class_=lambda c: c and any(
                                   k in c.lower()
                                   for k in ["sighting", "report", "event", "item", "card"]
                               )):
        text = card.get_text(" ", strip=True)
        parsed_date = _parse_date(text)
        if parsed_date:
            results.append({
                "date": parsed_date,
                "beach_name": _extract_beach_name_from_text(text),
                "jellyfish_present": 1,
                "species": "unknown",
                "intensity": "",
                "source": "meduzot",
            })

    return results


def _parse_date(text: str):
    """Try multiple date formats; return date or None."""
    text = str(text).strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d.%m.%Y", "%d-%m-%Y",
                "%Y/%m/%d", "%d %b %Y", "%B %d, %Y"):
        try:
            # Slice with a small buffer (+4) to accommodate leading
            # whitespace or day-name prefixes before the date token.
            return datetime.strptime(text[:len(fmt) + 4].split()[0], fmt).date()
        except Exception:
            pass
    return None


def _extract_beach_name_from_text(text: str) -> str:
    """Attempt to pull a beach name from free text (best effort)."""
    for keyword in MEDUZOT_TO_MODEL_ID:
        if keyword in text.lower():
            return keyword.title()
    return text[:40] if text else "Unknown"


# ---------------------------------------------------------------------------
# Fallback data
# ---------------------------------------------------------------------------

def _get_fallback_sightings() -> list[dict]:
    """
    Return a representative set of jellyfish sighting records drawn from the
    citizen-science dataset.  These cover beaches 1-20, summer blooms in
    2022-2024, and include both bloom days (positive) and quiet days (negative)
    so evaluation metrics are meaningful.

    In production, these would be replaced by live scrapes from meduzot.co.il.
    """
    print("  Loading citizen-science data for fallback sightings …")

    try:
        X, y, metadata = load_jellyfish_data(lookback_days=DEFAULT_LOOKBACK_DAYS, forecast_days=1)
    except Exception as exc:
        print(f"  ⚠  Could not load citizen-science data: {exc}")
        return _get_hardcoded_fallback()

    meta_df = metadata.copy()
    meta_df["jellyfish_present"] = y.astype(int)

    # Filter to well-represented beaches and summer months (May–Oct)
    meta_df["month"] = pd.to_datetime(meta_df["forecast_date"]).dt.month
    meta_df = meta_df[meta_df["month"].between(5, 10)].copy()

    # Sample: up to 30 positives + 30 negatives from 2022-2024
    meta_df["year"] = pd.to_datetime(meta_df["forecast_date"]).dt.year
    recent = meta_df[meta_df["year"].between(2022, 2024)]

    positives = recent[recent["jellyfish_present"] == 1].sample(
        min(30, (recent["jellyfish_present"] == 1).sum()), random_state=42
    )
    negatives = recent[recent["jellyfish_present"] == 0].sample(
        min(30, (recent["jellyfish_present"] == 0).sum()), random_state=42
    )
    sampled = pd.concat([positives, negatives]).reset_index(drop=True)

    sightings = []
    for _, row in sampled.iterrows():
        beach_id = int(row["beach_id"])
        beach_info = MODEL_BEACHES.get(beach_id)
        if beach_info is None:
            continue
        sightings.append({
            "date": row["forecast_date"],
            "beach_name": row["beach_name"],
            "jellyfish_present": int(row["jellyfish_present"]),
            "species": "Rhizostoma pulmo",
            "intensity": "Some" if row["jellyfish_present"] else "None",
            "source": "citizen_science_fallback",
            "beach_id": beach_id,          # pre-mapped
        })

    print(
        f"  ✓ Fallback: {len(sightings)} records "
        f"({sum(s['jellyfish_present'] for s in sightings)} positive, "
        f"{sum(1 - s['jellyfish_present'] for s in sightings)} negative)"
    )
    return sightings


def _get_hardcoded_fallback() -> list[dict]:
    """Minimal hardcoded fallback when citizen-science data cannot be loaded."""
    records = [
        # date, beach_id, jellyfish_present
        ("2024-07-15", 14, 1), ("2024-07-16", 14, 1), ("2024-07-20", 14, 0),
        ("2024-08-02", 11, 1), ("2024-08-05", 11, 1), ("2024-08-10", 11, 0),
        ("2024-07-18",  9, 1), ("2024-07-22",  9, 0), ("2024-08-01",  9, 1),
        ("2024-08-12", 18, 1), ("2024-08-15", 18, 0), ("2024-08-20", 18, 1),
        ("2024-06-25",  4, 1), ("2024-06-30",  4, 0), ("2024-07-05",  4, 1),
        ("2023-07-10", 14, 1), ("2023-07-11", 14, 1), ("2023-07-15", 14, 0),
        ("2023-08-08", 11, 1), ("2023-08-09", 11, 0),
    ]
    sightings = []
    for date_str, beach_id, present in records:
        info = MODEL_BEACHES.get(beach_id, ("Unknown", 0, 0))
        sightings.append({
            "date": datetime.strptime(date_str, "%Y-%m-%d").date(),
            "beach_name": info[0],
            "jellyfish_present": present,
            "species": "Rhizostoma pulmo",
            "intensity": "Some" if present else "None",
            "source": "hardcoded_fallback",
            "beach_id": beach_id,
        })
    return sightings


# ---------------------------------------------------------------------------
# Beach-name matching
# ---------------------------------------------------------------------------

def _string_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def map_to_model_beach_id(beach_name: str) -> int | None:
    """
    Map a sighting beach name to the nearest model beach_id.

    Priority:
    1. Direct keyword lookup in MEDUZOT_TO_MODEL_ID
    2. Fuzzy match against MODEL_BEACHES names (threshold 0.55)
    """
    if not beach_name or beach_name.lower() in {"unknown", ""}:
        return None

    lower = beach_name.lower()

    # 1. Keyword lookup (substring match)
    for keyword, bid in MEDUZOT_TO_MODEL_ID.items():
        if keyword in lower:
            return bid

    # 2. Fuzzy match against known beach names
    best_id, best_score = None, 0.0
    for bid, (name, *_) in MODEL_BEACHES.items():
        score = _string_similarity(beach_name, name)
        if score > best_score:
            best_score = score
            best_id = bid

    if best_score >= 0.55:
        return best_id

    return None


# ---------------------------------------------------------------------------
# Model training (lightweight, only if models not present)
# ---------------------------------------------------------------------------

def ensure_models_trained(lookback_days: int, models_dir: str = "models") -> bool:
    """
    Train GRU (baseline) and JellyfishNet if their .pth files are missing.
    Returns True if models are ready.
    """
    gru_path  = os.path.join(models_dir, "gru_model.pth")
    jnet_path = os.path.join(models_dir, "jellyfishnet_model.pth")

    if os.path.exists(gru_path) and os.path.exists(jnet_path):
        print(f"✓ Models found in {models_dir}/")
        return True

    print("No trained models found – training GRU + JellyfishNet …")
    print("  (use --no-train to skip and load existing models only)")

    try:
        from jellyfish.train import train_all_models
        train_all_models(
            lookback_days=lookback_days,
            use_integrated_data=False,       # citizen-science only for speed
            include_live_xml=False,
            batch_size=32,
            learning_rate=0.001,
            dropout_prob=0.3,
            num_epochs=50,
            patience=10,
            hybrid_hidden_dim=64,
            model_names=["GRU", "JellyfishNet"],
            report_path="training_report_latest.json",
            output_dir=models_dir,
        )
        return os.path.exists(gru_path) and os.path.exists(jnet_path)
    except Exception as exc:
        print(f"❌ Training failed: {exc}")
        return False


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def run_predictions(
    predictor: JellyfishPredictor,
    sightings: list[dict],
    model_name: str,
    threshold: float = 0.5,
) -> list[dict]:
    """
    For each sighting record, predict jellyfish presence and record result.
    """
    results = []
    for s in sightings:
        beach_id = s.get("beach_id")
        if beach_id is None:
            results.append(None)
            continue

        try:
            pred = predictor.predict_for_beach_date(
                beach_id=beach_id,
                forecast_date=s["date"],
                model_name=model_name,
            )
        except Exception as exc:
            results.append(None)
            continue

        prob = pred.get("probability")
        if prob is None:
            results.append(None)
            continue

        predicted = int(prob >= threshold)
        actual    = int(s["jellyfish_present"])
        correct   = int(predicted == actual)

        results.append({
            "beach_id":    beach_id,
            "beach_name":  s["beach_name"],
            "date":        str(s["date"]),
            "actual":      actual,
            "probability": round(float(prob), 4),
            "predicted":   predicted,
            "correct":     correct,
            "model":       model_name,
            "source":      s.get("source", ""),
        })

    return results


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(results: list[dict]) -> dict:
    valid = [r for r in results if r is not None]
    if not valid:
        return {}

    y_true = np.array([r["actual"]    for r in valid])
    y_pred = np.array([r["predicted"] for r in valid])
    y_prob = np.array([r["probability"] for r in valid])

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    acc  = (tp + tn) / len(y_true) if y_true.size else 0
    prec = tp / (tp + fp) if tp + fp else 0
    rec  = tp / (tp + fn) if tp + fn else 0
    f1   = 2 * prec * rec / (prec + rec) if prec + rec else 0

    # ROC-AUC (skip if only one class)
    auc = float("nan")
    if len(np.unique(y_true)) > 1:
        from sklearn.metrics import roc_auc_score
        try:
            auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            pass

    return {
        "n_evaluated": len(valid),
        "n_correct":   int((y_pred == y_true).sum()),
        "accuracy":    round(acc,  4),
        "precision":   round(prec, 4),
        "recall":      round(rec,  4),
        "f1":          round(f1,   4),
        "auc":         round(auc,  4) if not np.isnan(auc) else None,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _model_display_name(name: str) -> str:
    return "GRU (Baseline)" if name == "GRU" else "JellyfishNet"


def make_plots(
    gru_results:  list[dict],
    jnet_results: list[dict],
    gru_metrics:  dict,
    jnet_metrics: dict,
    output_dir:   str = "reports",
) -> str:
    """
    Create a 2×2 figure with:
      [0,0] Bar chart – metric comparison
      [0,1] Confusion matrices (side by side)
      [1,0] Prediction timeline (probability vs date)
      [1,1] Per-beach accuracy
    Save to output_dir/real_sightings_evaluation.png
    """
    os.makedirs(output_dir, exist_ok=True)
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        "Jellyfish Bloom Predictor – Evaluation on Real Sightings\n"
        "(data: meduzot.co.il / citizen-science fallback)",
        fontsize=14, fontweight="bold", y=0.98,
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── [0,0] Metric comparison bar chart ───────────────────────────────────
    ax_metrics = fig.add_subplot(gs[0, 0])
    metric_names = ["Accuracy", "Precision", "Recall", "F1"]
    gru_vals  = [gru_metrics.get(k.lower(),  0) for k in metric_names]
    jnet_vals = [jnet_metrics.get(k.lower(), 0) for k in metric_names]
    x = np.arange(len(metric_names))
    width = 0.35
    bars_gru  = ax_metrics.bar(x - width/2, gru_vals,  width, label="GRU (Baseline)", color="#4C72B0", alpha=0.85)
    bars_jnet = ax_metrics.bar(x + width/2, jnet_vals, width, label="JellyfishNet",   color="#DD8452", alpha=0.85)
    ax_metrics.set_ylim(0, 1.05)
    ax_metrics.set_xticks(x)
    ax_metrics.set_xticklabels(metric_names)
    ax_metrics.set_ylabel("Score")
    ax_metrics.set_title("Model Performance Metrics")
    ax_metrics.legend(fontsize=9)
    ax_metrics.grid(axis="y", alpha=0.3)
    for bar in list(bars_gru) + list(bars_jnet):
        h = bar.get_height()
        ax_metrics.text(
            bar.get_x() + bar.get_width() / 2, h + 0.01,
            f"{h:.2f}", ha="center", va="bottom", fontsize=8,
        )

    # ── [0,1] Confusion matrices (two side-by-side imshow panels) ───────────
    inner_gs = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs[0, 1], wspace=0.4
    )

    def _draw_cm_imshow(sub_ax, metrics, name):
        cm = np.array([
            [metrics.get("tn", 0), metrics.get("fp", 0)],
            [metrics.get("fn", 0), metrics.get("tp", 0)],
        ])
        im = sub_ax.imshow(cm, interpolation="nearest", cmap="Blues", aspect="auto")
        sub_ax.set_title(name, fontsize=10, fontweight="bold")
        tick_labels = ["Neg", "Pos"]
        sub_ax.set_xticks([0, 1])
        sub_ax.set_yticks([0, 1])
        sub_ax.set_xticklabels(tick_labels, fontsize=9)
        sub_ax.set_yticklabels(tick_labels, fontsize=9)
        sub_ax.set_xlabel("Predicted", fontsize=8)
        sub_ax.set_ylabel("Actual", fontsize=8)
        cell_labels = [["TN", "FP"], ["FN", "TP"]]
        thresh = cm.max() / 2.0
        for r in range(2):
            for c in range(2):
                color = "white" if cm[r, c] > thresh else "black"
                sub_ax.text(c, r, f"{cell_labels[r][c]}\n{cm[r, c]}",
                            ha="center", va="center", fontsize=11,
                            fontweight="bold", color=color)

    ax_cm_gru  = fig.add_subplot(inner_gs[0, 0])
    ax_cm_jnet = fig.add_subplot(inner_gs[0, 1])
    _draw_cm_imshow(ax_cm_gru,  gru_metrics,  "GRU (Baseline)")
    _draw_cm_imshow(ax_cm_jnet, jnet_metrics, "JellyfishNet")

    # ── [1,0] Prediction timeline ────────────────────────────────────────────
    ax_tl = fig.add_subplot(gs[1, 0])
    valid_gru  = [r for r in gru_results  if r is not None]
    valid_jnet = [r for r in jnet_results if r is not None]

    def _timeline_data(results):
        rows = sorted(results, key=lambda r: r["date"])
        dates = [datetime.strptime(r["date"], "%Y-%m-%d") for r in rows]
        probs = [r["probability"] for r in rows]
        acts  = [r["actual"]      for r in rows]
        return dates, probs, acts

    if valid_gru and valid_jnet:
        dates_g, probs_g, acts_g   = _timeline_data(valid_gru)
        dates_j, probs_j, _         = _timeline_data(valid_jnet)

        ax_tl.plot(dates_g, probs_g, "b-o", markersize=4, alpha=0.7,
                   label="GRU prob", linewidth=1)
        ax_tl.plot(dates_j, probs_j, "r-s", markersize=4, alpha=0.7,
                   label="JellyfishNet prob", linewidth=1)
        # Mark actual positives with vertical spans
        for d, a in zip(dates_g, acts_g):
            if a == 1:
                ax_tl.axvline(d, color="green", alpha=0.15, linewidth=4)
        ax_tl.axhline(0.5, color="grey", linestyle="--", linewidth=1, label="threshold=0.5")
        ax_tl.set_ylim(-0.05, 1.05)
        ax_tl.set_ylabel("Predicted probability")
        ax_tl.set_title("Prediction Timeline\n(green shading = actual sighting)")
        ax_tl.legend(fontsize=8)
        ax_tl.tick_params(axis="x", rotation=30)
        ax_tl.grid(alpha=0.2)

    # ── [1,1] Per-beach accuracy ─────────────────────────────────────────────
    ax_beach = fig.add_subplot(gs[1, 1])

    if valid_gru and valid_jnet:
        beach_names = sorted({r["beach_name"] for r in valid_gru})
        gru_acc_by_beach  = []
        jnet_acc_by_beach = []
        short_names       = []

        for bn in beach_names:
            g = [r for r in valid_gru  if r["beach_name"] == bn]
            j = [r for r in valid_jnet if r["beach_name"] == bn]
            if not g or not j:
                continue
            gru_acc_by_beach.append(np.mean([r["correct"] for r in g]))
            jnet_acc_by_beach.append(np.mean([r["correct"] for r in j]))
            # Shorten name for display
            short = bn.split("-")[0][:12] if "-" in bn else bn[:12]
            short_names.append(short)

        if short_names:
            x_b = np.arange(len(short_names))
            w_b = 0.35
            ax_beach.bar(x_b - w_b/2, gru_acc_by_beach,  w_b,
                         label="GRU (Baseline)", color="#4C72B0", alpha=0.85)
            ax_beach.bar(x_b + w_b/2, jnet_acc_by_beach, w_b,
                         label="JellyfishNet",   color="#DD8452", alpha=0.85)
            ax_beach.set_xticks(x_b)
            ax_beach.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
            ax_beach.set_ylim(0, 1.1)
            ax_beach.set_ylabel("Accuracy")
            ax_beach.set_title("Per-Beach Accuracy")
            ax_beach.legend(fontsize=8)
            ax_beach.grid(axis="y", alpha=0.3)

    out_path = os.path.join(output_dir, "real_sightings_evaluation.png")
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(
    sightings:    list[dict],
    gru_results:  list[dict],
    jnet_results: list[dict],
    gru_metrics:  dict,
    jnet_metrics: dict,
    output_dir:   str = "reports",
):
    os.makedirs(output_dir, exist_ok=True)

    # Merged prediction table
    records = []
    for g, j in zip(gru_results, jnet_results):
        if g is None or j is None:
            continue
        records.append({
            "date":           g["date"],
            "beach_id":       g["beach_id"],
            "beach_name":     g["beach_name"],
            "actual":         g["actual"],
            "source":         g["source"],
            "gru_probability":  g["probability"],
            "gru_predicted":    g["predicted"],
            "gru_correct":      g["correct"],
            "jnet_probability": j["probability"],
            "jnet_predicted":   j["predicted"],
            "jnet_correct":     j["correct"],
        })

    df = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, "real_sightings_predictions.csv")
    df.to_csv(csv_path, index=False)
    print(f"✓ Predictions saved: {csv_path}")

    # JSON summary
    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "n_sightings_scraped":  len(sightings),
        "n_evaluated":          len(records),
        "source_summary": {
            s: sum(1 for x in sightings if x.get("source") == s)
            for s in {x.get("source", "") for x in sightings}
        },
        "GRU_baseline": gru_metrics,
        "JellyfishNet": jnet_metrics,
    }
    json_path = os.path.join(output_dir, "real_sightings_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"✓ Summary saved:     {json_path}")

    return csv_path, json_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate jellyfish models against real sightings from meduzot.co.il"
    )
    parser.add_argument("--no-train",  action="store_true",
                        help="Skip training; require existing model files")
    parser.add_argument("--lookback",  type=int, default=DEFAULT_LOOKBACK_DAYS,
                        help=f"Lookback window in days (default: {DEFAULT_LOOKBACK_DAYS})")
    parser.add_argument("--pages",     type=int, default=5,
                        help="Number of pages to scrape from meduzot.co.il (default: 5)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Decision threshold for binary prediction (default: 0.5)")
    parser.add_argument("--models-dir", default="models",
                        help="Directory for model .pth files (default: models/)")
    parser.add_argument("--output-dir", default="reports",
                        help="Output directory for results (default: reports/)")
    args = parser.parse_args()

    print("=" * 80)
    print("JELLYFISH BLOOM PREDICTOR – REAL SIGHTINGS EVALUATION")
    print("=" * 80)
    print()

    # ── Step 1: Scrape ───────────────────────────────────────────────────────
    print("STEP 1: Fetching real jellyfish sightings from meduzot.co.il")
    print("-" * 80)
    sightings = scrape_meduzot(pages=args.pages)

    # Map beach names → model beach IDs for records without a pre-mapped ID
    mapped = 0
    unmapped_names = set()
    for s in sightings:
        if "beach_id" not in s:
            bid = map_to_model_beach_id(s.get("beach_name", ""))
            if bid is not None:
                s["beach_id"] = bid
                mapped += 1
            else:
                unmapped_names.add(s.get("beach_name", ""))

    # Keep only records with a valid beach_id
    sightings = [s for s in sightings if "beach_id" in s]

    positives = sum(s["jellyfish_present"] for s in sightings)
    negatives = len(sightings) - positives
    print(f"\nTotal sightings after mapping: {len(sightings)}")
    print(f"  Positive (jellyfish present): {positives}")
    print(f"  Negative (no jellyfish):      {negatives}")
    if unmapped_names:
        print(f"  Unmapped beach names: {', '.join(sorted(unmapped_names)[:10])}")
    print()

    if not sightings:
        print("❌ No sightings could be mapped to model beaches. Exiting.")
        sys.exit(1)

    # ── Step 2: Train (if needed) ────────────────────────────────────────────
    print("STEP 2: Ensuring models are trained")
    print("-" * 80)
    if not args.no_train:
        ready = ensure_models_trained(
            lookback_days=args.lookback,
            models_dir=args.models_dir,
        )
        if not ready:
            print("❌ Models not available. Run 'python scripts/train.py' manually.")
            sys.exit(1)
    else:
        print("  --no-train flag set; using pre-existing model files.")
    print()

    # ── Step 3: Load predictor ───────────────────────────────────────────────
    print("STEP 3: Loading predictor and data cache")
    print("-" * 80)
    predictor = JellyfishPredictor(device="cpu")
    predictor.load_data_cache(
        lookback_days=args.lookback,
        forecast_days=1,
        use_integrated_data=False,
        include_live_xml=False,
    )

    gru_path  = os.path.join(args.models_dir, "gru_model.pth")
    jnet_path = os.path.join(args.models_dir, "jellyfishnet_model.pth")

    for model_name, model_path in [("GRU", gru_path), ("JellyfishNet", jnet_path)]:
        try:
            predictor.load_model(model_name, model_path)
        except Exception as exc:
            print(f"❌ Could not load {model_name} from {model_path}: {exc}")
            sys.exit(1)
    print()

    # ── Step 4: Run predictions ───────────────────────────────────────────────
    print("STEP 4: Running predictions")
    print("-" * 80)

    # Determine per-model threshold from training report (if available)
    thresholds = {"GRU": args.threshold, "JellyfishNet": args.threshold}
    try:
        with open("training_report_latest.json") as f:
            report = json.load(f)
        for mname in ("GRU", "JellyfishNet"):
            t = report.get("results", {}).get(mname, {}).get("threshold")
            if t is not None:
                thresholds[mname] = float(t)
                print(f"  Using trained threshold for {mname}: {t:.3f}")
    except Exception:
        pass

    print(f"\n  Predicting with GRU (baseline) …")
    gru_results  = run_predictions(predictor, sightings, "GRU",
                                   threshold=thresholds["GRU"])

    print(f"  Predicting with JellyfishNet …")
    jnet_results = run_predictions(predictor, sightings, "JellyfishNet",
                                   threshold=thresholds["JellyfishNet"])

    gru_valid  = [r for r in gru_results  if r is not None]
    jnet_valid = [r for r in jnet_results if r is not None]
    print(f"\n  GRU evaluated:        {len(gru_valid)} / {len(sightings)} records")
    print(f"  JellyfishNet evaluated:{len(jnet_valid)} / {len(sightings)} records")
    print()

    # ── Step 5: Metrics ──────────────────────────────────────────────────────
    print("STEP 5: Computing metrics")
    print("-" * 80)

    gru_metrics  = compute_metrics(gru_results)
    jnet_metrics = compute_metrics(jnet_results)

    def _print_metrics(name, m):
        print(f"\n  {name}:")
        print(f"    Evaluated:  {m.get('n_evaluated', 0)}")
        print(f"    Correct:    {m.get('n_correct', 0)}")
        print(f"    Accuracy:   {m.get('accuracy',  0):.4f}")
        print(f"    Precision:  {m.get('precision', 0):.4f}")
        print(f"    Recall:     {m.get('recall',    0):.4f}")
        print(f"    F1:         {m.get('f1',        0):.4f}")
        if m.get("auc") is not None:
            print(f"    AUC:        {m.get('auc'):.4f}")
        print(f"    TP={m.get('tp',0)} FP={m.get('fp',0)} "
              f"FN={m.get('fn',0)} TN={m.get('tn',0)}")

    _print_metrics("GRU (Baseline)", gru_metrics)
    _print_metrics("JellyfishNet",   jnet_metrics)

    # Summary comparison
    print()
    print("  ── Comparison ──")
    for metric in ("accuracy", "precision", "recall", "f1"):
        g = gru_metrics.get(metric, 0)
        j = jnet_metrics.get(metric, 0)
        winner = "JellyfishNet ✓" if j > g else ("GRU ✓" if g > j else "Tie")
        print(f"    {metric:10s}: GRU={g:.4f}  JellyfishNet={j:.4f}  → {winner}")
    print()

    # ── Step 6: Save results ──────────────────────────────────────────────────
    print("STEP 6: Saving results")
    print("-" * 80)
    csv_path, json_path = save_results(
        sightings, gru_results, jnet_results,
        gru_metrics, jnet_metrics,
        output_dir=args.output_dir,
    )

    # ── Step 7: Graphs ────────────────────────────────────────────────────────
    print()
    print("STEP 7: Generating graphs")
    print("-" * 80)
    plot_path = make_plots(
        gru_results, jnet_results,
        gru_metrics, jnet_metrics,
        output_dir=args.output_dir,
    )
    print(f"✓ Plot saved:        {plot_path}")

    print()
    print("=" * 80)
    print("DONE")
    print(f"  Results CSV:  {csv_path}")
    print(f"  Summary JSON: {json_path}")
    print(f"  Graph:        {plot_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
