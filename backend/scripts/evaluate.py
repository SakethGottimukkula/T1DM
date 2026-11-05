#!/usr/bin/env python
"""Evaluation script for XGBoost models and baselines.

Produces backend/models/metrics.json following the canonical evaluation rule:
rows where all 18 features and all 4 targets are non-null.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import Lasso, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost
from xgboost import XGBRegressor

_BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BACKEND_DIR))

from app.config import HORIZONS
from app.features import FEATURE_COLUMNS, build_features
from app.metrics import summarize
from scripts.train_models import _load_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Glucose Horizon models")
    parser.add_argument(
        "--train",
        default=os.getenv("TRAIN_CSV"),
        help="Path to training CSV (ohio_training_combined.csv)",
    )
    parser.add_argument(
        "--test",
        required=True,
        help="Path to testing CSV (ohio_testing_combined.csv)",
    )
    parser.add_argument(
        "--models-dir",
        default=str(_BACKEND_DIR / "models"),
        help="Directory containing trained model JSON files (default: backend/models)",
    )
    parser.add_argument(
        "--out",
        default=str(_BACKEND_DIR / "models" / "metrics.json"),
        help="Output path for metrics.json (default: backend/models/metrics.json)",
    )
    args = parser.parse_args()
    if not args.train:
        parser.error("--train is required (or set TRAIN_CSV env var)")
    return args


def main() -> None:
    args = parse_args()

    models_dir = Path(args.models_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load test data and apply canonical evaluation rule
    print(f"Loading test data from {args.test} ...")
    df_test_raw = _load_csv(args.test)
    n_test_rows_raw = len(df_test_raw)
    patients = sorted([int(p) for p in df_test_raw["patient_id"].unique()])

    df_test_features = build_features(df_test_raw)
    df_eval = df_test_features.dropna().reset_index(drop=True)
    n_eval_rows = len(df_eval)
    X_eval = df_eval[FEATURE_COLUMNS]

    print(f"  Test rows raw: {n_test_rows_raw}, eval rows after dropna: {n_eval_rows}")

    # 2. Load train data for baselines and row counts
    print(f"Loading training data from {args.train} ...")
    df_train_raw = _load_csv(args.train)
    df_train_features = build_features(df_train_raw)
    df_train_valid = df_train_features.dropna().reset_index(drop=True)
    n_train_rows = len(df_train_valid)
    X_train = df_train_valid[FEATURE_COLUMNS]

    print(f"  Train rows valid after dropna: {n_train_rows}")

    # 3. Evaluate XGBoost models from disk
    models_info: dict[str, dict[str, str]] = {}
    horizons_info: dict[str, dict] = {}
    feature_importance_horizons: dict[str, dict[str, float]] = {}

    for hz in HORIZONS:
        hz_str = str(hz)
        model_filename = f"model_{hz}.json"
        model_path = models_dir / model_filename

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        with open(model_path, "rb") as f:
            model_bytes = f.read()
        sha256_hash = hashlib.sha256(model_bytes).hexdigest()
        models_info[hz_str] = {"file": model_filename, "sha256": sha256_hash}

        model = XGBRegressor()
        model.load_model(str(model_path))

        target_col = f"target_{hz}"
        y_true = df_eval[target_col].values
        y_pred = model.predict(X_eval)

        overall_metrics = summarize(y_true, y_pred)

        per_patient_metrics: dict[str, dict] = {}
        for pid in patients:
            p_mask = df_eval["patient_id"] == pid
            p_true = df_eval.loc[p_mask, target_col].values
            p_pred = y_pred[p_mask]
            per_patient_metrics[str(pid)] = summarize(p_true, p_pred)

        horizons_info[hz_str] = {
            "overall": overall_metrics,
            "per_patient": per_patient_metrics,
        }

        # Feature importances (normalized gain from scikit-learn API)
        fi_dict = {
            col: round(float(imp), 6)
            for col, imp in zip(FEATURE_COLUMNS, model.feature_importances_)
        }
        feature_importance_horizons[hz_str] = fi_dict

    feature_importance_data = {
        "importance_type": "gain",
        **feature_importance_horizons,
    }

    # 4. Compute Baselines (persistence, ridge, lasso, arima=null)
    print("Fitting baseline models (Persistence, Ridge, Lasso) ...")
    persistence_metrics: dict[str, dict] = {}
    ridge_metrics: dict[str, dict] = {}
    lasso_metrics: dict[str, dict] = {}

    for hz in HORIZONS:
        hz_str = str(hz)
        target_col = f"target_{hz}"
        y_eval_true = df_eval[target_col].values

        # Persistence: prediction is current glucose reading
        y_pred_persist = df_eval["glucose"].values
        persistence_metrics[hz_str] = summarize(y_eval_true, y_pred_persist)

        # Ridge baseline
        y_train_target = df_train_valid[target_col].values
        ridge_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge()),
        ])
        ridge_pipe.fit(X_train, y_train_target)
        y_pred_ridge = ridge_pipe.predict(X_eval)
        ridge_metrics[hz_str] = summarize(y_eval_true, y_pred_ridge)

        # Lasso baseline
        lasso_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", Lasso(max_iter=10000)),
        ])
        lasso_pipe.fit(X_train, y_train_target)
        y_pred_lasso = lasso_pipe.predict(X_eval)
        lasso_metrics[hz_str] = summarize(y_eval_true, y_pred_lasso)

    baselines_data = {
        "note": "Untuned defaults; trained on the same training rows and scored on the same evaluation rows as the XGBoost models.",
        "persistence": persistence_metrics,
        "ridge": ridge_metrics,
        "lasso": lasso_metrics,
        "arima": None,
    }

    # 5. Assemble JSON payload
    output_data = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "xgboost_version": xgboost.__version__,
        "sklearn_version": sklearn.__version__,
        "evaluation_rule": "rows where all 18 features and all 4 targets are non-null (identical to Model/XGBoost.ipynb)",
        "feature_columns": FEATURE_COLUMNS,
        "patients": patients,
        "n_train_rows": n_train_rows,
        "n_test_rows_raw": n_test_rows_raw,
        "n_eval_rows": n_eval_rows,
        "models": models_info,
        "horizons": horizons_info,
        "feature_importance": feature_importance_data,
        "baselines": baselines_data,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print(f"Metrics saved to {out_path}")

    # 6. Print summary table to terminal (ASCII only)
    print("\n" + "=" * 68)
    print("EVALUATION SUMMARY TABLE (XGBoost vs Baselines)")
    print("=" * 68)
    print(f"{'Horizon':<10} | {'Model':<12} | {'RMSE':<8} | {'MAE':<8} | {'Clarke A+B (%)':<14}")
    print("-" * 68)

    for hz in HORIZONS:
        hz_str = str(hz)
        xgb_m = horizons_info[hz_str]["overall"]
        per_m = persistence_metrics[hz_str]
        rdg_m = ridge_metrics[hz_str]
        las_m = lasso_metrics[hz_str]

        print(f"{hz:>3} min    | {'XGBoost':<12} | {xgb_m['rmse']:>8.2f} | {xgb_m['mae']:>8.2f} | {xgb_m['a_plus_b_pct']:>13.2f}%")
        print(f"{'':<10} | {'Persistence':<12} | {per_m['rmse']:>8.2f} | {per_m['mae']:>8.2f} | {per_m['a_plus_b_pct']:>13.2f}%")
        print(f"{'':<10} | {'Ridge':<12} | {rdg_m['rmse']:>8.2f} | {rdg_m['mae']:>8.2f} | {rdg_m['a_plus_b_pct']:>13.2f}%")
        print(f"{'':<10} | {'Lasso':<12} | {las_m['rmse']:>8.2f} | {las_m['mae']:>8.2f} | {las_m['a_plus_b_pct']:>13.2f}%")
        print("-" * 68)


if __name__ == "__main__":
    main()
