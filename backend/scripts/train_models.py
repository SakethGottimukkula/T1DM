#!/usr/bin/env python
"""Train the four horizon-specific XGBoost models.

Usage
-----
    python -m scripts.train_models --train path/to/ohio_training_combined.csv
    python -m scripts.train_models --train train.csv --test test.csv

Or from the repo root:
    python backend/scripts/train_models.py --train train.csv [--test test.csv]

Environment variable ``TRAIN_CSV`` may be used instead of ``--train``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

# ---------------------------------------------------------------------------
# Ensure the app package is importable when running as a script
# ---------------------------------------------------------------------------
_BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BACKEND_DIR))

from app.config import HORIZONS, HORIZON_SHIFTS  # noqa: E402
from app.ega import classify_array  # noqa: E402
from app.features import FEATURE_COLUMNS, build_features  # noqa: E402


def _load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%d-%m-%Y %H:%M:%S")
    df = df.sort_values(["patient_id", "timestamp"]).reset_index(drop=True)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Glucose Horizon XGBoost models")
    parser.add_argument(
        "--train",
        default=os.getenv("TRAIN_CSV"),
        help="Path to training CSV (ohio_training_combined.csv)",
    )
    parser.add_argument(
        "--test",
        default=None,
        help="Optional path to testing CSV for evaluation",
    )
    parser.add_argument(
        "--out",
        default=str(_BACKEND_DIR / "models"),
        help="Output directory for model files (default: backend/models)",
    )
    args = parser.parse_args()

    if not args.train:
        parser.error("--train is required (or set TRAIN_CSV env var)")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load & prepare training data
    # ------------------------------------------------------------------
    print(f"Loading training data from {args.train} ...")
    df_train = _load_csv(args.train)
    print(f"  {len(df_train)} rows, {df_train['patient_id'].nunique()} patients")

    df_train = build_features(df_train)
    df_train = df_train.dropna()

    X_train = df_train[FEATURE_COLUMNS]
    print(f"  {len(X_train)} valid training rows after feature engineering")

    # ------------------------------------------------------------------
    # Train one model per horizon
    # ------------------------------------------------------------------
    models: dict[int, XGBRegressor] = {}
    for hz in HORIZONS:
        target_col = f"target_{hz}"
        print(f"\nTraining {hz}-min model ...")
        model = XGBRegressor()
        model.fit(X_train, df_train[target_col])
        model_path = out_dir / f"model_{hz}.json"
        model.save_model(str(model_path))
        models[hz] = model
        print(f"  Saved → {model_path}")

    # Save feature column list
    fc_path = out_dir / "feature_columns.json"
    with open(fc_path, "w") as f:
        json.dump(FEATURE_COLUMNS, f, indent=2)
    print(f"\nFeature columns → {fc_path}")

    # ------------------------------------------------------------------
    # Optional evaluation on test set
    # ------------------------------------------------------------------
    if args.test:
        print(f"\n{'='*60}")
        print(f"Evaluating on {args.test} ...")
        df_test = _load_csv(args.test)
        df_test = build_features(df_test)
        df_test = df_test.dropna()
        X_test = df_test[FEATURE_COLUMNS]
        print(f"  {len(X_test)} valid test rows")

        for hz in HORIZONS:
            target_col = f"target_{hz}"
            y_true = df_test[target_col].values
            y_pred = models[hz].predict(X_test)
            rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            mae = float(mean_absolute_error(y_true, y_pred))
            _, counts = classify_array(y_true.tolist(), y_pred.tolist())
            total = sum(counts.values())
            ab_pct = (counts["A"] + counts["B"]) / total * 100 if total else 0
            print(
                f"  {hz:>3d} min  |  RMSE {rmse:6.2f}  |  MAE {mae:6.2f}  "
                f"|  Clarke A+B {ab_pct:.2f}%"
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
