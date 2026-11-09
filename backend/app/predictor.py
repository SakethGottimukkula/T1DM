"""Model loading and inference."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from .config import HORIZONS, MODEL_DIR
from .features import FEATURE_COLUMNS

logger = logging.getLogger(__name__)


class ModelStore:
    """Loads and caches the four horizon-specific XGBoost models."""

    def __init__(self, model_dir: Path | None = None) -> None:
        self._dir = Path(model_dir) if model_dir else MODEL_DIR
        self._models: dict[int, xgb.XGBRegressor] = {}
        self._sha256: dict[int, str] = {}
        self._size_bytes: dict[int, int] = {}
        self._feature_columns: list[str] = FEATURE_COLUMNS

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_all(self) -> None:
        """Attempt to load every horizon model. Missing files are logged
        as warnings rather than raising, so the app can still start."""

        # Optionally load the saved feature column list for sanity check
        fc_path = self._dir / "feature_columns.json"
        if fc_path.exists():
            with open(fc_path, "r") as f:
                saved = json.load(f)
            if saved != self._feature_columns:
                logger.warning(
                    "feature_columns.json differs from code — using code order"
                )

        for hz in HORIZONS:
            model_file = self._dir / f"model_{hz}.json"
            if not model_file.exists():
                logger.warning("Model file not found: %s", model_file)
                continue
            try:
                model = xgb.XGBRegressor()
                model.load_model(str(model_file))
                self._models[hz] = model
                raw_bytes = model_file.read_bytes()
                self._sha256[hz] = hashlib.sha256(raw_bytes).hexdigest().lower()
                self._size_bytes[hz] = len(raw_bytes)
                logger.info("Loaded model for %d-min horizon from %s", hz, model_file)
            except Exception:
                logger.exception("Failed to load %s", model_file)

    # ------------------------------------------------------------------
    # Info helpers
    # ------------------------------------------------------------------

    @property
    def loaded_horizons(self) -> list[int]:
        return sorted(self._models.keys())

    def is_loaded(self, horizon: int) -> bool:
        return horizon in self._models

    def model_sha256(self, horizon: int) -> str | None:
        return self._sha256.get(horizon)

    def model_size(self, horizon: int) -> int | None:
        return self._size_bytes.get(horizon)

    @property
    def feature_columns(self) -> list[str]:
        return list(self._feature_columns)

    @property
    def model_dir(self) -> Path:
        return self._dir

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, features: dict[str, float], horizon: int) -> float:
        """Return a single glucose prediction for the given horizon.

        Parameters
        ----------
        features : dict
            Feature name → value mapping for a **single** row.
        horizon : int
            One of 30, 60, 90, 120.

        Returns
        -------
        float
            Predicted glucose in mg/dL.
        """

        if horizon not in self._models:
            raise RuntimeError(f"Model for {horizon}-min horizon is not loaded")

        row = np.array([[features[col] for col in self._feature_columns]])
        pred = self._models[horizon].predict(row)
        return float(pred[0])


    def predict_batch(self, X: pd.DataFrame, horizon: int) -> np.ndarray:
        """Return predictions for a batch of feature rows for the given horizon.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame containing the 18 feature columns in FEATURE_COLUMNS order.
        horizon : int
            One of 30, 60, 90, 120.

        Returns
        -------
        np.ndarray
            Array of predicted glucose values in mg/dL.
        """
        if horizon not in self._models:
            raise RuntimeError(f"Model for {horizon}-min horizon is not loaded")

        features_df = X[self._feature_columns]
        preds = self._models[horizon].predict(features_df)
        return np.asarray(preds, dtype=float)
