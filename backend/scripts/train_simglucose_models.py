#!/usr/bin/env python
"""Generate virtual-patient traces with SimGlucose and train multimodal models."""

from __future__ import annotations

import sys
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))
from app.config import HORIZONS
from app.simglucose_features import SIM_FEATURE_COLUMNS, build_sim_features


def _simulate_patient(name: str, seed: int, days: int, start: datetime, sensor_params: dict) -> pd.DataFrame:
    from simglucose.actuator.pump import InsulinPump
    from simglucose.controller.basal_bolus_ctrller import BBController
    from simglucose.patient.t1dpatient import T1DPatient
    from simglucose.sensor.cgm import CGMSensor
    from simglucose.simulation.env import T1DSimEnv
    from simglucose.simulation.scenario_gen import RandomScenario
    from simglucose.simulation.sim_engine import SimObj

    group, index = name.split("#")
    i = int(index)
    patient = T1DPatient.withName(name)
    sensor = CGMSensor(pd.Series(sensor_params), seed=seed)
    pump = InsulinPump.withName("Insulet")
    scenario = RandomScenario(start_time=start, seed=seed)
    env = T1DSimEnv(patient, sensor, pump, scenario)
    simulation = SimObj(env, BBController(), timedelta(days=days), animate=False)
    simulation.simulate()
    trace = simulation.results().reset_index().rename(columns={
        "Time": "timestamp", "BG": "bg", "CGM": "cgm", "CHO": "cho"
    })
    trace["patient_id"] = name
    return trace[["timestamp", "patient_id", "bg", "cgm", "cho", "insulin"]]


def simulate_traces(days: int = 35) -> pd.DataFrame:
    from simglucose.sensor.cgm import SENSOR_PARA_FILE

    start = datetime(2020, 1, 1)
    sensor_params = pd.read_csv(SENSOR_PARA_FILE)
    dexcom_params = sensor_params.loc[sensor_params.Name == "Dexcom"].iloc[0].copy()
    dexcom_params["sample_time"] = 5.0
    params = dexcom_params.to_dict()
    tasks = []
    for i in range(1, 11):
        for group in ("adult", "adolescent", "child"):
            name = f"{group}#{i:03d}"
            tasks.append((name, i, days, start, params))
    traces = []
    with ProcessPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(_simulate_patient, *task) for task in tasks]
        for future in as_completed(futures):
            traces.append(future.result())
            print(f"Simulated {len(traces)} / {len(tasks)} virtual patients.", flush=True)
    return pd.concat(traces, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=35, help="Simulation duration per virtual patient (default: 35, about 10k five-minute rows)")
    parser.add_argument("--reuse-existing", action="store_true",
                        help="Train from backend/data/simglucose_training.csv without running the simulator again")
    args = parser.parse_args()
    dataset_path = BACKEND / "data" / "simglucose_training.csv"
    if args.reuse_existing:
        if not dataset_path.is_file():
            parser.error(f"No existing SimGlucose dataset found at {dataset_path}")
        data = pd.read_csv(dataset_path, parse_dates=["timestamp"])
        print(f"Reusing raw SimGlucose dataset at {dataset_path}", flush=True)
    else:
        data = simulate_traces(days=args.days)
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(dataset_path, index=False)
        print(f"Saved raw SimGlucose dataset to {dataset_path}", flush=True)
    rows = []
    for _, group in data.groupby("patient_id"):
        group = group.set_index("timestamp").sort_index()
        # Keep the same five-minute grid used by the Ohio feature pipeline.
        group = group.resample("5min").agg({"bg": "mean", "cgm": "mean", "insulin": "sum", "cho": "sum"}).dropna()
        group = group.reset_index()
        group["patient_id"] = _
        rows.append(build_sim_features(group))
    engineered = pd.concat(rows, ignore_index=True)
    engineered = engineered.dropna(subset=SIM_FEATURE_COLUMNS + [f"target_{hz}" for hz in HORIZONS])
    rng = np.random.default_rng(42)
    patients = engineered["patient_id"].unique().to_numpy(dtype=str, copy=True)
    rng.shuffle(patients)
    test_patients = set(patients[-6:])
    train = engineered[~engineered["patient_id"].isin(test_patients)]
    test = engineered[engineered["patient_id"].isin(test_patients)]
    print(f"Training rows: {len(train):,} across {train['patient_id'].nunique()} virtual patients; "
          f"held-out rows: {len(test):,} across {test['patient_id'].nunique()} patients.", flush=True)
    out = BACKEND / "models" / "simglucose"
    out.mkdir(parents=True, exist_ok=True)
    eval_dir = out / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "patient_ids.json").write_text(__import__("json").dumps(sorted(test_patients), indent=2))
    eval_metrics = {}
    for hz in HORIZONS:
        eval_model = XGBRegressor(n_estimators=350, max_depth=7, learning_rate=0.05,
                                  subsample=0.85, colsample_bytree=0.85, objective="reg:squarederror",
                                  random_state=42, n_jobs=-1)
        eval_model.fit(train[SIM_FEATURE_COLUMNS], train[f"target_{hz}"])
        eval_model.save_model(str(eval_dir / f"model_{hz}.json"))
        pred = eval_model.predict(test[SIM_FEATURE_COLUMNS])
        rmse = float(np.sqrt(mean_squared_error(test[f"target_{hz}"], pred)))
        print(f"{hz:>3} min RMSE on held-out virtual patients: {rmse:.2f} mg/dL")
        eval_metrics[hz] = rmse

    # Keep the held-out evaluation above, then fit the deployable weights on
    # every simulated patient so the upload model can learn from all available
    # simulation traces. These final weights do not have a fresh independent test score.
    all_data = engineered
    print(f"Refitting deployment models on all {all_data['patient_id'].nunique()} virtual patients.", flush=True)
    for hz in HORIZONS:
        model = XGBRegressor(n_estimators=350, max_depth=7, learning_rate=0.05,
                             subsample=0.85, colsample_bytree=0.85, objective="reg:squarederror",
                             random_state=42, n_jobs=-1)
        model.fit(all_data[SIM_FEATURE_COLUMNS], all_data[f"target_{hz}"])
        model.save_model(str(out / f"model_{hz}.json"))
        print(f"Saved {hz}-min deployment model trained on all patients.", flush=True)
    (out / "feature_columns.json").write_text(__import__("json").dumps(SIM_FEATURE_COLUMNS, indent=2))
    print(f"Saved four horizon models to {out}")


if __name__ == "__main__":
    main()
