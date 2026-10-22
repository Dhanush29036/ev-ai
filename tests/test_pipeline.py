"""
Unit and integration tests for EV XAI Planner.
pytest tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure src is on path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


# ─── Schema tests ─────────────────────────────────────────────────────────────

def test_schema_rename():
    """Schema mapper correctly renames columns."""
    from evplan.data.schema import build_rename_map, map_schema
    from evplan.utils.config import load_schema

    schema = load_schema()
    df = pd.DataFrame({
        "Energy (kWh)": [10, 20, 30],
        "City": ["Delhi", "Mumbai", "Bangalore"],
        "Latitude": [28.6, 19.0, 12.9],
    })
    rename_map = build_rename_map(df, schema["standard_columns"])
    assert "energy_consumed_kwh" in rename_map.values() or any(
        v == "energy_consumed_kwh" for v in rename_map.values()
    ) or "Energy (kWh)" in rename_map, f"Expected energy column rename, got: {rename_map}"


def test_schema_map_passthrough():
    """Already-named columns pass through unchanged."""
    from evplan.data.schema import map_schema

    df = pd.DataFrame({"energy_consumed_kwh": [1, 2], "city": ["A", "B"]})
    result = map_schema(df, "demand")
    assert "energy_consumed_kwh" in result.columns


# ─── Cleaning tests ───────────────────────────────────────────────────────────

def test_remove_duplicates():
    """Duplicate removal works."""
    from evplan.data.clean import remove_duplicates

    df = pd.DataFrame({"a": [1, 1, 2], "b": [3, 3, 4]})
    result = remove_duplicates(df)
    assert len(result) == 2


def test_handle_missing_drops_target_nans():
    """Rows with missing target are dropped."""
    from evplan.data.clean import handle_missing

    df = pd.DataFrame({
        "energy_consumed_kwh": [1.0, None, 3.0],
        "hour": [1, 2, 3],
    })
    result = handle_missing(df, target_col="energy_consumed_kwh")
    assert len(result) == 2
    assert result["energy_consumed_kwh"].notna().all()


def test_outlier_flagging():
    """Outlier flagging adds _outlier_flag column."""
    from evplan.data.clean import treat_outliers

    data = [1.0] * 50 + [1000.0]  # extreme outlier
    df = pd.DataFrame({"energy_consumed_kwh": data})
    result = treat_outliers(df, target_col="energy_consumed_kwh")
    assert "energy_consumed_kwh_outlier_flag" in result.columns


# ─── Feature engineering tests ────────────────────────────────────────────────

def test_time_features_added():
    """Time features are correctly added."""
    from evplan.features.engineer import add_time_features

    df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2023-01-15 14:00", "2023-07-20 08:30"]),
        "energy_consumed_kwh": [10.0, 20.0],
    })
    result = add_time_features(df)
    for col in ["hour", "day_of_week", "month", "is_weekend", "hour_sin", "hour_cos"]:
        assert col in result.columns, f"Missing feature: {col}"


def test_cyclic_encoding_range():
    """Cyclic sin/cos encodings stay in [-1, 1]."""
    from evplan.features.engineer import add_time_features

    df = pd.DataFrame({
        "timestamp": pd.date_range("2023-01-01", periods=24, freq="h"),
        "energy_consumed_kwh": np.random.rand(24),
    })
    result = add_time_features(df)
    assert result["hour_sin"].between(-1.01, 1.01).all()
    assert result["hour_cos"].between(-1.01, 1.01).all()


def test_lag_no_leakage():
    """Lag features do not use future values (no leakage)."""
    from evplan.features.engineer import add_lag_features

    n = 50
    df = pd.DataFrame({
        "timestamp": pd.date_range("2023-01-01", periods=n, freq="h"),
        "energy_consumed_kwh": np.arange(float(n)),
    })
    result = add_lag_features(df, target_col="energy_consumed_kwh")
    lag_col = "energy_consumed_kwh_lag_1"
    if lag_col in result.columns:
        # lag_1 at row i should equal value at row i-1
        for i in range(2, 10):
            expected = df["energy_consumed_kwh"].iloc[i - 1]
            actual = result[lag_col].iloc[i]
            assert abs(actual - expected) < 1e-6, f"Leakage at row {i}: expected {expected}, got {actual}"


# ─── Model split tests ────────────────────────────────────────────────────────

def test_chronological_split_no_shuffle():
    """Chronological split preserves order and does not shuffle."""
    from evplan.models.split import chronological_split

    n = 100
    df = pd.DataFrame({
        "timestamp": pd.date_range("2023-01-01", periods=n, freq="h"),
        "energy_consumed_kwh": np.arange(float(n)),
    })
    train, val, test = chronological_split(df, target_col="energy_consumed_kwh",
                                            test_size=0.2, val_size=0.1)
    # Must be non-overlapping and in order
    assert train.index.max() < val.index.min()
    assert val.index.max() < test.index.min()
    assert len(train) + len(val) + len(test) == n


def test_leakage_check_detects_perfect_copy():
    """Leakage check flags a feature that is a copy of the target."""
    from evplan.models.split import check_leakage

    n = 100
    y = pd.Series(np.random.rand(n))
    X = pd.DataFrame({"normal": np.random.rand(n), "copy_of_target": y.values})
    suspicious = check_leakage(X, y, threshold=0.98)
    assert "copy_of_target" in suspicious


# ─── Metrics tests ────────────────────────────────────────────────────────────

def test_metrics_perfect_prediction():
    """Perfect prediction gives RMSE=0, R²=1."""
    from evplan.models.evaluate import compute_metrics

    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    m = compute_metrics(y, y)
    assert abs(m["rmse"]) < 1e-10
    assert abs(m["r2"] - 1.0) < 1e-10
    assert abs(m["mae"]) < 1e-10


def test_metrics_all_keys_present():
    """All expected metrics are returned."""
    from evplan.models.evaluate import compute_metrics

    y = np.array([1.0, 2.0, 3.0])
    yhat = np.array([1.1, 2.1, 2.9])
    m = compute_metrics(y, yhat)
    for key in ["rmse", "mae", "r2", "mape", "smape"]:
        assert key in m


# ─── Scoring tests ────────────────────────────────────────────────────────────

def test_scoring_monotonicity():
    """Higher demand zones get higher demand component."""
    from evplan.recommend.scoring import compute_priority_scores

    df = pd.DataFrame({
        "zone_id": ["A", "B", "C"],
        "predicted_demand_kwh": [100.0, 500.0, 1000.0],
        "station_count": [5, 5, 5],
        "gi_z_score": [0.0, 0.0, 0.0],
        "ev_adoption_score": [0.5, 0.5, 0.5],
        "demand_supply_ratio": [20.0, 100.0, 200.0],
        "zone_lat": [28.6, 19.0, 12.9],
        "zone_lon": [77.2, 72.9, 77.6],
    })
    scored = compute_priority_scores(df)
    # Higher demand → higher comp_demand
    assert scored.loc[scored["zone_id"] == "C", "comp_demand"].values[0] > \
           scored.loc[scored["zone_id"] == "A", "comp_demand"].values[0]


def test_ranking_determinism():
    """Ranking with same inputs gives same output."""
    from evplan.recommend.scoring import rank_zones

    df = pd.DataFrame({
        "zone_id": ["A", "B", "C"],
        "predicted_demand_kwh": [100.0, 500.0, 1000.0],
        "station_count": [5, 5, 5],
        "gi_z_score": [0.5, 1.5, 2.5],
        "ev_adoption_score": [0.3, 0.5, 0.8],
        "demand_supply_ratio": [20.0, 100.0, 200.0],
        "zone_lat": [28.6, 19.0, 12.9],
        "zone_lon": [77.2, 72.9, 77.6],
    })
    r1 = rank_zones(df)
    r2 = rank_zones(df)
    assert list(r1["zone_id"]) == list(r2["zone_id"])


# ─── Integration test ─────────────────────────────────────────────────────────

def test_full_mini_pipeline():
    """Integration: runs the full pipeline on a small synthetic dataset."""
    from evplan.data.clean import clean_demand
    from evplan.features.engineer import engineer_features
    from evplan.models.split import chronological_split, get_X_y
    from evplan.models.evaluate import compute_metrics
    from sklearn.ensemble import RandomForestRegressor

    # Build synthetic demand dataset
    np.random.seed(42)
    n = 500
    ts = pd.date_range("2023-01-01", periods=n, freq="h")
    df = pd.DataFrame({
        "timestamp": ts,
        "energy_consumed_kwh": np.abs(np.sin(np.arange(n) * np.pi / 12) * 50 + np.random.randn(n) * 5 + 30),
        "city": np.random.choice(["Delhi", "Mumbai", "Bangalore"], n),
    })

    # Clean
    df_clean, _ = clean_demand(df)
    assert len(df_clean) > 0

    # Features
    df_feat = engineer_features(df_clean)
    assert len(df_feat.columns) > 5

    # Split
    target = "energy_consumed_kwh"
    train, val, test = chronological_split(df_feat, target_col=target)
    feat_cols = [c for c in df_feat.columns if c != target and pd.api.types.is_numeric_dtype(df_feat[c])]

    X_train, y_train = get_X_y(train, target, feat_cols)
    X_test, y_test = get_X_y(test, target, feat_cols)

    # Train
    model = RandomForestRegressor(n_estimators=20, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate
    metrics = compute_metrics(y_test, y_pred)
    assert metrics["rmse"] >= 0
    assert -1 <= metrics["r2"] <= 1.01  # some slack for tiny datasets

    print(f"\n✅ Mini-pipeline test passed: RMSE={metrics['rmse']:.3f}, R²={metrics['r2']:.4f}")
