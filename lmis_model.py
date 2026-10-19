"""
lmis_model.py - the ML model. Reads maharashtra_lmis_sample.csv, learns from it, forecasts the next quarters.

Model (one per target: openings, jobseekers), fitted on ALL 120 district x occupation series together:
    log(y) = series_level + quarter_seasonality + (occupation trend + district trend) * time + noise
fitted with Ridge regression. Pooling is the point: each series has only 10 quarters, so it borrows
seasonality and trend from its occupation/district peers instead of guessing from 10 noisy points.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

TARGETS = ["estimated_job_openings", "estimated_available_jobseekers"]
KEYS = ["district", "occupation"]
ALPHA_LEVEL, ALPHA_REST = (
    0.01,
    3.0,
)  # ridge: barely shrink series levels, shrink trends/seasonality more


def _qidx(period: pd.Series) -> np.ndarray:
    """Convert quarter labels such as '2024Q1' to integer time indices."""
    years = period.str[:4].astype(int).to_numpy()
    quarters = period.str[-1].astype(int).to_numpy()
    return (years - 2024) * 4 + (quarters - 1)


def _label(i: int) -> str:
    return f"{2024 + i // 4}Q{i % 4 + 1}"


def _design(df: pd.DataFrame, time_indices: np.ndarray, columns=None):
    series_ids = (df["district"] + "|" + df["occupation"]).to_numpy()
    feature_parts = {
        "level": pd.get_dummies(pd.Series(series_ids), prefix="level").astype(float),
        "seasonality": pd.get_dummies(
            pd.Series((time_indices % 4).astype(str)), prefix="season"
        ).astype(float),
        "occupation_trend": pd.get_dummies(
            df["occupation"].reset_index(drop=True), prefix="occupation"
        )
        .astype(float)
        .mul(time_indices, axis=0),
        "district_trend": pd.get_dummies(
            df["district"].reset_index(drop=True), prefix="district"
        )
        .astype(float)
        .mul(time_indices, axis=0),
    }
    feature_matrix = pd.concat(feature_parts.values(), axis=1)
    if columns is not None:
        feature_matrix = feature_matrix.reindex(columns=columns, fill_value=0.0)
    return feature_matrix


def fit_predict(hist: pd.DataFrame, horizon: int = 4, band_z: float = 1.2816):
    """Forecast each target and include an 80% interval for each forecast."""
    history = hist.reset_index(drop=True)
    required_columns = {"period", *KEYS, *TARGETS}
    missing_columns = sorted(required_columns.difference(history.columns))
    if missing_columns:
        raise ValueError(f"Forecast history is missing columns: {missing_columns}")
    if history.empty:
        raise ValueError("Forecast history must contain at least one row.")
    if horizon < 1:
        raise ValueError("Forecast horizon must be at least one quarter.")

    time_indices = _qidx(history["period"])
    last_time_index = time_indices.max()
    series_keys = history[KEYS].drop_duplicates().reset_index(drop=True)
    future = pd.concat(
        [
            series_keys.assign(t=last_time_index + offset)
            for offset in range(1, horizon + 1)
        ],
        ignore_index=True,
    )
    forecasts = future.copy()
    forecasts["period"] = forecasts["t"].map(_label)

    for target in TARGETS:
        log_values = np.log(history[target].clip(lower=1).to_numpy(float))
        feature_matrix = _design(history, time_indices)
        penalties = np.array(
            [
                ALPHA_LEVEL if column.startswith("level_") else ALPHA_REST
                for column in feature_matrix.columns
            ]
        )
        scaled_features = feature_matrix.to_numpy() / np.sqrt(penalties / ALPHA_REST)
        model = Ridge(alpha=ALPHA_REST, fit_intercept=True).fit(
            scaled_features, log_values
        )
        residuals = log_values - model.predict(scaled_features)
        degrees_of_freedom = max(
            len(log_values) - (np.abs(model.coef_) > 1e-9).sum(), 1
        )
        residual_scale = np.sqrt((residuals**2).sum() / degrees_of_freedom)

        future_features = _design(
            future, future["t"].to_numpy(), columns=feature_matrix.columns
        ).to_numpy() / np.sqrt(penalties / ALPHA_REST)
        log_forecasts = model.predict(future_features)
        forecast_steps = future["t"].to_numpy() - last_time_index
        forecast_scale = residual_scale * np.sqrt(1 + 0.15 * (forecast_steps - 1))

        forecasts[target] = np.exp(log_forecasts + forecast_scale**2 / 2)
        forecasts[target + "_lo"] = np.exp(log_forecasts - band_z * forecast_scale)
        forecasts[target + "_hi"] = np.exp(log_forecasts + band_z * forecast_scale)
    return forecasts.drop(columns="t")


# ---------------------------------------------------------------- honest evaluation
def backtest(df: pd.DataFrame, origins=(5, 6, 7, 8), horizon=2):
    """Rolling-origin: hide the last quarters, forecast them, compare to what really happened; vs 2 naive baselines."""
    history = df.copy()
    history["t"] = _qidx(history["period"])
    rows = []
    for origin in origins:
        training = history[history.t < origin]
        testing = history[(history.t >= origin) & (history.t < origin + horizon)]
        if testing.empty:
            continue
        predictions = fit_predict(training.drop(columns="t"), horizon=horizon)
        predictions["t"] = _qidx(predictions["period"])
        merged = testing.merge(predictions, on=KEYS + ["t"], suffixes=("", "_p"))
        for tgt in TARGETS:
            last_quarter = training[training.t == origin - 1].set_index(KEYS)[tgt]
            last_four_quarters = (
                training[training.t >= origin - 4].groupby(KEYS)[tgt].mean()
            )
            seasonal_baseline = training.set_index(KEYS + ["t"])[tgt]
            for _, row in merged.iterrows():
                series_key = (row["district"], row["occupation"])
                seasonal_value = seasonal_baseline.get(
                    (*series_key, row["t"] - 4), np.nan
                )
                rows.append(
                    dict(
                        target=tgt,
                        origin=origin,
                        h=int(row["t"] - origin + 1),
                        actual=row[tgt],
                        model=row[tgt + "_p"],
                        last=last_quarter[series_key],
                        mean4=last_four_quarters[series_key],
                        seasonal_naive=seasonal_value,
                        lo=row.get(tgt + "_lo", np.nan),
                        hi=row.get(tgt + "_hi", np.nan),
                    )
                )
    return pd.DataFrame(rows)
