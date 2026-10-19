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
ALPHA_LEVEL, ALPHA_REST = 0.01, 3.0     # ridge: barely shrink series levels, shrink trends/seasonality more


def _qidx(period: pd.Series) -> np.ndarray:                       # '2024Q1' -> 0, '2024Q2' -> 1 ...
    y = period.str[:4].astype(int).to_numpy(); q = period.str[-1].astype(int).to_numpy()
    return (y - 2024) * 4 + (q - 1)


def _label(i: int) -> str:
    return f"{2024 + i // 4}Q{i % 4 + 1}"


def _design(df: pd.DataFrame, t: np.ndarray, cols=None):
    sid = (df["district"] + "|" + df["occupation"]).to_numpy()
    parts = {
        "L": pd.get_dummies(pd.Series(sid), prefix="L").astype(float),                          # series level
        "S": pd.get_dummies(pd.Series((t % 4).astype(str)), prefix="S").astype(float),          # quarter-of-year
        "O": pd.get_dummies(df["occupation"].reset_index(drop=True), prefix="O").astype(float).mul(t, axis=0),   # occupation trend
        "D": pd.get_dummies(df["district"].reset_index(drop=True), prefix="D").astype(float).mul(t, axis=0),     # district trend
    }
    X = pd.concat(parts.values(), axis=1)
    if cols is not None:
        X = X.reindex(columns=cols, fill_value=0.0)
    return X


def fit_predict(hist: pd.DataFrame, horizon: int = 4, band_z: float = 1.2816):
    """hist: rows with period, district, occupation + TARGETS. Returns long table of forecasts (+80% band for openings)."""
    hist = hist.reset_index(drop=True)
    t = _qidx(hist["period"]); last = t.max()
    keys = hist[KEYS].drop_duplicates().reset_index(drop=True)
    fut = pd.concat([keys.assign(t=last + h) for h in range(1, horizon + 1)], ignore_index=True)
    out = fut.copy(); out["period"] = out["t"].map(_label)
    for tgt in TARGETS:
        y = np.log(hist[tgt].clip(lower=1).to_numpy(float))
        X = _design(hist, t)
        pen = np.array([ALPHA_LEVEL if c.startswith("L_") else ALPHA_REST for c in X.columns])
        Xs = X.to_numpy() / np.sqrt(pen / ALPHA_REST)           # per-column ridge strength through rescaling
        m = Ridge(alpha=ALPHA_REST, fit_intercept=True).fit(Xs, y)
        resid = y - m.predict(Xs)
        dof = max(len(y) - (np.abs(m.coef_) > 1e-9).sum(), 1)
        sigma = np.sqrt((resid ** 2).sum() / dof)
        Xf = _design(fut, fut["t"].to_numpy(), cols=X.columns).to_numpy() / np.sqrt(pen / ALPHA_REST)
        mu = m.predict(Xf)
        h = fut["t"].to_numpy() - last
        s = sigma * np.sqrt(1 + 0.15 * (h - 1))
        out[tgt] = np.exp(mu + s ** 2 / 2)                       # mean of the lognormal
        out[tgt + "_lo"] = np.exp(mu - band_z * s)
        out[tgt + "_hi"] = np.exp(mu + band_z * s)
    return out.drop(columns="t")


# ---------------------------------------------------------------- honest evaluation
def backtest(df: pd.DataFrame, origins=(5, 6, 7, 8), horizon=2):
    """Rolling-origin: hide the last quarters, forecast them, compare to what really happened; vs 2 naive baselines."""
    df = df.copy(); df["t"] = _qidx(df["period"]); rows = []
    for o in origins:                                            # o = number of quarters the model may see
        tr, te = df[df.t < o], df[(df.t >= o) & (df.t < o + horizon)]
        if te.empty: continue
        p = fit_predict(tr.drop(columns="t"), horizon=horizon)
        p["t"] = _qidx(p["period"]); mg = te.merge(p, on=KEYS + ["t"], suffixes=("", "_p"))
        for tgt in TARGETS:
            last1 = tr[tr.t == o - 1].set_index(KEYS)[tgt]
            last4 = tr[tr.t >= o - 4].groupby(KEYS)[tgt].mean()
            snaive = tr.set_index(KEYS + ["t"])[tgt]
            for _, r in mg.iterrows():
                k = (r["district"], r["occupation"]); a = r[tgt]
                sn = snaive.get((*k, r["t"] - 4), np.nan)
                rows.append(dict(target=tgt, origin=o, h=int(r["t"] - o + 1), actual=a, model=r[tgt + "_p"], last=last1[k], mean4=last4[k], seasonal_naive=sn,
                                 lo=r.get(tgt + "_lo", np.nan), hi=r.get(tgt + "_hi", np.nan)))
    return pd.DataFrame(rows)
