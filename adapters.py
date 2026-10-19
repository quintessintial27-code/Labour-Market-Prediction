"""
adapters.py  -  THE ONLY FILE THAT KNOWS WHERE DATA COMES FROM.

frontend.py never contains data, district names, trades or numbers. It asks this file for three tables:

    series   monthly demand / supply, history + forecast       (feeds the graph)
    summary  one row per district x trade                       (feeds the chips + seat recommendations)
    skills   one row per recommended skill                      (feeds the skill recommendation section)

Pick where they come from with one environment variable:

    LMIS_SOURCE=lmis   (default)  maharashtra_lmis_sample.csv -> ML forecast (lmis_model.py) -> the 3 tables
    LMIS_SOURCE=mock              synthetic demo data generated below
    LMIS_SOURCE=csv               read series.csv / summary.csv / skills.csv from LMIS_DATA_DIR (default ./data_real)
    LMIS_SOURCE=api               GET  {LMIS_API_URL}/series , /summary , /skills   (JSON list of records)

    e.g.  LMIS_SOURCE=csv streamlit run frontend.py

TO PLUG IN YOUR REAL MODELS
    * forecast model       -> produce the `series` table (columns below)
    * recommendation model -> fill  summary.seats_suggested  and  summary.reason
    * skill model          -> produce the `skills` table
  Write them out as CSV (or serve them from an API). No frontend edit is needed as long as the column names match.

--------------------------------------------------------------------------------------------------------
TABLE CONTRACT   (R = required, O = optional; extra columns are ignored)
--------------------------------------------------------------------------------------------------------
series   R district, trade, month (YYYY-MM-01), kind ("history" | "forecast"), demand, supply
         O sector, demand_lo, demand_hi   (uncertainty band, forecast rows only)

summary  R district, trade, demand_12m, supply_12m          (next-12-month totals)
         O sector, gap_pct   = (demand-supply)/demand      -> computed if missing
         O status  one of shortage | watch_short | balanced | watch_over | saturation   -> computed if missing
         O level   one of critical | high | watch | ok                                  -> computed if missing
         O severity 0-100                                                               -> computed if missing
         O seats_now, seats_suggested, reason                <- your recommendation model fills these

skills   R district, skill
         O related_trade, shortage_pct, demand_trend, suggested_course, nsqf_level, priority, reason
--------------------------------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE = os.environ.get("LMIS_SOURCE", "lmis").lower()
SAMPLE_CSV = Path(
    os.environ.get(
        "LMIS_SAMPLE_CSV", Path(__file__).parent / "maharashtra_lmis_sample.csv"
    )
)
DATA_DIR = Path(os.environ.get("LMIS_DATA_DIR", "data_real"))
API_URL = os.environ.get("LMIS_API_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.environ.get("LMIS_API_KEY", "")

# thresholds used ONLY to fill in status/level when your data does not provide them
FLAG, WATCH, CRITICAL = 0.25, 0.15, 0.45

REQUIRED = {
    "series": ["district", "trade", "month", "kind", "demand", "supply"],
    "summary": ["district", "trade", "demand_12m", "supply_12m"],
    "skills": ["district", "skill"],
}


# =====================================================================================================
# PUBLIC ENTRY POINT  (frontend calls this and nothing else)
# =====================================================================================================
def load_tables():
    """Returns (series, summary, skills, meta). meta = dict(data_kind, last_updated, skills_source)."""
    if SOURCE == "csv":
        series, summary, skills, meta = _from_csv()
    elif SOURCE == "api":
        series, summary, skills, meta = _from_api()
    elif SOURCE == "lmis":
        series, summary, skills, meta = _from_lmis_sample()
    else:
        series, summary, skills, meta = _mock()
    series = _check("series", series)
    summary = _check("summary", summary)
    skills = _check("skills", skills)
    series["month"] = pd.to_datetime(series["month"])
    return series, _normalise_summary(summary), skills, meta


# =====================================================================================================
# REAL DATA CONNECTORS
# =====================================================================================================
def _from_csv():
    def read_table(name, required=True):
        csv_path = DATA_DIR / f"{name}.csv"
        if not csv_path.exists():
            if required:
                raise FileNotFoundError(
                    f"{csv_path} not found. Expected series.csv, summary.csv "
                    "(and optionally skills.csv)."
                )
            return pd.DataFrame(columns=REQUIRED[name])
        return pd.read_csv(csv_path)

    metadata_path = DATA_DIR / "meta.json"
    meta = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    meta.setdefault("data_kind", f"CSV files in {DATA_DIR}")
    meta.setdefault("skills_source", "model")
    return (
        read_table("series"),
        read_table("summary"),
        read_table("skills", required=False),
        meta,
    )


def _from_api():
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError(
            "LMIS_SOURCE=api requires the 'requests' package. "
            "Install project dependencies with `pip install -r requirements.txt`."
        ) from exc

    h = {"X-API-Key": API_KEY} if API_KEY else {}

    def get_table(path):
        response = requests.get(f"{API_URL}{path}", headers=h, timeout=30)
        response.raise_for_status()
        return pd.DataFrame(response.json())

    try:
        skills = get_table("/skills")
    except requests.HTTPError as exc:
        if exc.response is None or exc.response.status_code != 404:
            raise
        skills = pd.DataFrame(columns=REQUIRED["skills"])
    return (
        get_table("/series"),
        get_table("/summary"),
        skills,
        {"data_kind": f"API {API_URL}", "skills_source": "model"},
    )


# =====================================================================================================
# LMIS SAMPLE CSV  ->  ML FORECAST  ->  the three tables
# =====================================================================================================
CLOSE_SHARE = 0.5  # seat rule: each training cycle tries to close this share of the forecast gap (tune me)
FORECAST_Q = 4  # quarters to forecast


def _from_lmis_sample():
    import lmis_model as model

    if not SAMPLE_CSV.exists():
        raise FileNotFoundError(
            f"{SAMPLE_CSV} not found. Put maharashtra_lmis_sample.csv next to "
            "adapters.py or set LMIS_SAMPLE_CSV."
        )
    sample = pd.read_csv(SAMPLE_CSV)
    required_sample_columns = {
        "period",
        "district",
        "occupation",
        "sector",
        "required_skills",
        "estimated_job_openings",
        "estimated_available_jobseekers",
        "training_seats_available",
        "training_completions_estimate",
        "indicative_skill_level_not_verified",
    }
    missing_sample_columns = sorted(required_sample_columns.difference(sample.columns))
    if missing_sample_columns:
        raise ValueError(
            f"Sample CSV is missing required columns: {missing_sample_columns}"
        )

    sample["_t"] = model._qidx(sample["period"])
    forecast = model.fit_predict(sample, horizon=FORECAST_Q)
    openings_column = "estimated_job_openings"
    jobseekers_column = "estimated_available_jobseekers"
    sector_by_occupation = (
        sample.drop_duplicates("occupation").set_index("occupation")["sector"]
    )

    def to_months(table, quarter_column):
        quarter_starts = pd.PeriodIndex(
            table[quarter_column].str.replace("Q", "-Q"), freq="Q"
        ).start_time
        return pd.concat(
            [
                table.assign(month=quarter_starts + pd.DateOffset(months=month_offset))
                for month_offset in range(3)
            ],
            ignore_index=True,
        )

    # Distribute each quarterly total evenly across its three calendar months.
    history_monthly = to_months(sample.assign(q=sample["period"]), "q")
    history_series = pd.DataFrame(
        dict(
            district=history_monthly["district"],
            trade=history_monthly["occupation"],
            sector=history_monthly["sector"],
            month=history_monthly["month"],
            kind="history",
            demand=history_monthly[openings_column] / 3,
            supply=history_monthly[jobseekers_column] / 3,
            demand_lo=np.nan,
            demand_hi=np.nan,
        )
    )
    forecast_monthly = to_months(forecast.assign(q=forecast["period"]), "q")
    forecast_series = pd.DataFrame(
        dict(
            district=forecast_monthly["district"],
            trade=forecast_monthly["occupation"],
            sector=forecast_monthly["occupation"].map(sector_by_occupation),
            month=forecast_monthly["month"],
            kind="forecast",
            demand=forecast_monthly[openings_column] / 3,
            supply=forecast_monthly[jobseekers_column] / 3,
            demand_lo=forecast_monthly[openings_column + "_lo"] / 3,
            demand_hi=forecast_monthly[openings_column + "_hi"] / 3,
        )
    )
    series = (
        pd.concat([history_series, forecast_series], ignore_index=True)
        .sort_values(["district", "trade", "month"])
        .reset_index(drop=True)
    )

    # ---- summary: next-12-month totals + seat recommendation
    summary_data = (
        forecast.groupby(["district", "occupation"])
        .agg(
            demand_12m=(openings_column, "sum"),
            supply_12m=(jobseekers_column, "sum"),
        )
        .reset_index()
    )
    recent_training = (
        sample[sample["_t"] >= sample["_t"].max() - 3]
        .groupby(["district", "occupation"])
        .agg(
            seats_now=("training_seats_available", "mean"),
            completions=("training_completions_estimate", "sum"),
            seats_4q=("training_seats_available", "sum"),
            openings_last12=(openings_column, "sum"),
        )
        .reset_index()
    )
    summary_data = summary_data.merge(
        recent_training, on=["district", "occupation"]
    )
    completion_rate = (summary_data["completions"] / summary_data["seats_4q"]).clip(
        0.3, 1
    )
    quarterly_gap = (
        summary_data["demand_12m"] - summary_data["supply_12m"]
    ) / FORECAST_Q  # missing (+) / surplus (-) people per quarter
    summary_data["seats_now"] = summary_data["seats_now"].round()
    summary_data["seats_suggested"] = np.clip(
        summary_data["seats_now"] + CLOSE_SHARE * quarterly_gap / completion_rate,
        summary_data["seats_now"] * 0.5,
        summary_data["seats_now"] * 2,
    ).round()
    gap_percent = (
        summary_data["demand_12m"] - summary_data["supply_12m"]
    ) / summary_data["demand_12m"]
    summary_data["reason"] = [
        f"Forecast openings {forecast_openings / previous_openings - 1:+.0%} vs last 4 quarters; "
        f"{'demand exceeds' if gap >= 0 else 'jobseekers exceed'} candidates by {abs(gap):.0%}."
        for forecast_openings, previous_openings, gap in zip(
            summary_data["demand_12m"], summary_data["openings_last12"], gap_percent
        )
    ]
    summary = summary_data.rename(columns={"occupation": "trade"}).assign(
        sector=lambda table: table["trade"].map(sector_by_occupation)
    )[
        [
            "district",
            "trade",
            "sector",
            "demand_12m",
            "supply_12m",
            "seats_now",
            "seats_suggested",
            "reason",
            "openings_last12",
        ]
    ]

    # ---- skills: the CSV only lists skills per OCCUPATION, so this is attribution, not a skill-demand model
    nsqf_by_occupation = (
        sample.drop_duplicates("occupation")
        .set_index("occupation")["indicative_skill_level_not_verified"]
        .str.extract(r"(\d)")[0]
        .astype(float)
    )
    skills_by_occupation = (
        sample.drop_duplicates("occupation")
        .set_index("occupation")["required_skills"]
        .str.split(";")
        .map(lambda skills: [skill.strip() for skill in skills])
    )
    skill_rows = []
    for _, recommendation in summary.iterrows():
        shortage_pct = (
            recommendation["demand_12m"] - recommendation["supply_12m"]
        ) / recommendation["demand_12m"]
        if shortage_pct < FLAG:
            continue
        demand_trend = (
            recommendation["demand_12m"] / recommendation["openings_last12"] - 1
        )
        recommended_skills = [
            skill
            for skill in skills_by_occupation[recommendation["trade"]]
            if skill.lower() != "problem solving"
        ][:3]
        for skill in recommended_skills:
            skill_rows.append(
                dict(
                    district=recommendation["district"],
                    skill=skill,
                    related_trade=recommendation["trade"],
                    shortage_pct=shortage_pct,
                    demand_trend=demand_trend,
                    suggested_course=np.nan,
                    nsqf_level=nsqf_by_occupation[recommendation["trade"]],
                    priority="critical" if shortage_pct >= CRITICAL else "high",
                    reason=(
                        f"Listed skill for {recommendation['trade']}; forecast "
                        f"openings exceed candidates by {shortage_pct:.0%}."
                    ),
                )
            )
    skills = pd.DataFrame(
        skill_rows,
        columns=REQUIRED["skills"]
        + [
            "related_trade",
            "shortage_pct",
            "demand_trend",
            "suggested_course",
            "nsqf_level",
            "priority",
            "reason",
        ],
    )

    # ---- honest evaluation shown in the app
    backtest = model.backtest(
        sample.drop(columns="_t"), origins=(5, 6, 7, 8), horizon=2
    )

    def weighted_absolute_percentage_error(rows, prediction_column):
        return float(
            abs(rows[prediction_column] - rows["actual"]).sum()
            / rows["actual"].sum()
        )

    openings_backtest = backtest[backtest.target == openings_column]
    jobseekers_backtest = backtest[backtest.target == jobseekers_column]
    validation = dict(
        openings_model=weighted_absolute_percentage_error(
            openings_backtest, "model"
        ),
        openings_mean4=weighted_absolute_percentage_error(
            openings_backtest, "mean4"
        ),
        openings_seasonal=weighted_absolute_percentage_error(
            openings_backtest.dropna(subset=["seasonal_naive"]), "seasonal_naive"
        ),
        seekers_model=weighted_absolute_percentage_error(
            jobseekers_backtest, "model"
        ),
        seekers_mean4=weighted_absolute_percentage_error(
            jobseekers_backtest, "mean4"
        ),
        coverage80=float(
            (
                (openings_backtest.actual >= openings_backtest.lo)
                & (openings_backtest.actual <= openings_backtest.hi)
            ).mean()
        ),
    )
    meta = dict(
        data_kind="SYNTHETIC DEMO CSV (maharashtra_lmis_sample.csv) + ML forecast",
        skills_source="DERIVED from occupation skill lists",
        last_updated=sample["period"].iloc[sample["_t"].argmax()],
        validation=validation,
        forecast_quarters=sorted(forecast["period"].unique()),
    )
    return series, summary, skills, meta


# =====================================================================================================
# HELPERS
# =====================================================================================================
def _check(name, df):
    missing = [c for c in REQUIRED[name] if c not in df.columns]
    if missing:
        raise ValueError(
            f"'{name}' table is missing required column(s): {missing}. Found: {list(df.columns)}"
        )
    return df.copy()


def _status(p):
    return (
        "shortage"
        if p >= FLAG
        else (
            "saturation"
            if p <= -FLAG
            else (
                "watch_short"
                if p >= WATCH
                else "watch_over" if p <= -WATCH else "balanced"
            )
        )
    )


def _normalise_summary(s: pd.DataFrame) -> pd.DataFrame:
    """Fill every optional column the frontend uses, so partial data still works."""
    s = s.copy()
    if "sector" not in s:
        s["sector"] = ""
    if "gap_pct" not in s:
        s["gap_pct"] = (s["demand_12m"] - s["supply_12m"]) / s["demand_12m"].clip(
            lower=1e-9
        )
    if "status" not in s:
        s["status"] = s["gap_pct"].map(_status)
    if "level" not in s:
        a = s["gap_pct"].abs()
        s["level"] = np.select(
            [a >= CRITICAL, a >= FLAG, a >= WATCH], ["critical", "high", "watch"], "ok"
        )
    if "severity" not in s:
        size = (s["demand_12m"] - s["supply_12m"]).abs().rank(pct=True)
        s["severity"] = (
            100 * np.minimum(1, s["gap_pct"].abs() / 0.6) * (0.5 + 0.5 * size)
        )
    for c in ("seats_now", "seats_suggested"):
        if c not in s:
            s[c] = np.nan
    if "reason" not in s:
        s["reason"] = ""
    return s


# =====================================================================================================
# MOCK DATA  (everything hardcoded lives HERE and only here; delete this section when you go live)
# =====================================================================================================
_DISTRICTS = {  # district -> sector weights
    "Mumbai": (0.7, 0.6, 2.2, 2.0),
    "Pune": (0.5, 2.2, 2.4, 1.6),
    "Thane": (1.4, 1.0, 1.2, 1.2),
    "Nagpur": (0.6, 0.5, 1.0, 1.3),
    "Nashik": (0.4, 1.3, 0.4, 0.7),
    "Chhatrapati Sambhajinagar": (0.5, 1.6, 0.4, 0.6),
    "Kolhapur": (1.3, 0.8, 0.3, 0.5),
    "Solapur": (1.5, 0.3, 0.2, 0.5),
    "Satara": (0.3, 0.6, 0.2, 0.3),
    "Sangli": (0.5, 0.4, 0.15, 0.4),
    "Amravati": (0.5, 0.2, 0.15, 0.3),
    "Nanded": (0.3, 0.2, 0.1, 0.3),
}
_SECTORS = ["Textile & Apparel", "Automotive & EV", "IT-ITeS", "Healthcare"]
# (sector idx, trade, yearly base, yearly growth, seasonality, ratio range)
_TRADES = [
    (0, "Tailor", 900, -0.05, "tex", (0.95, 1.15)),
    (0, "Sewing Machine Operator", 1400, -0.02, "tex", (0.95, 1.15)),
    (0, "Power Loom Operator", 800, -0.04, "tex", (0.95, 1.15)),
    (0, "Garment Quality Checker", 500, 0.02, "tex", (0.9, 1.15)),
    (1, "Auto Service Technician", 900, 0.03, "auto", (0.9, 1.15)),
    (1, "EV Technician", 250, 0.38, "auto", (1.1, 1.5)),
    (1, "CNC Operator", 700, 0.06, "auto", (0.9, 1.15)),
    (1, "Welder", 1000, 0.02, "auto", (0.9, 1.15)),
    (2, "Data Entry Operator", 1100, -0.10, "it", (0.85, 1.05)),
    (2, "Customer Care Executive", 1500, -0.03, "it", (0.95, 1.15)),
    (2, "Junior Software Tester", 600, 0.05, "it", (0.95, 1.2)),
    (3, "General Duty Assistant", 1000, 0.16, "health", (1.0, 1.3)),
    (3, "Phlebotomist", 450, 0.10, "health", (0.95, 1.25)),
    (3, "Home Health Aide", 500, 0.20, "health", (1.05, 1.35)),
]
_STORIES = {  # (district, trade) -> supply/demand ratio (>1 oversupply, <1 shortage)
    ("Solapur", "Tailor"): 1.6,
    ("Kolhapur", "Sewing Machine Operator"): 1.5,
    ("Thane", "Power Loom Operator"): 1.45,
    ("Mumbai", "Data Entry Operator"): 1.55,
    ("Mumbai", "Customer Care Executive"): 1.45,
    ("Pune", "Junior Software Tester"): 1.4,
    ("Pune", "EV Technician"): 0.55,
    ("Nagpur", "General Duty Assistant"): 0.65,
    ("Nashik", "Welder"): 0.65,
    ("Chhatrapati Sambhajinagar", "CNC Operator"): 0.65,
    ("Amravati", "Phlebotomist"): 0.7,
    ("Nanded", "Home Health Aide"): 0.7,
    ("Thane", "EV Technician"): 1.05,
    ("Nashik", "EV Technician"): 1.0,
}
_SEASON = {
    "tex": [0.9, 0.9, 1, 0.9, 0.85, 0.85, 0.95, 1.15, 1.3, 1.35, 1.1, 0.8],
    "auto": [1, 1, 1.25, 1, 0.9, 0.85, 0.9, 0.95, 1.05, 1.2, 1, 0.9],
    "it": [1.25, 1.2, 1.15, 1, 0.9, 0.9, 0.95, 0.95, 1, 1, 0.9, 0.8],
    "health": [1, 1, 1, 0.95, 0.95, 0.95, 1, 1, 1.05, 1.05, 1.05, 1],
}
_MOCK_SKILLS = {  # trade -> (skill, suggested course, NSQF)
    "EV Technician": [
        ("EV battery diagnostics & repair", "Electric Vehicle Battery Technician", 5),
        (
            "Charging-station installation & upkeep",
            "EV Charging Infrastructure Technician",
            4,
        ),
    ],
    "General Duty Assistant": [
        ("Patient mobility & transfer", "General Duty Assistant (Advanced)", 4),
        ("Basic life support", "First Aid & BLS Certification", 4),
    ],
    "Home Health Aide": [
        ("Elderly & post-surgery home care", "Home Health Aide", 4),
        ("Medication assistance", "Medication Assistance Add-on", 4),
    ],
    "Phlebotomist": [("Sample collection & handling", "Phlebotomy Technician", 4)],
    "CNC Operator": [
        ("CNC programming (G-code)", "CNC Programmer", 5),
        ("Precision measurement", "Quality Inspection Add-on", 4),
    ],
    "Welder": [("MIG / TIG welding", "Welder (MIG/TIG)", 4)],
    "Auto Service Technician": [
        ("Hybrid & EV servicing basics", "Auto Service Technician + EV Add-on", 4)
    ],
    "Junior Software Tester": [("Test automation", "Automation Testing Associate", 5)],
    "Customer Care Executive": [
        ("Multilingual voice & chat support", "Customer Care Executive (Digital)", 4)
    ],
    "Garment Quality Checker": [
        ("Technical-textile quality inspection", "Technical Textile Inspector", 4)
    ],
}


def _round5(x):
    return int(np.round(x / 5) * 5)


def _mock():
    rng = np.random.default_rng(7)
    months = pd.date_range("2023-10-01", periods=36, freq="MS")
    fut = pd.date_range(months[-1] + pd.offsets.MonthBegin(1), periods=12, freq="MS")
    S, rows, skills = [], [], []
    for dname, w in _DISTRICTS.items():
        for si, trade, base, g, seas, rr in _TRADES:
            ratio = _STORIES.get((dname, trade), rng.uniform(*rr))
            season = np.array(_SEASON[seas]) / np.mean(_SEASON[seas])
            trend = lambda k: base / 12 * w[si] * (1 + g) ** (k / 12)
            d = np.array(
                [
                    trend(i) * season[m.month - 1] * rng.lognormal(0, 0.06)
                    for i, m in enumerate(months)
                ]
            )
            sup = np.array(
                [ratio * trend(i - 9) * rng.lognormal(0, 0.04) for i in range(36)]
            )
            growth = d[-12:].sum() / d[:12].sum()
            growth = growth**0.5 if growth > 0 else 1.0  # crude year-on-year estimate
            fd = np.array([d[-12 + k] * growth for k in range(12)])
            fs = np.full(12, sup[-3:].mean())
            band = fd * 0.07 * np.sqrt(1 + 0.1 * (np.arange(12) + 1))
            sector = _SECTORS[si]
            S.append(
                pd.DataFrame(
                    dict(
                        district=dname,
                        trade=trade,
                        sector=sector,
                        month=months,
                        kind="history",
                        demand=d,
                        supply=sup,
                        demand_lo=np.nan,
                        demand_hi=np.nan,
                    )
                )
            )
            S.append(
                pd.DataFrame(
                    dict(
                        district=dname,
                        trade=trade,
                        sector=sector,
                        month=fut,
                        kind="forecast",
                        demand=fd,
                        supply=fs,
                        demand_lo=fd - band,
                        demand_hi=fd + band,
                    )
                )
            )
            D, Sup = fd.sum(), fs.sum()
            gp = (D - Sup) / D
            seats_now = _round5(Sup / 0.70)
            sugg = _round5(np.clip(D / 0.70, seats_now * 0.5, seats_now * 2.0))
            rows.append(
                dict(
                    district=dname,
                    trade=trade,
                    sector=sector,
                    demand_12m=D,
                    supply_12m=Sup,
                    seats_now=seats_now,
                    seats_suggested=sugg,
                    reason=(
                        f"Demand {D / d[-12:].sum() - 1:+.0%} vs last 12 months; "
                        f"{'supply exceeds demand' if gp < 0 else 'demand exceeds supply'} by {abs(gp):.0%}."
                    ),
                )
            )
            if gp >= FLAG and trade in _MOCK_SKILLS:
                for skill, course, nsqf in _MOCK_SKILLS[trade]:
                    skills.append(
                        dict(
                            district=dname,
                            skill=skill,
                            related_trade=trade,
                            shortage_pct=gp,
                            demand_trend=D / d[-12:].sum() - 1,
                            suggested_course=course,
                            nsqf_level=nsqf,
                            priority="critical" if gp >= CRITICAL else "high",
                            reason=f"{trade} demand is outrunning trained supply by {gp:.0%}; this skill closes the gap.",
                        )
                    )
    meta = dict(
        data_kind="SYNTHETIC (mock)",
        skills_source="MOCK",
        last_updated=str(months[-1].date()),
    )
    return (
        pd.concat(S, ignore_index=True),
        pd.DataFrame(rows),
        pd.DataFrame(
            skills,
            columns=REQUIRED["skills"]
            + [
                "related_trade",
                "shortage_pct",
                "demand_trend",
                "suggested_course",
                "nsqf_level",
                "priority",
                "reason",
            ],
        ),
        meta,
    )
