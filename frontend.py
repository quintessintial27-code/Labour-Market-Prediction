"""
frontend.py  -  SURFACE ONLY (layout + styling). No data, no district names, no numbers in this file.
All data comes from adapters.load_tables().  Edit freely.

Run:   streamlit run frontend.py
Layout:
    [ district chips  ◀ ▶  (slide sideways) ]
    [ graph (left, wide) ]   [ recommendations (right, main part) ]
    [ skill recommendations (AI) ]
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import adapters

st.set_page_config(page_title="LMIS Maharashtra", page_icon="📊", layout="wide")

ALL = "__all__"
ALL_TRADES = "__all_trades__"

# presentation only (colour + symbol per status; the STATUS VALUES come from the data)
STYLE = {
    "shortage":    dict(color="#C0392B", sym="▲▲", label="Acute shortage"),
    "watch_short": dict(color="#E57373", sym="▲",  label="Watch: shortage"),
    "balanced":    dict(color="#2E9E5B", sym="●",  label="Balanced"),
    "watch_over":  dict(color="#F0B26B", sym="▼",  label="Watch: oversupply"),
    "saturation":  dict(color="#D9730D", sym="▼▼", label="Saturation risk"),
}
FALLBACK = dict(color="#888", sym="•", label="")
st_ = lambda s: STYLE.get(s, FALLBACK)

# ------------------------------------------------------------------------------------------ styling
st.markdown("""
<style>
.block-container {padding-top: 1.6rem; max-width: 1500px;}
/* ---- district chip row: one line, slides sideways INSIDE its own box ---- */
[data-testid="stColumn"]:has(.st-key-district_row), .st-key-district_row, .st-key-district_row > div, .st-key-district_row [data-testid="stRadio"],
.st-key-district_row [data-testid="stRadio"] > div {min-width: 0 !important; width: 100% !important; max-width: 100% !important;}
.st-key-district_row {overflow: hidden;}
.st-key-district_row div[role="radiogroup"] {flex-wrap: nowrap !important; overflow-x: auto; overflow-y: hidden; gap: .6rem; padding: .2rem .2rem .8rem;}
.st-key-district_row div[role="radiogroup"] > div {flex: 0 0 auto;}
.st-key-district_row label[data-testid="stRadioOption"] {background: #fff; border: 1.5px solid #D5DBE5; border-radius: 14px;
    padding: .65rem 1.1rem; cursor: pointer; white-space: nowrap; box-shadow: 0 1px 2px rgba(0,0,0,.05); margin: 0;
    color: #1F2937 !important;}
.st-key-district_row label[data-testid="stRadioOption"] * {color: #1F2937 !important;}
.st-key-district_row label[data-testid="stRadioOption"] > div > div:first-child {display: none;}   /* hide the round radio dot */
.st-key-district_row label[data-testid="stRadioOption"]:hover {border-color: #1F5FBF;}
.st-key-district_row label[data-testid="stRadioOption"]:has(input:checked) {background: #1F5FBF; border-color: #1F5FBF;}
.st-key-district_row label[data-testid="stRadioOption"]:has(input:checked) * {color: #fff !important;}
.st-key-district_row label[data-testid="stRadioOption"]:focus-within {outline: 3px solid #FFD400; outline-offset: 2px;}
.st-key-district_row div[role="radiogroup"]::-webkit-scrollbar {height: 8px;}
.st-key-district_row div[role="radiogroup"]::-webkit-scrollbar-thumb {background: #B8C2D3; border-radius: 8px;}
.tag {display:inline-block; padding:.1rem .55rem; border-radius:999px; font-size:.78rem; font-weight:600; color:#fff;}
.muted {color:#6B7280; font-size:.85rem;}
.rec-title {font-weight:700; font-size:1.02rem;}
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=900, show_spinner="Loading data ...")      # re-reads the source every 15 min
def load():
    return adapters.load_tables()


series, summary, skills, meta = load()

# ------------------------------------------------------------------------------------------ header
h1, h2 = st.columns([4, 2])
h1.title("Labour Market Intelligence – Maharashtra")
h2.markdown(f"<div style='text-align:right;padding-top:1.2rem' class='muted'>Data: <b>{meta.get('data_kind', '')}</b>"
            f"{' · updated ' + str(meta['last_updated']) if meta.get('last_updated') else ''}</div>", unsafe_allow_html=True)
if "SYNTHETIC" in str(meta.get("data_kind", "")).upper():
    st.warning("Prototype running on SYNTHETIC demo data.", icon="⚠️")

# ------------------------------------------------------------------------------------------ district chips
by_d = summary.groupby("district").apply(lambda g: pd.Series(dict(
    n_alert=int(g["level"].isin(["critical", "high"]).sum()),
    worst=g.sort_values("severity", ascending=False)["status"].iloc[0],
    gap=(g["demand_12m"].sum() - g["supply_12m"].sum()) / max(g["demand_12m"].sum(), 1e-9))), include_groups=False)
options = [ALL] + sorted(by_d.index)
st.session_state.setdefault("district", ALL)


def step(delta):
    i = options.index(st.session_state["district"])
    st.session_state["district"] = options[(i + delta) % len(options)]


def chip(d):
    if d == ALL:
        return "All Maharashtra"
    r = by_d.loc[d]
    return f"{d}  ·  {st_(r['worst'])['sym']} {int(r['n_alert'])} alert{'s' if r['n_alert'] != 1 else ''}" if r["n_alert"] else f"{d}  ·  ● OK"


a, b, c = st.columns([0.5, 0.5, 14], vertical_alignment="center")
a.button("◀", on_click=step, args=(-1,), help="Previous district")
b.button("▶", on_click=step, args=(1,), help="Next district")
with c.container(key="district_row"):
    st.radio("District", options, format_func=chip, horizontal=True, key="district", label_visibility="collapsed")
district = st.session_state["district"]
D = summary if district == ALL else summary[summary["district"] == district]
title_place = "Maharashtra" if district == ALL else district

# ------------------------------------------------------------------------------------------ graph + recommendations
left, right = st.columns([3, 2], gap="large")

with left:
    st.subheader(f"{title_place}: demand vs supply")
    f1, f2 = st.columns([3, 2])
    trades = sorted(D["trade"].unique())
    trade = f1.selectbox("Trade", [ALL_TRADES] + trades, format_func=lambda v: "All trades" if v == ALL_TRADES else v)
    view = f2.radio("View", ["Rolling 12 months", "Monthly"], horizontal=True)

    sel = D if trade == ALL_TRADES else D[D["trade"] == trade]
    tot_d, tot_s = sel["demand_12m"].sum(), sel["supply_12m"].sum()
    gp = (tot_d - tot_s) / max(tot_d, 1e-9)
    k = st.columns(3)
    k[0].metric("Expected openings (12m)", f"{tot_d:,.0f}")
    k[1].metric("Job-ready candidates (12m)", f"{tot_s:,.0f}")
    k[2].metric("Net gap", f"{tot_d - tot_s:+,.0f}", f"{gp * 100:+.0f}%", delta_color="off")

    s = series if district == ALL else series[series["district"] == district]
    if trade != ALL_TRADES:
        s = s[s["trade"] == trade]
    g = s.groupby(["month", "kind"], as_index=False)[["demand", "supply", "demand_lo", "demand_hi"]].sum(min_count=1).sort_values("month")
    g = g.reset_index(drop=True)
    hw = ((g["demand_hi"] - g["demand_lo"]) / 2).fillna(0)
    if view.startswith("Rolling"):
        g["demand"], g["supply"] = g["demand"].rolling(12).sum(), g["supply"].rolling(12).sum()
        hw12 = hw.rolling(12, min_periods=1).sum()
        is_f = g["kind"] == "forecast"
        g["demand_lo"], g["demand_hi"] = np.where(is_f, g["demand"] - hw12, np.nan), np.where(is_f, g["demand"] + hw12, np.nan)
        ylab = "per rolling 12 months"
    else:
        for col in ["demand", "supply", "demand_lo", "demand_hi"]:
            g[col] = g[col].rolling(3, min_periods=3).mean()
        ylab = "per month (3-month average)"
    g = g.dropna(subset=["demand", "supply"])
    hist, fc = g[g["kind"] == "history"], g[g["kind"] == "forecast"]

    fig = go.Figure()
    if len(fc) and fc["demand_lo"].notna().any():
        fig.add_trace(go.Scatter(x=list(fc["month"]) + list(fc["month"][::-1]), y=list(fc["demand_hi"]) + list(fc["demand_lo"][::-1]),
                                 fill="toself", fillcolor="rgba(192,57,43,.16)", line=dict(width=0), hoverinfo="skip", name="80% range"))
    for col, name, colr in (("demand", "Demand", "#C0392B"), ("supply", "Supply", "#1F5FBF")):
        fig.add_trace(go.Scatter(x=hist["month"], y=hist[col], name=f"{name} · history", line=dict(color=colr, width=3)))
        if len(fc) and len(hist):
            fig.add_trace(go.Scatter(x=[hist["month"].iloc[-1]] + list(fc["month"]), y=[hist[col].iloc[-1]] + list(fc[col]),
                                     name=f"{name} · forecast", line=dict(color=colr, width=3, dash="dash")))
    if len(fc):
        fig.add_vrect(x0=fc["month"].min().to_pydatetime(), x1=fc["month"].max().to_pydatetime(), fillcolor="rgba(128,128,128,.10)", line_width=0)
    fig.update_layout(height=430, margin=dict(l=0, r=0, t=10, b=0), yaxis_title=ylab, legend=dict(orientation="h", y=-0.15),
                      template="plotly_white")
    st.plotly_chart(fig)
    with st.expander("Data table (accessible alternative to the chart)"):
        st.dataframe(g.round(1), hide_index=True)

with right:
    st.subheader("Recommendations")
    st.caption("Seat changes for the next training cycle, most urgent first.")
    R = D[D["level"].isin(["critical", "high", "watch"])].sort_values("severity", ascending=False)
    n_show = st.slider("Show", 3, 12, 6, label_visibility="collapsed", help="How many recommendations to show")
    if R.empty:
        st.success("No recommendations: this selection looks balanced.")
    for _, r in R.head(n_show).iterrows():
        sty = st_(r["status"])
        has_seats = pd.notna(r["seats_now"]) and pd.notna(r["seats_suggested"])
        delta = (r["seats_suggested"] - r["seats_now"]) if has_seats else None
        with st.container(border=True):
            top = f"<span class='tag' style='background:{sty['color']}'>{sty['sym']} {sty['label']}</span>"
            where = f" · {r['district']}" if district == ALL else ""
            st.markdown(f"<div class='rec-title'>{r['trade']}<span class='muted'>{where}</span> &nbsp; {top}</div>", unsafe_allow_html=True)
            if has_seats:
                verb = "Reduce" if delta < 0 else "Increase" if delta > 0 else "Keep"
                st.markdown(f"**{verb}** seats: {int(r['seats_now']):,} → **{int(r['seats_suggested']):,}** "
                            f"<span class='muted'>({delta:+,.0f})</span>", unsafe_allow_html=True)
            else:
                st.markdown("<span class='muted'>Seat suggestion not provided by the model yet.</span>", unsafe_allow_html=True)
            if r["reason"]:
                st.caption(r["reason"])
            st.progress(min(max(float(r["severity"]), 0), 100) / 100, text=f"Severity {r['severity']:.0f}/100 · gap {r['gap_pct'] * 100:+.0f}%")

# ------------------------------------------------------------------------------------------ skill recommendations
st.markdown("---")
sh1, sh2 = st.columns([4, 1])
sh1.subheader("Skills to train for (AI-predicted shortages)")
if str(meta.get("skills_source", "")).upper() == "MOCK":
    sh2.markdown("<div style='text-align:right'><span class='tag' style='background:#888'>MOCK suggestions</span></div>", unsafe_allow_html=True)
K = skills if district == ALL else skills[skills["district"] == district]
if "shortage_pct" in K:
    K = K.sort_values("shortage_pct", ascending=False)
if K.empty:
    st.info("No skill recommendations for this selection yet. Connect your skill model in adapters.py (the `skills` table).")
else:
    cards = K.head(9).to_dict("records")
    for i in range(0, len(cards), 3):
        cols = st.columns(3)
        for col, r in zip(cols, cards[i:i + 3]):
            with col.container(border=True):
                st.markdown(f"<div class='rec-title'>{r['skill']}</div>", unsafe_allow_html=True)
                bits = []
                if pd.notna(r.get("related_trade")):
                    bits.append(f"Trade: {r['related_trade']}")
                if district == ALL:
                    bits.append(r["district"])
                if bits:
                    st.caption(" · ".join(bits))
                m = st.columns(2)
                if pd.notna(r.get("shortage_pct")):
                    m[0].metric("Shortage", f"{r['shortage_pct'] * 100:.0f}%")
                if pd.notna(r.get("demand_trend")):
                    m[1].metric("Demand trend", f"{r['demand_trend'] * 100:+.0f}%")
                if pd.notna(r.get("suggested_course")):
                    lvl = f" (NSQF L{int(r['nsqf_level'])})" if pd.notna(r.get("nsqf_level")) else ""
                    st.markdown(f"**Course:** {r['suggested_course']}{lvl}")
                if pd.notna(r.get("reason")) and r.get("reason"):
                    st.caption(r["reason"])

# ------------------------------------------------------------------------------------------ model honesty panel
v = meta.get("validation")
if v:
    with st.expander("How good is the forecast? (back-test on hidden quarters)"):
        st.markdown(
            f"Hid the last quarters, forecast them, compared with what the CSV says happened (error = weighted abs. % error, lower is better).\n\n"
            f"| | ML model | 'average of last 4 quarters' | 'same quarter last year' |\n|---|---|---|---|\n"
            f"| Openings | **{v['openings_model']:.1%}** | {v['openings_mean4']:.1%} | {v['openings_seasonal']:.1%} |\n"
            f"| Jobseekers | **{v['seekers_model']:.1%}** | {v['seekers_mean4']:.1%} | n/a |\n\n"
            f"80% range actually contained the truth **{v['coverage80']:.0%}** of the time. "
            f"Trained on SYNTHETIC data: this measures how well the model learns the demo's patterns, not the real labour market.")
