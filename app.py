import json
import math
import time
from dataclasses import dataclass
import os

import requests
import streamlit as st
import plotly.graph_objects as go

# ---------- Page & basic theming ----------
st.set_page_config(page_title="Smart Investment Advisor", page_icon="💹", layout="wide")

CUSTOM_CSS = """
<style>
body, .stApp {
    background: radial-gradient(circle at top left, #141826, #050608) !important;
    color: #f4f4f7 !important;
}
.block-card {
    background: rgba(20, 24, 40, 0.92);
    border-radius: 14px;
    padding: 18px 22px;
    box-shadow: 0 12px 30px rgba(0,0,0,0.35);
    border: 1px solid rgba(255,255,255,0.04);
}
.metric-card > div {
    background: rgba(8, 12, 24, 0.9) !important;
    border-radius: 12px !important;
}
.mascot-box {
    border-radius: 16px;
    padding: 14px 18px;
    margin-top: 8px;
    background: linear-gradient(120deg, #1b4332, #081c15);
    color: #f8fff4;
    border: 1px solid rgba(163, 255, 163, 0.3);
}
.mascot-box.sad {
    background: linear-gradient(120deg, #641220, #2b050b);
    border-color: rgba(255, 163, 163, 0.4);
}
.mascot-box.neutral {
    background: linear-gradient(120deg, #22333b, #11151c);
    border-color: rgba(200, 200, 200, 0.3);
}
.mascot-title {
    font-weight: 700;
    font-size: 1.05rem;
}
.mascot-emoji {
    font-size: 1.6rem;
    margin-right: 6px;
}
.goal-pill {
    display: inline-block;
    padding: 4px 9px;
    margin: 2px 4px 2px 0;
    border-radius: 999px;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.18);
    font-size: 0.8rem;
}
.priority-label {
    font-size: 0.8rem;
    opacity: 0.85;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------- Helpers for compatibility ----------

def get_query_param(name: str, default=None):
    """Safe wrapper around query params for different Streamlit versions."""
    try:
        # Newer versions
        params = st.query_params
    except AttributeError:
        # Older versions
        params = st.experimental_get_query_params()
    if isinstance(params, dict):
        val = params.get(name, default)
        if isinstance(val, list):
            return val[0] if val else default
        return val
    return default

def get_openrouter_key():
    """Try env var first, then st.secrets; never crash if not present."""
    env_key = os.getenv("OPENROUTER_API_KEY")
    if env_key:
        return env_key
    try:
        return st.secrets["OPENROUTER_API_KEY"]
    except Exception:
        return None

# ---------- URL query param (risk used only as seed if present) ----------
RISK_OPTIONS = ["Low", "Medium", "High"]
risk_hint = get_query_param("risk", "Medium") or "Medium"
risk_hint = str(risk_hint).title()
if risk_hint not in RISK_OPTIONS:
    risk_hint = "Medium"

# ---------- Financial helpers ----------

def fv_lump_sum(P, r, n):
    return P * ((1 + r) ** n)

def fv_annuity(PMT, r, n):
    if r == 0:
        return PMT * n
    return PMT * (((1 + r) ** n - 1) / r)

def annual_to_monthly_rate(annual_rate):
    return (1 + annual_rate) ** (1 / 12) - 1

def discount_to_present(future_value, inflation_rate, years):
    return future_value / ((1 + inflation_rate) ** years)

def project_corpus(current_savings, monthly_invest, annual_return, years):
    m = int(years * 12)
    rm = annual_to_monthly_rate(float(annual_return))
    fv_current = fv_lump_sum(float(current_savings), rm, m)
    fv_stream = fv_annuity(float(monthly_invest), rm, m)
    return fv_current + fv_stream

def project_fd_corpus(principal, annual_rate, years):
    return principal * ((1 + annual_rate) ** years)

def inflation_adjust_series(values, inflation_rate, years_list):
    adj = []
    for v, yr in zip(values, years_list):
        adj.append(discount_to_present(v, inflation_rate, yr))
    return adj

def suggest_allocation_from_risk(risk_level):
    if risk_level == "Low":
        return {"Equity": 0.20, "Debt": 0.65, "Gold": 0.10, "Cash": 0.05}
    if risk_level == "High":
        return {"Equity": 0.75, "Debt": 0.15, "Gold": 0.05, "Cash": 0.05}
    return {"Equity": 0.55, "Debt": 0.30, "Gold": 0.10, "Cash": 0.05}

def expected_return_from_risk(risk_level):
    return {"Low": 0.07, "Medium": 0.11, "High": 0.14}[risk_level]

def infer_risk(age, horizon, goals, monthly_invest, income):
    savings_rate = 0 if income <= 0 else monthly_invest / max(income, 1)
    long_horizon = horizon >= 15
    many_growth_goals = any(g in goals for g in ["Wealth Creation", "Retirement"])
    score = 0
    if age < 35:
        score += 2
    elif age <= 50:
        score += 1
    if long_horizon:
        score += 2
    if savings_rate >= 0.3:
        score += 1
    if many_growth_goals:
        score += 1
    if score <= 2:
        return "Low", score
    if score <= 4:
        return "Medium", score
    return "High", score

def normalize_allocation(raw_alloc, default_alloc):
    keys = ["Equity", "Debt", "Gold", "Cash"]
    out = {}
    for k in keys:
        try:
            out[k] = float(raw_alloc.get(k, default_alloc.get(k, 0.0)))
        except Exception:
            out[k] = float(default_alloc.get(k, 0.0))
    s = sum(out.values())
    if s <= 0:
        return default_alloc
    return {k: v / s for k, v in out.items()}

# ---------- Mascot logic ----------

@dataclass
class MascotState:
    name: str          # "Wealth Monster" or "Money Monster"
    mood: str          # "happy" | "neutral" | "sad"
    headline: str
    detail: str

def pick_mascot(portfolio_cagr, fd_cagr, risk_level, portfolio_future, fd_future, inflation_adj_future):
    strength_ratio = portfolio_future / max(fd_future, 1)
    real_growth_ratio = inflation_adj_future / max(fd_future, 1)

    if portfolio_cagr >= 0.12 and strength_ratio >= 1.3 and real_growth_ratio >= 1.1:
        mood = "happy"
        headline = "Long‑term wealth path looks powerful."
        detail = "Your growth beats conservative FDs after inflation, with risk calibrated for higher upside."
    elif portfolio_cagr <= 0.08 or strength_ratio <= 1.0:
        mood = "sad"
        headline = "Growth is too close to FD levels."
        detail = "Consider nudging equity exposure or tenure to avoid inflation quietly eroding your goals."
    else:
        mood = "neutral"
        headline = "Decent trajectory, but can be sharpened."
        detail = "Balancing growth and stability is working, though small tweaks may unlock better real returns."

    wealth_monster = MascotState(
        name="Wealth Monster",
        mood=mood,
        headline=headline,
        detail=detail,
    )

    if risk_level == "High" and portfolio_cagr > fd_cagr + 0.04:
        mm_mood = "happy"
        mm_head = "Risk-return balance looks bold but controlled."
        mm_detail = "Volatility risk is higher, yet expected reward meaningfully outpaces safe alternatives."
    elif risk_level == "Low" and portfolio_cagr < fd_cagr + 0.015:
        mm_mood = "sad"
        mm_head = "Too defensive relative to your horizon."
        mm_detail = "Safety is high, but inflation may still chip away at your purchasing power."
    else:
        mm_mood = "neutral"
        mm_head = "Risk is broadly aligned with your profile."
        mm_detail = "Stay disciplined with SIPs, rebalancing, and periodic reviews as life goals change."

    money_monster = MascotState(
        name="Money Monster",
        mood=mm_mood,
        headline=mm_head,
        detail=mm_detail,
    )
    return wealth_monster, money_monster

def render_mascot_box(state: MascotState):
    mood_class = {
        "happy": "",
        "sad": "sad",
        "neutral": "neutral",
    }.get(state.mood, "neutral")
    emoji = {"happy": "🟢", "sad": "🔴", "neutral": "🟡"}.get(state.mood, "🟡")
    st.markdown(
        f"""
        <div class="mascot-box {mood_class}">
          <div class="mascot-title"><span class="mascot-emoji">{emoji}</span>{state.name}</div>
          <div style="font-size:0.9rem;margin-top:4px;">{state.headline}</div>
          <div style="font-size:0.8rem;margin-top:2px;opacity:0.85;">{state.detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------- LLM call (safe) ----------

def call_llm_with_backoff(prompt, risk_level, alloc_seed, max_retries=3):
    api_key = get_openrouter_key()
    if not api_key:
        return {
            "advice": (
                "Maintain disciplined SIPs, keep emergency cash aside, and align your equity-debt mix "
                "with goal priority and time horizon. Use this only as a research preview."
            ),
            "expected_return": expected_return_from_risk(risk_level),
            "allocation": alloc_seed,
        }

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": "openrouter/auto",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a fiduciary financial advisor for Indian retail investors. "
                    "Return strictly valid JSON with keys: advice (string, <=120 words, research/preview only), "
                    "expected_return (number 0-1), allocation (object with keys Equity, Debt, Gold, Cash that sum to 1)."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.3,
        "seed": 11,
    }
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=60)
            if resp.status_code == 200:
                j = resp.json()
                content = j["choices"][0]["message"]["content"]
                return json.loads(content)
        except Exception:
            pass
        time.sleep((2 ** attempt) + (0.1 * attempt))
    return {
        "advice": (
            "Use diversified equity and debt funds, map each goal to its horizon, "
            "and review allocations once a year. Research preview only."
        ),
        "expected_return": expected_return_from_risk(risk_level),
        "allocation": alloc_seed,
    }

# ---------- Layout: Landing + embedded app ----------

st.markdown(
    "<p style='font-size:0.9rem; opacity:0.8; margin-bottom:0.5rem;'>"
    "Enter your details below to see personalized, inflation‑aware investment projections and allocation suggestions."
    "</p>",
    unsafe_allow_html=True,
)

tab_main, tab_compare = st.tabs(["Main Advisor", "Inflation & Comparison"])

# ---------- Main tab ----------
with tab_main:
    st.markdown('<div class="block-card">', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
        horizon = st.number_input("Investment Horizon (years)", min_value=1, max_value=60, value=15, step=1)
    with col2:
        income = st.number_input("Monthly Income (₹)", min_value=0, value=100000, step=1000)
        monthly_invest = st.number_input("Monthly Investment (₹)", min_value=0, value=20000, step=500)
    with col3:
        current_savings = st.number_input("Current Savings / Principal (₹)", min_value=0, value=0, step=5000)
        inflation_rate = st.number_input(
            "Expected Inflation (p.a., %)", min_value=0.0, max_value=15.0, value=5.0, step=0.5
        ) / 100.0

    st.markdown("---")

    all_goals = ["Wealth Creation", "Retirement", "Education", "Home Purchase", "Emergency Fund", "Other"]
    selected_goals = st.multiselect(
        "Select your investment goals (multi‑select). First selected is treated as highest priority.",
        options=all_goals,
        default=["Wealth Creation"],
        help="You can re‑order by removing and re‑adding; the earliest item is considered top priority.",
    )

    if selected_goals:
        st.markdown("<span class='priority-label'>Goal priority (left = highest):</span>", unsafe_allow_html=True)
        for idx, g in enumerate(selected_goals, start=1):
            st.markdown(f"<span class='goal-pill'>#{idx} · {g}</span>", unsafe_allow_html=True)
    else:
        st.info("Select at least one goal to run the advisor.")

    fd_rate = st.number_input(
        "Comparable FD rate (p.a., %)", min_value=1.0, max_value=12.0, value=7.0, step=0.25
    ) / 100.0

    submitted = st.button("Get Research Preview")

    st.markdown("</div>", unsafe_allow_html=True)

    if submitted and selected_goals:
        top_goal = selected_goals[0]

        inferred_risk, risk_score = infer_risk(
            age=age,
            horizon=horizon,
            goals=selected_goals,
            monthly_invest=monthly_invest,
            income=income,
        )

        base_alloc = suggest_allocation_from_risk(inferred_risk)
        base_exp_ret = expected_return_from_risk(inferred_risk)

        profile = {
            "age": age,
            "income": income,
            "horizon_years": horizon,
            "monthly_invest": monthly_invest,
            "current_savings": current_savings,
            "inflation_rate": inflation_rate,
            "fd_rate": fd_rate,
            "goals": selected_goals,
            "top_goal": top_goal,
            "inferred_risk": inferred_risk,
        }
        prompt = (
            "Create high‑level research‑only guidance for an Indian retail investor.\n"
            f"Inputs (JSON): {json.dumps(profile)}\n"
            "Focus on the top‑priority goal first. Return JSON: advice, expected_return, allocation{Equity,Debt,Gold,Cash}."
        )

        with st.spinner("Crunching numbers and asking the advisor (preview only)..."):
            llm = call_llm_with_backoff(prompt, risk_level=inferred_risk, alloc_seed=base_alloc)

        exp_ret = float(llm.get("expected_return", base_exp_ret))
        alloc = normalize_allocation(llm.get("allocation", {}), base_alloc)
        advice = str(llm.get("advice", ""))

        corpus_port = project_corpus(current_savings, monthly_invest, exp_ret, horizon)
        # simple FD proxy: same total contribution treated as lump sum (years 0 for extra compounding safety)
        corpus_fd = project_fd_corpus(current_savings + monthly_invest * 12 * horizon, fd_rate, 0)
        real_corpus_port = discount_to_present(corpus_port, inflation_rate, horizon)
        real_principal = (
            discount_to_present(current_savings, inflation_rate, horizon) if current_savings > 0 else 0.0
        )

        wealth_monster, money_monster = pick_mascot(
            portfolio_cagr=exp_ret,
            fd_cagr=fd_rate,
            risk_level=inferred_risk,
            portfolio_future=corpus_port,
            fd_future=corpus_fd,
            inflation_adj_future=real_corpus_port,
        )

        # ---------- KPI cards ----------
        st.markdown('<div class="block-card metric-card">', unsafe_allow_html=True)
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Expected CAGR (portfolio)", f"{exp_ret*100:.1f}%", f"Risk: {inferred_risk}")
        k2.metric("Future Value (Portfolio)", f"₹ {corpus_port:,.0f}", "Nominal")
        k3.metric("Future Value (FD proxy)", f"₹ {corpus_fd:,.0f}", None)
        k4.metric(
            "Inflation‑adjusted Portfolio",
            f"₹ {real_corpus_port:,.0f}",
            f"{inflation_rate*100:.1f}% inflation",
        )
        k5.metric("Inflation‑adjusted Principal", f"₹ {real_principal:,.0f}", None)
        st.markdown("</div>", unsafe_allow_html=True)

        # ---------- Mascots ----------
        m1, m2 = st.columns(2)
        with m1:
            render_mascot_box(wealth_monster)
        with m2:
            render_mascot_box(money_monster)

        # ---------- Advice ----------
        st.markdown('<div class="block-card">', unsafe_allow_html=True)
        st.subheader("Advisor Note (Research / Preview)")
        st.write(advice)
        st.markdown(
            "<span style='font-size:0.8rem;opacity:0.7;'>Not registered as investment advice; use to frame questions for a qualified advisor.</span>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # ---------- Allocation pie ----------
        st.markdown('<div class="block-card">', unsafe_allow_html=True)
        st.subheader("Suggested Allocation (Portfolio vs FD & Cash)")

        combined_labels = list(alloc.keys()) + ["Fixed Deposits"]
        combined_values = [v * 100 for v in alloc.values()] + [10.0]  # illustrative FD slice
        fig_pie = go.Figure(
            data=[
                go.Pie(
                    labels=combined_labels,
                    values=combined_values,
                    hole=0.45,
                    textinfo="label+percent",
                    textposition="inside",
                    hoverinfo="label+percent+value",
                )
            ]
        )
        fig_pie.update_layout(
            height=420,
            margin=dict(l=0, r=0, t=20, b=0),
            showlegend=True,
            uniformtext_minsize=10,
            uniformtext_mode="show",
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # ---------- Year‑by‑year growth ----------
        st.markdown('<div class="block-card">', unsafe_allow_html=True)
        st.subheader("Year‑by‑Year Projected Growth")

        years = list(range(1, horizon + 1))
        rm = annual_to_monthly_rate(exp_ret)
        balances_nominal = []
        bal = float(current_savings)
        for year in years:
            for _ in range(12):
                bal = bal * (1 + rm) + float(monthly_invest)
            balances_nominal.append(bal)
        balances_real = inflation_adjust_series(balances_nominal, inflation_rate, years)

        line = go.Figure()
        line.add_trace(
            go.Scatter(x=years, y=balances_nominal, mode="lines+markers", name="Portfolio (nominal)")
        )
        line.add_trace(
            go.Scatter(
                x=years, y=balances_real, mode="lines+markers", name="Portfolio (inflation‑adjusted)"
            )
        )
        line.update_layout(
            xaxis_title="Year",
            yaxis_title="Corpus (₹)",
            height=420,
            margin=dict(l=0, r=0, t=20, b=0),
        )
        st.plotly_chart(line, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # ---------- Mutual fund suggestions ----------
        st.markdown('<div class="block-card">', unsafe_allow_html=True)
        st.subheader("Mutual Fund Shortlist (For Research / Preview Only)")

        risk_bucket = inferred_risk
        mf_universe = [
            {"name": "Index Wealth Builder Fund", "category": "Equity Index", "risk": "High", "five_year_cagr": 0.15},
            {"name": "Balanced Stability Fund", "category": "Aggressive Hybrid", "risk": "Medium", "five_year_cagr": 0.12},
            {"name": "Debt Shield Fund", "category": "Short Duration Debt", "risk": "Low", "five_year_cagr": 0.08},
            {"name": "FlexiCap Opportunity Fund", "category": "Flexi Cap Equity", "risk": "High", "five_year_cagr": 0.14},
            {"name": "Conservative Income Fund", "category": "Conservative Hybrid", "risk": "Low", "five_year_cagr": 0.09},
        ]

        filtered = [
            mf for mf in mf_universe if (mf["risk"] == risk_bucket or risk_bucket == "Medium")
        ]
        filtered = sorted(filtered, key=lambda x: x["five_year_cagr"], reverse=True)[:3]

        st.markdown(
            "<span style='font-size:0.8rem;opacity:0.7;'>Illustrative names and returns, ranked by past 5‑year CAGR. "
            "Use as a research starting point only; not a recommendation.</span>",
            unsafe_allow_html=True,
        )
        for mf in filtered:
            st.markdown(
                f"- **{mf['name']}** · {mf['category']} · "
                f"Risk: {mf['risk']} · 5‑yr CAGR ~ {mf['five_year_cagr']*100:.1f}%"
            )
        st.markdown("</div>", unsafe_allow_html=True)

        # ---------- Save context for comparison ----------
        st.session_state["latest_context"] = {
            "years": years,
            "balances_nominal": balances_nominal,
            "balances_real": balances_real,
            "fd_future": corpus_fd,
            "fd_real": discount_to_present(corpus_fd, inflation_rate, horizon),
            "port_future": corpus_port,
            "port_real": real_corpus_port,
            "inflation_rate": inflation_rate,
            "fd_rate": fd_rate,
            "exp_ret": exp_ret,
            "risk_level": inferred_risk,
            "wealth_monster": wealth_monster,
            "money_monster": money_monster,
            "selected_goals": selected_goals,
        }

# ---------- Comparison tab ----------
with tab_compare:
    st.markdown('<div class="block-card">', unsafe_allow_html=True)
    st.subheader("Inflation‑Adjusted vs Nominal Comparison")

    ctx = st.session_state.get("latest_context")
    if not ctx:
        st.info("Run the Main Advisor first to populate comparison data.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Nominal Portfolio FV", f"₹ {ctx['port_future']:,.0f}")
        c2.metric("Real Portfolio FV", f"₹ {ctx['port_real']:,.0f}")
        c3.metric("Nominal FD Proxy", f"₹ {ctx['fd_future']:,.0f}")
        c4.metric("Real FD Proxy", f"₹ {ctx['fd_real']:,.0f}")

        comp_fig = go.Figure()
        comp_fig.add_trace(
            go.Bar(name="Portfolio (nominal)", x=["Future Value"], y=[ctx["port_future"]])
        )
        comp_fig.add_trace(
            go.Bar(
                name="Portfolio (real, inflation‑adjusted)",
                x=["Future Value"],
                y=[ctx["port_real"]],
            )
        )
        comp_fig.add_trace(
            go.Bar(name="FD (nominal)", x=["Future Value"], y=[ctx["fd_future"]])
        )
        comp_fig.add_trace(
            go.Bar(
                name="FD (real, inflation‑adjusted)",
                x=["Future Value"],
                y=[ctx["fd_real"]],
            )
        )
        comp_fig.update_layout(
            barmode="group",
            yaxis_title="Corpus (₹)",
            height=420,
            margin=dict(l=0, r=0, t=20, b=0),
        )
        st.plotly_chart(comp_fig, use_container_width=True)

        st.markdown("#### Goal & Risk Summary")
        st.write(
            f"Top goal priority: **{ctx['selected_goals'][0]}** "
            f"with overall risk profile inferred as **{ctx['risk_level']}**."
        )

        st.markdown("#### Mascots React")
        mc1, mc2 = st.columns(2)
        with mc1:
            render_mascot_box(ctx["wealth_monster"])
        with mc2:
            render_mascot_box(ctx["money_monster"])

        st.markdown(
            "<span style='font-size:0.8rem;opacity:0.7;'>Inflation quietly reduces the real value of both FDs and portfolios. "
            "The gap between nominal and real bars above shows the impact.</span>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
