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


/* Gold scrollbar */
* {
    scrollbar-width: thin;
    scrollbar-color: #d4af37 transparent;
}
::-webkit-scrollbar {
    width: 10px;
}
::-webkit-scrollbar-track {
    background: rgba(5, 6, 8, 0.9);
}
::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #f1c40f, #d4af37);
    border-radius: 10px;
    border: 1px solid rgba(0,0,0,0.4);
}
::-webkit-scrollbar-thumb:hover {
    background: #f1c40f;
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


/* Gold metric for inflation‑adjusted portfolio */
.gold-metric {
    background: linear-gradient(135deg, #f1c40f 0%, #d4af37 40%, #8a6d1d 100%);
    border-radius: 12px;
    padding: 10px 14px;
    color: #050608;
    box-shadow: 0 8px 22px rgba(0,0,0,0.45);
}
.gold-metric-label {
    font-size: 0.8rem;
    font-weight: 600;
    opacity: 0.95;
}
.gold-metric-value {
    font-size: 1.3rem;
    font-weight: 800;
    margin-top: 4px;
}
.gold-metric-subtext {
    font-size: 0.75rem;
    margin-top: 2px;
    opacity: 0.9;
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
    try:
        params = st.query_params
    except AttributeError:
        params = st.experimental_get_query_params()
    if isinstance(params, dict):
        val = params.get(name, default)
        if isinstance(val, list):
            return val[0] if val else default
        return val
    return default


def get_openrouter_key():
    env_key = os.getenv("OPENROUTER_API_KEY")
    if env_key:
        return env_key
    try:
        return st.secrets["OPENROUTER_API_KEY"]
    except Exception:
        return None


# ---------- URL query param (risk seed) ----------
RISK_OPTIONS = ["Low", "Medium", "High"]
risk_hint = get_query_param("risk", "Medium") or "Medium"
risk_hint = str(risk_hint).title()
if risk_hint not in RISK_OPTIONS:
    risk_hint = "Medium"


# ---------- Rule-based helpers (fallback only) ----------

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
    return [discount_to_present(v, inflation_rate, yr) for v, yr in zip(values, years_list)]


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


# ---------- LLM call: GenAI is primary engine ----------

FIXED_MODEL = "google/gemini-flash-1.5"


def call_llm_with_backoff(profile, max_retries=2):
    api_key = get_openrouter_key()
    if not api_key:
        return None

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    system_content = (
        "You are a fiduciary financial advisor AND quantitative planner for Indian retail investors. "
        "Use only the JSON profile provided by the user. "
        "Assume annual return and interest rates are compounded monthly. "
        "Use standard time-value-of-money formulas (future value of lump sum, future value of annuity, "
        "present value with inflation discounting). "
        "All outputs are research / preview only; never guarantee returns. "
        "Write a detailed, structured advisor note in plain text that clearly uses the user's inputs "
        "(age, goals, horizon, monthly investment, current savings, FD rate, inflation). "
        "Focus on risk profiling, suggested allocation, expected returns, and inflation impact. "
    )

    body = {
        "model": FIXED_MODEL,
        "messages": [
            {"role": "system", "content": system_content},
            {
                "role": "user",
                "content": (
                    "Here is the investor profile as JSON:\n"
                    f"{json.dumps(profile)}\n\n"
                    "Write only a detailed advisor note in plain text. "
                    "Do NOT return JSON, just human-readable advice."
                ),
            },
        ],
        "temperature": 0.4,
        "seed": 11,
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=45)
            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                return content
        except Exception:
            pass
        time.sleep((2 ** attempt) + 0.1 * attempt)

    return None


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

        rule_risk, risk_score = infer_risk(
            age=age,
            horizon=horizon,
            goals=selected_goals,
            monthly_invest=monthly_invest,
            income=income,
        )
        base_alloc = suggest_allocation_from_risk(rule_risk)
        base_exp_ret = expected_return_from_risk(rule_risk)

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
            "rule_based_risk": rule_risk,
            "rule_based_allocation": base_alloc,
            "rule_based_expected_return": base_exp_ret,
        }

        with st.spinner("Crunching numbers and asking the advisor (preview only)..."):
            llm_advice = call_llm_with_backoff(profile)

        # Numbers still from rules (since text mode can’t safely drive charts)
        inferred_risk = rule_risk
        exp_ret = base_exp_ret
        alloc = base_alloc

        if not llm_advice:
            advice = (
                f"Based on your age ({age}), monthly investment of roughly ₹{monthly_invest:,.0f}, "
                f"current savings of about ₹{current_savings:,.0f}, and a {horizon}-year horizon, "
                f"this preview assumes a {inferred_risk.lower()} risk profile with an expected CAGR "
                f"of around {exp_ret*100:.1f}% per year. Use this purely as research, not as a guarantee."
            )
        else:
            advice = llm_advice

        corpus_port = project_corpus(current_savings, monthly_invest, exp_ret, horizon)
        corpus_fd = project_fd_corpus(current_savings + monthly_invest * 12 * horizon, fd_rate, horizon)
        real_corpus_port = discount_to_present(corpus_port, inflation_rate, horizon)

        years = list(range(1, horizon + 1))
        balances_nominal = []
        rm = annual_to_monthly_rate(exp_ret)
        bal = float(current_savings)
        for year in years:
            for _ in range(12):
                bal = bal * (1 + rm) + float(monthly_invest)
            balances_nominal.append(bal)
        balances_real = inflation_adjust_series(balances_nominal, inflation_rate, years)

        mf_shortlist = []
        goal_plans = []
        risk_explanation = ""

        # ---------- KPI cards ----------
        st.markdown('<div class="block-card metric-card">', unsafe_allow_html=True)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Expected CAGR (portfolio)", f"{exp_ret*100:.1f}%", f"Risk: {inferred_risk}")
        k2.metric("Future Value (Portfolio)", f"₹ {corpus_port:,.0f}", "Before inflation")
        k3.metric("Future Value (FD proxy)", f"₹ {corpus_fd:,.0f}", None)

        with k4:
            st.markdown(
                f"""
                <div class="gold-metric">
                  <div class="gold-metric-label">Inflation‑adjusted Future Value – Portfolio</div>
                  <div class="gold-metric-value">₹ {real_corpus_port:,.0f}</div>
                  <div class="gold-metric-subtext">{inflation_rate*100:.1f}% inflation assumed</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

        # ---------- Risk explanation ----------
        if risk_explanation:
            st.markdown(
                f"<div class='block-card' style='font-size:0.8rem;opacity:0.8;'>"
                f"<strong>Why this risk level?</strong> {risk_explanation}"
                f"</div>",
                unsafe_allow_html=True,
            )

        # ---------- Advisor note ----------
        st.markdown('<div class="block-card">', unsafe_allow_html=True)
        st.subheader("Advisor Note (Research / Preview)")
        st.write(advice)
        st.markdown(
            "<span style='font-size:0.8rem;opacity:0.7;'>"
            "This is a research-oriented projection based on your inputs. "
            "Always cross‑check and consult a qualified advisor before making decisions."
            "</span>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # ---------- Goal‑wise plan ----------
        if goal_plans:
            st.markdown('<div class="block-card">', unsafe_allow_html=True)
            st.subheader("Goal‑wise Plan (GenAI‑based, research only)")
            for gp in sorted(goal_plans, key=lambda x: x.get("priority_rank", 999)):
                st.markdown(
                    f"- **#{gp.get('priority_rank', '?')} · {gp.get('goal', '')}** — "
                    f"Target future value: ₹ {gp.get('target_future_value', 0):,.0f} "
                    f"(inflation‑adjusted: ₹ {gp.get('target_inflation_adjusted', 0):,.0f}); "
                    f"suggested monthly investment: ₹ {gp.get('suggested_monthly_investment', 0):,.0f}"
                )
            st.markdown(
                "<span style='font-size:0.8rem;opacity:0.7;'>Targets are GenAI estimates, not guarantees. "
                "Use them to discuss with a human advisor.</span>",
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        # ---------- Allocation pie ----------
        st.markdown('<div class="block-card">', unsafe_allow_html=True)
        st.subheader("Suggested Allocation (Portfolio vs FD & Cash)")

        combined_labels = list(alloc.keys()) + ["Fixed Deposits"]
        combined_values = [v * 100 for v in alloc.values()] + [10.0]
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

        line = go.Figure()
        line.add_trace(
            go.Scatter(x=years, y=balances_nominal, mode="lines+markers", name="Portfolio (future value)")
        )
        line.add_trace(
            go.Scatter(
                x=years,
                y=balances_real,
                mode="lines+markers",
                name="Portfolio (inflation‑adjusted future value)",
            )
        )
        line.update_layout(
            xaxis_title="Year",
            yaxis_title="Corpus (₹)",
            height=420,
            margin=dict(l=0, r=0, t=20, b=0),
        )
        st.plotly_chart(line, use_container_width=True)

        st.markdown(
            "<span style='font-size:0.8rem;opacity:0.7;'>"
            "Inflation quietly reduces the real value of both FD returns and portfolio returns over time. "
            "The gap between the future value and inflation‑adjusted lines above shows this impact."
            "</span>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # ---------- Mutual fund suggestions ----------
        st.markdown('<div class="block-card">', unsafe_allow_html=True)
        st.subheader("Mutual Fund Shortlist (For Research / Preview Only)")

        if mf_shortlist:
            for mf in mf_shortlist[:3]:
                st.markdown(
                    f"- **{mf.get('name','')}** · {mf.get('category','')} · "
                    f"Risk: {mf.get('risk','')} · 5‑year CAGR ~ {mf.get('five_year_cagr',0)*100:.1f}% "
                    f"— {mf.get('reason','')}"
                )
        else:
            fallback_mfs = [
                {"name": "Index Wealth Builder Fund", "category": "Equity Index", "risk": "High", "five_year_cagr": 0.15},
                {"name": "Balanced Stability Fund", "category": "Aggressive Hybrid", "risk": "Medium", "five_year_cagr": 0.12},
                {"name": "Debt Shield Fund", "category": "Short Duration Debt", "risk": "Low", "five_year_cagr": 0.08},
            ]
            for mf in fallback_mfs:
                st.markdown(
                    f"- **{mf['name']}** · {mf['category']} · "
                    f"Risk: {mf['risk']} · 5‑year CAGR ~ {mf['five_year_cagr']*100:.1f}%"
                )

        st.markdown(
            "<span style='font-size:0.8rem;opacity:0.7;'>Illustrative only; not a recommendation. "
            "Always cross‑check factsheets and consult a qualified advisor.</span>",
            unsafe_allow_html=True,
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
            "selected_goals": selected_goals,
        }


# ---------- Comparison tab ----------
with tab_compare:
    st.markdown('<div class="block-card">', unsafe_allow_html=True)
    st.subheader("Inflation‑Adjusted vs Future Value")

    ctx = st.session_state.get("latest_context")
    if not ctx:
        st.info("Run the Main Advisor first to populate comparison data.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Future Value – Portfolio", f"₹ {ctx['port_future']:,.0f}")
        c2.metric("Inflation‑adjusted Future Value – Portfolio", f"₹ {ctx['port_real']:,.0f}")
        c3.metric("Future Value – FD proxy", f"₹ {ctx['fd_future']:,.0f}")
        c4.metric("Inflation‑adjusted Future Value – FD proxy", f"₹ {ctx['fd_real']:,.0f}")

        comp_fig = go.Figure()
        comp_fig.add_trace(
            go.Bar(name="Portfolio – Future Value", x=["Future Value"], y=[ctx["port_future"]])
        )
        comp_fig.add_trace(
            go.Bar(
                name="Portfolio – Inflation‑adjusted Future Value",
                x=["Future Value"],
                y=[ctx["port_real"]],
            )
        )
        comp_fig.add_trace(
            go.Bar(name="FD – Future Value", x=["Future Value"], y=[ctx["fd_future"]])
        )
        comp_fig.add_trace(
            go.Bar(
                name="FD – Inflation‑adjusted Future Value",
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

        st.markdown(
            "<span style='font-size:0.8rem;opacity:0.7;'>"
            "Inflation quietly reduces the real value of both FDs and portfolios. "
            "The gap between the future value and inflation‑adjusted bars above shows this impact."
            "</span>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
