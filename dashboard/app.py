from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from main import run_pipeline  # noqa: E402
from models.domain_mapper import map_domain_insight  # noqa: E402
from utils.preprocessing import RISK_WEIGHT_MAP  # noqa: E402

TITLE = "AI Risk Intelligence System"
DOMAINS = {
    "Public Health": "health_risk",
    "Environment": "climate_risk",
    "Education": "education_risk",
    "Policy": "policy_risk",
}


@st.cache_data
def load_predictions(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path)


def get_predictions() -> pd.DataFrame:
    predictions_path = BASE_DIR / "outputs" / "predictions.csv"
    required_cols = {
        "health_risk",
        "climate_risk",
        "education_risk",
        "policy_risk",
        "final_score",
        "explanation_text",
        "model_confidence",
        "income_norm",
        "pollution_norm",
        "education_level_norm",
        "hospital_access_norm",
    }
    if not predictions_path.exists():
        run_pipeline()
    mtime = predictions_path.stat().st_mtime if predictions_path.exists() else 0.0
    df = load_predictions(str(predictions_path), mtime)
    if not required_cols.issubset(df.columns):
        run_pipeline()
        mtime = predictions_path.stat().st_mtime if predictions_path.exists() else 0.0
        df = load_predictions(str(predictions_path), mtime)
    return df


def _compute_top_driver(df: pd.DataFrame, label: str) -> pd.Series:
    contributions = []
    for feature, weight, direction in RISK_WEIGHT_MAP[label]:
        values = df[f"{feature}_norm"].to_numpy(dtype=np.float32)
        if direction == "inverse":
            values = 1 - values
        contributions.append(weight * values)
    contributions = np.vstack(contributions).T
    top_indices = np.argmax(contributions, axis=1)
    top_features = [RISK_WEIGHT_MAP[label][idx][0] for idx in top_indices]
    return pd.Series(top_features)


def _recompute_risks(df: pd.DataFrame) -> pd.DataFrame:
    income = df["income_norm"].to_numpy(dtype=np.float32)
    pollution = df["pollution_norm"].to_numpy(dtype=np.float32)
    education = df["education_level_norm"].to_numpy(dtype=np.float32)
    hospital = df["hospital_access_norm"].to_numpy(dtype=np.float32)

    df = df.copy()
    df["health_risk"] = 0.5 * pollution + 0.3 * (1 - hospital) + 0.2 * (1 - income)
    df["climate_risk"] = 0.7 * pollution + 0.2 * (1 - education) + 0.1 * (1 - income)
    df["education_risk"] = 0.6 * (1 - education) + 0.3 * (1 - income) + 0.1 * (1 - hospital)
    df["policy_risk"] = 0.4 * (1 - income) + 0.4 * pollution + 0.2 * (1 - education)

    for label in ["health_risk", "climate_risk", "education_risk", "policy_risk"]:
        df[f"top_driver_{label}"] = _compute_top_driver(df, label)

    df["final_score"] = (
        0.3 * df["health_risk"]
        + 0.3 * df["climate_risk"]
        + 0.2 * df["education_risk"]
        + 0.2 * df["policy_risk"]
    ) * 100.0
    df["final_score"] = df["final_score"].clip(0.0, 100.0)

    insights = df.apply(lambda row: map_domain_insight(row), axis=1)
    df["explanation_text"] = insights.apply(lambda item: item[0])
    df["severity_level"] = insights.apply(lambda item: item[1])

    return df


def apply_simulation(df: pd.DataFrame, improvement_pct: int) -> pd.DataFrame:
    if improvement_pct <= 0:
        return df.copy()

    df_sim = df.copy()
    delta = (improvement_pct / 100.0) * 0.4
    df_sim["pollution_norm"] = np.clip(df_sim["pollution_norm"] - delta, 0.0, 1.0)
    df_sim["hospital_access_norm"] = np.clip(df_sim["hospital_access_norm"] + delta, 0.0, 1.0)

    df_sim["pollution"] = df_sim["pollution"] * (1 - 0.5 * improvement_pct / 100.0)
    df_sim["hospital_access"] = np.clip(
        df_sim["hospital_access"] * (1 + 0.4 * improvement_pct / 100.0), 0.0, 1.0
    )

    df_sim = _recompute_risks(df_sim)
    return df_sim


def is_california(label: str | None) -> bool:
    if not label:
        return False
    text = str(label).lower()
    return "california" in text or ", ca" in text or " ca," in text


def policy_impact_scenarios(df: pd.DataFrame, steps: list[int]) -> pd.DataFrame:
    records = []
    for step in steps:
        simulated = apply_simulation(df, step)
        records.append(
            {
                "Policy Improvement (%)": step,
                "Avg Health Risk": simulated["health_risk"].mean(),
                "Avg Climate Risk": simulated["climate_risk"].mean(),
                "Avg Education Risk": simulated["education_risk"].mean(),
                "Avg Policy Risk": simulated["policy_risk"].mean(),
                "Avg Final Score": simulated["final_score"].mean(),
            }
        )
    return pd.DataFrame(records)


def score_color(score: float) -> str:
    if score < 40:
        return "#16a34a"
    if score < 70:
        return "#f59e0b"
    return "#dc2626"


st.set_page_config(page_title=TITLE, layout="wide")

st.markdown(
    "<h1 style='text-align:center;'>🏛️ AI Risk Intelligence System</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center; color:#6b7280;'>Government AI Decision System</p>",
    unsafe_allow_html=True,
)

preds = get_predictions()

control_left, control_mid, control_right = st.columns([2, 1, 1])
with control_left:
    selected_domain = st.selectbox("Domain Focus", list(DOMAINS.keys()))
with control_mid:
    demo_mode = st.toggle("Demo Mode: ON", value=True)
with control_right:
    improvement = st.slider("Simulate improvement", 0, 100, 0)

sim_df = apply_simulation(preds, improvement)

if "region_name" in sim_df.columns:
    sim_df["region_display"] = (
        sim_df["region_name"].fillna("Unknown").astype(str)
        + " ("
        + sim_df["region_id"].astype(str)
        + ")"
    )
else:
    sim_df["region_display"] = sim_df["region_id"].astype(str)

if demo_mode:
    selected_row = sim_df.sort_values("final_score", ascending=False).iloc[0]
    st.info("Demo Mode active: highlighting the highest-risk region.")
else:
    region_display = st.selectbox("Select Region", sim_df["region_display"].tolist())
    selected_row = sim_df[sim_df["region_display"] == region_display].iloc[0]

risk_col = DOMAINS[selected_domain]

headline_color = score_color(float(selected_row["final_score"]))

headline, right_metrics = st.columns([2, 1])
with headline:
    st.markdown(
        f"""
        <div style="text-align:center; padding:18px; border-radius:14px; border:1px solid #e2e8f0; background:#f8fafc;">
          <div style="font-size:22px; color:#334155;">AI Risk Score</div>
          <div style="font-size:48px; font-weight:700; color:{headline_color};">{selected_row['final_score']:.1f} / 100</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with right_metrics:
    confidence_pct = float(selected_row.get("model_confidence", 0.0)) * 100
    st.metric("Model Confidence", f"{confidence_pct:.1f}%")
    st.metric("Severity", str(selected_row.get("severity_level", "Unknown")))

st.subheader("AI Insight")
if demo_mode:
    st.write(f"This region is at risk because: {selected_row['explanation_text']}")
else:
    st.write(selected_row["explanation_text"])

st.subheader("Top 3 Highest-Risk Regions")
top3 = sim_df.nlargest(3, "final_score")[
    ["region_display", "final_score", "severity_level", "explanation_text"]
]
st.dataframe(top3, width="stretch")

st.subheader("California vs Rest of US (Public Health Risk)")
if "region_name" in sim_df.columns:
    ca_mask = sim_df["region_name"].apply(is_california)
    if ca_mask.any() and (~ca_mask).any():
        ca_avg = sim_df.loc[ca_mask, "health_risk"].mean()
        us_avg = sim_df.loc[~ca_mask, "health_risk"].mean()
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(["California", "Rest of US"], [ca_avg, us_avg], color=["#3b82f6", "#f97316"])
        ax.set_ylabel("Avg Public Health Risk", fontsize=13)
        ax.set_title("California vs Rest of US (Public Health Risk)", fontsize=15)
        ax.tick_params(axis="x", labelsize=12)
        ax.tick_params(axis="y", labelsize=12)
        st.pyplot(fig)

        st.write("Top drivers in California")
        ca_drivers = sim_df.loc[ca_mask, "top_driver_health_risk"].value_counts()
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(ca_drivers.index, ca_drivers.values, color="#8b5cf6")
        ax.set_ylabel("Count", fontsize=13)
        ax.set_title("Top Drivers in California (Public Health)", fontsize=15)
        ax.tick_params(axis="x", rotation=30, labelsize=11)
        ax.tick_params(axis="y", labelsize=12)
        st.pyplot(fig)
    else:
        st.info("Not enough California vs non-California data to compare.")
else:
    st.info("Region names not available for state-level comparison.")

metrics_left, metrics_right = st.columns(2)
with metrics_left:
    avg_risk = float(sim_df[risk_col].mean())
    st.metric(f"Average {selected_domain} Risk", f"{avg_risk:.2f}")
    st.write(f"Total regions: {len(sim_df)}")

with metrics_right:
    st.write("Top Risk Drivers")
    driver_col = f"top_driver_{risk_col}"
    if driver_col in sim_df:
        driver_counts = sim_df[driver_col].value_counts().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(driver_counts.index, driver_counts.values, color="#22c55e")
        ax.set_ylabel("Count", fontsize=13)
        ax.set_title(f"Top Drivers for {selected_domain}", fontsize=15)
        ax.tick_params(axis="x", rotation=30, labelsize=11)
        ax.tick_params(axis="y", labelsize=12)
        st.pyplot(fig)
    else:
        st.write("Driver data unavailable.")

st.subheader("California Areas vs Nationwide (Public Health Risk)")
if "region_name" in sim_df.columns:
    ca_mask = sim_df["region_name"].apply(is_california)
    if ca_mask.any():
        ca_df = (
            sim_df.loc[ca_mask, ["region_name", "health_risk"]]
            .groupby("region_name")["health_risk"]
            .mean()
            .sort_values(ascending=False)
            .head(12)
        )
        ca_df.index = ca_df.index.str.replace(", United States", "", regex=False)
        us_avg = sim_df["health_risk"].mean()
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(ca_df.index, ca_df.values, color="#0ea5e9")
        ax.axhline(us_avg, color="#fb7185", linestyle="--", linewidth=2, label=f"US Avg {us_avg:.3f}")
        ax.set_ylabel("Avg Public Health Risk", fontsize=13)
        ax.set_title("California Areas vs US Average", fontsize=15)
        ax.tick_params(axis="x", rotation=35, labelsize=10)
        ax.tick_params(axis="y", labelsize=12)
        ax.legend()
        st.pyplot(fig)
    else:
        st.info("No California regions found.")
else:
    st.info("Region names not available for state-level comparison.")

st.subheader("Policy Improvement vs Public Health Risk")
policy_steps = list(range(1, 101))
policy_df = policy_impact_scenarios(preds, policy_steps)
policy_indexed = policy_df.set_index("Policy Improvement (%)")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(policy_indexed.index, policy_indexed["Avg Health Risk"], color="#ef4444", linewidth=2.5)
ax.set_xlabel("Policy Improvement (%)", fontsize=13)
ax.set_ylabel("Avg Public Health Risk", fontsize=13)
ax.set_title("Policy Improvement vs Public Health Risk", fontsize=15)
ax.tick_params(axis="x", labelsize=11)
ax.tick_params(axis="y", labelsize=12)
st.pyplot(fig)
start = policy_df["Avg Health Risk"].iloc[0]
end = policy_df["Avg Health Risk"].iloc[-1]
drop_pct = ((start - end) / start * 100) if start > 0 else 0.0
st.caption(
    f"From 1% to 100% policy improvement, average public health risk drops by {drop_pct:.1f}%."
)
st.success("This curve shows policy change can materially reduce public health risk nationwide.")

st.subheader("California Public Health Risk Heatmap by District and Month")
if "region_name" in sim_df.columns:
    ca_mask = sim_df["region_name"].apply(is_california)
else:
    ca_mask = pd.Series([False] * len(sim_df))

if ca_mask.any() and "created_at" in sim_df.columns:
    ca_df = sim_df.loc[ca_mask].copy()
    ca_df["created_at"] = pd.to_datetime(ca_df["created_at"], errors="coerce", utc=True)
    ca_df["created_at"] = ca_df["created_at"].dt.tz_convert(None)
    ca_df = ca_df.dropna(subset=["created_at"])
    ca_df["month"] = ca_df["created_at"].dt.to_period("M").dt.to_timestamp()
    if "locality" in ca_df.columns:
        district = ca_df["locality"]
    else:
        district = ca_df["region_name"].str.split(",").str[0]
    ca_df["district"] = district.fillna("Unknown District").astype(str)

    if not ca_df.empty:
        top_districts = ca_df["district"].value_counts().head(25).index.tolist()
        ca_df = ca_df[ca_df["district"].isin(top_districts)]
        pivot = ca_df.pivot_table(
            index="district",
            columns="month",
            values="health_risk",
            aggfunc="mean",
        )
        pivot = pivot.sort_index()
        if not pivot.empty:
            month_index = pd.date_range(pivot.columns.min(), pivot.columns.max(), freq="MS")
            pivot = pivot.reindex(month_index, axis=1)
            pivot = pivot.interpolate(axis=1, limit_direction="both")
            pivot = pivot.apply(lambda row: row.fillna(row.mean()), axis=1)
            overall_mean = float(ca_df["health_risk"].mean())
            pivot = pivot.fillna(overall_mean)
        heat = pivot.to_numpy()

        fig_height = max(6, len(pivot.index) * 0.35)
        fig, ax = plt.subplots(figsize=(12, fig_height))
        cmap = plt.cm.magma.copy()
        cmap.set_bad(color="#111827")
        img = ax.imshow(heat, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index.tolist(), fontsize=9)

        months = pivot.columns.tolist()
        tick_step = max(1, len(months) // 12)
        xticks = list(range(0, len(months), tick_step))
        xlabels = [months[i].strftime("%b %Y") for i in xticks]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, rotation=30, ha="right", fontsize=9)
        ax.set_xlabel("Month", fontsize=12)
        ax.set_ylabel("District", fontsize=12)
        ax.set_title("California Public Health Risk by District Over Time", fontsize=15)
        colorbar = fig.colorbar(img, ax=ax, fraction=0.02, pad=0.02)
        colorbar.set_label("Public Health Risk (0–1)")
        st.pyplot(fig)
        st.caption(
            "Rows are districts; columns are months. Colors reflect mean public health risk (0–1)."
        )
    else:
        st.info("No California records with timestamps available for the heatmap.")
else:
    st.info("California data or timestamps not available for a district-by-month heatmap.")

st.subheader("Sample Predictions")
columns = ["region_display", risk_col, "final_score", f"top_driver_{risk_col}", "cluster"]
columns = [col for col in columns if col in sim_df]
st.dataframe(sim_df[columns].head(20), width="stretch")
