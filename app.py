import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


STYLE_FEATURES = [
    "slpm", "str_acc", "sapm", "str_def",
    "td_avg", "td_acc", "td_def", "sub_avg",
    "strike_aggression", "strike_efficiency",
    "grappling_aggression", "grappling_efficiency",
    "submission_threat", "striker_ratio"
]

STYLE_COLORS = {
    "Pure Striker":               "#e74c3c",
    "Balanced Grappler":          "#2ecc71",
    "Elite Wrestler":             "#3498db",
    "Submission Specialist":      "#9b59b6",
    "Counter/Defensive Fighter":  "#f39c12",
}


@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/fighters_clustered.csv")
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.drop_duplicates(subset=["name"])
    df = df.dropna(subset=STYLE_FEATURES + ["win_rate"])
    df = df.reset_index(drop=True)
    return df


@st.cache_resource
def train_model(df):
    fighters = df.copy()
    n = len(fighters)
    np.random.seed(42)
    n_pairs = min(20000, n * 10)
    matchups = []

    for _ in range(n_pairs):
        i, j = np.random.choice(n, 2, replace=False)
        a = fighters.iloc[i]
        b = fighters.iloc[j]
        row = {}
        for feat in STYLE_FEATURES:
            row[f"diff_{feat}"] = float(a[feat]) - float(b[feat])
        row["cluster_a"] = float(a["cluster"])
        row["cluster_b"] = float(b["cluster"])
        row["same_cluster"] = float(a["cluster"] == b["cluster"])
        row["experience_diff"] = float(a["total_record_fights"]) - float(b["total_record_fights"])
        row["label"] = int(float(a["win_rate"]) > float(b["win_rate"]))
        matchups.append(row)

    df_matchups = pd.DataFrame(matchups)
    feature_cols = [c for c in df_matchups.columns if c != "label"]
    X = df_matchups[feature_cols].values.astype(float)
    y = df_matchups["label"].values.astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)

    return model, scaler, feature_cols


def make_radar_chart(fighter_a, fighter_b, df):
    radar_features = ["slpm", "str_acc", "str_def",
                      "td_avg", "td_def", "sub_avg", "striker_ratio"]
    labels = ["Strike Output", "Strike Acc", "Strike Def",
              "Takedowns", "TD Defense", "Submissions", "Striker Ratio"]

    a = df[df["name"] == fighter_a].iloc[0]
    b = df[df["name"] == fighter_b].iloc[0]

    # normalize each feature to 0-1 across all fighters
    vals_a = []
    vals_b = []
    for feat in radar_features:
        col_min = df[feat].min()
        col_max = df[feat].max()
        if col_max > col_min:
            vals_a.append((float(a[feat]) - col_min) / (col_max - col_min))
            vals_b.append((float(b[feat]) - col_min) / (col_max - col_min))
        else:
            vals_a.append(0.0)
            vals_b.append(0.0)

    angles = np.linspace(0, 2 * np.pi, len(radar_features), endpoint=False).tolist()
    angles += angles[:1]
    vals_a += vals_a[:1]
    vals_b += vals_b[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    color_a = STYLE_COLORS.get(a["style"], "#e74c3c")
    color_b = STYLE_COLORS.get(b["style"], "#3498db")

    ax.plot(angles, vals_a, "o-", linewidth=2.5, color=color_a, label=fighter_a)
    ax.fill(angles, vals_a, alpha=0.15, color=color_a)
    ax.plot(angles, vals_b, "o--", linewidth=2.5, color=color_b, label=fighter_b,
            dashes=(5, 3))
    ax.fill(angles, vals_b, alpha=0.10, color=color_b)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    ax.set_title("Style Comparison", size=13, fontweight="bold", pad=20)

    plt.tight_layout()
    return fig


def predict_matchup(fighter_a, fighter_b, df, model, scaler, feature_cols):
    a = df[df["name"] == fighter_a].iloc[0]
    b = df[df["name"] == fighter_b].iloc[0]

    row = {}
    for feat in STYLE_FEATURES:
        row[f"diff_{feat}"] = float(a[feat]) - float(b[feat])
    row["cluster_a"] = float(a["cluster"])
    row["cluster_b"] = float(b["cluster"])
    row["same_cluster"] = float(a["cluster"] == b["cluster"])
    row["experience_diff"] = float(a["total_record_fights"]) - float(b["total_record_fights"])

    X_pred = pd.DataFrame([row])[feature_cols].values.astype(float)
    X_pred_scaled = scaler.transform(X_pred)

    prob = model.predict_proba(X_pred_scaled)[0]
    winner = fighter_a if prob[1] > 0.5 else fighter_b
    confidence = max(prob) * 100

    return winner, confidence, prob


def main():
    st.set_page_config(
        page_title="UFC Fight Predictor",
        page_icon="🥊",
        layout="wide"
    )

    st.title("🥊 UFC Fighter Style Analyzer & Fight Predictor")
    st.markdown("*Predicts fight outcomes based on fighter style profiles derived from career statistics*")

    # load data and train model
    with st.spinner("Loading fighter data..."):
        df = load_data()

    with st.spinner("Training prediction model..."):
        model, scaler, feature_cols = train_model(df)

    fighter_names = sorted(df["name"].tolist())

    # ── sidebar ───────────────────────────────────────────────────────────────
    st.sidebar.title("Select Fighters")
    fighter_a = st.sidebar.selectbox("Fighter A", fighter_names, index=fighter_names.index("Khabib Nurmagomedov") if "Khabib Nurmagomedov" in fighter_names else 0)
    fighter_b = st.sidebar.selectbox("Fighter B", fighter_names, index=fighter_names.index("Conor McGregor") if "Conor McGregor" in fighter_names else 1)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Model accuracy:** ~63%")
    st.sidebar.markdown("**Fighters in database:** 3,084")
    st.sidebar.markdown("**Style archetypes:** 5")

    if fighter_a == fighter_b:
        st.warning("Please select two different fighters.")
        return

    # ── fighter profiles ──────────────────────────────────────────────────────
    a = df[df["name"] == fighter_a].iloc[0]
    b = df[df["name"] == fighter_b].iloc[0]

    col1, col2, col3 = st.columns([2, 1, 2])

    with col1:
        st.subheader(fighter_a)
        style_color_a = STYLE_COLORS.get(a["style"], "#e74c3c")
        st.markdown(f"**Style:** :{style_color_a}[{a['style']}]")
        st.metric("Win Rate", f"{float(a['win_rate']):.1%}")
        st.metric("Strike Output (SLpM)", f"{float(a['slpm']):.2f}")
        st.metric("Takedown Avg", f"{float(a['td_avg']):.2f}")
        st.metric("Sub Avg", f"{float(a['sub_avg']):.2f}")
        st.metric("Total Fights", f"{int(a['total_record_fights'])}")

    with col2:
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        st.markdown("### VS", unsafe_allow_html=False)

    with col3:
        st.subheader(fighter_b)
        style_color_b = STYLE_COLORS.get(b["style"], "#3498db")
        st.markdown(f"**Style:** :{style_color_b}[{b['style']}]")
        st.metric("Win Rate", f"{float(b['win_rate']):.1%}")
        st.metric("Strike Output (SLpM)", f"{float(b['slpm']):.2f}")
        st.metric("Takedown Avg", f"{float(b['td_avg']):.2f}")
        st.metric("Sub Avg", f"{float(b['sub_avg']):.2f}")
        st.metric("Total Fights", f"{int(b['total_record_fights'])}")

    st.markdown("---")

    # ── prediction ────────────────────────────────────────────────────────────
    winner, confidence, prob = predict_matchup(
        fighter_a, fighter_b, df, model, scaler, feature_cols
    )

    st.subheader("🏆 Prediction")
    pred_col1, pred_col2 = st.columns(2)

    with pred_col1:
        st.success(f"**Predicted Winner: {winner}**")
        st.metric("Confidence", f"{confidence:.1f}%")

    with pred_col2:
        st.progress(prob[1], text=f"{fighter_a}: {prob[1]:.1%}")
        st.progress(prob[0], text=f"{fighter_b}: {prob[0]:.1%}")

    st.markdown("---")

    # ── radar chart ───────────────────────────────────────────────────────────
    st.subheader("📊 Style Comparison")
    fig = make_radar_chart(fighter_a, fighter_b, df)
    st.pyplot(fig)

    st.markdown("---")

    # ── style breakdown table ─────────────────────────────────────────────────
    st.subheader("📋 Detailed Stats Comparison")
    stats_display = {
        "Metric": ["Strike Output (SLpM)", "Strike Accuracy",
                   "Strikes Absorbed (SApM)", "Strike Defense",
                   "Takedown Avg", "Takedown Accuracy",
                   "Takedown Defense", "Submission Avg",
                   "Striker Ratio", "Win Rate"],
        fighter_a: [
            f"{float(a['slpm']):.2f}",
            f"{float(a['str_acc']):.1%}",
            f"{float(a['sapm']):.2f}",
            f"{float(a['str_def']):.1%}",
            f"{float(a['td_avg']):.2f}",
            f"{float(a['td_acc']):.1%}",
            f"{float(a['td_def']):.1%}",
            f"{float(a['sub_avg']):.2f}",
            f"{float(a['striker_ratio']):.1%}",
            f"{float(a['win_rate']):.1%}",
        ],
        fighter_b: [
            f"{float(b['slpm']):.2f}",
            f"{float(b['str_acc']):.1%}",
            f"{float(b['sapm']):.2f}",
            f"{float(b['str_def']):.1%}",
            f"{float(b['td_avg']):.2f}",
            f"{float(b['td_acc']):.1%}",
            f"{float(b['td_def']):.1%}",
            f"{float(b['sub_avg']):.2f}",
            f"{float(b['striker_ratio']):.1%}",
            f"{float(b['win_rate']):.1%}",
        ]
    }
    st.dataframe(pd.DataFrame(stats_display), use_container_width=True)

    st.markdown("---")
    st.caption("Built using UFC Stats data | Style clustering via KMeans | Prediction via Gradient Boosting | ~63% accuracy")


if __name__ == "__main__":
    main()

