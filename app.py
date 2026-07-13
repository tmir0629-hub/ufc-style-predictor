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
    "Pure Striker":               "#D4001A",
    "Balanced Grappler":          "#F5A623",
    "Elite Wrestler":             "#3a86ff",
    "Submission Specialist":      "#9b59b6",
    "Counter/Defensive Fighter":  "#2ecc71",
}

UFC5_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;600;700;800;900&family=Barlow:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    background-color: #080808 !important;
    color: #ffffff !important;
    font-family: 'Barlow', sans-serif !important;
}

.stApp {
    background-color: #080808 !important;
    background-image:
        radial-gradient(ellipse at top, #1a0005 0%, transparent 60%),
        repeating-linear-gradient(
            0deg,
            transparent,
            transparent 2px,
            rgba(255,255,255,0.01) 2px,
            rgba(255,255,255,0.01) 4px
        );
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

h1, h2, h3, .display {
    font-family: 'Barlow Condensed', sans-serif !important;
    text-transform: uppercase;
    letter-spacing: 2px;
}

/* fighter select dropdowns */
.stSelectbox > div > div {
    background-color: #111111 !important;
    border: 1px solid #D4001A !important;
    border-radius: 2px !important;
    color: #ffffff !important;
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 1.1rem !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* header bar */
.ufc-header {
    border-bottom: 2px solid #D4001A;
    padding: 24px 0 20px 0;
    margin-bottom: 32px;
    text-align: center;
}

.ufc-title {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 3.2rem;
    font-weight: 900;
    text-transform: uppercase;
    letter-spacing: 6px;
    color: #ffffff;
    line-height: 1;
}

.ufc-subtitle {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.85rem;
    font-weight: 600;
    color: #D4001A;
    letter-spacing: 4px;
    text-transform: uppercase;
    margin-top: 4px;
}

/* fighter card */
.fighter-card {
    background: linear-gradient(160deg, #141414 60%, #1a0005 100%);
    border: 1px solid #222222;
    border-top: 3px solid #D4001A;
    border-radius: 2px;
    padding: 24px 20px;
    position: relative;
    overflow: hidden;
}

.fighter-card::before {
    content: '';
    position: absolute;
    top: -40px;
    right: -40px;
    width: 120px;
    height: 120px;
    background: rgba(212, 0, 26, 0.04);
    transform: rotate(45deg);
}

.fighter-number {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 5rem;
    font-weight: 900;
    color: rgba(212, 0, 26, 0.08);
    position: absolute;
    top: -10px;
    right: 12px;
    line-height: 1;
    pointer-events: none;
}

.fighter-name-display {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 2rem;
    font-weight: 900;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #ffffff;
    line-height: 1.1;
    margin-bottom: 4px;
}

.fighter-style-tag {
    display: inline-block;
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 1px;
    margin-bottom: 20px;
}

.stat-bar-section {
    margin-top: 16px;
}

.stat-bar-row {
    margin-bottom: 10px;
}

.stat-bar-header {
    display: flex;
    justify-content: space-between;
    margin-bottom: 3px;
}

.stat-bar-label {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.72rem;
    font-weight: 600;
    color: #888888;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.stat-bar-value {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.72rem;
    font-weight: 700;
    color: #ffffff;
}

.stat-bar-track {
    background: #1e1e1e;
    border-radius: 1px;
    height: 5px;
    overflow: hidden;
}

.stat-bar-fill {
    height: 100%;
    border-radius: 1px;
    box-shadow: 0 0 6px currentColor;
}

/* VS divider */
.vs-block {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100%;
    padding-top: 48px;
}

.vs-word {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 4rem;
    font-weight: 900;
    color: #D4001A;
    letter-spacing: 4px;
    text-shadow:
        0 0 20px rgba(212,0,26,0.8),
        0 0 40px rgba(212,0,26,0.4);
    line-height: 1;
}

.vs-line {
    width: 2px;
    height: 60px;
    background: linear-gradient(to bottom, transparent, #D4001A, transparent);
    margin: 8px 0;
}

/* prediction section */
.prediction-panel {
    background: linear-gradient(135deg, #111111, #0d0005);
    border: 1px solid #222222;
    border-left: 4px solid #D4001A;
    border-radius: 2px;
    padding: 28px 32px;
    margin: 28px 0;
    position: relative;
    overflow: hidden;
}

.pred-label {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.75rem;
    font-weight: 700;
    color: #D4001A;
    letter-spacing: 4px;
    text-transform: uppercase;
    margin-bottom: 6px;
}

.pred-winner {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 2.8rem;
    font-weight: 900;
    text-transform: uppercase;
    letter-spacing: 3px;
    color: #ffffff;
    line-height: 1;
}

.pred-confidence {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.8rem;
    font-weight: 600;
    color: #888888;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 4px;
}

.prob-bar-track {
    background: #1e1e1e;
    border-radius: 1px;
    height: 8px;
    margin: 16px 0 6px 0;
    overflow: hidden;
    display: flex;
}

.prob-label-row {
    display: flex;
    justify-content: space-between;
}

.prob-name {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
}

/* section headers */
.section-eyebrow {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.72rem;
    font-weight: 700;
    color: #D4001A;
    letter-spacing: 4px;
    text-transform: uppercase;
    margin: 36px 0 4px 0;
}

.section-heading {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #ffffff;
    border-bottom: 1px solid #222222;
    padding-bottom: 10px;
    margin-bottom: 20px;
}

/* comparison stat rows */
.comp-row {
    display: flex;
    align-items: center;
    padding: 10px 0;
    border-bottom: 1px solid #1a1a1a;
    gap: 12px;
}

.comp-val {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    width: 80px;
    text-align: right;
}

.comp-val-b {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    width: 80px;
    text-align: left;
}

.comp-label {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.75rem;
    font-weight: 600;
    color: #888888;
    text-transform: uppercase;
    letter-spacing: 1px;
    flex: 1;
    text-align: center;
}

.comp-bar-track {
    width: 100px;
    background: #1e1e1e;
    height: 4px;
    border-radius: 1px;
    overflow: hidden;
}

/* footer */
.ufc-footer {
    text-align: center;
    padding: 32px 0 16px 0;
    border-top: 1px solid #1a1a1a;
    margin-top: 48px;
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.72rem;
    font-weight: 600;
    color: #333333;
    letter-spacing: 3px;
    text-transform: uppercase;
}
</style>
"""


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


def stat_bar_html(label, value, max_value, color="#D4001A", show_pct=False):
    pct = min((value / max_value) * 100, 100) if max_value > 0 else 0
    display = f"{value:.1%}" if show_pct else f"{value:.2f}"
    return f"""
    <div class="stat-bar-row">
        <div class="stat-bar-header">
            <span class="stat-bar-label">{label}</span>
            <span class="stat-bar-value">{display}</span>
        </div>
        <div class="stat-bar-track">
            <div class="stat-bar-fill"
                 style="width:{pct:.1f}%; background:{color};
                        box-shadow: 0 0 8px {color}88;">
            </div>
        </div>
    </div>
    """


def fighter_card_html(name, row, df, side="left"):
    color = STYLE_COLORS.get(row["style"], "#D4001A")
    number = "01" if side == "left" else "02"

    slpm_max   = float(df["slpm"].max())
    td_max     = float(df["td_avg"].max())
    sub_max    = float(df["sub_avg"].max())

    bars = (
        stat_bar_html("Strike Output (SLpM)", float(row["slpm"]),   slpm_max, color) +
        stat_bar_html("Strike Accuracy",      float(row["str_acc"]), 1.0,     color, show_pct=True) +
        stat_bar_html("Strike Defense",       float(row["str_def"]), 1.0,     color, show_pct=True) +
        stat_bar_html("Takedown Avg",         float(row["td_avg"]),  td_max,  color) +
        stat_bar_html("Takedown Defense",     float(row["td_def"]),  1.0,     color, show_pct=True) +
        stat_bar_html("Submission Avg",       float(row["sub_avg"]), sub_max, color)
    )

    wins   = int(row["wins"])   if "wins"   in row.index and not pd.isna(row["wins"])   else 0
    losses = int(row["losses"]) if "losses" in row.index and not pd.isna(row["losses"]) else 0
    draws  = int(row["draws"])  if "draws"  in row.index and not pd.isna(row["draws"])  else 0
    record = f"{wins}-{losses}-{draws}"
    return f"""

    <div class="fighter-card">
        <div class="fighter-number">{number}</div>
        <div class="fighter-name-display">{name}</div>
        <div class="fighter-style-tag"
             style="background:{color}22; color:{color};
                    border: 1px solid {color}55;">
            {row["style"]}
        </div>
        <div style="font-family:'Barlow Condensed',sans-serif;
                    font-size:0.8rem; color:#666666;
                    letter-spacing:2px; text-transform:uppercase;
                    margin-bottom:4px;">Record</div>
        <div style="font-family:'Barlow Condensed',sans-serif;
                    font-size:1.6rem; font-weight:900;
                    color:#ffffff; margin-bottom:20px;
                    letter-spacing:2px;">{record}</div>
        <div class="stat-bar-section">{bars}</div>
    </div>
    """


def make_radar_chart(fighter_a, fighter_b, df):
    radar_features = ["slpm", "str_acc", "str_def",
                      "td_avg", "td_def", "sub_avg", "striker_ratio"]
    labels = ["Strike\nOutput", "Strike\nAcc", "Strike\nDef",
              "Takedowns", "TD\nDefense", "Submissions", "Striker\nRatio"]

    a = df[df["name"] == fighter_a].iloc[0]
    b = df[df["name"] == fighter_b].iloc[0]

    vals_a, vals_b = [], []
    for feat in radar_features:
        col_min = df[feat].min()
        col_max = df[feat].max()
        if col_max > col_min:
            vals_a.append((float(a[feat]) - col_min) / (col_max - col_min))
            vals_b.append((float(b[feat]) - col_min) / (col_max - col_min))
        else:
            vals_a.append(0.0)
            vals_b.append(0.0)

    angles = np.linspace(0, 2 * np.pi, len(radar_features),
                         endpoint=False).tolist()
    angles += angles[:1]
    vals_a += vals_a[:1]
    vals_b += vals_b[:1]

    color_a = STYLE_COLORS.get(a["style"], "#D4001A")
    color_b = STYLE_COLORS.get(b["style"], "#3a86ff")

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("#111111")
    ax.set_facecolor("#111111")

    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels([])
    ax.yaxis.grid(True, color="#2a2a2a", linewidth=0.8)
    ax.xaxis.grid(True, color="#2a2a2a", linewidth=0.8)
    ax.spines["polar"].set_color("#2a2a2a")

    ax.plot(angles, vals_a, "o-", linewidth=2.5,
            color=color_a, label=fighter_a, zorder=3)
    ax.fill(angles, vals_a, alpha=0.20, color=color_a)

    ax.plot(angles, vals_b, "o--", linewidth=2.5,
            color=color_b, label=fighter_b, dashes=(5, 3), zorder=3)
    ax.fill(angles, vals_b, alpha=0.10, color=color_b)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=8, color="#aaaaaa",
                       fontfamily="sans-serif")

    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.4, 1.15),
        frameon=True,
        facecolor="#080808",
        edgecolor="#333333",
        labelcolor="#ffffff",
        fontsize=9
    )
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
    row["experience_diff"] = (float(a["total_record_fights"]) -
                              float(b["total_record_fights"]))
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
    st.markdown(UFC5_CSS, unsafe_allow_html=True)

    # header
    st.markdown("""
        <div class="ufc-header">
            <div class="ufc-title">UFC Fight Predictor</div>
            <div class="ufc-subtitle">Style-Based Matchup Analysis</div>
        </div>
    """, unsafe_allow_html=True)

    with st.spinner("Loading fighters..."):
        df = load_data()
    with st.spinner("Training model..."):
        model, scaler, feature_cols = train_model(df)

    fighter_names = sorted(df["name"].tolist())

    # fighter selectors
    sel1, sel2, sel3 = st.columns([5, 1, 5])
    with sel1:
        fighter_a = st.selectbox(
            "Fighter A", fighter_names,
            index=fighter_names.index("Khabib Nurmagomedov")
            if "Khabib Nurmagomedov" in fighter_names else 0,
            label_visibility="collapsed"
        )
    with sel2:
        st.markdown(
            "<div style='text-align:center; padding-top:6px; "
            "font-family:Barlow Condensed,sans-serif; font-weight:900; "
            "font-size:1.2rem; color:#D4001A; letter-spacing:3px;'>VS</div>",
            unsafe_allow_html=True
        )
    with sel3:
        fighter_b = st.selectbox(
            "Fighter B", fighter_names,
            index=fighter_names.index("Conor McGregor")
            if "Conor McGregor" in fighter_names else 1,
            label_visibility="collapsed"
        )

    if fighter_a == fighter_b:
        st.warning("Select two different fighters.")
        return

    a = df[df["name"] == fighter_a].iloc[0]
    b = df[df["name"] == fighter_b].iloc[0]

    # fighter cards
    st.markdown("<div style='margin-top:20px;'></div>",
                unsafe_allow_html=True)
    card1, card2 = st.columns(2)
    with card1:
        st.markdown(fighter_card_html(fighter_a, a, df, "left"),
                    unsafe_allow_html=True)
    with card2:
        st.markdown(fighter_card_html(fighter_b, b, df, "right"),
                    unsafe_allow_html=True)

    st.markdown("""
        <div style="display:flex; align-items:center; margin: 20px 0;">
            <div style="flex:1; height:1px; background:linear-gradient(90deg, transparent, #D4001A);"></div>
            <div style="font-family:'Barlow Condensed',sans-serif; font-size:2rem;
                        font-weight:900; color:#D4001A; letter-spacing:6px;
                        padding: 0 24px;
                        text-shadow: 0 0 20px rgba(212,0,26,0.8);">VS</div>
            <div style="flex:1; height:1px; background:linear-gradient(90deg, #D4001A, transparent);"></div>
        </div>
    """, unsafe_allow_html=True)

    # prediction
    winner, confidence, prob = predict_matchup(
        fighter_a, fighter_b, df, model, scaler, feature_cols
    )
    prob_a = prob[1] * 100
    prob_b = prob[0] * 100

    st.markdown("""
        <div class="section-eyebrow">Model Output</div>
        <div class="section-heading">Prediction</div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
        <div class="prediction-panel">
            <div class="pred-label">Predicted Winner</div>
            <div class="pred-winner">{winner}</div>
            <div class="pred-confidence">{confidence:.1f}% confidence</div>
            <div style='display:flex; justify-content:space-between;
                        margin-top:20px; margin-bottom:5px;'>
                <span class="prob-name" style="color:#D4001A;">{fighter_a}</span>
                <span class="prob-name" style="color:#3a86ff;">{fighter_b}</span>
            </div>
            <div class="prob-bar-track">
                <div style="width:{prob_a:.1f}%;
                            background:linear-gradient(90deg,#D4001A,#ff4444);
                            height:100%; box-shadow:0 0 8px #D4001A88;">
                </div>
                <div style="width:{prob_b:.1f}%;
                            background:linear-gradient(90deg,#2255cc,#3a86ff);
                            height:100%; box-shadow:0 0 8px #3a86ff88;">
                </div>
            </div>
            <div class="prob-label-row">
                <span style="font-family:'Barlow Condensed',sans-serif;
                             font-size:0.9rem; font-weight:700;
                             color:#D4001A;">{prob_a:.1f}%</span>
                <span style="font-family:'Barlow Condensed',sans-serif;
                             font-size:0.9rem; font-weight:700;
                             color:#3a86ff;">{prob_b:.1f}%</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # radar chart
    st.markdown("""
        <div class="section-eyebrow">Visualization</div>
        <div class="section-heading">Style Comparison</div>
    """, unsafe_allow_html=True)

    r1, r2, r3 = st.columns([1, 4, 1])
    with r2:
        fig = make_radar_chart(fighter_a, fighter_b, df)
        st.pyplot(fig, use_container_width=True)

    # detailed stats comparison
    st.markdown("""
        <div class="section-eyebrow">Breakdown</div>
        <div class="section-heading">Detailed Stats</div>
    """, unsafe_allow_html=True)

    stats = [
        ("Strike Output (SLpM)", "slpm",         ".2f",  False),
        ("Strike Accuracy",      "str_acc",       ".1%",  True),
        ("Strikes Absorbed",     "sapm",          ".2f",  False),
        ("Strike Defense",       "str_def",       ".1%",  True),
        ("Takedown Avg",         "td_avg",        ".2f",  False),
        ("Takedown Accuracy",    "td_acc",        ".1%",  True),
        ("Takedown Defense",     "td_def",        ".1%",  True),
        ("Submission Avg",       "sub_avg",       ".2f",  False),
        ("Striker Ratio",        "striker_ratio", ".1%",  True),
        ("Win Rate",             "win_rate",      ".1%",  True),
    ]

    header_html = f"""
        <div style="display:flex; align-items:center; padding:8px 0 12px 0;
                    border-bottom:1px solid #D4001A; margin-bottom:4px;">
            <div style="width:90px; text-align:right;
                        font-family:'Barlow Condensed',sans-serif;
                        font-size:0.9rem; font-weight:800;
                        color:#D4001A; text-transform:uppercase;
                        letter-spacing:1px;">{fighter_a}</div>
            <div style="flex:1; text-align:center;
                        font-family:'Barlow Condensed',sans-serif;
                        font-size:0.7rem; color:#555555;
                        text-transform:uppercase; letter-spacing:2px;">Stat</div>
            <div style="width:90px; text-align:left;
                        font-family:'Barlow Condensed',sans-serif;
                        font-size:0.9rem; font-weight:800;
                        color:#3a86ff; text-transform:uppercase;
                        letter-spacing:1px;">{fighter_b}</div>
        </div>
    """

    rows_html = ""
    for label, col, fmt, is_pct in stats:
        val_a = float(a[col])
        val_b = float(b[col])
        fa = format(val_a, fmt)
        fb = format(val_b, fmt)
        highlight_a = "#ffffff"
        highlight_b = "#ffffff"
        if val_a > val_b:
            highlight_a = "#F5A623"
        elif val_b > val_a:
            highlight_b = "#F5A623"

        col_max = float(df[col].max())
        bar_a = min((val_a / col_max) * 100, 100) if col_max > 0 else 0
        bar_b = min((val_b / col_max) * 100, 100) if col_max > 0 else 0

        rows_html += f"""
        <div class="comp-row">
            <div style="width:90px; text-align:right;">
                <span style="font-family:'Barlow Condensed',sans-serif;
                             font-size:1rem; font-weight:700;
                             color:{highlight_a};">{fa}</span>
            </div>
            <div style="width:80px; padding-right:6px;">
                <div class="comp-bar-track" style="margin-left:auto;">
                    <div style="width:{bar_a:.0f}%; height:100%;
                                background:#D4001A;
                                box-shadow:0 0 4px #D4001A88;
                                margin-left:auto;
                                float:right;"></div>
                </div>
            </div>
            <div class="comp-label">{label}</div>
            <div style="width:80px; padding-left:6px;">
                <div class="comp-bar-track">
                    <div style="width:{bar_b:.0f}%; height:100%;
                                background:#3a86ff;
                                box-shadow:0 0 4px #3a86ff88;"></div>
                </div>
            </div>
            <div style="width:90px; text-align:left;">
                <span style="font-family:'Barlow Condensed',sans-serif;
                             font-size:1rem; font-weight:700;
                             color:{highlight_b};">{fb}</span>
            </div>
        </div>
        """

    st.markdown(f"""
        <div style="background:#111111; border:1px solid #1e1e1e;
                    border-radius:2px; padding:20px 24px;">
            {header_html}
            {rows_html}
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="ufc-footer">
            UFC Fight Predictor &nbsp;·&nbsp;
            Style-Based ML Model &nbsp;·&nbsp;
            Data from UFCStats.com &nbsp;·&nbsp;
            ~63% Accuracy
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

