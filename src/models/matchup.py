import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
import os
warnings.filterwarnings("ignore")


STYLE_FEATURES = [
    "slpm", "str_acc", "sapm", "str_def",
    "td_avg", "td_acc", "td_def", "sub_avg",
    "strike_aggression", "strike_efficiency",
    "grappling_aggression", "grappling_efficiency",
    "submission_threat", "striker_ratio"
]


def load_data():
    df = pd.read_csv("data/processed/fighters_clustered.csv")
    print(f"Loaded {len(df)} fighters")
    return df


def build_matchup_dataset(df):
    print("Building matchup dataset...")
    keep = ["name", "cluster", "style", "win_rate", "total_record_fights"] + STYLE_FEATURES
    fighters = df[keep].copy()
    fighters = fighters[fighters["total_record_fights"] >= 5]
    fighters = fighters.drop_duplicates(subset=["name"])
    fighters = fighters.dropna(subset=STYLE_FEATURES + ["win_rate"])
    fighters = fighters.reset_index(drop=True)
    print(f"Fighters ready: {len(fighters)}")

    np.random.seed(42)
    n = len(fighters)
    n_pairs = min(20000, n * 10)
    matchups = []

    for _ in range(n_pairs):
        i, j = np.random.choice(n, 2, replace=False)
        a = fighters.iloc[i].copy()
        b = fighters.iloc[j].copy()
        a_winrate = float(fighters.at[fighters.index[i], "win_rate"])
        b_winrate = float(fighters.at[fighters.index[j], "win_rate"])
        row = {}
        for feat in STYLE_FEATURES:
            row[f"diff_{feat}"] = a[feat] - b[feat]
        row["cluster_a"] = a["cluster"]
        row["cluster_b"] = b["cluster"]
        row["same_cluster"] = int(a["cluster"] == b["cluster"])
        row["experience_diff"] = a["total_record_fights"] - b["total_record_fights"]
        row["label"] = int(a_winrate > b_winrate)
        matchups.append(row)

    df_matchups = pd.DataFrame(matchups)
    print(f"Built {len(df_matchups)} matchup pairs")
    print(f"Class balance: {df_matchups['label'].mean():.3f}")
    return df_matchups


def train_models(df_matchups):
    feature_cols = [c for c in df_matchups.columns if c != "label"]
    X = df_matchups[feature_cols].values.astype(float)
    y = df_matchups["label"].values.astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    models = {
        "Logistic Regression":  LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest":        RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting":    GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\n=== MODEL COMPARISON (5-fold cross validation) ===")
    best_score = 0
    best_name = ""
    for name, model in models.items():
        scores = cross_val_score(model, X_scaled, y, cv=cv, scoring="accuracy")
        print(f"{name:25s} | mean={scores.mean():.4f} | std={scores.std():.4f}")
        if scores.mean() > best_score:
            best_score = scores.mean()
            best_name = name

    print(f"\nBest model: {best_name} ({best_score:.4f})")
    return models, X_scaled, y, scaler, feature_cols


def plot_feature_importance(X_scaled, y, feature_cols):
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_scaled, y)

    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1][:15]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        range(len(indices)),
        importances[indices][::-1],
        color="steelblue",
        alpha=0.8
    )
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_cols[i] for i in indices][::-1])
    ax.set_xlabel("Feature Importance")
    ax.set_title("Top Features for Fight Outcome Prediction")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig("outputs/figures/feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved outputs/figures/feature_importance.png")

    print("\nTop 10 features:")
    for i in indices[:10]:
        print(f"  {feature_cols[i]:35s} {importances[i]:.4f}")

    return rf


def predict_matchup(fighter_a, fighter_b, df, rf, scaler, feature_cols):
    a = df[df["name"] == fighter_a]
    b = df[df["name"] == fighter_b]

    if len(a) == 0:
        print(f"Not found: {fighter_a}")
        return
    if len(b) == 0:
        print(f"Not found: {fighter_b}")
        return

    a = a.iloc[0]
    b = b.iloc[0]

    row = {}
    for feat in STYLE_FEATURES:
        row[f"diff_{feat}"] = a[feat] - b[feat]
    row["cluster_a"] = a["cluster"]
    row["cluster_b"] = b["cluster"]
    row["same_cluster"] = int(a["cluster"] == b["cluster"])
    row["experience_diff"] = a["total_record_fights"] - b["total_record_fights"]

    X_pred = pd.DataFrame([row])[feature_cols].values.astype(float)
    X_pred_scaled = scaler.transform(X_pred)

    prob = rf.predict_proba(X_pred_scaled)[0]
    winner = fighter_a if prob[1] > 0.5 else fighter_b
    confidence = max(prob) * 100

    print(f"\n{'='*50}")
    print(f"  {fighter_a} ({a['style']})")
    print(f"  vs")
    print(f"  {fighter_b} ({b['style']})")
    print(f"  Predicted winner: {winner}")
    print(f"  Confidence: {confidence:.1f}%")
    print(f"{'='*50}")


def main():
    os.makedirs("outputs/figures", exist_ok=True)

    df = load_data()
    df_matchups = build_matchup_dataset(df)
    models, X_scaled, y, scaler, feature_cols = train_models(df_matchups)
    rf = plot_feature_importance(X_scaled, y, feature_cols)

    print("\n=== SAMPLE MATCHUP PREDICTIONS ===")
    fights = [
        ("Khabib Nurmagomedov", "Conor McGregor"),
        ("Israel Adesanya", "Alex Pereira"),
        ("Jon Jones", "Francis Ngannou"),
        ("Georges St-Pierre", "Kamaru Usman"),
        ("Max Holloway", "Dustin Poirier"),
    ]
    for a, b in fights:
        predict_matchup(a, b, df, rf, scaler, feature_cols)


if __name__ == "__main__":
    main()