import pandas as pd
import numpy as np
import os


def engineer_style_features(df):
    """
    Create style ratio features from raw career statistics.
    These ratios capture HOW a fighter fights, not just how much.
    """
    print("Engineering style features...")
    feat = df.copy()

    # ── striking volume and aggression ────────────────────────────────────────
    # SLpM = significant strikes landed per minute (offensive output)
    # SApM = significant strikes absorbed per minute (how much they get hit)
    # Aggression ratio: how much do they dish out vs absorb?
    feat["strike_aggression"] = feat["slpm"] / (feat["slpm"] + feat["sapm"]).replace(0, np.nan)

    # Striking efficiency: accuracy × volume — rewards both precision and output
    feat["strike_efficiency"] = feat["slpm"] * feat["str_acc"]

    # Defensive efficiency: defense × inverse absorption
    feat["defensive_efficiency"] = feat["str_def"] / (feat["sapm"] + 1)

    # ── grappling profile ─────────────────────────────────────────────────────
    # Takedown aggression: how often do they attempt takedowns?
    feat["grappling_aggression"] = feat["td_avg"]

    # Takedown success when attempted
    feat["grappling_efficiency"] = feat["td_avg"] * feat["td_acc"]

    # Submission threat: average submission attempts per 15 minutes
    feat["submission_threat"] = feat["sub_avg"]

    # ── fighting style ratios ─────────────────────────────────────────────────
    # Striker vs Grappler ratio
    # High value = striker, low value = grappler
    total_activity = feat["slpm"] + feat["td_avg"] + feat["sub_avg"]
    feat["striker_ratio"] = feat["slpm"] / total_activity.replace(0, np.nan)
    feat["grappler_ratio"] = (feat["td_avg"] + feat["sub_avg"]) / total_activity.replace(0, np.nan)

    # ── finishing ability ─────────────────────────────────────────────────────
    # Win rate as a proxy for overall effectiveness
    feat["win_rate"] = feat["win_rate"].fillna(0)

    # ── experience ───────────────────────────────────────────────────────────
    feat["total_record_fights"] = feat["total_record_fights"].fillna(0)

    # ── physical features ─────────────────────────────────────────────────────
    # Normalize reach by height — reach advantage relative to body size
    feat["reach_to_height"] = feat["reach_inches"] / feat["height_inches"]

    # ── stance encoding ──────────────────────────────────────────────────────
    stance_map = {
        "Orthodox": 0,
        "Southpaw": 1,
        "Switch": 2,
        "Open Stance": 3,
        "Unknown": -1
    }
    feat["stance_encoded"] = feat["stance"].map(stance_map).fillna(-1)

    print("Style features engineered successfully")
    return feat


def select_modeling_features(df):
    """
    Select only the features we will use for clustering.
    Drop rows with too many missing values in key features.
    """
    # core style features for clustering
    style_features = [
        "slpm",              # striking output
        "str_acc",           # striking accuracy
        "sapm",              # strikes absorbed
        "str_def",           # strike defense
        "td_avg",            # takedown average
        "td_acc",            # takedown accuracy
        "td_def",            # takedown defense
        "sub_avg",           # submission attempts
        "strike_aggression", # offensive vs defensive ratio
        "strike_efficiency", # accuracy × volume
        "grappling_aggression",  # takedown frequency
        "grappling_efficiency",  # takedown success
        "submission_threat", # submission danger
        "striker_ratio",     # striking vs grappling balance
        "win_rate",          # overall effectiveness
    ]

    # keep identifier columns too
    id_cols = ["name", "url", "weight_lbs", "stance", "total_record_fights",
               "wins", "losses", "height_inches", "reach_inches", "age"]

    all_cols = id_cols + style_features
    available_cols = [c for c in all_cols if c in df.columns]
    df_model = df[available_cols].copy()

    # drop rows missing any of the core style features
    before = len(df_model)
    df_model = df_model.dropna(subset=style_features)
    after = len(df_model)
    print(f"Dropped {before - after} rows with missing style features")
    print(f"Modeling dataset: {after} fighters")

    return df_model, style_features


def main():
    os.makedirs("data/processed", exist_ok=True)

    print("Loading cleaned data...")
    df = pd.read_csv("data/processed/fighters_with_stats.csv")
    print(f"Loaded {len(df)} fighters with stats")

    # engineer features
    df_feat = engineer_style_features(df)

    # select modeling features
    df_model, style_features = select_modeling_features(df_feat)

    # save
    df_feat.to_csv("data/processed/fighters_engineered.csv", index=False)
    df_model.to_csv("data/processed/fighters_modeling.csv", index=False)

    print(f"\nSaved engineered dataset to data/processed/fighters_engineered.csv")
    print(f"Saved modeling dataset to data/processed/fighters_modeling.csv")

    print(f"\nStyle feature summary:")
    print(df_model[style_features].describe().round(3).to_string())

    print(f"\nSample fighters:")
    sample_names = ["Khabib Nurmagomedov", "Jon Jones", "Israel Adesanya",
                    "Conor McGregor", "Anderson Silva"]
    for name in sample_names:
        row = df_model[df_model["name"] == name]
        if len(row) > 0:
            print(f"\n{name}:")
            print(row[style_features].to_string())


if __name__ == "__main__":
    main()