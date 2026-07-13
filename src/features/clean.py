import pandas as pd
import numpy as np
import os

def parse_height(val):
    """Convert '6' 0"' to inches as a float. Returns NaN if missing."""
    if val == "--" or pd.isna(val):
        return np.nan
    try:
        val = val.replace('"', '').strip()
        parts = val.split("'")
        feet = int(parts[0].strip())
        inches = int(parts[1].strip()) if parts[1].strip() else 0
        return feet * 12 + inches
    except:
        return np.nan


def parse_weight(val):
    """Convert '155 lbs.' to float. Returns NaN if missing."""
    if val == "--" or pd.isna(val):
        return np.nan
    try:
        return float(val.replace("lbs.", "").strip())
    except:
        return np.nan


def parse_reach(val):
    """Convert '72"' to float. Returns NaN if missing."""
    if val == "--" or pd.isna(val):
        return np.nan
    try:
        return float(val.replace('"', '').strip())
    except:
        return np.nan


def parse_record(val):
    """Convert '10-15-0' to separate wins, losses, draws columns."""
    try:
        parts = val.strip().split("-")
        wins = int(parts[0])
        losses = int(parts[1])
        draws = int(parts[2]) if len(parts) > 2 else 0
        return wins, losses, draws
    except:
        return np.nan, np.nan, np.nan


def parse_percentage(val):
    """Convert '38%' to 0.38 as float. Returns NaN if missing."""
    if val == "--" or pd.isna(val):
        return np.nan
    try:
        return float(str(val).replace("%", "").strip()) / 100
    except:
        return np.nan


def parse_age(dob_str):
    """Calculate age from date of birth string."""
    if not dob_str or dob_str == "--" or pd.isna(dob_str):
        return np.nan
    try:
        dob = pd.to_datetime(dob_str)
        today = pd.Timestamp.today()
        return (today - dob).days / 365.25
    except:
        return np.nan


def flag_no_stats(row):
    """Flag fighters who have all zero stats - means no recorded data."""
    return (row["slpm"] == 0.0 and
            row["sapm"] == 0.0 and
            row["td_avg"] == 0.0 and
            row["sub_avg"] == 0.0)


def clean_fighters(df):
    print("Cleaning fighter data...")
    print(f"Starting shape: {df.shape}")

    # ── physical attributes ───────────────────────────────────────────────────
    df["height_inches"] = df["height"].apply(parse_height)
    df["weight_lbs"] = df["weight"].apply(parse_weight)
    df["reach_inches"] = df["reach"].apply(parse_reach)
    df["age"] = df["dob"].apply(parse_age)

    # ── record ────────────────────────────────────────────────────────────────
    records = df["record"].apply(parse_record)
    df["wins"] = records.apply(lambda x: x[0])
    df["losses"] = records.apply(lambda x: x[1])
    df["draws"] = records.apply(lambda x: x[2])
    df["total_record_fights"] = df["wins"] + df["losses"] + df["draws"]
    df["win_rate"] = df["wins"] / df["total_record_fights"].replace(0, np.nan)

    # ── stance ────────────────────────────────────────────────────────────────
    df["stance"] = df["stance"].fillna("Unknown").replace("", "Unknown")

    # ── convert percentage strings ────────────────────────────────────────────
    df["str_acc"] = df["str_acc"].apply(parse_percentage)
    df["str_def"] = df["str_def"].apply(parse_percentage)
    df["td_acc"] = df["td_acc"].apply(parse_percentage)
    df["td_def"] = df["td_def"].apply(parse_percentage)

    # ── flag fighters with no recorded stats ──────────────────────────────────
    df["has_stats"] = ~df.apply(flag_no_stats, axis=1)

    # ── drop original string columns we have now replaced ────────────────────
    df = df.drop(columns=["height", "weight", "reach", "dob", "record"])

    print(f"Cleaning complete. Shape: {df.shape}")
    print(f"\nFighters with recorded stats: {df['has_stats'].sum()}")
    print(f"Fighters without stats (all zeros): {(~df['has_stats']).sum()}")
    print(f"\nMissing values after cleaning:")
    print(df.isnull().sum())

    return df


def main():
    os.makedirs("data/processed", exist_ok=True)

    print("Loading raw data...")
    df = pd.read_csv("data/raw/all_fighters.csv")
    print(f"Loaded {len(df)} fighters")

    df_clean = clean_fighters(df)

    # save two versions
    # full dataset including fighters with no stats
    df_clean.to_csv("data/processed/fighters_clean.csv", index=False)
    print(f"\nSaved full cleaned dataset to data/processed/fighters_clean.csv")

    # filtered dataset - only fighters with actual recorded stats
    df_with_stats = df_clean[df_clean["has_stats"] == True].copy()
    df_with_stats.to_csv("data/processed/fighters_with_stats.csv", index=False)
    print(f"Saved stats-only dataset to data/processed/fighters_with_stats.csv")
    print(f"Fighters with stats: {len(df_with_stats)}")

    print("\nSample of cleaned data:")
    cols_to_show = ["name", "height_inches", "weight_lbs", "reach_inches",
                    "age", "wins", "losses", "win_rate", "slpm", "str_acc",
                    "td_avg", "td_def", "stance", "has_stats"]
    print(df_with_stats[cols_to_show].head(10).to_string())


if __name__ == "__main__":
    main()