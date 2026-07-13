import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
import os
import warnings
warnings.filterwarnings("ignore")


# ── load data ─────────────────────────────────────────────────────────────────

def load_data():
    df = pd.read_csv("data/processed/fighters_modeling.csv")
    print(f"Loaded {len(df)} fighters")
    return df


STYLE_FEATURES = [
    "slpm", "str_acc", "sapm", "str_def",
    "td_avg", "td_acc", "td_def", "sub_avg",
    "strike_aggression", "strike_efficiency",
    "grappling_aggression", "grappling_efficiency",
    "submission_threat", "striker_ratio", "win_rate"
]


# ── scale features ────────────────────────────────────────────────────────────

def scale_features(df):
    """Standardize features to zero mean and unit variance."""
    scaler = StandardScaler()
    X = scaler.fit_transform(df[STYLE_FEATURES])
    print(f"Scaled {X.shape[1]} features for {X.shape[0]} fighters")
    return X, scaler


# ── find optimal k ────────────────────────────────────────────────────────────

def find_optimal_k(X, k_range=range(2, 11)):
    """
    Test k from 2 to 10.
    Plot both inertia (elbow method) and silhouette score.
    """
    print("Testing different values of k...")
    inertias = []
    silhouette_scores = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        inertias.append(kmeans.inertia_)
        score = silhouette_score(X, labels)
        silhouette_scores.append(score)
        print(f"  k={k} | inertia={kmeans.inertia_:.1f} | silhouette={score:.4f}")

    # plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # elbow plot
    ax1.plot(list(k_range), inertias, "bo-", linewidth=2, markersize=8)
    ax1.set_xlabel("Number of Clusters (k)")
    ax1.set_ylabel("Inertia")
    ax1.set_title("Elbow Method — Finding Optimal k")
    ax1.grid(True, alpha=0.3)

    # silhouette plot
    ax2.plot(list(k_range), silhouette_scores, "ro-", linewidth=2, markersize=8)
    ax2.set_xlabel("Number of Clusters (k)")
    ax2.set_ylabel("Silhouette Score")
    ax2.set_title("Silhouette Score — Higher is Better")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs("outputs/figures", exist_ok=True)
    plt.savefig("outputs/figures/optimal_k.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved outputs/figures/optimal_k.png")

    best_k = list(k_range)[silhouette_scores.index(max(silhouette_scores))]
    print(f"\nBest k by silhouette score: {best_k}")
    return best_k, inertias, silhouette_scores


# ── run final clustering ──────────────────────────────────────────────────────

def run_clustering(X, df, k):
    """Run KMeans with chosen k and assign cluster labels."""
    print(f"\nRunning final KMeans with k={k}...")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = kmeans.fit_predict(X)
    df = df.copy()
    df["cluster"] = labels

    score = silhouette_score(X, labels)
    print(f"Final silhouette score: {score:.4f}")
    print(f"\nCluster sizes:")
    print(df["cluster"].value_counts().sort_index())
    return df, kmeans


# ── analyze clusters ──────────────────────────────────────────────────────────

def analyze_clusters(df):
    """Print mean feature values per cluster and list notable fighters."""
    print("\n=== CLUSTER PROFILES ===")
    cluster_means = df.groupby("cluster")[STYLE_FEATURES].mean().round(3)
    print(cluster_means.to_string())

    print("\n=== NOTABLE FIGHTERS PER CLUSTER ===")
    notable = [
        "Khabib Nurmagomedov", "Jon Jones", "Israel Adesanya",
        "Conor McGregor", "Anderson Silva", "Georges St-Pierre",
        "Demetrious Johnson", "Nate Diaz", "Tony Ferguson",
        "Charles Oliveira", "Max Holloway", "Dustin Poirier",
        "Kamaru Usman", "Francis Ngannou", "Alex Pereira"
    ]
    for name in notable:
        row = df[df["name"] == name]
        if len(row) > 0:
            cluster = row["cluster"].values[0]
            print(f"  {name}: Cluster {cluster}")


# ── visualize with PCA ────────────────────────────────────────────────────────

def visualize_clusters(X, df):
    """Reduce to 2D with PCA and plot clusters."""
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)

    explained = pca.explained_variance_ratio_
    print(f"\nPCA variance explained: {explained[0]:.1%} + {explained[1]:.1%} = {sum(explained):.1%}")

    df_plot = df.copy()
    df_plot["pca1"] = X_2d[:, 0]
    df_plot["pca2"] = X_2d[:, 1]

    k = df["cluster"].nunique()
    colors = cm.tab10(np.linspace(0, 1, k))

    fig, ax = plt.subplots(figsize=(12, 8))

    for cluster_id in sorted(df_plot["cluster"].unique()):
        mask = df_plot["cluster"] == cluster_id
        ax.scatter(
            df_plot[mask]["pca1"],
            df_plot[mask]["pca2"],
            c=[colors[cluster_id]],
            label=f"Cluster {cluster_id}",
            alpha=0.6,
            s=30
        )

    # annotate notable fighters
    notable = [
        "Khabib Nurmagomedov", "Israel Adesanya",
        "Conor McGregor", "Jon Jones", "Georges St-Pierre"
    ]
    for name in notable:
        row = df_plot[df_plot["name"] == name]
        if len(row) > 0:
            ax.annotate(
                name,
                (row["pca1"].values[0], row["pca2"].values[0]),
                fontsize=8,
                fontweight="bold",
                xytext=(5, 5),
                textcoords="offset points"
            )

    ax.set_xlabel(f"PC1 ({explained[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({explained[1]:.1%} variance)")
    ax.set_title("UFC Fighter Style Clusters (PCA Visualization)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig("outputs/figures/fighter_clusters.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved outputs/figures/fighter_clusters.png")


# ── radar chart per cluster ───────────────────────────────────────────────────

def plot_radar_charts(df):
    """Plot a radar chart showing the style profile of each cluster."""
    radar_features = [
        "slpm", "str_acc", "str_def",
        "td_avg", "td_def", "sub_avg", "striker_ratio"
    ]
    labels = [
        "Strike Output", "Strike Acc", "Strike Def",
        "Takedowns", "TD Defense", "Submissions", "Striker Ratio"
    ]

    cluster_means = df.groupby("cluster")[radar_features].mean()

    # normalize each feature to 0-1 for radar
    for col in radar_features:
        col_min = cluster_means[col].min()
        col_max = cluster_means[col].max()
        if col_max > col_min:
            cluster_means[col] = (cluster_means[col] - col_min) / (col_max - col_min)

    k = len(cluster_means)
    angles = np.linspace(0, 2 * np.pi, len(radar_features), endpoint=False).tolist()
    angles += angles[:1]

    fig, axes = plt.subplots(1, k, figsize=(4 * k, 5), subplot_kw=dict(polar=True))
    if k == 1:
        axes = [axes]

    colors = cm.tab10(np.linspace(0, 1, k))

    for idx, (cluster_id, row) in enumerate(cluster_means.iterrows()):
        values = row.tolist()
        values += values[:1]

        ax = axes[idx]
        ax.plot(angles, values, "o-", linewidth=2, color=colors[idx])
        ax.fill(angles, values, alpha=0.25, color=colors[idx])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, size=8)
        ax.set_ylim(0, 1)
        ax.set_title(f"Cluster {cluster_id}", size=11, fontweight="bold", pad=15)
        ax.set_yticks([])

    plt.suptitle("UFC Fighter Style Profiles by Cluster", size=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("outputs/figures/radar_charts.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved outputs/figures/radar_charts.png")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs("outputs/figures", exist_ok=True)

    df = load_data()
    X, scaler = scale_features(df)

    # find optimal k — run this for the paper
    best_k, inertias, silhouette_scores = find_optimal_k(X)

    # we force k=5 for more meaningful style archetypes
    # k=2 is statistically optimal but too coarse for style analysis
    CHOSEN_K = 5
    print(f"\nNote: Using k={CHOSEN_K} for richer style archetypes")
    print(f"(k=2 is statistically optimal but only separates strikers vs grapplers)")

    # run clustering with k=5
    df_clustered, kmeans = run_clustering(X, df, CHOSEN_K)

    # analyze
    analyze_clusters(df_clustered)

    # visualize
    visualize_clusters(X, df_clustered)
    plot_radar_charts(df_clustered)

    # save clustered dataset
    df_clustered.to_csv("data/processed/fighters_clustered.csv", index=False)
    print(f"\nSaved clustered dataset to data/processed/fighters_clustered.csv")
    print("\nDone. Check outputs/figures/ for your plots.")


if __name__ == "__main__":
    main()