import pandas as pd

df = pd.read_csv("data/processed/fighters_clustered.csv")

cluster_names = {
    0: "Pure Striker",
    1: "Submission Specialist",
    2: "Balanced Grappler",
    3: "Counter/Defensive Fighter",
    4: "Elite Wrestler"
}

df["style"] = df["cluster"].map(cluster_names)

df.to_csv("data/processed/fighters_clustered.csv", index=False)

print("Style distribution:")
print(df["style"].value_counts())
print()

notable = [
    "Khabib Nurmagomedov", "Jon Jones", "Israel Adesanya",
    "Conor McGregor", "Georges St-Pierre", "Charles Oliveira",
    "Francis Ngannou", "Alex Pereira", "Max Holloway"
]

print("Notable fighter styles:")
for name in notable:
    row = df[df["name"] == name]
    if len(row) > 0:
        print(f"  {name}: {row['style'].values[0]}")