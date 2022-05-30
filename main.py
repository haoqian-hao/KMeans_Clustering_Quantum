from matplotlib.axes import Axes
from kmeans import solve
import numpy as np
import pandas as pd


def get_distance(i, j, df, num_colors):
    x = df.iloc[i, :num_colors].to_numpy()
    y = df.iloc[j, :num_colors].to_numpy()
    return np.linalg.norm(x - y)


def get_congestion():
    raw_file = "duplicates_removed.csv"
    df = pd.read_csv(raw_file)
    df = df.drop(columns=["id"])
    
    df["timestamp"] = [0]*len(df["CI"])
    
    df = df.loc[0:199]
    
    return df


df = get_congestion()
print(df)
N = len(df['CI'])
K = 4


df = df.round(3)
df.drop_duplicates(subset=["CI"])

print(df)
print(len(df["CI"]))

dist_matrix = np.zeros((N, N))
for i in range(N):
    print("create matrix: " + str(i))
    for j in range(N):
        if i != j:
            dist_matrix[i][j] = get_distance(i, j, df, num_colors=K)

label_results = solve(N, K, dist_matrix, gamma_distinct=1, gamma_multiple=10)

print(label_results)
print(len(label_results))