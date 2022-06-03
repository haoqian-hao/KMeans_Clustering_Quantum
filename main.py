from cProfile import label
from relabel import get_intervals
from kmeans import solve
import numpy as np
import pandas as pd


def get_distance(i, j, df, num_colors):
    x = df.iloc[i, :num_colors].to_numpy()
    y = df.iloc[j, :num_colors].to_numpy()
    return np.linalg.norm(x - y)


def get_congestion(start, end):
    raw_file = "duplicates_removed.csv"
    df = pd.read_csv(raw_file)
    df = df.drop(columns=["id"])
    df = df.loc[start:end]
    return df


if __name__ == '__main__':
        
    start = 0
    end = 3
    K = 4
    
    df = get_congestion(start, end)
    df = df.round(3)
    df.drop_duplicates(subset=["CI"])

    
    print(df)
    
    N = len(df['CI'])

    dist_matrix = np.zeros((N, N))
    print("creating matrix ...")
    for i in range(N):
        for j in range(N):
            if i != j:
                dist_matrix[i][j] = get_distance(i, j, df, num_colors=K)
    print("matrix created")

    label_results = solve(N, K, dist_matrix, gamma_distinct=1, gamma_multiple=10)

    print("result: ...")
    print(label_results)
    
    print("thresholds ...")
    print(get_intervals(K,start,df["CI"], label_results))
    