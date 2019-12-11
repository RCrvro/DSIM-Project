import re
import os
import pandas as pd
import numpy as np
from scipy.spatial import distance


DEFAULT_METHOD = "YOLO"
DEFAULT_DISTANCE = "cosine"
CLASSES = [re.sub("\..+", "", filename)
           for filename in os.listdir(os.path.join("./data/",
                                                   DEFAULT_METHOD))]


def normalize(x):
    lim, Lim = x.min(), x.max()
    return (x + lim) / (Lim + lim)

def dist(x, y, metric):
    dist_fn = {
        "euclidean": distance.euclidean,
        "cosine": distance.cosine,
        "manhattan": distance.cityblock
    }
    return dist_fn[metric](x, y)

def get_best_distance(X, y, metric, best_low):
    best_fn = np.min if best_low else np.max
    return best_fn([dist(x, y, metric) for x in X])

def update_scores(df, filename, score, metric=DEFAULT_DISTANCE):
    best_low = (metric in {"euclidean", "manhattan"})
    rif = list(df[df["filename"] == filename]["features"].values)
    dist = df["features"].map(
        lambda x: get_best_distance(rif, x, metric, best_low))
    df["score"] *= score * normalize(dist)
    df["score"] = normalize(df["score"])
    return df

def open_dataset(category, method=DEFAULT_METHOD):
    df = pd.read_csv("./data/{}/{}.csv".format(method, category),
                     names=["filename", "ratio", "features"],
                     header=None)
    return df

def intersect(df_list):
    files_list = [set(df["filename"]) for df in df_list]
    selected_files = set.intersection(*files_list)
    return [df[df["filename"].map(lambda x: x in selected_files)]
            for df in df_list]

def get_higher(df):
    return df.groupby(["filename"]) \
             .max(level="score") \
             .sort_values(by=["score"], ascending=False) \
             .index

def get_images_from_text(text):
    text = re.sub(r"\s+", " ", text.lower())
    selected_classes = [c for c in CLASSES
                        if re.search(r"\b" + re.escape(c) + r"\b", text)]
    df_list = intersect([open_dataset(c) for c in selected_classes])
    for df, c in zip(df_list, selected_classes):
        df["features"] = df["features"].map(lambda x: np.array(eval(x)))
        df["score"] = normalize(df["ratio"])
        df["class"] = c
    return pd.concat([df[["filename", "class", "score", "features"]]
                      for df in df_list])
