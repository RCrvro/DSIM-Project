import cv2
import re
import os
import pandas as pd
import numpy as np
from scipy.spatial import distance


DEFAULT_METHOD = "VGG"
DEFAULT_DISTANCE = "cosine"
CLASSES = [re.sub(r"\..+", "", filename)
           for filename in os.listdir(os.path.join("./data/",
                                                   DEFAULT_METHOD))]
SIFT_MATCHER = cv2.BFMatcher()


def sift_distance(img1, img2):
    matches = SIFT_MATCHER.match(img1, img2)
    return np.mean([m.distance for m in matches])


def normalize(x):
    lim, Lim = x.min(), x.max()
    return (x - lim) / (Lim - lim)


def dist(x, y, metric):
    dist_fn = {
        "euclidean": distance.euclidean,
        "cosine": distance.cosine,
        "manhattan": distance.cityblock,
        "sift": sift_distance
    }
    return dist_fn[metric](x, y)


def get_best_distance(X, y, metric):
    return np.min([dist(x, y, metric) for x in X])


def update_scores(df, filename, score, metric=DEFAULT_DISTANCE):
    if not score:
        return df
    classes = list(df["class"].unique())
    # sorry, it is very bad written (a for D: cycle)
    # but I cannot do in other ways
    rif = {c: [] for c in classes}
    for _, row in df[df["filename"] == filename].iterrows():
        rif[row["class"]].append(row["features"])
    dist = df.apply(
        lambda row: get_best_distance(rif[row["class"]], row["features"],
                                      metric),
        axis=1)
    dist = 1 - normalize(dist)
    df["score"] = normalize(df["score"] * (1 - np.abs(score)) +
                            (score * dist))
    return df


def open_dataset(category, method):
    df = pd.read_csv("./data/{}/{}.csv".format(method, category),
                     names=["filename", "ratio", "features"],
                     header=None)
    df = df[df["features"].map(lambda x: "nan" not in x)]
    return df


def intersect(df_list):
    files_list = [set(df["filename"]) for df in df_list]
    selected_files = set.intersection(*files_list)
    return [df[df["filename"].map(lambda x: x in selected_files)]
            for df in df_list]


def get_higher(df):
    return df.groupby(["filename"]) \
             .mean() \
             .sort_values(by=["score"], ascending=False) \
             .index


def get_images_from_text(text, method=DEFAULT_METHOD):
    text = re.sub(r"\s+", " ", text.lower())
    selected_classes = [c for c in CLASSES
                        if re.search(r"\b" + re.escape(c) + r"\b", text)]
    df_list = intersect([open_dataset(c, method) for c in selected_classes])
    for df, c in zip(df_list, selected_classes):
        df["features"] = df["features"].map(lambda x: np.array(eval(x)).astype("float32"))
        df["score"] = normalize(df["ratio"])
        df["class"] = c
    return pd.concat([df[["filename", "class", "score", "features"]]
                      for df in df_list])
