"""Step 1: load the Shanghai delivery / pickup CSVs and pick a compact set of orders."""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
FILES = {"delivery": "delivery_sh.csv", "pickup": "pickup_sh.csv"}

# delivery_sh.csv has no time window column, so we create one:
# earliest = accept time, latest = accept time + this many minutes.
DELIVERY_WINDOW_MIN = 480


def _to_minutes(series):
    """'07-08 09:00:00' -> minutes since midnight (date part is ignored)."""
    t = pd.to_datetime(series, format="%m-%d %H:%M:%S", errors="coerce")
    return t.dt.hour * 60 + t.dt.minute


def load_raw(mode):
    path = os.path.join(DATA_DIR, FILES[mode])
    return pd.read_csv(path, dtype={"ds": str})


def list_days(mode="pickup", top=10):
    """Days (ds) with the most orders, useful for choosing a demo day."""
    return load_raw(mode)["ds"].value_counts().head(top)


def select_orders(mode="pickup", n=20, ds=None, region_id=None, seed=42):
    """Return n orders of ONE day that are close to each other (a realistic service area)."""
    df = load_raw(mode).dropna(subset=["lng", "lat", "accept_time"])
    if region_id is not None:
        df = df[df["region_id"] == region_id]
    if ds is None:
        ds = df["ds"].value_counts().index[0]
    df = df[df["ds"] == ds].copy()

    if mode == "pickup":
        df["ready"] = _to_minutes(df["time_window_start"])
        df["due"] = _to_minutes(df["time_window_end"])
    else:
        acc = _to_minutes(df["accept_time"])
        df["ready"] = acc
        df["due"] = acc + DELIVERY_WINDOW_MIN
    df = df.dropna(subset=["ready", "due"])

    # one stop per location (~10 m), many orders share the same building
    df["_key"] = df["lng"].round(4).astype(str) + "_" + df["lat"].round(4).astype(str)
    df = df.drop_duplicates("_key")

    # pick a random anchor order (seed) and keep its n nearest neighbours
    anchor = df.sample(1, random_state=seed).iloc[0]
    coslat = np.cos(np.radians(anchor["lat"]))
    df["_d"] = (df["lat"] - anchor["lat"]) ** 2 + ((df["lng"] - anchor["lng"]) * coslat) ** 2
    df = df.nsmallest(n, "_d")

    cols = ["order_id", "lat", "lng", "ready", "due", "courier_id", "aoi_id"]
    out = df[cols].reset_index(drop=True)
    out.attrs["ds"] = ds
    return out