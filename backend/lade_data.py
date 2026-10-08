"""Step 1: load the Shanghai delivery / pickup CSVs and pick a compact set of orders."""
import os
import threading
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

NEEDED = {"order_id", "region_id", "courier_id", "lng", "lat", "aoi_id",
          "accept_time", "time_window_start", "time_window_end", "ds"}
_CACHE, _LOCK = {}, threading.Lock()


def load_raw(mode):
    """Read the CSV once, then reuse it (the file is big, so this saves ~a minute per run)."""
    with _LOCK:
        if mode not in _CACHE:
            path = os.path.join(DATA_DIR, FILES[mode])
            print(f"Loading {FILES[mode]} ...", flush=True)
            df = pd.read_csv(path, dtype={"ds": str}, usecols=lambda c: c in NEEDED)
            print(f"  {len(df):,} rows loaded", flush=True)
            _CACHE[mode] = df
        return _CACHE[mode]



def list_days(mode="pickup", top=10):
    """Days (ds) with the most orders, useful for choosing a demo day."""
    return load_raw(mode)["ds"].value_counts().head(top)
def select_orders(mode="pickup", n=20, ds=None, region_id=None, seed=42,
                  radius_km=6.0, slot_span_min=180):
    """Return n orders of ONE day, in one service area (radius_km) and one time slot (slot_span_min)."""
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

    # keep one busy time slot: orders whose window opens close to the most common opening time
    common = df["ready"].mode().iloc[0]
    df = df[(df["ready"] - common).abs() <= slot_span_min]

    # random anchor order (seed), then n random orders within radius_km of it
    anchor = df.sample(1, random_state=seed).iloc[0]
    coslat = np.cos(np.radians(anchor["lat"]))
    dkm = 111.2 * np.sqrt((df["lat"] - anchor["lat"]) ** 2 + ((df["lng"] - anchor["lng"]) * coslat) ** 2)
    df = df[dkm <= radius_km]
    if len(df) > n:
        df = df.sample(n=n, random_state=seed)

    cols = ["order_id", "lat", "lng", "ready", "due", "courier_id", "aoi_id"]
    out = df[cols].reset_index(drop=True)
    out.attrs["ds"] = ds
    return out

