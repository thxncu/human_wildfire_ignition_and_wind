# -*- coding: utf-8 -*-
"""
Panel construction.

Builds a balanced station-by-day panel over the sample window with
cause-stratified daily fire counts merged onto daily weather covariates.

Pipeline:
  (1) load_weather   : concatenate regional daily weather, derive fire-weather
                       variables (dry spell, antecedent precipitation, lags).
  (2) classify_cause : map source cause descriptions to analysis categories.
  (3) map_to_station : project fire coordinates to lat/lon and assign the
                       nearest weather station via a haversine BallTree.
  (4) build_panel    : assemble the full station-day grid with stratified
                       counts and weather, plus derived outcomes.

Expected input columns
----------------------
Weather files (data/weather/*.csv): sid, date, tmax, rh_min, gust, precip
Fire file (data/fire_records.csv) : objt_id, occu_date, occu_tm, end_year,
    end_mt, end_de, end_tm, amount, x, y, sgg_cd, cause_desc
See data/data_dictionary.md for full definitions.
"""
import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pyproj import Transformer
from sklearn.neighbors import BallTree

import config as C

# Defensive rename map: lets the loader also accept files that still carry the
# original source-language column headers.
_WEATHER_RENAME = {
    "지점번호": "sid", "날짜": "date",
    "최고기온(°C)": "tmax", "최저상대습도(%)": "rh_min",
    "최대순간풍속(m/s)": "gust", "일강수량(mm)": "precip",
}


# ---------------------------------------------------------------------------
# 1. Weather and fire-weather variables
# ---------------------------------------------------------------------------
def load_weather():
    paths = [os.path.join(C.WEATHER_DIR, f + ".csv") for f in C.REGION_FILES]
    paths = [p for p in paths if os.path.exists(p)]
    if paths:                                   # concatenate regional files
        w = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    else:                                       # fallback: single combined file
        w = pd.read_csv(os.path.join(C.WEATHER_DIR, C.WEATHER_COMBINED))
    w = w.rename(columns=_WEATHER_RENAME)
    w = w[["sid", "date", "tmax", "rh_min", "gust", "precip"]].copy()
    w["date"] = pd.to_datetime(w["date"])
    w["precip"] = w["precip"].fillna(0.0)
    w = w.drop_duplicates(["sid", "date"]).sort_values(["sid", "date"]).reset_index(drop=True)

    g = w.groupby("sid", group_keys=False)

    def dry_spell(s):
        # Consecutive days since the last day with precipitation >= threshold.
        rained = (s >= C.RAIN_THRESH).values
        out, c = np.empty(len(s)), 0
        for i, r in enumerate(rained):
            c = 0 if r else c + 1
            out[i] = c
        return pd.Series(out, index=s.index)

    w["dryspell"] = g["precip"].apply(dry_spell)
    w["ante7"] = g["precip"].apply(lambda s: s.shift(1).rolling(7, min_periods=1).sum())
    w["ante30"] = g["precip"].apply(lambda s: s.shift(1).rolling(30, min_periods=1).sum())
    for k in range(1, C.MAX_LAG + 1):
        w[f"precip_l{k}"] = g["precip"].shift(k)
    return w


# ---------------------------------------------------------------------------
# 2. Cause classification
# ---------------------------------------------------------------------------
def classify_cause(s):
    """
    Map a source cause description to one analysis category.

    Categories
    ----------
    natural       : lightning ignition
    unknown       : unknown / under investigation / unspecified "other"
    burning       : escaped intentional burning (refuse, field/levee burning)
    accidental    : accidental human ignition (carelessness, discarded
                    cigarette, sparks, recreational visitors)
    other_human   : remaining human causes (e.g., arson, industrial)

    The matched tokens are the source-language descriptions used by the data
    provider; they are retained verbatim because they are data values rather
    than code. See data/data_dictionary.md for the full token list.
    """
    s = str(s)
    if "낙뢰" in s:                                            # lightning
        return "natural"
    if any(k in s for k in ["미상", "조사중"]) or s.startswith("기타"):  # unknown / under investigation / "other"
        return "unknown"
    if ("소각" in s) or ("두렁" in s):                          # intentional burning / field-levee burning
        return "burning"
    if any(k in s for k in ["실화", "담뱃불", "불씨", "부주의"]):  # accidental human ignition
        return "accidental"
    return "other_human"                                       # remaining human causes (e.g., arson, industrial)


def map_fire_to_station(fire):
    # Project planar coordinates (EPSG:3857) to geographic (EPSG:4326),
    # then assign each fire to its nearest station by great-circle distance.
    tr = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    lon, lat = tr.transform(fire["x"].values, fire["y"].values)
    fire = fire.assign(lon=lon, lat=lat)
    sids = np.array(list(C.STATION_COORDS.keys()))
    scoords = np.radians(np.array([C.STATION_COORDS[s] for s in sids]))
    tree = BallTree(scoords, metric="haversine")
    dist, idx = tree.query(np.radians(fire[["lat", "lon"]].values), k=1)
    return fire.assign(sid=sids[idx[:, 0]], dist_km=dist[:, 0] * 6371.0)


def load_fire():
    f = pd.read_csv(C.FIRE_PATH)
    if "cause_desc" not in f.columns and "resn" in f.columns:
        f = f.rename(columns={"resn": "cause_desc"})
    f["date"] = pd.to_datetime(f["occu_date"].astype(str).str.zfill(8),
                               format="%Y%m%d", errors="coerce")
    # Winsorize burned area at the 99th percentile of positive values.
    pos = f.loc[f["amount"] > 0, "amount"]
    cap = pos.quantile(0.99)
    f["area"] = f["amount"].clip(upper=cap).fillna(0.0)
    f["cause"] = f["cause_desc"].apply(classify_cause)
    f["cause_top"] = np.where(f["cause"] == "natural", "natural",
                       np.where(f["cause"] == "unknown", "unknown", "human"))
    return map_fire_to_station(f)


# ---------------------------------------------------------------------------
# 3. Station-day panel with stratified counts
# ---------------------------------------------------------------------------
def build_panel():
    w, f = load_weather(), load_fire()
    fsub = f[(f.date >= C.START) & (f.date <= C.END)].copy()
    days = pd.date_range(C.START, C.END, freq="D")
    sids = list(C.STATION_COORDS.keys())
    panel = pd.MultiIndex.from_product([sids, days], names=["sid", "date"]).to_frame(index=False)

    base = fsub.groupby(["sid", "date"]).agg(
        fires=("objt_id", "count"), area_sum=("area", "sum")).reset_index()
    panel = panel.merge(base, on=["sid", "date"], how="left")

    # Stratified daily counts: human / natural, and burning / accidental.
    for col, mask in {
        "fires_hum":  fsub.cause_top == "human",
        "fires_nat":  fsub.cause_top == "natural",
        "fires_burn": fsub.cause == "burning",
        "fires_acc":  fsub.cause == "accidental",
    }.items():
        c = fsub[mask].groupby(["sid", "date"]).size().rename(col).reset_index()
        panel = panel.merge(c, on=["sid", "date"], how="left")

    cnt_cols = ["fires", "fires_hum", "fires_nat", "fires_burn", "fires_acc"]
    panel[cnt_cols] = panel[cnt_cols].fillna(0).astype(int)
    panel["area_sum"] = panel["area_sum"].fillna(0.0)

    wcols = (["sid", "date", "tmax", "rh_min", "gust", "dryspell", "ante7", "ante30"]
             + [f"precip_l{k}" for k in range(1, C.MAX_LAG + 1)])
    panel = panel.merge(w[wcols], on=["sid", "date"], how="left").dropna(
        subset=["tmax", "rh_min", "gust"])
    panel["year"] = panel.date.dt.year
    panel["month"] = panel.date.dt.month
    panel["log_area"] = np.log1p(panel["area_sum"])
    panel["fire_ind"] = (panel.fires > 0).astype(int)
    return panel, f


if __name__ == "__main__":
    panel, f = build_panel()
    print(f"[panel] station-days {len(panel):,} | "
          f"fire-days {(panel.fires > 0).sum():,} ({(panel.fires > 0).mean() * 100:.1f}%)")
    print("[cause, top-level]", f["cause_top"].value_counts().to_dict())
    print("[cause, detailed] ", f["cause"].value_counts().to_dict())
