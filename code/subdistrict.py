# -*- coding: utf-8 -*-
"""
Sub-district (sgg) resolution design.

Motivation
----------
The station-day design has only 23 clusters, which is what makes the headline
contrast fragile under conservative inference. Fires carry a sub-district code
(sgg_cd), so the same contrast can be estimated on a finer spatial grid with
an order of magnitude more clusters.

Design
------
  - Each sub-district is located at the centroid of its own fire coordinates
    (the fire-weighted centroid; no administrative boundary file is needed).
  - Daily weather is interpolated to that centroid by inverse-distance
    weighting (IDW, power 2) over the K nearest synoptic stations.
  - The dry spell is recomputed from interpolated precipitation, so it is a
    property of the sub-district rather than of a station.
  - Counts are stratified by cause on the sub-district-day grid.
  - The stacked contrast is estimated with sgg-by-cause and month-by-cause
    fixed effects, clustered on sub-district.

Memory note: the grid is ~256 x 3653 station-days, doubled by stacking, so
columns are held as float32 and only the analysis variables are retained.
"""
import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pyfixest as pf
from pyproj import Transformer
from sklearn.neighbors import BallTree

import config as C
from build_panel import load_weather, load_fire

K_NEIGHBORS = int(os.environ.get("IDW_K", 3))
POWER = 2.0
VARS = ["dryspell", "rh_min", "gust", "tmax"]


def sgg_centroids(f):
    """Fire-weighted centroid (lat/lon) of each sub-district."""
    g = f.groupby("sgg_cd")[["lat", "lon"]].mean()
    return g


def idw_weights(cent, k=K_NEIGHBORS, power=POWER):
    sids = np.array(list(C.STATION_COORDS.keys()))
    scoords = np.radians(np.array([C.STATION_COORDS[s] for s in sids]))
    tree = BallTree(scoords, metric="haversine")
    dist, idx = tree.query(np.radians(cent[["lat", "lon"]].values), k=k)
    dist_km = dist * 6371.0
    w = 1.0 / np.power(np.maximum(dist_km, 1e-6), power)
    w = w / w.sum(axis=1, keepdims=True)
    return sids[idx], w, dist_km


def build_sgg_panel():
    w, f = load_weather(), load_fire()
    fsub = f[(f.date >= C.START) & (f.date <= C.END)].copy()
    fsub = fsub[fsub.cause.isin(["burning", "accidental"])].copy()

    cent = sgg_centroids(fsub)
    sgg_list = cent.index.to_numpy()
    nb_sids, nb_w, nb_d = idw_weights(cent)
    print(f"[sgg] sub-districts: {len(sgg_list)} | "
          f"mean distance to nearest station: {nb_d[:, 0].mean():.1f} km", flush=True)

    days = pd.date_range(C.START, C.END, freq="D")
    # station x day matrices for each weather field
    wsub = w[(w.date >= C.START) & (w.date <= C.END)]
    fields = {}
    for v in ["tmax", "rh_min", "gust", "precip"]:
        M = (wsub.pivot_table(index="date", columns="sid", values=v)
             .reindex(index=days))
        M = M.interpolate(limit_direction="both")     # fill isolated station gaps
        fields[v] = M

    # IDW to each sub-district: (days x sgg)
    out = {}
    for v, M in fields.items():
        cols = list(M.columns)
        pos = {s: i for i, s in enumerate(cols)}
        A = M.to_numpy(dtype=np.float32)              # days x stations
        res = np.empty((len(days), len(sgg_list)), dtype=np.float32)
        for j in range(len(sgg_list)):
            ids = [pos[s] for s in nb_sids[j] if s in pos]
            ww = nb_w[j][: len(ids)]
            ww = ww / ww.sum()
            res[:, j] = A[:, ids] @ ww.astype(np.float32)
        out[v] = res
    del fields

    # dry spell from interpolated precipitation, per sub-district
    P = out["precip"]
    dry = np.zeros_like(P, dtype=np.float32)
    c = np.zeros(P.shape[1], dtype=np.float32)
    for t in range(P.shape[0]):
        rained = P[t] >= C.RAIN_THRESH
        c = np.where(rained, 0.0, c + 1.0)
        dry[t] = c

    panel = pd.DataFrame({
        "sgg": np.repeat(sgg_list, len(days)),
        "date": np.tile(days.to_numpy(), len(sgg_list)),
        "tmax": out["tmax"].T.ravel(),
        "rh_min": out["rh_min"].T.ravel(),
        "gust": out["gust"].T.ravel(),
        "dryspell": dry.T.ravel(),
    })
    del out, dry, P

    # cause-stratified counts
    for col, mask in {"fires_burn": fsub.cause == "burning",
                      "fires_acc": fsub.cause == "accidental"}.items():
        cnt = (fsub[mask].groupby(["sgg_cd", "date"]).size()
               .rename(col).reset_index()
               .rename(columns={"sgg_cd": "sgg"}))
        panel = panel.merge(cnt, on=["sgg", "date"], how="left")
    panel[["fires_burn", "fires_acc"]] = panel[["fires_burn", "fires_acc"]].fillna(0).astype(np.int16)
    panel["year"] = panel.date.dt.year.astype(np.int16)
    panel["month"] = panel.date.dt.month.astype(np.int8)
    panel = panel.dropna(subset=["tmax", "rh_min", "gust"])
    return panel


def stacked(panel):
    keep = ["sgg", "year", "month", *VARS]
    b = panel[keep].copy(); b["count"] = panel["fires_burn"].to_numpy(); b["acc"] = np.int8(0)
    a = panel[keep].copy(); a["count"] = panel["fires_acc"].to_numpy();  a["acc"] = np.int8(1)
    st = pd.concat([b, a], ignore_index=True)
    del b, a
    for v in VARS:
        st[f"acc_{v}"] = (st["acc"].to_numpy() * st[v].to_numpy()).astype(np.float32)
    rhs = " + ".join(VARS) + " + " + " + ".join(f"acc_{v}" for v in VARS)
    return pf.fepois(f"count ~ {rhs} | sgg^acc + month^acc + year",
                     data=st, vcov={"CRV1": "sgg"})


if __name__ == "__main__":
    panel = build_sgg_panel()
    print(f"[sgg] panel rows: {len(panel):,} | clusters: {panel.sgg.nunique()} | "
          f"burning {panel.fires_burn.sum():,} accidental {panel.fires_acc.sum():,}", flush=True)

    # separate cause-specific models
    print("\n=== separate cause-stratified PPML (gust) ===", flush=True)
    for name, dep in {"burning": "fires_burn", "accidental": "fires_acc"}.items():
        m = pf.fepois(f"{dep} ~ dryspell + rh_min + gust + tmax | sgg + year + month",
                      data=panel, vcov={"CRV1": "sgg"})
        print(f"  {name:11s} gust = {m.coef()['gust']:+.4f}  (p={m.pvalue()['gust']:.4f})", flush=True)

    md = stacked(panel)
    print("\n===== (1) SUB-DISTRICT RESOLUTION: stacked acc_gust =====", flush=True)
    for v in VARS:
        n = f"acc_{v}"
        b, se, p = md.coef()[n], md.se()[n], md.pvalue()[n]
        star = "  <-- headline" if v == "gust" else ""
        print(f"  {n:13s} = {b:+.4f}  SE {se:.4f}  p {p:.4f}  "
              f"95% CI [{b-1.96*se:+.4f}, {b+1.96*se:+.4f}]{star}", flush=True)
    print(f"\n  clusters = {panel.sgg.nunique()}  (station design: 23)", flush=True)
    print("  paper Table 4 reports: 0.045, p = 0.007", flush=True)
