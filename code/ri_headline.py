# -*- coding: utf-8 -*-
"""
Randomization inference for the headline contrast (acc_gust).

Motivation
----------
The wild cluster bootstrap in analysis.py is applied only to the linear
(LPM / OLS) specifications; pyfixest raises NotImplementedError for Poisson.
The headline stacked-Poisson contrast therefore rests on analytic CRV1 errors
with only 23 station clusters. Randomization inference sidesteps the
few-cluster problem entirely.

Design
------
Sharp null: conditional on station-year-month, the cause label of a fire is
independent of that day's weather (i.e., no differential weather response).

Permutation preserves, exactly:
  - the total number of human fires on each station-day (so the common
    weather-occurrence relation is untouched),
  - the number of accidental and burning fires within each station-year-month
    block (so cause-specific baselines and seasonality are untouched),
and randomizes only WHICH DAY within the block carries which cause label,
which is exactly what the null says is uninformative about gust.

p_RI = (1 + #{|lambda_perm| >= |lambda_obs|}) / (1 + R), two-sided.
"""
import warnings
warnings.filterwarnings("ignore")
import os
import sys
import time

import numpy as np
import pandas as pd
import pyfixest as pf

import config as C
from build_panel import build_panel

R = int(os.environ.get("RI_REPS", 999))
SEED = 20260716
OUT = os.path.join(C.OUTPUT_DIR, "ri_headline_draws.csv")

VARS = C.VARS  # dryspell, rh_min, gust, tmax


def stacked_fit(panel, burn_counts, acc_counts):
    """Stacked Poisson; returns the accidental-minus-burning gust difference."""
    keep = ["sid", "date", "year", "month", *VARS]
    b = panel[keep].copy()
    b["count"] = burn_counts
    b["acc"] = 0
    a = panel[keep].copy()
    a["count"] = acc_counts
    a["acc"] = 1
    st = pd.concat([b, a], ignore_index=True)
    for v in VARS:
        st[f"acc_{v}"] = st["acc"] * st[v]
    rhs = " + ".join(VARS) + " + " + " + ".join(f"acc_{v}" for v in VARS)
    m = pf.fepois(f"count ~ {rhs} | sid^acc + month^acc + year",
                  data=st, vcov={"CRV1": "sid"})
    return m


def counts_from_labels(fires, panel_index):
    """Rebuild (fires_burn, fires_acc) aligned to the panel's (sid,date) index."""
    g = (fires.groupby(["sid", "date", "cause_perm"]).size()
         .unstack("cause_perm").reindex(panel_index, fill_value=0))
    burn = g["burning"].fillna(0).to_numpy() if "burning" in g else np.zeros(len(panel_index))
    acc = g["accidental"].fillna(0).to_numpy() if "accidental" in g else np.zeros(len(panel_index))
    return burn.astype(int), acc.astype(int)


def main():
    panel, f = build_panel()
    panel = panel.reset_index(drop=True)
    pidx = pd.MultiIndex.from_arrays([panel["sid"], panel["date"]])

    # fires used for the contrast: burning + accidental, inside the window,
    # and only those that land on a station-day present in the panel.
    fs = f[(f.date >= C.START) & (f.date <= C.END)].copy()
    fs = fs[fs.cause.isin(["burning", "accidental"])].copy()
    fs["year"] = fs.date.dt.year
    fs["month"] = fs.date.dt.month
    key = pd.MultiIndex.from_arrays([fs["sid"], fs["date"]])
    fs = fs[key.isin(pidx)].copy()
    print(f"[ri] fires in contrast: {len(fs):,} "
          f"(burning {(fs.cause=='burning').sum():,}, accidental {(fs.cause=='accidental').sum():,})",
          flush=True)

    # observed
    fs["cause_perm"] = fs["cause"]
    b0, a0 = counts_from_labels(fs, pidx)
    m0 = stacked_fit(panel, b0, a0)
    obs = m0.coef()["acc_gust"]
    obs_p = m0.pvalue()["acc_gust"]
    print(f"[ri] observed acc_gust = {obs:+.5f}  (analytic CRV1 p = {obs_p:.4f})", flush=True)

    # permutation blocks: station x year x month
    block = fs.groupby(["sid", "month"]).indices   # pooled across years: matches sid^acc + month^acc FE
    labels = fs["cause"].to_numpy().copy()
    rng = np.random.default_rng(SEED)

    draws = []
    t0 = time.time()
    for r in range(1, R + 1):
        perm = labels.copy()
        for _, pos in block.items():
            if len(pos) > 1:
                perm[pos] = rng.permutation(labels[pos])
        fs["cause_perm"] = perm
        bb, aa = counts_from_labels(fs, pidx)
        try:
            m = stacked_fit(panel, bb, aa)
            draws.append(float(m.coef()["acc_gust"]))
        except Exception as e:
            print(f"  [rep {r}] failed: {repr(e)[:60]}", flush=True)
            continue
        if r % 10 == 0:
            d = np.array(draws)
            p = (1 + (np.abs(d) >= abs(obs)).sum()) / (1 + len(d))
            el = time.time() - t0
            print(f"  [rep {r:4d}/{R}] p_RI so far = {p:.4f} | "
                  f"perm mean {d.mean():+.4f} sd {d.std():.4f} | "
                  f"{el/r:.2f}s/rep, elapsed {el/60:.1f}m", flush=True)
            pd.DataFrame({"draw": draws}).to_csv(OUT, index=False)

    d = np.array(draws)
    p = (1 + (np.abs(d) >= abs(obs)).sum()) / (1 + len(d))
    pd.DataFrame({"draw": draws}).to_csv(OUT, index=False)
    print("\n===== RANDOMIZATION INFERENCE: acc_gust =====", flush=True)
    print(f"  observed            : {obs:+.5f}", flush=True)
    print(f"  analytic CRV1 p     : {obs_p:.4f}", flush=True)
    print(f"  permutation draws   : {len(d)}", flush=True)
    print(f"  perm mean / sd      : {d.mean():+.5f} / {d.std():.5f}", flush=True)
    print(f"  |perm| >= |obs|     : {(np.abs(d) >= abs(obs)).sum()}", flush=True)
    print(f"  p_RI (two-sided)    : {p:.4f}", flush=True)
    print(f"  perm 2.5/97.5 pct   : {np.percentile(d,2.5):+.4f} / {np.percentile(d,97.5):+.4f}", flush=True)


if __name__ == "__main__":
    main()
