# -*- coding: utf-8 -*-
"""
Section 4.6 -- the contrast informs attribution, not forecasting.

No code for this section existed in the replication package, so this is a fresh
implementation written to the description in the methods: classifiers are
trained on 2015-2022 and evaluated on 2023-2024, and the quantity of interest is
the change in AUC from adding wind to a model that already contains dryness,
humidity, temperature, and fixed effects.

Three quantities:

  (a) occurrence   per cause, does wind help separate fire-days from quiet days?
  (b) cause        among human fires, does wind help say which cause it was?
  (c) composition  on the windiest days, what share of human ignitions is
                   accidental?

Design choices, all deliberate and none free:

  - Logistic regression, no penalty, matching the log-link of the occurrence
    model rather than introducing a second functional form.
  - Fixed effects enter as station and month indicators, the same absorption the
    Poisson model uses, so "adding wind" is the only difference between the two
    specifications compared.
  - Continuous predictors are standardised on the training fold only. The test
    fold never touches the scaler.
  - Weather is the observed value for the day. A genuine day-ahead system would
    use a forecast; treating observed weather as a perfect forecast is the
    generous case for wind, which is the right way round when the finding is
    that wind does not help.
"""
import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import config as C
from build_panel import build_panel

BASE = ["dryspell", "rh_min", "tmax"]
WIND = BASE + ["gust"]
TRAIN_END, TEST_START = 2022, 2023


def design(df, cols):
    """Weather columns plus station and month indicators."""
    X = pd.concat([df[cols].reset_index(drop=True),
                   pd.get_dummies(df["sid"].reset_index(drop=True), prefix="s", dtype=float),
                   pd.get_dummies(df["month"].reset_index(drop=True), prefix="m", dtype=float)],
                  axis=1)
    return X


def auc(df, y, cols):
    tr, te = df.year <= TRAIN_END, df.year >= TEST_START
    X = design(df, cols)
    # align columns, standardise the continuous block on the training fold only
    sc = StandardScaler().fit(X.loc[tr.to_numpy(), cols])
    Xtr, Xte = X[tr.to_numpy()].copy(), X[te.to_numpy()].copy()
    Xtr[cols] = sc.transform(Xtr[cols])
    Xte[cols] = sc.transform(Xte[cols])
    m = LogisticRegression(penalty=None, max_iter=5000)
    m.fit(Xtr, y[tr.to_numpy()])
    return roc_auc_score(y[te.to_numpy()], m.predict_proba(Xte)[:, 1])


if __name__ == "__main__":
    panel, fires = build_panel()
    panel = panel.reset_index(drop=True)
    fs = fires[(fires.date >= C.START) & (fires.date <= C.END)].copy()

    print("=== (a) day-ahead occurrence: does wind help? ===")
    print(f"{'cause':<12}{'base':>9}{'+ wind':>9}{'change':>10}")
    occ = {}
    for cause, col in [("burning", "fires_burn"), ("accidental", "fires_acc")]:
        y = (panel[col] > 0).astype(int).to_numpy()
        a0 = auc(panel, y, BASE)
        a1 = auc(panel, y, WIND)
        occ[cause] = (a0, a1)
        print(f"{cause:<12}{a0:>9.4f}{a1:>9.4f}{a1-a0:>+10.4f}")
    print("  manuscript: about +0.0005 accidental, about -0.0002 burning, base 0.86 to 0.89")

    print("\n=== (b) cause discrimination among human fires ===")
    h = fs[fs.cause.isin(["burning", "accidental"])].copy()
    w = panel.set_index(["sid", "date"])[["dryspell", "rh_min", "tmax", "gust"]]
    idx = pd.MultiIndex.from_arrays([h.sid, h.date])
    h = h.join(w.reindex(idx).set_index(h.index)).dropna(subset=["gust"])
    h["year"] = h.date.dt.year
    h["month"] = h.date.dt.month
    y = (h.cause == "accidental").astype(int).to_numpy()
    b0 = auc(h, y, BASE)
    b1 = auc(h, y, WIND)
    print(f"  base {b0:.3f} -> with wind {b1:.3f}  (change {b1-b0:+.3f})")
    print(f"  test fires: {int((h.year >= TEST_START).sum()):,}")
    print("  manuscript: 0.557 -> 0.560")

    print("\n=== (c) composition on the windiest days ===")
    q = pd.qcut(panel["gust"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    gq = panel.assign(q=q).set_index(["sid", "date"])["q"]
    h2 = fs[fs.cause.isin(["burning", "accidental"])].copy()
    h2["q"] = gq.reindex(pd.MultiIndex.from_arrays([h2.sid, h2.date])).to_numpy()
    tab = (h2.dropna(subset=["q"]).groupby("q", observed=True)["cause"]
           .apply(lambda s: (s == "accidental").mean()))
    n = h2.dropna(subset=["q"]).groupby("q", observed=True).size()
    for k in tab.index:
        print(f"  {k}  accidental share {tab[k]:.1%}  (n={n[k]:,})")
    print("  manuscript: about 70% in the windiest quarter, 64 to 66% on calmer days")

    pd.DataFrame({
        "quantity": ["occurrence_burning_base", "occurrence_burning_wind",
                     "occurrence_accidental_base", "occurrence_accidental_wind",
                     "cause_auc_base", "cause_auc_wind"],
        "value": [occ["burning"][0], occ["burning"][1],
                  occ["accidental"][0], occ["accidental"][1], b0, b1]}
    ).to_csv(os.path.join(C.OUTPUT_DIR, "forecast_auc.csv"), index=False)
    tab.rename("accidental_share").to_csv(os.path.join(C.OUTPUT_DIR, "gust_quartile_share.csv"))
    print("\nwritten: output/forecast_auc.csv, output/gust_quartile_share.csv")
