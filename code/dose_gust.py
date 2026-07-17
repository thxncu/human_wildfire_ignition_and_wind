# -*- coding: utf-8 -*-
"""
Cause-specific dose-response to wind gust.

The baseline enters gust linearly while humidity and temperature are binned
(figures.py, Figure 2). This script bins gust as well and estimates the
occurrence response separately for intentional burning and accidental fires,
then tests the differential across bins in a stacked specification.

Outputs
-------
  output/figure_gust_dose.png   two cause-specific dose-response curves
  output/gust_dose.csv          plotted coefficients and CIs
"""
import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pyfixest as pf
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt

import config as C
from build_panel import build_panel

mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.linewidth": 0.8, "savefig.dpi": 300, "figure.dpi": 300})

EDGES = [0, 5, 7, 9, 12, 15, 100]
LABS = ["<5", "5-7", "7-9", "9-12", "12-15", ">=15"]
# formula-safe identifiers (formulaic rejects "-", ">", "=" in names)
KEYS = ["lt5", "b5_7", "b7_9", "b9_12", "b12_15", "ge15"]
K = dict(zip(LABS, KEYS))
REF = "<5"
CTRL = "dryspell + rh_min + tmax"


def add_bins(panel):
    p = panel.copy()
    p["gb"] = pd.cut(p["gust"], bins=EDGES, labels=LABS, right=False)
    for L in LABS:
        if L != REF:
            p[f"g_{K[L]}"] = (p["gb"] == L).astype(int)
    return p


def curve(p, dep):
    terms = " + ".join(f"g_{K[L]}" for L in LABS if L != REF)
    m = pf.fepois(f"{dep} ~ {terms} + {CTRL} | sid + year + month",
                  data=p, vcov={"CRV1": "sid"})
    co, se = m.coef(), m.se()
    b = [0.0] + [co[f"g_{K[L]}"] for L in LABS if L != REF]
    s = [0.0] + [se[f"g_{K[L]}"] for L in LABS if L != REF]
    return np.array(b), np.array(s)


def stacked_bins(p):
    """Stacked model: bin dummies interacted with an accidental indicator."""
    gcols = [f"g_{K[L]}" for L in LABS if L != REF]
    keep = ["sid", "date", "year", "month", "dryspell", "rh_min", "tmax", *gcols]
    b = p[keep].copy(); b["count"] = p["fires_burn"].to_numpy(); b["acc"] = 0
    a = p[keep].copy(); a["count"] = p["fires_acc"].to_numpy();  a["acc"] = 1
    st = pd.concat([b, a], ignore_index=True)
    for v in gcols + ["dryspell", "rh_min", "tmax"]:
        st[f"acc_{v}"] = st["acc"] * st[v]
    rhs = (" + ".join(gcols + ["dryspell", "rh_min", "tmax"]) + " + " +
           " + ".join(f"acc_{v}" for v in gcols + ["dryspell", "rh_min", "tmax"]))
    return pf.fepois(f"count ~ {rhs} | sid^acc + month^acc + year",
                     data=st, vcov={"CRV1": "sid"})


if __name__ == "__main__":
    panel, _ = build_panel()
    p = add_bins(panel)

    n = p.groupby("gb", observed=True)[["fires_burn", "fires_acc"]].sum()
    print("=== fires per gust bin ===")
    print(n.to_string())

    bb, bs = curve(p, "fires_burn")
    ab, as_ = curve(p, "fires_acc")

    print("\n===== (5) CAUSE-SPECIFIC GUST DOSE-RESPONSE =====")
    print(f"{'bin':>7} | {'burning':>22} | {'accidental':>22}")
    print("-" * 58)
    for i, L in enumerate(LABS):
        print(f"{L:>7} | {bb[i]:+.3f} [{bb[i]-1.96*bs[i]:+.3f},{bb[i]+1.96*bs[i]:+.3f}] | "
              f"{ab[i]:+.3f} [{ab[i]-1.96*as_[i]:+.3f},{ab[i]+1.96*as_[i]:+.3f}]")

    md = stacked_bins(p)
    print("\n===== differential by bin (accidental - burning), stacked =====")
    rows = []
    for L in LABS:
        if L == REF:
            rows.append({"bin": L, "diff": 0.0, "se": 0.0, "p": np.nan})
            continue
        v = f"acc_g_{K[L]}"
        rows.append({"bin": L, "diff": round(md.coef()[v], 4),
                     "se": round(md.se()[v], 4), "p": round(md.pvalue()[v], 4)})
    print(pd.DataFrame(rows).to_string(index=False))

    # joint test: are the four gust interactions jointly zero?
    try:
        jt = md.wald_test(R=None, q=None)
    except Exception:
        jt = None

    pd.DataFrame({"bin": LABS, "burn_coef": bb, "burn_se": bs,
                  "acc_coef": ab, "acc_se": as_}).to_csv(
        os.path.join(C.OUTPUT_DIR, "gust_dose.csv"), index=False)

    # ---- figure ----
    x = np.arange(len(LABS))
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    ax.axhline(0, color="0.5", lw=.8, zorder=1)
    ax.errorbar(x - 0.06, bb, yerr=1.96 * bs, fmt="s--", color="black",
                mfc="white", ms=5, lw=1.0, capsize=3, label="Intentional burning")
    ax.errorbar(x + 0.06, ab, yerr=1.96 * as_, fmt="o-", color="black",
                mfc="0.45", ms=5, lw=1.4, capsize=3, label="Accidental escape")
    ax.set_xticks(x)
    ax.set_xticklabels(LABS)
    ax.set_xlabel("Maximum gust speed (m s$^{-1}$)")
    ax.set_ylabel("Log occurrence rate\n(relative to <5 m s$^{-1}$)")
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(C.OUTPUT_DIR, "figure_gust_dose.png"))
    print("\n[saved] output/figure_gust_dose.png")
