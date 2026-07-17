# -*- coding: utf-8 -*-
"""
Figures.

Generates three figures at 300 DPI (grayscale, print-ready) and writes the
underlying plotted values to output/figure_data.json.

  figure_1_lag.png       distributed-lag rainfall suppression of occurrence
  figure_2_dose.png      nonlinear dose-response for humidity and temperature
  figure_3_exposure.png  occurrence by day of week, by cause

Run from the code/ directory:  python figures.py
"""
import os
import json
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

panel, f = build_panel()
out = {}
OUT = C.OUTPUT_DIR

# ---- Figure 1: distributed-lag rainfall suppression ----
rhs_dl = "rh_min + gust + tmax + " + " + ".join([f"precip_l{k}" for k in range(1, 8)])
m = pf.fepois(f"fires ~ {rhs_dl} | sid+year+month", data=panel, vcov={"CRV1": "sid"})
co, se = m.coef(), m.se()
lags = list(range(1, 8))
b = [co[f"precip_l{k}"] for k in lags]
s = [se[f"precip_l{k}"] for k in lags]
out["dl"] = {"b": b, "s": s}
fig, ax = plt.subplots(figsize=(5.4, 3.6))
ax.axhline(0, color="0.5", lw=.8, zorder=1)
ax.errorbar(lags, b, yerr=[1.96 * x for x in s], fmt="o-", color="black",
            capsize=2.5, lw=1.4, ms=5, mfc="white", mec="black", mew=1.3, zorder=3)
ax.set_xlabel("Days since rainfall (lag)")
ax.set_ylabel("Change in log fire count per mm")
ax.set_xticks(lags)
ax.grid(alpha=.3, lw=.5)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "figure_1_lag.png"))
plt.close(fig)

# ---- Figure 2: nonlinear dose-response (humidity, temperature) ----
rh_lab = ["<20", "20-30", "30-40", "40-50", "50-60", "60+"]
tx_lab = ["<5", "5-10", "10-15", "15-20", "20-25", "25-30", "30+"]
panel["rh_bin"] = pd.cut(panel.rh_min, [0, 20, 30, 40, 50, 60, 200], labels=rh_lab, right=False)
panel["tx_bin"] = pd.cut(panel.tmax, [-50, 5, 10, 15, 20, 25, 30, 100], labels=tx_lab, right=False)
mb = pf.fepois(
    "fires ~ C(rh_bin, contr.treatment(base='60+')) + "
    "C(tx_bin, contr.treatment(base='15-20')) + dryspell + gust | sid+year+month",
    data=panel, vcov={"CRV1": "sid"})
co, se = mb.coef(), mb.se()


def grab(pref, labs, base):
    xs, bs, ss = [], [], []
    for l in labs:
        if l == base:
            xs.append(l); bs.append(0.0); ss.append(0.0); continue
        k = [c for c in co.index if pref in c and f"[T.{l}]" in c]
        if k:
            xs.append(l); bs.append(co[k[0]]); ss.append(se[k[0]])
    return xs, np.array(bs), np.array(ss)


rx, rb, rs = grab("rh_bin", rh_lab, "60+")
tx, tb, ts = grab("tx_bin", tx_lab, "15-20")
out["rh_bin"] = {"x": rx, "b": rb.tolist(), "s": rs.tolist()}
out["tx_bin"] = {"x": tx, "b": tb.tolist(), "s": ts.tolist()}
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.5))
ax[0].axhline(0, color="0.5", lw=.8)
ax[0].errorbar(range(len(rx)), rb, yerr=1.96 * rs, fmt="s-", color="black",
               capsize=2.5, lw=1.4, ms=5, mfc="white", mec="black", mew=1.3)
ax[0].set_xticks(range(len(rx))); ax[0].set_xticklabels(rx, fontsize=8)
ax[0].set_xlabel("Minimum relative humidity (%)")
ax[0].set_ylabel("Log fire rate vs reference")
ax[0].grid(alpha=.3, lw=.5)
ax[0].text(.04, .92, "(a)", transform=ax[0].transAxes, fontweight="bold")
ax[1].axhline(0, color="0.5", lw=.8)
ax[1].errorbar(range(len(tx)), tb, yerr=1.96 * ts, fmt="^-", color="black",
               capsize=2.5, lw=1.4, ms=5, mfc="white", mec="black", mew=1.3)
ax[1].set_xticks(range(len(tx))); ax[1].set_xticklabels(tx, fontsize=8)
ax[1].set_xlabel("Maximum temperature (C)")
ax[1].set_ylabel("Log fire rate vs reference")
ax[1].grid(alpha=.3, lw=.5)
ax[1].text(.04, .92, "(b)", transform=ax[1].transAxes, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "figure_2_dose.png"))
plt.close(fig)

# ---- Figure 3: occurrence by day of week, by cause (grayscale) ----
panel["dow"] = panel.date.dt.dayofweek
g = panel.groupby("dow")[["fires_burn", "fires_acc"]].mean() * 1000
x = np.arange(7); w = 0.38
lab = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
out["dow"] = {"burn": g["fires_burn"].tolist(), "acc": g["fires_acc"].tolist()}
fig, ax = plt.subplots(figsize=(6.2, 3.6))
ax.bar(x - w/2, g["fires_burn"], w, label="Intentional burning",
       color="white", edgecolor="black", hatch="////", lw=.9)
ax.bar(x + w/2, g["fires_acc"], w, label="Accidental escape",
       color="0.45", edgecolor="black", lw=.9)
ax.set_xticks(x); ax.set_xticklabels(lab)
ax.set_ylabel("Fires per 1,000 station-days")
ax.legend(frameon=False, fontsize=9)
ax.grid(axis="y", alpha=.3, lw=.5)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "figure_3_exposure.png"))
plt.close(fig)

json.dump(out, open(os.path.join(OUT, "figure_data.json"), "w"))
print(f"figures written to {OUT}")
