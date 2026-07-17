# -*- coding: utf-8 -*-
"""
Main analysis.

(1) Cause-stratified occurrence models (Poisson PPML with station, year, and
    month fixed effects; station-clustered standard errors).
(2) Stacked interaction specification that formally tests whether the
    occurrence response to gust speed differs between accidental and
    intentional-burning fires (the headline contrast), reporting the
    interaction coefficient with a 95% confidence interval.
(3) Wild cluster bootstrap robustness for inference with a small number of
    station clusters, applied to a linear probability model of occurrence and
    to an OLS model of log burned area.

Run from the code/ directory:  python analysis.py
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pyfixest as pf

import config as C
from build_panel import build_panel

FE = "| sid + year + month"


def tidy(model, name):
    co, se, pv = model.coef(), model.se(), model.pvalue()
    return pd.DataFrame({(name, "coef"): co[C.VARS].round(4),
                         (name, "se"): se[C.VARS].round(4),
                         (name, "p"): pv[C.VARS].round(4)})


def stacked_difference(panel):
    """
    Pool burning and accidental counts and interact every weather covariate
    with an accidental indicator. Fixed effects are interacted with the
    indicator (station-by-cause, month-by-cause) so the contrast is identified
    within cause. The acc_gust coefficient is the accidental-minus-burning
    difference in the gust response.
    """
    keep = ["sid", "date", "year", "month", *C.VARS]
    b = panel[keep].copy(); b["count"] = panel["fires_burn"]; b["acc"] = 0
    a = panel[keep].copy(); a["count"] = panel["fires_acc"];  a["acc"] = 1
    st = pd.concat([b, a], ignore_index=True)
    for v in C.VARS:
        st[f"acc_{v}"] = st["acc"] * st[v]
    rhs = " + ".join(C.VARS) + " + " + " + ".join(f"acc_{v}" for v in C.VARS)
    m = pf.fepois(f"count ~ {rhs} | sid^acc + month^acc + year^acc",
                  data=st, vcov={"CRV1": "sid"})
    return m


def wcb_table(model, label):
    rows = []
    for v in C.VARS:
        wb = model.wildboottest(param=v, reps=C.WCB_REPS, seed=42)
        rows.append({"var": v, "coef": round(model.coef()[v], 4),
                     "p_cluster_robust": round(model.pvalue()[v], 4),
                     "p_wild_bootstrap": round(float(wb["Pr(>|t|)"]), 4)})
    out = pd.DataFrame(rows).set_index("var")
    out.columns = pd.MultiIndex.from_product([[label], out.columns])
    return out


if __name__ == "__main__":
    panel, f = build_panel()
    print(f"[panel] station-days {len(panel):,} | "
          f"fire-days {(panel.fires > 0).sum():,} ({(panel.fires > 0).mean() * 100:.1f}%)")
    print("[cause, top-level]", f["cause_top"].value_counts().to_dict())

    # (1) Cause-stratified occurrence (Poisson PPML)
    specs = {"all": "fires", "human": "fires_hum", "natural": "fires_nat",
             "burning": "fires_burn", "accidental": "fires_acc"}
    tabs = []
    for name, dep in specs.items():
        try:
            m = pf.fepois(f"{dep} ~ {C.RHS} {FE}", data=panel, vcov={"CRV1": "sid"})
            tabs.append(tidy(m, name))
        except Exception as e:
            print(f"  [{name}] estimation skipped (sparse): {repr(e)[:70]}")
    print("\n===== (1) Cause-stratified occurrence (Poisson FE; coef = log-rate marginal effect) =====")
    print(pd.concat(tabs, axis=1).to_string())

    # (2) Stacked interaction: accidental-minus-burning gust difference
    md = stacked_difference(panel)
    b = md.coef()["acc_gust"]; se = md.se()["acc_gust"]; p = md.pvalue()["acc_gust"]
    lo, hi = b - 1.96 * se, b + 1.96 * se
    print("\n===== (2) Stacked interaction: accidental vs intentional-burning gust response =====")
    print(f"  acc_gust (difference) = {b:+.4f}  SE {se:.4f}  p {p:.3f}  95% CI [{lo:+.4f}, {hi:+.4f}]")

    # (3) Wild cluster bootstrap (few-cluster inference)
    fire_days = panel[panel.fires > 0].copy()
    m_spread = pf.feols(f"log_area ~ {C.RHS} {FE}", data=fire_days, vcov={"CRV1": "sid"})
    m_lpm = pf.feols(f"fire_ind ~ {C.RHS} {FE}", data=panel, vcov={"CRV1": "sid"})
    print(f"\n===== (3) Wild cluster bootstrap ({len(C.STATION_COORDS)} station clusters, reps={C.WCB_REPS}) =====")
    print(pd.concat([wcb_table(m_lpm, "occurrence_LPM"),
                     wcb_table(m_spread, "spread_logArea")], axis=1).to_string())
