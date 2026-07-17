# -*- coding: utf-8 -*-
"""
The two Table 4 rows that need the author's own data, re-estimated under the
symmetric (cause-specific year) fixed-effects specification.

  (3) day-of-week and public-holiday fixed effects
  (8) control for incoming-visitor exposure, 2022 to 2024
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pyfixest as pf

import config as C
from build_panel import build_panel
from holidays_kr import holidays

VARS = C.VARS
VIS = "/mnt/user-data/uploads/visitors_sgg_2022_2025.csv"


def stack(p, extra_cols=()):
    keep = ["sid", "date", "year", "month", *VARS, *extra_cols]
    b = p[keep].copy(); b["count"] = p["fires_burn"].to_numpy(); b["acc"] = 0
    a = p[keep].copy(); a["count"] = p["fires_acc"].to_numpy();  a["acc"] = 1
    st = pd.concat([b, a], ignore_index=True)
    for v in VARS:
        st[f"acc_{v}"] = st["acc"] * st[v]
    return st


def fit(st, fe, extra=""):
    rhs = " + ".join(VARS) + " + " + " + ".join(f"acc_{v}" for v in VARS) + extra
    return pf.fepois(f"count ~ {rhs} | {fe}", data=st, vcov={"CRV1": "sid"})


def show(tag, m):
    b, p = m.coef()["acc_gust"], m.pvalue()["acc_gust"]
    print(f"  {tag:<52} acc_gust = {b:+.4f}  p = {p:.4f}", flush=True)
    return b, p


panel, f = build_panel()
H = set(holidays())
panel["dow"] = panel.date.dt.dayofweek
panel["hol"] = panel.date.isin(H).astype(int)
print(f"[panel] holiday station-days: {panel.hol.sum():,} of {len(panel):,}\n", flush=True)

# ---------------------------------------------------- row 3: dow + holiday FE
print("=== ROW 3: day-of-week and public-holiday fixed effects ===", flush=True)
st = stack(panel, ("dow", "hol"))
show("(reference) no exposure FE", fit(st, "sid^acc + month^acc + year^acc"))
show("dow + holiday FE, common across causes",
     fit(st, "sid^acc + month^acc + year^acc + dow + hol"))
b3, p3 = show("dow + holiday FE, cause-specific",
              fit(st, "sid^acc + month^acc + year^acc + dow^acc + hol^acc"))
print("  paper row (old spec, dow only): 0.041 / 0.020\n", flush=True)

# --------------------------------------------- row 8: visitor exposure control
print("=== ROW 8: incoming-visitor exposure, 2022 to 2024 ===", flush=True)
m = f.dropna(subset=["sgg_cd", "sid"]).copy()
m["sgg_cd"] = m["sgg_cd"].astype(str).str.zfill(5)
sgg2sid = m.groupby("sgg_cd")["sid"].agg(lambda s: s.value_counts().index[0])

v = pd.read_csv(VIS, encoding="utf-8-sig", dtype=str,
                usecols=["signguCode", "touDivCd", "touNum", "baseYmd"])
v = v[v.touDivCd == "2"].copy()
v["sgg"] = v.signguCode.str.zfill(5)
v["tou"] = pd.to_numeric(v.touNum, errors="coerce")
v["date"] = pd.to_datetime(v.baseYmd, format="%Y%m%d", errors="coerce")
v["sid"] = v.sgg.map(sgg2sid)
exp = (v.dropna(subset=["sid"]).groupby(["sid", "date"])["tou"]
       .sum().rename("exposure").reset_index())
del v
print(f"  visitor series: {exp.date.min().date()} .. {exp.date.max().date()}, "
      f"{len(exp):,} station-days", flush=True)

lo, hi = exp.date.min(), exp.date.max()
p = panel[(panel.date >= lo) & (panel.date <= hi)].copy()
p = p.merge(exp, on=["sid", "date"], how="left").dropna(subset=["exposure"])
p["log_exp"] = np.log1p(p["exposure"])
print(f"  merged panel: {len(p):,} station-days | accidental fires "
      f"{int(p.fires_acc.sum()):,} | burning {int(p.fires_burn.sum()):,}", flush=True)

st2 = stack(p, ("log_exp",))
st2["acc_log_exp"] = st2["acc"] * st2["log_exp"]
FE = "sid^acc + month^acc + year^acc"
show("(a) subsample, no exposure control", fit(st2, FE))
show("(b) + exposure main effect", fit(st2, FE, " + log_exp"))
b8, p8 = show("(c) + exposure main + accidental-specific",
              fit(st2, FE, " + log_exp + acc_log_exp"))
print("  paper row (old spec): 0.041 / 0.017, n accidental 729", flush=True)
