# -*- coding: utf-8 -*-
"""
Exposure-control robustness.

Realized fire counts reflect human presence as well as flammability. This
module merges a daily station-level exposure series (incoming non-resident
visitor counts, summed from sub-district records) onto the panel and checks
whether controlling for exposure changes the accidental-minus-burning gust
contrast from the stacked specification.

The visitor file is optional and is not redistributed in bulk; obtain it with
code/collect_visitors.py from the public open-data API, or point VISITOR_PATH
in config.py to a local copy. Expected columns are described in
data/data_dictionary.md.

Run from the code/ directory:  python exposure_control.py
"""
import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pyfixest as pf

import config as C
from build_panel import build_panel

VARS = C.VARS


def load_exposure(fires):
    if not os.path.exists(C.VISITOR_PATH):
        raise SystemExit(
            f"Visitor file not found at {C.VISITOR_PATH}. "
            "See README; run collect_visitors.py or set VISITOR_PATH.")
    # Map each sub-district to its modal nearest station (from the fire records).
    m = fires.dropna(subset=["sgg_cd", "sid"]).copy()
    m["sgg_cd"] = m["sgg_cd"].astype(str).str.zfill(5)
    sgg2sid = m.groupby("sgg_cd")["sid"].agg(lambda x: x.value_counts().index[0])

    v = pd.read_csv(C.VISITOR_PATH, dtype=str)
    v = v[v.touDivCd == "2"].copy()                 # non-resident visitors only
    v["sgg"] = v.signguCode.astype(str).str.zfill(5)
    v["tou"] = pd.to_numeric(v.touNum, errors="coerce")
    v["date"] = pd.to_datetime(v.baseYmd, format="%Y%m%d", errors="coerce")
    v["sid"] = v.sgg.map(sgg2sid)
    exp = (v.dropna(subset=["sid"]).groupby(["sid", "date"])["tou"]
           .sum().rename("exposure").reset_index())
    return exp


def stacked(p, extra=""):
    keep = ["sid", "date", "year", "month", "exposure", "log_exp", *VARS]
    b = p[keep].copy(); b["count"] = p["fires_burn"]; b["acc"] = 0
    a = p[keep].copy(); a["count"] = p["fires_acc"];  a["acc"] = 1
    st = pd.concat([b, a], ignore_index=True)
    for v in VARS:
        st[f"acc_{v}"] = st["acc"] * st[v]
    st["acc_log_exp"] = st["acc"] * st["log_exp"]
    rhs = " + ".join(VARS) + " + " + " + ".join(f"acc_{v}" for v in VARS) + extra
    return pf.fepois(f"count ~ {rhs} | sid^acc + month^acc + year",
                     data=st, vcov={"CRV1": "sid"})


if __name__ == "__main__":
    panel, f = build_panel()
    exp = load_exposure(f)

    lo, hi = exp.date.min(), exp.date.max()
    p = panel[(panel.date >= lo) & (panel.date <= hi)].copy()
    p = p.merge(exp, on=["sid", "date"], how="left").dropna(subset=["exposure"])
    p["log_exp"] = np.log1p(p["exposure"])
    print(f"[panel] {len(p)} station-days | {p.date.min().date()}..{p.date.max().date()} "
          f"| stations {p.sid.nunique()}")
    print(f"[fire-days] accidental={int((p.fires_acc > 0).sum())}, "
          f"burning={int((p.fires_burn > 0).sum())}")

    print("\n=== acc_gust (accidental-minus-burning gust response): before vs after exposure control ===")
    m0 = stacked(p)
    print(f"(a) subsample, no exposure control       : acc_gust={m0.coef()['acc_gust']:+.4f}  p={m0.pvalue()['acc_gust']:.3f}")
    m1 = stacked(p, extra=" + log_exp")
    print(f"(b) + exposure main effect               : acc_gust={m1.coef()['acc_gust']:+.4f}  p={m1.pvalue()['acc_gust']:.3f} | log_exp={m1.coef()['log_exp']:+.4f} (p={m1.pvalue()['log_exp']:.3f})")
    m2 = stacked(p, extra=" + log_exp + acc_log_exp")
    print(f"(c) + exposure main + accidental-specific : acc_gust={m2.coef()['acc_gust']:+.4f}  p={m2.pvalue()['acc_gust']:.3f} | acc_log_exp={m2.coef()['acc_log_exp']:+.4f} (p={m2.pvalue()['acc_log_exp']:.3f})")

    print("\n=== Reference: does exposure itself predict occurrence (single-equation) ===")
    pa = p.copy(); pa["count"] = pa["fires_acc"]
    ma = pf.fepois(f"count ~ {' + '.join(VARS)} + log_exp | sid + month + year", data=pa, vcov={"CRV1": "sid"})
    print(f"  accidental: log_exp={ma.coef()['log_exp']:+.4f} (p={ma.pvalue()['log_exp']:.3f}), gust={ma.coef()['gust']:+.4f} (p={ma.pvalue()['gust']:.3f})")
    pb = p.copy(); pb["count"] = pb["fires_burn"]
    mb = pf.fepois(f"count ~ {' + '.join(VARS)} + log_exp | sid + month + year", data=pb, vcov={"CRV1": "sid"})
    print(f"  burning   : log_exp={mb.coef()['log_exp']:+.4f} (p={mb.pvalue()['log_exp']:.3f}), gust={mb.coef()['gust']:+.4f} (p={mb.pvalue()['gust']:.3f})")
