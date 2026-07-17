# -*- coding: utf-8 -*-
"""
Suppression-duration analysis.

Examines time from ignition to containment for human-caused fires:
  (A) duration by cause (accidental vs intentional burning), with a
      Mann-Whitney comparison;
  (B) whether gust speed lengthens duration and whether that effect differs
      by cause (log-duration regression with a gust-by-accidental interaction
      and fixed effects);
  (B2) mean duration under low vs high gust by cause;
  (C) an illustrative counterfactual for the accidental suppression burden on
      high-gust days.

Run from the code/ directory:  python suppression.py
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pyfixest as pf
from scipy import stats

import config as C
from build_panel import build_panel

VARS = C.VARS


def to_minutes(tm):
    # Parse an HHMM-style clock value into minutes since midnight.
    try:
        s = str(int(float(tm)))
    except Exception:
        return np.nan
    if len(s) == 0 or len(s) > 4:
        return np.nan
    s = s.zfill(3)
    mm, hh = int(s[-2:]), int(s[:-2])
    if mm > 59 or hh > 23:
        return np.nan
    return hh * 60 + mm


if __name__ == "__main__":
    panel, f = build_panel()
    fs = f[(f.date >= C.START) & (f.date <= C.END)].copy()

    # Build ignition and containment datetimes; duration in hours.
    fs["ig_min"] = fs["occu_tm"].apply(to_minutes)
    fs["en_min"] = fs["end_tm"].apply(to_minutes)
    fs["ig"] = (pd.to_datetime(fs["occu_date"], format="%Y%m%d", errors="coerce")
                + pd.to_timedelta(fs["ig_min"], unit="m"))
    fs["en"] = (pd.to_datetime(dict(year=fs["end_year"].astype(float),
                                    month=fs["end_mt"].astype(float),
                                    day=fs["end_de"].astype(float)), errors="coerce")
                + pd.to_timedelta(fs["en_min"], unit="m"))
    fs["dur_h"] = (fs["en"] - fs["ig"]).dt.total_seconds() / 3600

    # Attach same-day station weather.
    dayw = panel.set_index(["sid", "date"])[VARS]
    fs = fs.merge(dayw, left_on=["sid", "date"], right_index=True, how="left")

    hum = fs[fs.cause.isin(["burning", "accidental"])].copy()
    hum = hum[(hum.dur_h > 0) & (hum.dur_h < 48)].dropna(subset=["gust", "dur_h"])
    print(f"[fires] human, 0<dur<48h: {len(hum)} "
          f"(accidental {(hum.cause=='accidental').sum()}, burning {(hum.cause=='burning').sum()})")

    print("\n=== (A) Duration by cause (ignition to containment, hours) ===")
    for c, lab in [("burning", "burning"), ("accidental", "accidental")]:
        d = hum[hum.cause == c].dur_h
        print(f"  {lab:11s}: n={len(d):4d} mean={d.mean():.2f}h median={d.median():.2f}h p90={d.quantile(.9):.2f}h")
    db = hum[hum.cause == "burning"].dur_h
    da = hum[hum.cause == "accidental"].dur_h
    print(f"  Mann-Whitney p={stats.mannwhitneyu(da, db).pvalue:.4f}")

    print("\n=== (B) Does gust lengthen duration, and does the effect differ by cause? ===")
    hum["acc"] = (hum.cause == "accidental").astype(int)
    hum["ldur"] = np.log(hum.dur_h)
    hum["gust_acc"] = hum.gust * hum.acc
    hum["mon"] = hum.date.dt.month
    hum["yr"] = hum.date.dt.year
    # Rule 2: a model that asks whether the gust effect differs by cause must let
    # every baseline differ by cause too. With common station, month, and year
    # effects this reads +0.009 (p = 0.014) and contradicts the manuscript; with
    # cause-specific ones it is -0.009 (p = 0.42), which is what the text says.
    for _v in ["rh_min", "tmax", "dryspell"]:
        hum[f"acc_{_v}"] = hum[_v] * hum["acc"]
    m = pf.feols("ldur ~ gust + gust_acc + rh_min + acc_rh_min + tmax + acc_tmax "
                 "+ dryspell + acc_dryspell | sid^acc + mon^acc + yr^acc",
                 data=hum, vcov={"CRV1": "sid"})
    print(f"  gust (burning baseline)   = {m.coef()['gust']:+.4f} (p={m.pvalue()['gust']:.3f})")
    print(f"  gust x accidental (extra) = {m.coef()['gust_acc']:+.4f} (p={m.pvalue()['gust_acc']:.3f})")

    print("\n=== (B2) Mean duration under low vs high gust, by cause ===")
    gq = hum.gust.quantile(.75)
    for c, lab in [("accidental", "accidental"), ("burning", "burning")]:
        d = hum[hum.cause == c]
        lo = d[d.gust < gq].dur_h.mean()
        hi = d[d.gust >= gq].dur_h.mean()
        print(f"  {lab:11s}: low gust {lo:.2f}h -> high gust (top 25%) {hi:.2f}h "
              f"(change {hi-lo:+.2f}h, {100*(hi-lo)/lo:+.0f}%)")

    print("\n=== (C) Where the accidental suppression burden falls ===")
    # The manuscript says "the windiest quarter of station-DAYS", so the cut must be
    # a quartile of the panel, not of the fires. Fires cluster on windy days, so the
    # fire-level 75th percentile sits higher and returns 30% where the day-level cut
    # returns 28%. Two definitions, one sentence -- this is the day-level one.
    acc = hum[hum.cause == "accidental"]
    brn = hum[hum.cause == "burning"]
    yrs = hum.yr.nunique()
    q4 = pd.qcut(panel["gust"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    qmap = panel.assign(q=q4).set_index(["sid", "date"])["q"]
    acc_q = qmap.reindex(pd.MultiIndex.from_arrays([acc.sid, acc.date])).to_numpy()
    print(f"  annual control-hours: accidental {acc.dur_h.sum()/yrs:.0f}, "
          f"burning {brn.dur_h.sum()/yrs:.0f}")
    share = acc.dur_h[acc_q == "Q4"].sum() / acc.dur_h.sum()
    print(f"  share of accidental control-hours on the windiest quarter of station-days: "
          f"{share:.1%}")
    print("  No counterfactual is computed. The design identifies how occurrence moves")
    print("  with wind, not how it moves with a patrol; see burden.py for why the")
    print("  burden decomposition is too noisy to carry an interpretation.")
