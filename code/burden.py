# -*- coding: utf-8 -*-
"""
Section 4.8 -- operational stakes, under rule 2.

Two things the old suppression.py got wrong, both from the same cause:

  1. Its duration regression absorbed station, month, and year effects that were
     common across causes while asking whether the gust effect differs by cause.
     That is the asymmetry that manufactured a spurious dry-spell difference in
     the occurrence model, and it manufactures one here too: the gust-by-cause
     interaction reads +0.009 (p = 0.014) under common fixed effects and
     -0.009 (p = 0.42) once every baseline is cause-specific. The manuscript's
     text -- that the lengthening is common to both causes -- was right; its
     code was not.

  2. Its counterfactual multiplied the high-gust burden by 0.30 while the
     manuscript reported 30% as the *share* of the burden falling on windy days.
     Two different quantities, both 30%, one paragraph. Here the prevention rate
     is a named argument and the scenario is reported as a schedule, so the
     reader can see which assumption produces which number.

Outputs: output/burden_by_quartile.csv, output/prevention_scenarios.csv
"""
import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pyfixest as pf

import config as C
from build_panel import build_panel
from suppression import to_minutes

VARS = C.VARS


def human_fires():
    """Human fires with a usable ignition-to-containment duration and day weather."""
    panel, f = build_panel()
    fs = f[(f.date >= C.START) & (f.date <= C.END)].copy()
    fs["ig_min"] = fs["occu_tm"].apply(to_minutes)
    fs["en_min"] = fs["end_tm"].apply(to_minutes)
    fs["ig"] = (pd.to_datetime(fs["occu_date"], format="%Y%m%d", errors="coerce")
                + pd.to_timedelta(fs["ig_min"], unit="m"))
    fs["en"] = (pd.to_datetime(dict(year=fs["end_year"].astype(float),
                                    month=fs["end_mt"].astype(float),
                                    day=fs["end_de"].astype(float)), errors="coerce")
                + pd.to_timedelta(fs["en_min"], unit="m"))
    fs["dur_h"] = (fs["en"] - fs["ig"]).dt.total_seconds() / 3600
    fs = fs.merge(panel.set_index(["sid", "date"])[VARS],
                  left_on=["sid", "date"], right_index=True, how="left")
    h = fs[fs.cause.isin(["burning", "accidental"])].copy()
    h = h[(h.dur_h > 0) & (h.dur_h < 48)].dropna(subset=["gust", "dur_h"])
    h["acc"] = (h.cause == "accidental").astype(int)
    h["ldur"] = np.log(h.dur_h)
    h["mon"], h["yr"] = h.date.dt.month, h.date.dt.year
    return panel, h


if __name__ == "__main__":
    panel, h = human_fires()
    yrs = h.yr.nunique()
    print(f"[fires] {len(h):,} human fires with duration "
          f"(accidental {int(h.acc.sum()):,}, burning {int((1-h.acc).sum()):,}), {yrs} years")

    # ---- is the lengthening selective? (rule 2: every baseline cause-specific)
    h["gust_acc"] = h.gust * h.acc
    for v in ["rh_min", "tmax", "dryspell"]:
        h[f"acc_{v}"] = h[v] * h["acc"]
    m = pf.feols("ldur ~ gust + gust_acc + rh_min + acc_rh_min + tmax + acc_tmax "
                 "+ dryspell + acc_dryspell | sid^acc + mon^acc + yr^acc",
                 data=h, vcov={"CRV1": "sid"})
    print("\n=== is the wind lengthening selective? (rule 2) ===")
    print(f"  gust x accidental = {m.coef()['gust_acc']:+.4f}  p = {m.pvalue()['gust_acc']:.3f}"
          f"  -> {'selective' if m.pvalue()['gust_acc'] < .05 else 'common to both causes'}")

    # the interaction is not needed, so report the common effect
    mc = pf.feols("ldur ~ gust + rh_min + tmax + dryspell | sid^acc + mon^acc + yr^acc",
                  data=h, vcov={"CRV1": "sid"})
    g = mc.coef()["gust"]
    print(f"  common gust effect on duration = {g:+.4f} per m/s "
          f"({100*(np.exp(g)-1):+.1f}% per m/s, p = {mc.pvalue()['gust']:.3f})")

    # ---- does the burden itself respond selectively to wind?
    # Table 3 asks the question of occurrence. The same question can be put to
    # control-hours, which combine occurrence and duration in one outcome.
    idx = pd.MultiIndex.from_arrays([panel["sid"], panel["date"]])

    def hours_per_day(cause):
        g = h[h.cause == cause].groupby(["sid", "date"])["dur_h"].sum()
        return g.reindex(idx, fill_value=0.0).to_numpy()

    HB, HA = hours_per_day("burning"), hours_per_day("accidental")
    print("\n=== elasticity of the suppression burden with respect to gust ===")
    print(f"  annual control-hours: burning {HB.sum()/yrs:.0f}, accidental {HA.sum()/yrs:.0f}")
    for lab, y in [("burning", HB), ("accidental", HA)]:
        mm = pf.fepois(f"H ~ {C.RHS} | sid + year + month",
                       data=panel.assign(H=y), vcov={"CRV1": "sid"})
        print(f"  {lab:<11} {mm.coef()['gust']:+.4f} per m/s "
              f"({100*(np.exp(mm.coef()['gust'])-1):+.1f}%/m/s, p={mm.pvalue()['gust']:.3f})")

    keep = ["sid", "date", "year", "month", *VARS]
    bb = panel[keep].copy(); bb["H"] = HB; bb["acc"] = 0
    aa = panel[keep].copy(); aa["H"] = HA; aa["acc"] = 1
    stk = pd.concat([bb, aa], ignore_index=True)
    for v in VARS:
        stk[f"acc_{v}"] = stk["acc"] * stk[v]
    rhs = " + ".join(VARS) + " + " + " + ".join(f"acc_{v}" for v in VARS)
    mb = pf.fepois(f"H ~ {rhs} | sid^acc + month^acc + year^acc",
                   data=stk, vcov={"CRV1": "sid"})
    gb, seb = mb.coef()["acc_gust"], mb.se()["acc_gust"]
    from scipy import stats as _st
    OCC = 0.0399   # the occurrence difference this test would have to see
    lo, hi = gb - 1.96*seb, gb + 1.96*seb
    pw = 1 - _st.norm.cdf(1.96 - OCC/seb) + _st.norm.cdf(-1.96 - OCC/seb)
    print(f"  accidental - burning = {gb:+.4f}  SE {seb:.4f}  p = {mb.pvalue()['acc_gust']:.3f}"
          f"  95% CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"  cannot reject 0        (p = {mb.pvalue()['acc_gust']:.2f})")
    z2 = (gb - OCC) / seb
    print(f"  cannot reject {OCC:.3f}    (p = {2*(1-_st.norm.cdf(abs(z2))):.2f})")
    print(f"  power against {OCC:.3f}: {pw:.0%}   MDE = {2.8*seb:.3f}")
    print("  DO NOT report this as evidence of no difference. The interval contains both")
    print("  zero and the value the occurrence contrast implies, so the test separates")
    print("  nothing: control-hours carry the variance of duration on top of the variance")
    print("  of counts, and 23 clusters cannot pay for both. The duration regression above")
    print("  is the informative one -- its interval excludes +0.040 with 96% power. The")
    print("  cause-specific claim therefore rests on occurrence, where it is estimated")
    print("  precisely, and this decomposition stays out of the manuscript.")

    # ---- where the burden sits
    print("\n=== accidental suppression burden by gust quartile ===")
    q = pd.qcut(panel["gust"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    qmap = panel.assign(q=q).set_index(["sid", "date"])["q"]
    h["q"] = qmap.reindex(pd.MultiIndex.from_arrays([h.sid, h.date])).to_numpy()
    days = panel.assign(q=q).groupby("q", observed=True).size()

    acc = h[h.acc == 1]
    tot = acc.dur_h.sum() / yrs
    rows = []
    for k in ["Q1", "Q2", "Q3", "Q4"]:
        a = acc[acc.q == k]
        b = h[(h.acc == 0) & (h.q == k)]
        hours = a.dur_h.sum() / yrs
        rows.append({
            "quartile": k,
            "station_days": int(days[k]),
            "acc_fires_per_yr": round(len(a) / yrs, 1),
            "acc_hours_per_yr": round(hours, 1),
            "share_of_acc_hours": round(a.dur_h.sum() / acc.dur_h.sum(), 3),
            "hours_per_1000_station_days": round(1000 * a.dur_h.sum() / days[k], 1),
            "acc_share_of_human_hours": round(
                a.dur_h.sum() / (a.dur_h.sum() + b.dur_h.sum()), 3),
        })
    bq = pd.DataFrame(rows)
    print(bq.to_string(index=False))
    print(f"\n  annual accidental burden: {tot:.0f} control-hours/yr")
    print(f"  Q4 carries {bq.loc[3,'share_of_acc_hours']:.1%} of it on "
          f"{days['Q4']/days.sum():.1%} of station-days "
          f"-> {bq.loc[3,'share_of_acc_hours']/(days['Q4']/days.sum()):.2f}x the average day")

    # ---- the scenario, with the prevention rate named rather than buried
    print("\n=== illustrative prevention schedule on the windiest quarter ===")
    q4_hours = acc[acc.q == "Q4"].dur_h.sum() / yrs
    sc = []
    for rate in [0.10, 0.20, 0.30]:
        saved = rate * q4_hours
        sc.append({"prevention_rate_on_Q4": rate,
                   "hours_avoided_per_yr": round(saved, 1),
                   "share_of_annual_accidental_burden": round(saved / tot, 3)})
        print(f"  remove {rate:.0%} of Q4 accidental ignitions -> "
              f"{saved:5.1f} control-hours/yr avoided ({saved/tot:.1%} of the annual burden)")
    print("\n  The rate is an assumption, not an estimate: the design identifies how "
          "occurrence\n  moves with wind, not how it moves with a patrol.")

    bq.to_csv(os.path.join(C.OUTPUT_DIR, "burden_by_quartile.csv"), index=False)
    pd.DataFrame(sc).to_csv(os.path.join(C.OUTPUT_DIR, "prevention_scenarios.csv"), index=False)
    print("\nwritten: output/burden_by_quartile.csv, output/prevention_scenarios.csv")
