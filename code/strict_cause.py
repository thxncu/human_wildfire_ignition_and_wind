# -*- coding: utf-8 -*-
"""
Stricter cause classification (robustness) -- Rule 1.

Strictness is a FILTER applied on top of classify_cause, never a second
taxonomy. A fire keeps the label classify_cause already gave it, and is
retained only if that label is unambiguous:

  burning     an explicit burning token together with an agricultural or
              refuse token
  accidental  an explicit agent (visitor, hiker, grave-visitor, cooking)
              together with an explicit carelessness/escape token

Everything else is dropped from the contrast sample. Because the label is
never rewritten, nesting holds by construction:

    strict_burning    subset of  burning
    strict_accidental subset of  accidental

This matters. The earlier rule evaluated its own burning and accidental
branches in sequence and so could re-label: four grave-visitor fires
("성묘객 실화(유품소각)" and similar) are burning under classify_cause but
were flipped to accidental, because the burning branch demanded an
agricultural token and the accidental branch then matched. Ambiguous records
should leave the sample, not change sides.
"""
AG_WASTE = ["논", "밭", "두렁", "영농", "부산물", "쓰레기", "폐기물"]
AGENTS = ["입산", "등산", "성묘", "취사"]
ACTS = ["실화", "부주의", "실수"]


def strict_keep(cause_desc, cause):
    """True if `cause` (from classify_cause) is unambiguous for this record."""
    s = "" if cause_desc is None else str(cause_desc)
    has = lambda kws: any(k in s for k in kws)
    if cause == "burning":
        return ("소각" in s) and has(AG_WASTE)
    if cause == "accidental":
        return has(AGENTS) and has(ACTS)
    return False
