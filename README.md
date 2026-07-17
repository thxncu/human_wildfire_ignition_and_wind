# Setup — `...\산불\Jupyter\`

Extract so the folder looks exactly like this. `code\` is not optional: `config.py`
takes the **parent** of the folder it sits in as the package root, so it has to be
one level down for `data\` to resolve to `Jupyter\data`.

```
Jupyter\
├── wildfire_unified.ipynb
├── code\            <- all .py here (13 files)
├── data\            <- see below
└── output\          <- created on first run, do not make it by hand
```

Nothing needs editing. `PKG = "."` is already the default, and every data path is
derived from `code/config.py` rather than restated in the notebook.

## What goes in `data\`

**The data are not distributed with this package.** All three sources are
public, but their release terms do not permit redistribution. `data_dictionary.md`
records what to obtain, from where, with which filters, and as of when.

| put here | obtain from |
|---|---|
| `fire_records.csv` | Korea Forest Service national forest-fire statistics. Either the English headers or the provider's own will do; `load_fire` renames `resn` to `cause_desc` itself. |
| `weather\region_*.csv` (6 files) | KMA synoptic (ASOS) archive, the 23 stations in `config.STATION_COORDS`. A single combined `weather\weather_all.csv` also works. Korean headers are accepted. |
| `visitors.csv` | Korea Tourism Organization open data portal, via `code\collect_visitors.py`. Needs a service key issued to you; **never commit the key**. Only the exposure-control row of Table 4 depends on it. |
| `data_dictionary.md` | ships with this package — the provenance record |

You do not need to hunt for the right vintage by hand. The notebook's first
check asserts the counts the manuscript reports (5,473 fires; 4,605 human;
2,580 accidental; 1,322 burning; 21 lightning), so a different extract halts
the run with a named discrepancy instead of quietly producing different
numbers.

**Not needed:** the holiday calendar (it is inside `holidays_kr.py`),
`산불통계데이터.csv` (2022-2025 only, no `objt_id`), and
`산림청_산불상태별_이력.xlsx` (2022+ suppression times, no cause field).

## Running

Run the cells in order. The first code cell prints the resolved paths and stops
with an assertion if anything is missing:

```
package root : ...\산불\Jupyter        <- must be the notebook's folder
DATA_DIR     : ...\산불\Jupyter\data
  OK      fire_records  ...
```

If `package root` shows `...\산불` instead, the .py files are still outside `code\`.

Runtime: about 30 minutes with `RI_REPS = 999`. Set it to 220 for a quick pass —
no permutation reaches the observed value either way, so the conclusion is the
same and only the resolution of the p-value changes.

## The two checks

- **CHECK A** — `strict_x ⊆ x`. Strict filters, never re-labels.
- **CHECK B** — `λ_v == β_accidental,v − β_burning,v` for every weather variable.
  This is the one that matters: it fails the moment a fixed effect stops being
  cause-specific, which is what made the dry-spell difference look significant.

Both are `assert`s. If they pass, the classification and the fixed effects each
have exactly one rule.

## Superseded

`재난_산불_wildfire_manuscript.ipynb` should not be run for numbers. Its
classification returns 2,538 accidental / 1,339 burning against this paper's
2,580 / 1,322, so it predates the final rule, and its day-of-week row never
included the public holidays the manuscript claims. Keep it in a `_superseded\`
folder for history.

## Files in `code\`

Unchanged from the replication package: `config.py`, `build_panel.py`,
`analysis.py`, `figures.py`, `suppression.py`, `exposure_control.py`,
`collect_visitors.py`.

New: `strict_cause.py` (strict as a filter), `holidays_kr.py` (2015–2024,
election days and the 2020 temporary holiday included), `subdistrict.py`
(256-cluster design), `dose_gust.py` (Fig. 4), `ri_headline.py`,
`table4_rows.py`.
