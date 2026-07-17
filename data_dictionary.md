# Data: provenance and dictionary

**No data are distributed with this package.** The three sources are publicly
accessible, but the terms on which they are released do not permit
redistribution. This file records what to obtain, from where, with which
filters, and as of when, so that the analysis inputs can be rebuilt exactly.

Place the rebuilt files under `data/` as named below, then run
`wildfire_unified.ipynb`. The first check in that notebook asserts the record
counts the manuscript reports, so an extract of a different vintage stops the
run with a named discrepancy rather than passing silently into different
numbers.

---

## Provenance

All three sources are pinned below, each extracted on **27 June 2026**. That
date is what fixes the vintage: the counts in the check further down are the
counts these extracts yield, and a later extract will differ because the
underlying statistics are revised.

### 1. Wildfire incidents → `data/fire_records.csv`

| | |
|---|---|
| Provider | Ministry of the Interior and Safety, 생활안전지도 (Life Safety Map). The incident records themselves are compiled by the Korea Forest Service and local governments; the Life Safety Map is the release channel and is where the extract used here came from. |
| Dataset / access | 산불발생이력 (wildfire-history) layer, `https://www.safemap.go.kr/opna/data/dataViewRenew.do?objtId=207`, catalogued at `https://www.data.go.kr/data/15149598/openapi.do`. Downloaded as the full extract (`산불발생이력_전체데이터.csv`); columns `objt_id`, `occu_date`, `occu_tm`, `end_year/mt/de/tm`, `amount`, `x`, `y` (EPSG:3857), `sgg_cd`, `resn`. |
| **Date of extraction** | **2026-06-27** |
| Filters applied in code | ignition date within 2015-01-01 to 2024-12-31 (`config.START`, `config.END`); each fire assigned to its nearest of the 23 stations in `config.STATION_COORDS` |
| Rows | 6,676 in the source file; 5,473 inside the analysis window |

Either the English headers below or the provider's own will do:
`build_panel.load_fire` renames `resn` to `cause_desc` itself.

Two related Korea Forest Service datasets were examined and are **not** used, for
the reasons given, and are recorded here so that a reader does not mistake them
for the input: 산림청_산불통계데이터 (`https://www.data.go.kr/data/15121380/fileData.do`)
covers 2022-2025 only and carries no incident identifier; 산림청_산불상태별 이력
(`https://www.data.go.kr/data/15121205/fileData.do`) begins in 2022 and carries
no cause field.

### 2. Station weather → `data/weather/region_*.csv`

| | |
|---|---|
| Provider | Korea Meteorological Administration, National Climate Data Center |
| Dataset | 종관기상관측 (ASOS), daily values. Portal: `https://data.kma.go.kr/data/grnd/selectAsosRltmList.do?pgmNo=36`. Equivalent API: 기상청_지상(종관, ASOS) 일자료 조회서비스, `https://www.data.go.kr/data/15059093/openapi.do`, endpoint `http://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList` |
| Query at source | the 23 stations in `config.STATION_COORDS`, 2015-01-01 to 2024-12-31; elements: maximum temperature, minimum relative humidity, maximum instantaneous wind gust, daily precipitation |
| **Date of extraction** | **2026-06-27** |
| Filters applied in code | station-days missing any of `tmax`, `rh_min`, `gust` are dropped; the balanced grid is 23 stations x 3,653 days = 83,970 station-days |

Six regional files (`region_capital`, `region_gangwon`, `region_chungcheong`,
`region_jeolla`, `region_gyeongsang`, `region_jeju`), or one combined
`data/weather/weather_all.csv`, which the loader falls back to. The portal's
Korean headers (지점번호, 최고기온(°C), 최저상대습도(%), 최대순간풍속(m/s),
일강수량(mm)) are accepted and renamed automatically.

### 3. Incoming visitors → `data/visitors.csv`

Needed only for the exposure-control row of Table 4.

| | |
|---|---|
| Provider | Korea Tourism Organization (agency code B551011), 한국관광 데이터랩 |
| Dataset / service | DataLabService, operation `locgoRegnVisitrDDList`. Endpoint `http://apis.data.go.kr/B551011/DataLabService/locgoRegnVisitrDDList`, catalogued on `https://www.data.go.kr`. Counts derive from mobile-network data and are day-level unique visitors per sub-district; a visitor staying three days counts three times. |
| Access | API; **requires a service key issued to the user**. `code/collect_visitors.py` performs the collection and is resumable. The key is read from `WILDFIRE_API_KEY` and is not distributed. |
| Query at source | sub-district (signgu) daily visitor counts, 2022-01-01 onward |
| **Date of extraction** | **2026-06-27** |
| Filters applied in code | `touDivCd == "2"` (non-resident visitors only); the merge to station-days restricts the row to 2022-2024 |

Because the key is personal, a reader without one cannot reproduce this row.
That is a limitation of the source, not of the code.

---

## Vintage check

`wildfire_unified.ipynb` asserts these counts inside the analysis window. They
identify the extract the manuscript reports and are the reason a wrong vintage
cannot pass unnoticed.

| quantity | value |
|---|---|
| fires, 2015-2024 | 5,473 |
| human-caused | 4,605 |
| accidental | 2,580 |
| intentional burning | 1,322 |
| lightning | 21 |
| unknown | 847 |
| other human | 703 |
| station-days | 83,970 |

---

## Dictionary

### `data/fire_records.csv`

| column | type | description |
|---|---|---|
| objt_id | integer | Incident identifier (one row per fire). |
| occu_date | integer | Ignition date, `YYYYMMDD`. |
| occu_tm | integer | Ignition clock time, `HHMM`. |
| end_year | integer | Containment year. |
| end_mt | integer | Containment month. |
| end_de | integer | Containment day. |
| end_tm | integer | Containment clock time, `HHMM`. |
| amount | float | Burned area. Winsorized at the 99th percentile of positive values in code. |
| x | float | Ignition x-coordinate, projected CRS (EPSG:3857). |
| y | float | Ignition y-coordinate, projected CRS (EPSG:3857). |
| sgg_cd | integer | Sub-district administrative code (5 digits). |
| cause_desc | string | Source cause description, original language. Accepted as `resn`. |

### `data/weather/region_*.csv`

| column | type | description |
|---|---|---|
| sid | integer | Station identifier; coordinates in `config.STATION_COORDS`. |
| date | date | Calendar date. |
| tmax | float | Daily maximum temperature (°C). |
| rh_min | float | Daily minimum relative humidity (%). |
| gust | float | Daily maximum instantaneous wind gust (m/s). |
| precip | float | Daily total precipitation (mm). |

Derived in `build_panel.py`: `dryspell` (consecutive days since precipitation
≥ `config.RAIN_THRESH` = 1.0 mm), `ante7` / `ante30` (antecedent 7- and 30-day
precipitation), `precip_l1`..`precip_l7` (lags).

### `data/visitors.csv`

| column | type | description |
|---|---|---|
| query_ymd | string | Query date, `YYYYMMDD`. |
| signguCode | string | Sub-district code (5 digits; matched to `sgg_cd`). |
| signguNm | string | Sub-district name, original language. |
| daywkDivCd | string | Day-of-week division code. |
| daywkDivNm | string | Day-of-week division name, original language. |
| touDivCd | string | Visitor type: 1 resident, 2 non-resident, 3 foreign. Analysis uses 2. |
| touDivNm | string | Visitor type name, original language. |
| touNum | numeric | Visitor count. |
| baseYmd | string | Reference date, `YYYYMMDD`. |

---

## Cause classification

One taxonomy, in `build_panel.classify_cause`. Matching is on source-language
tokens in `cause_desc`, in this order:

| category | tokens | meaning |
|---|---|---|
| natural | 낙뢰 | Lightning. |
| unknown | 미상, 조사중, or beginning with 기타 | Unknown, under investigation, unspecified. |
| burning | 소각, 두렁 | Escaped intentional burning (refuse, field-margin). |
| accidental | 실화, 담뱃불, 불씨, 부주의 | Accidental human ignition. |
| other_human | remainder | Other human causes (arson, industrial). |

`cause_top` collapses these to `natural`, `unknown`, and `human`.

### Stricter classification (Table 4)

`strict_cause.strict_keep` is a **filter** on the labels above, never a second
taxonomy: a fire keeps whatever `classify_cause` gave it and is retained only
if that label is unambiguous. Nesting therefore holds by construction, and the
notebook asserts it.

| label | retained when `cause_desc` contains |
|---|---|
| burning | 소각 **and** one of 논, 밭, 두렁, 영농, 부산물, 쓰레기, 폐기물 |
| accidental | one of 입산, 등산, 성묘, 취사 **and** one of 실화, 부주의, 실수 |

Everything else leaves the contrast sample. This yields 1,748 accidental and
1,206 burning fires on the panel.

---

## Public holidays

Not a data file. The 2015-2024 calendar, including substitute holidays,
election days, and the 2020 temporary holiday, is embedded in
`code/holidays_kr.py` (172 dates).
