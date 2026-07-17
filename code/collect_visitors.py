"""
Optional collector for the daily sub-district visitor series used in the
exposure-control robustness check.

Source: national open-data portal, regional visitor statistics
(operation: locgoRegnVisitrDDList). One request returns all sub-districts for
a given date across three visitor types; the non-resident type (touDivCd == 2)
is used as the exposure proxy.

Authentication
--------------
Supply your own service key via the WILDFIRE_API_KEY environment variable or
the --key argument. No key is distributed with this package.

Usage
-----
  pip install requests
  export WILDFIRE_API_KEY="your-service-key"
  python collect_visitors.py --probe --ymd 20230401
  python collect_visitors.py --start 20220101 --end 20241231 --out ../data/visitors.csv

Output columns: query_ymd, signguCode, signguNm, daywkDivCd, daywkDivNm,
                touDivCd, touDivNm, touNum, baseYmd  (see data_dictionary.md).
The collector appends and is resumable: re-running continues from dates not
already present in the output file.
"""
import argparse
import csv
import os
import time
from datetime import date, timedelta

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

OP = "locgoRegnVisitrDDList"
NUM_ROWS = 1000
REQ_CAP = 950          # per-session call cap (protects a development quota)
SLEEP = 0.3
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
FIELDS = ["signguCode", "signguNm", "daywkDivCd", "daywkDivNm",
          "touDivCd", "touDivNm", "touNum", "baseYmd"]


def make_session():
    s = requests.Session()
    s.headers.update({"User-Agent": UA})
    retry = Retry(total=5, backoff_factor=1.0,
                  status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"])
    s.mount("http://", HTTPAdapter(max_retries=retry))
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


def call_api(s, base, key, ymd, page=1):
    params = {"serviceKey": key, "numOfRows": NUM_ROWS, "pageNo": page,
              "MobileOS": "ETC", "MobileApp": "research",
              "startYmd": ymd, "endYmd": ymd, "_type": "json"}
    r = s.get(f"{base}/{OP}", params=params, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code} | body (first 300): {r.text[:300]}")
    try:
        j = r.json()
    except ValueError:
        raise RuntimeError(f"non-JSON response (first 400): {r.text[:400]}")
    resp = j.get("response", {})
    header = resp.get("header", {})
    body = resp.get("body", {}) or {}
    items = body.get("items", "")
    if not items:
        item_list = []
    else:
        node = items.get("item", [])
        item_list = node if isinstance(node, list) else [node]
    total = int(body.get("totalCount", 0) or 0)
    return header, item_list, total


def daterange(a, b):
    while a <= b:
        yield a
        a += timedelta(days=1)


def parse_ymd(s):
    return date(int(s[:4]), int(s[4:6]), int(s[6:8]))


def done_dates(path):
    seen = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if row.get("baseYmd"):
                    seen.add(row["baseYmd"])
    return seen


def probe(s, base, key, ymd):
    print(f"[probe] {OP} ymd={ymd} base={base}")
    header, items, total = call_api(s, base, key, ymd)
    print("[probe] header =", header, "| totalCount =", total, "| page-1 items =", len(items))
    if items:
        print("[probe] first item:", items[0])
        miss = [f for f in FIELDS if f not in items[0]]
        print("[probe] missing fields:", miss if miss else "none (schema matches)")
    else:
        print("[probe] 0 rows. Check header resultCode (00 ok / 30 key not registered / 22 quota).")


def collect(s, base, key, start, end, out):
    seen = done_dates(out)
    new = not os.path.exists(out)
    calls = 0
    with open(out, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["query_ymd"] + FIELDS)
        if new:
            w.writeheader()
        for d in daterange(start, end):
            ymd = d.strftime("%Y%m%d")
            if ymd in seen:
                continue
            if calls >= REQ_CAP:
                print(f"[stop] session cap reached; resume from {ymd} on next run.")
                break
            try:
                page, got = 1, 0
                while True:
                    header, items, total = call_api(s, base, key, ymd, page)
                    calls += 1
                    code = str(header.get("resultCode", ""))
                    if code not in ("00", "0000", ""):
                        print(f"[warn] {ymd} resultCode={code} msg={header.get('resultMsg')}")
                        break
                    for it in items:
                        w.writerow({"query_ymd": ymd, **{k: it.get(k, "") for k in FIELDS}})
                        got += 1
                    if got >= total or len(items) < NUM_ROWS:
                        break
                    page += 1
                    time.sleep(SLEEP)
                print(f"[ok] {ymd}: {got} rows (calls={calls})")
            except Exception as e:
                print(f"[err] {ymd}: {e}  -> retry on next run")
            f.flush()
            time.sleep(SLEEP)
    print(f"[done] -> {out} (this session {calls} calls)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--ymd", default="20230401")
    ap.add_argument("--start", default="20220101")
    ap.add_argument("--end", default="20241231")
    ap.add_argument("--out", default="../data/visitors.csv")
    ap.add_argument("--https", action="store_true", help="force https (default follows guide: http)")
    ap.add_argument("--key", default=os.environ.get("WILDFIRE_API_KEY", ""),
                    help="service key; defaults to the WILDFIRE_API_KEY environment variable")
    a = ap.parse_args()
    if not a.key:
        raise SystemExit("No service key. Set WILDFIRE_API_KEY or pass --key.")
    base = ("https" if a.https else "http") + "://apis.data.go.kr/B551011/DataLabService"
    s = make_session()
    if a.probe:
        probe(s, base, a.key, a.ymd)
    else:
        collect(s, base, a.key, parse_ymd(a.start), parse_ymd(a.end), a.out)


if __name__ == "__main__":
    main()
