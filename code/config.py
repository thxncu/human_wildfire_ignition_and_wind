# -*- coding: utf-8 -*-
"""
Configuration: paths, constants, and station coordinates.

All paths are resolved relative to the package root so the package runs
without modification after extraction. Override DATA_DIR with the
WILDFIRE_DATA_DIR environment variable if the data live elsewhere.
"""
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("WILDFIRE_DATA_DIR", os.path.join(PKG_ROOT, "data"))
WEATHER_DIR = os.path.join(DATA_DIR, "weather")
FIRE_PATH = os.path.join(DATA_DIR, "fire_records.csv")
VISITOR_PATH = os.path.join(DATA_DIR, "visitors.csv")   # optional; see README
OUTPUT_DIR = os.path.join(PKG_ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Regional weather files (without extension). The loader concatenates these;
# if none are found it falls back to a single combined file "weather_all.csv".
REGION_FILES = ["region_capital", "region_gangwon", "region_chungcheong",
                "region_jeolla", "region_gyeongsang", "region_jeju"]
WEATHER_COMBINED = "weather_all.csv"

# ---------------------------------------------------------------------------
# Sample window and model constants
# ---------------------------------------------------------------------------
START, END = "2015-01-01", "2024-12-31"
RAIN_THRESH = 1.0     # mm; daily precipitation at or above this resets the dry spell
MAX_LAG = 7           # days; maximum precipitation lag retained
WCB_REPS = 9999       # wild cluster bootstrap replications

# Right-hand-side weather covariates shared across specifications
RHS = "dryspell + rh_min + gust + tmax"
VARS = ["dryspell", "rh_min", "gust", "tmax"]

# ---------------------------------------------------------------------------
# Weather-station coordinates: station_id -> (latitude, longitude)
# Standard surface-station coordinates. Verify against the current provider
# station registry before operational use.
# ---------------------------------------------------------------------------
STATION_COORDS = {
    101: (37.9026, 127.7357), 105: (37.7515, 128.8910), 108: (37.5714, 126.9658),
    112: (37.4776, 126.6624), 114: (37.3375, 127.9466), 119: (37.2723, 126.9853),
    130: (36.9918, 129.4128), 131: (36.6392, 127.4407), 133: (36.3720, 127.3721),
    135: (36.2204, 127.9946), 136: (37.5071, 129.1243), 143: (35.8780, 128.6526),
    146: (35.8409, 127.1190), 152: (35.5821, 129.3320), 155: (35.1701, 128.5727),
    156: (35.1729, 126.8916), 159: (35.1047, 129.0320), 165: (34.8172, 126.3814),
    168: (34.7393, 127.7406), 184: (33.5141, 126.5297), 189: (33.2461, 126.5653),
    192: (35.1641, 128.0401), 232: (36.7766, 127.1219),
}
