"""Tham so cau hinh dung chung cho toan bo ung dung."""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------- duong dan
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
OUT_FIG = ROOT / "outputs" / "figures"
OUT_RES = ROOT / "outputs" / "results"

for _p in (DATA_RAW, DATA_PROCESSED, OUT_FIG, OUT_RES):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- nguon data
JHU_BASE = (
    "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/"
    "csse_covid_19_data/csse_covid_19_time_series"
)

FILES = {
    "confirmed": "time_series_covid19_confirmed_global.csv",
    "deaths": "time_series_covid19_deaths_global.csv",
    "recovered": "time_series_covid19_recovered_global.csv",
}

# ---------------------------------------------------------------- mac dinh
DEFAULT_COUNTRY = "Vietnam"
DEFAULT_METRIC = "confirmed"      # confirmed | deaths | recovered
DEFAULT_HORIZON = 14              # so ngay du bao
SEASONAL_PERIOD = 7               # chu ky tuan (du lieu dich te theo ngay)
RANDOM_STATE = 42
