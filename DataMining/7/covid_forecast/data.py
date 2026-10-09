"""
Tai va tien xu ly du lieu COVID-19 (JHU CSSE time series).

Dinh dang goc cua JHU la "wide": moi dong la mot vung lanh tho,
moi cot (tu cot thu 5 tro di) la mot ngay -> gia tri LUY KE (cumulative).

    Province/State | Country/Region | Lat | Long | 1/22/20 | 1/23/20 | ...

Module nay chuyen no ve dang "long"/series chuan cua time series:
chi so la DatetimeIndex theo ngay, gia tri la so ca.
"""
from __future__ import annotations

import urllib.request
from typing import Iterable

import numpy as np
import pandas as pd

from .config import DATA_RAW, FILES, JHU_BASE

ID_COLS = ["Province/State", "Country/Region", "Lat", "Long"]


# --------------------------------------------------------------------- tai
def download(metric: str = "confirmed", force: bool = False) -> pd.DataFrame:
    """Tai file CSV cua JHU ve thu muc data/raw (cache lai neu da co)."""
    if metric not in FILES:
        raise ValueError(f"metric phai thuoc {list(FILES)}, nhan duoc {metric!r}")

    path = DATA_RAW / FILES[metric]
    if force or not path.exists():
        url = f"{JHU_BASE}/{FILES[metric]}"
        print(f"[data] Dang tai {url}")
        urllib.request.urlretrieve(url, path)
    return pd.read_csv(path)


def download_all(force: bool = False) -> dict[str, pd.DataFrame]:
    return {m: download(m, force) for m in FILES}


# ------------------------------------------------------------- wide -> long
def to_long(df_wide: pd.DataFrame, value_name: str = "cumulative") -> pd.DataFrame:
    """Chuyen bang wide cua JHU sang dang long (country, date, value)."""
    date_cols = [c for c in df_wide.columns if c not in ID_COLS]
    long = df_wide.melt(
        id_vars=ID_COLS, value_vars=date_cols,
        var_name="date", value_name=value_name,
    )
    long["date"] = pd.to_datetime(long["date"], format="%m/%d/%y")
    long = long.rename(columns={"Country/Region": "country",
                                "Province/State": "province"})
    return long[["country", "province", "Lat", "Long", "date", value_name]]


def list_countries(metric: str = "confirmed") -> list[str]:
    """Danh sach quoc gia/vung lanh tho co trong bo du lieu."""
    return sorted(download(metric)["Country/Region"].unique().tolist())


# ------------------------------------------------------------ series 1 nuoc
def country_series(
    metric: str = "confirmed",
    country: str | None = "Vietnam",
    provinces: Iterable[str] | None = None,
) -> pd.Series:
    """
    Tra ve chuoi LUY KE theo ngay cho mot quoc gia.

    country=None  -> tong toan cau.
    Mot so nuoc (Trung Quoc, Canada, Uc...) tach theo tinh/bang nen phai
    cong don lai; do do luon groupby theo ngay sau khi loc.
    """
    wide = download(metric)
    if country is not None:
        mask = wide["Country/Region"].str.lower() == country.lower()
        if not mask.any():
            raise KeyError(f"Khong tim thay quoc gia {country!r}")
        wide = wide[mask]
        if provinces is not None:
            wide = wide[wide["Province/State"].isin(list(provinces))]

    date_cols = [c for c in wide.columns if c not in ID_COLS]
    s = wide[date_cols].sum(axis=0)
    s.index = pd.to_datetime(s.index, format="%m/%d/%y")
    s = s.sort_index().astype("float64")
    s.index.freq = "D"
    s.name = f"{metric}_cumulative"
    return s


# --------------------------------------------------------------- bien doi
def to_daily_new(cumulative: pd.Series, clip_negative: bool = True) -> pd.Series:
    """
    Chuyen chuoi luy ke -> so ca MOI moi ngay (sai phan bac 1).

    Du lieu thuc te co nhung ngay am do co quan y te dieu chinh so lieu
    hoi to; mac dinh cat ve 0 vi so ca moi khong the am.
    """
    daily = cumulative.diff()
    daily.iloc[0] = cumulative.iloc[0]
    if clip_negative:
        daily = daily.clip(lower=0)
    daily.name = cumulative.name.replace("_cumulative", "_daily")
    return daily


def smooth(series: pd.Series, window: int = 7, center: bool = False) -> pd.Series:
    """Trung binh truot - khu nhieu "hieu ung cuoi tuan" cua bao cao dich te."""
    out = series.rolling(window=window, center=center, min_periods=1).mean()
    out.name = f"{series.name}_ma{window}"
    return out


def trim_leading_zeros(series: pd.Series, min_value: float = 1.0) -> pd.Series:
    """Bo doan dau chuoi toan 0 (truoc khi dich xuat hien tai quoc gia do)."""
    nz = series[series >= min_value]
    if nz.empty:
        return series
    return series.loc[nz.index[0]:]


def load_target(
    country: str | None = "Vietnam",
    metric: str = "confirmed",
    kind: str = "daily",          # daily | cumulative
    smooth_window: int | None = None,
    start: str | None = None,
    end: str | None = None,
    trim: bool = True,
) -> pd.Series:
    """
    Ham tien ich "mot cua": tra ve chuoi muc tieu da san sang de mo hinh hoa.

    kind="daily"       -> so ca moi / ngay (chuoi thuong dung de du bao)
    kind="cumulative"  -> so ca luy ke
    smooth_window=7    -> lam tron bang trung binh truot 7 ngay
    """
    s = country_series(metric=metric, country=country)
    if kind == "daily":
        s = to_daily_new(s)
    elif kind != "cumulative":
        raise ValueError("kind phai la 'daily' hoac 'cumulative'")

    if trim:
        s = trim_leading_zeros(s)
    if start is not None:
        s = s.loc[start:]
    if end is not None:
        s = s.loc[:end]
    if smooth_window:
        s = smooth(s, smooth_window)

    s = s.asfreq("D").interpolate(limit_direction="both")
    return s


def describe_series(s: pd.Series) -> pd.Series:
    """Tom tat nhanh dac diem chuoi (dung cho phan 'tim hieu du lieu')."""
    return pd.Series({
        "so_quan_sat": len(s),
        "bat_dau": s.index.min().date(),
        "ket_thuc": s.index.max().date(),
        "tan_suat": s.index.freqstr,
        "thieu_du_lieu": int(s.isna().sum()),
        "trung_binh": s.mean(),
        "do_lech_chuan": s.std(),
        "nho_nhat": s.min(),
        "lon_nhat": s.max(),
        "ngay_dinh_diem": s.idxmax().date(),
        "he_so_bien_thien": s.std() / (s.mean() + 1e-10),
    })
