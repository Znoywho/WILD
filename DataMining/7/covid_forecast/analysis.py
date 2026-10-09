"""
Phan tich chuoi thoi gian truoc khi du bao.

Quy trinh chuan (Box-Jenkins) gom 4 buoc duoc cai dat o day:
    1. Phan ra chuoi   : level / trend / seasonality / residual
    2. Kiem tra tinh dung (stationarity): ADF, KPSS, rolling mean-std
    3. Bien doi cho dung: sai phan, log, sai phan mua vu
    4. Xac dinh bac mo hinh: ACF / PACF
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL, seasonal_decompose
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf

from .config import SEASONAL_PERIOD


def _adfuller(s):
    """Boc adfuller de an FutureWarning ve dinh dang tra ve cua statsmodels."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return adfuller(s, autolag="AIC")


# ------------------------------------------------------------- 1. phan ra
def decompose(series: pd.Series, period: int = SEASONAL_PERIOD,
              method: str = "stl", model: str = "additive"):
    """
    Tach chuoi thanh Xu the (trend) + Mua vu (seasonal) + Phan du (resid).

    method="stl"     : STL - on dinh voi outlier, mua vu duoc phep thay doi
                       theo thoi gian (rat hop voi du lieu COVID nhieu dot song)
    method="classic" : phan ra co dien bang trung binh truot
    """
    s = series.dropna()
    if method == "stl":
        res = STL(s, period=period, robust=True).fit()
        return pd.DataFrame({
            "observed": s, "trend": res.trend,
            "seasonal": res.seasonal, "resid": res.resid,
        })
    res = seasonal_decompose(s, period=period, model=model)
    return pd.DataFrame({
        "observed": res.observed, "trend": res.trend,
        "seasonal": res.seasonal, "resid": res.resid,
    })


def seasonal_strength(series: pd.Series, period: int = SEASONAL_PERIOD) -> dict:
    """
    Do manh xu the / mua vu theo Wang-Smith-Hyndman (gia tri trong [0, 1]).
    Cang gan 1 thi thanh phan do cang ro.
    """
    d = decompose(series, period=period).dropna()
    var_r = d["resid"].var()
    f_trend = max(0.0, 1 - var_r / max((d["trend"] + d["resid"]).var(), 1e-10))
    f_seas = max(0.0, 1 - var_r / max((d["seasonal"] + d["resid"]).var(), 1e-10))
    return {"do_manh_xu_the": round(f_trend, 3), "do_manh_mua_vu": round(f_seas, 3)}


# ------------------------------------------------------- 2. tinh dung
def adf_test(series: pd.Series, alpha: float = 0.05) -> dict:
    """
    Augmented Dickey-Fuller.
    H0: chuoi CO nghiem don vi (khong dung).
    p-value < alpha  =>  bac bo H0  =>  chuoi DUNG.
    """
    s = series.dropna()
    stat, p, lags, nobs, crit, _ = _adfuller(s)
    return {
        "kiem_dinh": "ADF", "thong_ke": round(stat, 4), "p_value": round(p, 6),
        "so_tre": lags, "gia_tri_toi_han_5%": round(crit["5%"], 4),
        "ket_luan": "DUNG (stationary)" if p < alpha else "KHONG DUNG",
    }


def kpss_test(series: pd.Series, alpha: float = 0.05) -> dict:
    """
    KPSS - gia thuyet nguoc voi ADF.
    H0: chuoi DUNG.  p-value < alpha => bac bo => KHONG dung.
    Dung ca hai kiem dinh de ket luan chac chan hon.
    """
    s = series.dropna()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, p, lags, crit = kpss(s, regression="c", nlags="auto")
    return {
        "kiem_dinh": "KPSS", "thong_ke": round(stat, 4), "p_value": round(p, 6),
        "so_tre": lags, "gia_tri_toi_han_5%": round(crit["5%"], 4),
        "ket_luan": "KHONG DUNG" if p < alpha else "DUNG (stationary)",
    }


def stationarity_report(series: pd.Series) -> pd.DataFrame:
    return pd.DataFrame([adf_test(series), kpss_test(series)])


def rolling_stats(series: pd.Series, window: int = 30) -> pd.DataFrame:
    """Trung binh & do lech chuan truot - kiem tra tinh dung bang truc quan."""
    return pd.DataFrame({
        "goc": series,
        f"trung_binh_truot_{window}": series.rolling(window).mean(),
        f"do_lech_chuan_truot_{window}": series.rolling(window).std(),
    })


# --------------------------------------------------- 3. bien doi cho dung
def difference(series: pd.Series, order: int = 1, seasonal_lag: int = 0) -> pd.Series:
    """Sai phan thuong (order) va/hoac sai phan mua vu (seasonal_lag)."""
    out = series.copy()
    if seasonal_lag:
        out = out.diff(seasonal_lag)
    for _ in range(order):
        out = out.diff()
    return out.dropna()


def log_transform(series: pd.Series) -> pd.Series:
    """log(1+x) - on dinh phuong sai khi chuoi tang theo cap so nhan."""
    return np.log1p(series.clip(lower=0))


def inverse_log(series: pd.Series) -> pd.Series:
    return np.expm1(series)


def suggest_differencing(series: pd.Series, max_d: int = 2,
                         alpha: float = 0.05) -> int:
    """Tim bac sai phan d nho nhat de chuoi tro nen dung theo ADF."""
    test = series.dropna()
    for d in range(max_d + 1):
        if len(test) < 20:
            return d
        if _adfuller(test)[1] < alpha:
            return d
        test = test.diff().dropna()
    return max_d


# ----------------------------------------------------- 4. ACF / PACF
def acf_pacf(series: pd.Series, nlags: int = 40) -> pd.DataFrame:
    """
    ACF  - tu tuong quan: goi y bac q cua thanh phan MA
    PACF - tu tuong quan rieng phan: goi y bac p cua thanh phan AR
    Khoang tin cay xap xi +-1.96/sqrt(n): gia tri vuot ra ngoai la co y nghia.
    """
    s = series.dropna()
    n = len(s)
    a = acf(s, nlags=nlags, fft=True)
    p = pacf(s, nlags=min(nlags, n // 2 - 1))
    ci = 1.96 / np.sqrt(n)
    return pd.DataFrame({
        "lag": np.arange(len(a)),
        "acf": a,
        "pacf": np.pad(p, (0, len(a) - len(p)), constant_values=np.nan),
        "nguong_tren": ci, "nguong_duoi": -ci,
    })


def summary(series: pd.Series, period: int = SEASONAL_PERIOD) -> dict:
    """Bao cao phan tich tong hop cho mot chuoi."""
    return {
        "thong_ke_mo_ta": series.describe().round(2).to_dict(),
        "do_manh_thanh_phan": seasonal_strength(series, period),
        "tinh_dung": stationarity_report(series).to_dict("records"),
        "bac_sai_phan_de_xuat": suggest_differencing(series),
    }
