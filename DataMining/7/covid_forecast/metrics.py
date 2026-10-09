"""
Do do danh gia sai so du bao.

Khong co mot chi so nao "tot nhat" - moi chi so tra loi mot cau hoi khac:

    MAE   - sai lech trung binh tuyet doi, cung don vi voi du lieu, de giai thich
    RMSE  - phat nang cac sai so lon (binh phuong), nhay voi outlier
    MAPE  - sai so tuong doi (%), so sanh duoc giua cac chuoi khac thang do,
            nhung vo nghia khi gia tri thuc = 0 (rat hay gap o du lieu dich)
    sMAPE - ban doi xung cua MAPE, on dinh hon khi co gia tri 0
    R2    - ty le phuong sai giai thich duoc; < 0 nghia la te hon du bao bang trung binh
    MASE  - sai so chia cho sai so cua mo hinh naive theo mua vu;
            MASE < 1 => mo hinh tot hon naive. Day la chi so an toan nhat
            cho du lieu COVID vi co nhieu ngay bang 0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-10


def _prep(actual, predicted) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(actual, dtype="float64").ravel()
    p = np.asarray(predicted, dtype="float64").ravel()
    if a.shape != p.shape:
        raise ValueError(f"Kich thuoc khong khop: {a.shape} vs {p.shape}")
    return a, p


def mae(actual, predicted) -> float:
    a, p = _prep(actual, predicted)
    return float(np.mean(np.abs(a - p)))


def mse(actual, predicted) -> float:
    a, p = _prep(actual, predicted)
    return float(np.mean((a - p) ** 2))


def rmse(actual, predicted) -> float:
    return float(np.sqrt(mse(actual, predicted)))


def mape(actual, predicted) -> float:
    """Tinh theo %; bo qua cac diem co gia tri thuc = 0."""
    a, p = _prep(actual, predicted)
    mask = np.abs(a) > EPS
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((a[mask] - p[mask]) / a[mask])) * 100)


def smape(actual, predicted) -> float:
    a, p = _prep(actual, predicted)
    denom = (np.abs(a) + np.abs(p)) / 2
    mask = denom > EPS
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(a[mask] - p[mask]) / denom[mask]) * 100)


def r2(actual, predicted) -> float:
    a, p = _prep(actual, predicted)
    ss_res = np.sum((a - p) ** 2)
    ss_tot = np.sum((a - np.mean(a)) ** 2)
    if ss_tot < EPS:
        return float("nan")
    return float(1 - ss_res / ss_tot)


def mase(actual, predicted, insample=None, seasonality: int = 1) -> float:
    """
    MAE cua mo hinh / MAE cua du bao naive theo mua vu.

    Chuan Hyndman: mau so tinh tren TAP HUAN LUYEN (insample). Neu khong
    truyen insample thi tinh ngay tren chuoi thuc te cua tap kiem tra.
    """
    a, p = _prep(actual, predicted)
    base = np.asarray(insample if insample is not None else a, dtype="float64").ravel()
    if len(base) <= seasonality:
        return float("nan")
    scale = np.mean(np.abs(base[seasonality:] - base[:-seasonality]))
    if scale < EPS:
        return float("nan")
    return float(np.mean(np.abs(a - p)) / scale)


def bias(actual, predicted) -> float:
    """Sai so trung binh co dau: >0 = mo hinh du bao THAP hon thuc te."""
    a, p = _prep(actual, predicted)
    return float(np.mean(a - p))


def coverage(actual, lower, upper) -> float:
    """Ty le (%) gia tri thuc nam trong khoang du bao - danh gia do tin cay."""
    a = np.asarray(actual, dtype="float64").ravel()
    lo = np.asarray(lower, dtype="float64").ravel()
    hi = np.asarray(upper, dtype="float64").ravel()
    return float(np.mean((a >= lo) & (a <= hi)) * 100)


def evaluate(actual, predicted, insample=None, seasonality: int = 7) -> dict:
    """Bo chi so tieu chuan dung xuyen suot ung dung."""
    return {
        "MAE": mae(actual, predicted),
        "RMSE": rmse(actual, predicted),
        "MAPE": mape(actual, predicted),
        "sMAPE": smape(actual, predicted),
        "R2": r2(actual, predicted),
        "MASE": mase(actual, predicted, insample, seasonality),
        "Bias": bias(actual, predicted),
    }


def leaderboard(results: dict[str, dict], sort_by: str = "RMSE") -> pd.DataFrame:
    """Gop ket qua cua nhieu mo hinh thanh bang xep hang."""
    df = pd.DataFrame(results).T
    df.index.name = "Mo hinh"
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=sort_by != "R2")
    return df.round(3)
