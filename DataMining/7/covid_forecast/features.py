"""
Sinh dac trung de dua bai toan du bao ve bai toan hoc co giam sat.

Y tuong: mot mo hinh hoi quy binh thuong (Ridge, RandomForest, XGBoost...)
khong hieu "thoi gian". Ta phai tu tay bien chuoi y(t) thanh bang X -> y:

    y(t) = f( y(t-1), y(t-2), ..., y(t-p), trung binh truot, thu trong tuan, ... )

Day chinh la ky thuat "sliding window" / "lag embedding".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_LAGS = (1, 2, 3, 4, 5, 6, 7, 14, 21, 28)
DEFAULT_ROLL = (3, 7, 14, 28)


def make_lag_features(
    series: pd.Series,
    lags: tuple[int, ...] = DEFAULT_LAGS,
    rolling_windows: tuple[int, ...] = DEFAULT_ROLL,
    calendar: bool = True,
    trend: bool = True,
) -> pd.DataFrame:
    """
    Tao ma tran dac trung tu mot chuoi.

    lags            : cac do tre y(t-k) - "bo nho" ngan/dai han cua chuoi
    rolling_windows : trung binh & do lech chuan truot (chi tinh tren qua khu,
                      dung shift(1) de KHONG ro ri thong tin tuong lai)
    calendar        : thu trong tuan (ma hoa sin/cos), ngay trong thang
    trend           : chi so thoi gian tuyen tinh
    """
    df = pd.DataFrame({"y": series.astype("float64")}, index=series.index)

    for lag in lags:
        df[f"lag_{lag}"] = df["y"].shift(lag)

    past = df["y"].shift(1)                      # chi nhin vao qua khu
    for w in rolling_windows:
        df[f"roll_mean_{w}"] = past.rolling(w).mean()
        df[f"roll_std_{w}"] = past.rolling(w).std()
        df[f"roll_max_{w}"] = past.rolling(w).max()
    df["diff_1"] = past.diff()
    df["diff_7"] = past.diff(7)

    if calendar:
        dow = df.index.dayofweek
        df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
        df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
        df["is_weekend"] = (dow >= 5).astype(int)
        df["day_of_month"] = df.index.day
        df["month"] = df.index.month

    if trend:
        df["t"] = np.arange(len(df), dtype="float64")

    return df


def supervised_split(df_feat: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Tach bang dac trung thanh (X, y) va bo cac dong thieu do lag."""
    clean = df_feat.dropna()
    return clean.drop(columns="y"), clean["y"]


def build_row(history: pd.Series, columns: list[str],
              lags: tuple[int, ...], rolling_windows: tuple[int, ...],
              next_date: pd.Timestamp, t_index: float) -> pd.DataFrame:
    """
    Dung 1 dong dac trung cho ngay `next_date` tu lich su hien co.

    Ham nay phuc vu du bao DE QUY nhieu buoc: du bao ngay t+1, noi gia tri
    vua du bao vao lich su, roi lai dung dac trung cho ngay t+2...
    """
    row: dict[str, float] = {}
    vals = history.to_numpy()

    for lag in lags:
        row[f"lag_{lag}"] = vals[-lag] if len(vals) >= lag else np.nan
    for w in rolling_windows:
        win = vals[-w:] if len(vals) >= w else vals
        row[f"roll_mean_{w}"] = float(np.mean(win))
        row[f"roll_std_{w}"] = float(np.std(win, ddof=1)) if len(win) > 1 else 0.0
        row[f"roll_max_{w}"] = float(np.max(win))
    row["diff_1"] = float(vals[-1] - vals[-2]) if len(vals) >= 2 else 0.0
    row["diff_7"] = float(vals[-1] - vals[-8]) if len(vals) >= 8 else 0.0

    dow = next_date.dayofweek
    row["dow_sin"] = float(np.sin(2 * np.pi * dow / 7))
    row["dow_cos"] = float(np.cos(2 * np.pi * dow / 7))
    row["is_weekend"] = int(dow >= 5)
    row["day_of_month"] = next_date.day
    row["month"] = next_date.month
    row["t"] = t_index

    return pd.DataFrame([[row.get(c, 0.0) for c in columns]],
                        columns=columns, index=[next_date])


def make_windows(series: pd.Series, lookback: int, horizon: int = 1):
    """
    Cat chuoi thanh cac cua so (X, y) 3 chieu cho mang no-ron:
        X.shape = (n_mau, lookback, 1), y.shape = (n_mau, horizon)
    """
    v = series.to_numpy(dtype="float32")
    xs, ys = [], []
    for i in range(len(v) - lookback - horizon + 1):
        xs.append(v[i:i + lookback])
        ys.append(v[i + lookback:i + lookback + horizon])
    X = np.asarray(xs, dtype="float32")[..., None]
    y = np.asarray(ys, dtype="float32")
    return X, y
