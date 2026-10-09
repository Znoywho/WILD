"""
Danh gia mo hinh du bao mot cach TRUNG THUC.

Sai lam pho bien nhat khi lam time series la dung K-Fold ngau nhien:
lam vay la cho mo hinh "nhin thay tuong lai" roi bat no doan qua khu
(data leakage). Voi chuoi thoi gian, tap kiem tra LUON phai nam sau
tap huan luyen tren truc thoi gian.

Module nay cai dat hai so do danh gia:

  1. HOLDOUT        - cat mot lan: [====== train ======][= test =]
                      Nhanh, nhung ket qua phu thuoc vao dung mot doan test.

  2. ROLLING-ORIGIN - truot diem cat nhieu lan (con goi la time series CV,
                      walk-forward validation):
                          [== train ==][test]
                          [==== train ====][test]
                          [====== train ======][test]
                      Cho uoc luong on dinh hon vi trung binh tren nhieu
                      giai doan (song dich len, xuong, phang...).
                      expanding=True  -> tap train lon dan (mac dinh)
                      expanding=False -> cua so truot co dinh kich thuoc
"""
from __future__ import annotations

import copy
import time
import warnings
from typing import Iterator, Sequence

import numpy as np
import pandas as pd

from .config import DEFAULT_HORIZON, SEASONAL_PERIOD
from .metrics import evaluate, leaderboard
from .models import Forecaster


# ------------------------------------------------------------------ chia du lieu
def train_test_split(
    y: pd.Series,
    test_size: int | float = DEFAULT_HORIZON,
) -> tuple[pd.Series, pd.Series]:
    """
    Cat chuoi theo thu tu thoi gian (KHONG xao tron).

    test_size la so nguyen -> so ngay cuoi dung lam test.
    test_size la so thuc trong (0, 1) -> ty le phan tram.
    """
    n = len(y)
    k = int(round(n * test_size)) if isinstance(test_size, float) and 0 < test_size < 1 \
        else int(test_size)
    if not 0 < k < n:
        raise ValueError(f"test_size={test_size!r} khong hop le voi chuoi dai {n}")
    return y.iloc[:-k], y.iloc[-k:]


def rolling_origin_splits(
    y: pd.Series,
    horizon: int = DEFAULT_HORIZON,
    n_splits: int = 5,
    step: int | None = None,
    min_train: int | None = None,
    expanding: bool = True,
) -> Iterator[tuple[pd.Series, pd.Series]]:
    """
    Sinh lan luot cac cap (train, test) truot dan ve phia cuoi chuoi.

    step      : khoang cach giua hai diem cat (mac dinh = horizon, tuc la cac
                doan test khong chong nhau)
    min_train : so quan sat toi thieu cua tap train dau tien
    """
    n = len(y)
    step = horizon if step is None else step
    if min_train is None:
        min_train = max(3 * horizon, n - horizon - (n_splits - 1) * step)
    if min_train < horizon:
        raise ValueError("Chuoi qua ngan de backtest voi horizon nay")

    # cac diem cat: lui dan tu cuoi chuoi
    cuts = [n - horizon - i * step for i in range(n_splits)][::-1]
    cuts = [c for c in cuts if c >= min_train]
    if not cuts:
        raise ValueError("Khong tao duoc fold nao - giam n_splits hoac horizon")

    for cut in cuts:
        train = y.iloc[:cut] if expanding else y.iloc[max(0, cut - min_train):cut]
        yield train, y.iloc[cut:cut + horizon]


# --------------------------------------------------------------- chay 1 mo hinh
def _fit_predict(model: Forecaster, train: pd.Series, horizon: int) -> pd.Series:
    """Huan luyen tren ban sao sach cua mo hinh roi du bao horizon buoc."""
    m = copy.deepcopy(model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(train)
        return m.predict(horizon)


def holdout(
    model: Forecaster,
    y: pd.Series,
    test_size: int | float = DEFAULT_HORIZON,
    seasonality: int = SEASONAL_PERIOD,
) -> dict:
    """
    Danh gia mot mo hinh bang mot lan cat duy nhat.

    Tra ve dict gom: ten mo hinh, cac chi so sai so, chuoi du bao,
    khoang tin cay (neu mo hinh ho tro) va thoi gian chay.
    """
    train, test = train_test_split(y, test_size)
    t0 = time.perf_counter()
    m = copy.deepcopy(model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(train)
        pred = m.predict(len(test))
        interval = None
        try:
            interval = m.predict_interval(len(test))
        except Exception:
            interval = None
    elapsed = time.perf_counter() - t0

    pred.index = test.index                       # dong bo chi so de so sanh
    scores = evaluate(test, pred, insample=train, seasonality=seasonality)
    if interval is not None:
        interval.index = test.index
        from .metrics import coverage
        scores["Coverage95"] = coverage(test, interval["lower"], interval["upper"])

    return {
        "ten": model.name,
        "chi_so": scores,
        "du_bao": pred,
        "khoang_tin_cay": interval,
        "thuc_te": test,
        "train": train,
        "giay": round(elapsed, 2),
    }


def rolling_backtest(
    model: Forecaster,
    y: pd.Series,
    horizon: int = DEFAULT_HORIZON,
    n_splits: int = 5,
    step: int | None = None,
    expanding: bool = True,
    seasonality: int = SEASONAL_PERIOD,
) -> dict:
    """
    Danh gia mot mo hinh tren nhieu diem cat, tra ve chi so tung fold
    va trung binh cong cua chung.
    """
    rows, preds = [], []
    t0 = time.perf_counter()
    for i, (train, test) in enumerate(
        rolling_origin_splits(y, horizon, n_splits, step, expanding=expanding), start=1
    ):
        try:
            pred = _fit_predict(model, train, len(test))
            pred.index = test.index
        except Exception as exc:                  # mo hinh loi o fold nay
            rows.append({"fold": i, "cat_tai": test.index[0].date(), "loi": str(exc)[:80]})
            continue
        sc = evaluate(test, pred, insample=train, seasonality=seasonality)
        rows.append({"fold": i, "cat_tai": test.index[0].date(),
                     "co_train": len(train), **sc})
        preds.append(pred)

    per_fold = pd.DataFrame(rows)
    numeric = per_fold.select_dtypes("number").drop(columns=["fold", "co_train"],
                                                    errors="ignore")
    return {
        "ten": model.name,
        "tung_fold": per_fold.round(3),
        "trung_binh": numeric.mean().round(3).to_dict(),
        "do_lech_chuan": numeric.std().round(3).to_dict(),
        "du_bao_cac_fold": preds,
        "giay": round(time.perf_counter() - t0, 2),
    }


# ------------------------------------------------------------ so sanh nhieu mo hinh
def compare_holdout(
    models: dict[str, Forecaster] | Sequence[Forecaster],
    y: pd.Series,
    test_size: int | float = DEFAULT_HORIZON,
    seasonality: int = SEASONAL_PERIOD,
    sort_by: str = "RMSE",
    verbose: bool = True,
) -> dict:
    """
    Chay toan bo danh muc mo hinh tren cung mot lan cat holdout.

    Tra ve:
        bang     - DataFrame xep hang theo `sort_by`
        du_bao   - DataFrame moi cot la du bao cua mot mo hinh
        chi_tiet - dict ket qua day du tung mo hinh (co khoang tin cay)
        loi      - dict mo hinh chay that bai va ly do
    """
    items = models.items() if isinstance(models, dict) else [(m.name, m) for m in models]
    scores, preds, details, errors = {}, {}, {}, {}

    for label, model in items:
        try:
            res = holdout(model, y, test_size, seasonality)
        except Exception as exc:
            errors[label] = str(exc)[:120]
            if verbose:
                print(f"  [X] {label:42s} loi: {str(exc)[:60]}")
            continue
        scores[label] = {**res["chi_so"], "Giay": res["giay"]}
        preds[label] = res["du_bao"]
        details[label] = res
        if verbose:
            print(f"  [v] {label:42s} RMSE={res['chi_so']['RMSE']:10.2f}  "
                  f"MASE={res['chi_so']['MASE']:6.3f}  ({res['giay']}s)")

    return {
        "bang": leaderboard(scores, sort_by=sort_by),
        "du_bao": pd.DataFrame(preds),
        "chi_tiet": details,
        "loi": errors,
        "thuc_te": train_test_split(y, test_size)[1],
    }


def compare_rolling(
    models: dict[str, Forecaster] | Sequence[Forecaster],
    y: pd.Series,
    horizon: int = DEFAULT_HORIZON,
    n_splits: int = 5,
    step: int | None = None,
    expanding: bool = True,
    seasonality: int = SEASONAL_PERIOD,
    sort_by: str = "RMSE",
    verbose: bool = True,
) -> dict:
    """So sanh cac mo hinh bang rolling-origin (ket qua dang tin cay hon holdout)."""
    items = models.items() if isinstance(models, dict) else [(m.name, m) for m in models]
    means, folds, errors = {}, {}, {}

    for label, model in items:
        try:
            res = rolling_backtest(model, y, horizon, n_splits, step,
                                   expanding, seasonality)
            if not res["trung_binh"] or np.isnan(res["trung_binh"].get("RMSE", np.nan)):
                raise RuntimeError("khong fold nao chay thanh cong")
        except Exception as exc:
            errors[label] = str(exc)[:120]
            if verbose:
                print(f"  [X] {label:42s} loi: {str(exc)[:60]}")
            continue
        means[label] = {**res["trung_binh"], "Giay": res["giay"]}
        folds[label] = res["tung_fold"]
        if verbose:
            print(f"  [v] {label:42s} RMSE={res['trung_binh']['RMSE']:10.2f}  "
                  f"MASE={res['trung_binh'].get('MASE', float('nan')):6.3f}  ({res['giay']}s)")

    return {
        "bang": leaderboard(means, sort_by=sort_by),
        "tung_fold": folds,
        "loi": errors,
        "so_fold": n_splits,
        "horizon": horizon,
    }


# -------------------------------------------------------------------- tien ich
def best_model(result: dict, metric: str = "RMSE") -> str:
    """Ten mo hinh tot nhat trong bang xep hang tra ve boi compare_*."""
    bang = result["bang"]
    return str(bang[metric].idxmin() if metric != "R2" else bang[metric].idxmax())


def refit_and_forecast(
    model: Forecaster,
    y: pd.Series,
    horizon: int = DEFAULT_HORIZON,
) -> pd.DataFrame:
    """
    Buoc cuoi cung cua quy trinh: sau khi da chon duoc mo hinh tot nhat,
    huan luyen lai tren TOAN BO du lieu roi du bao tuong lai that su.
    """
    m = copy.deepcopy(model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(y)
        mean = m.predict(horizon)
        try:
            ci = m.predict_interval(horizon)
        except Exception:
            ci = None
    out = pd.DataFrame({"du_bao": mean.to_numpy()}, index=mean.index)
    if ci is not None:
        out["can_duoi"] = ci["lower"].to_numpy()
        out["can_tren"] = ci["upper"].to_numpy()
    out.index.name = "ngay"
    return out
