"""
Bo mo hinh du bao chuoi thoi gian.

Tat ca mo hinh dung chung mot giao dien (giong scikit-learn) de co the
so sanh cong bang trong cung mot vong thu nghiem:

    model = SomeForecaster(...)
    model.fit(y_train)                 # y_train: pd.Series co DatetimeIndex
    y_hat = model.predict(horizon)     # pd.Series cho horizon ngay ke tiep
    ci    = model.predict_interval(h)  # DataFrame lower/upper (neu ho tro)

Bon nhom mo hinh:
  1. BASELINE  - Naive, Seasonal Naive, Mean, Drift  (moc so sanh bat buoc)
  2. THONG KE  - SES, Holt, Holt-Winters, Theta, ARIMA, SARIMA, AutoARIMA, Prophet
  3. HOC MAY   - Ridge, RandomForest, GradientBoosting, KNN, SVR, XGBoost
                 (dung dac trung tre, du bao de quy nhieu buoc)
  4. DEEP      - LSTM (PyTorch)
"""
from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from . import features as F
from .config import RANDOM_STATE, SEASONAL_PERIOD

warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.CRITICAL)
logging.getLogger("prophet").setLevel(logging.CRITICAL)


def future_index(last_date: pd.Timestamp, horizon: int) -> pd.DatetimeIndex:
    """Sinh chi so ngay cho horizon buoc tiep theo."""
    return pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")


# ============================================================ lop truu tuong
class Forecaster(ABC):
    """Lop co so cho moi mo hinh du bao."""

    name: str = "Forecaster"
    supports_interval: bool = False

    def __init__(self) -> None:
        self.y_: pd.Series | None = None
        self.fitted_: pd.Series | None = None

    @abstractmethod
    def fit(self, y: pd.Series) -> "Forecaster":
        ...

    @abstractmethod
    def predict(self, horizon: int) -> pd.Series:
        ...

    def predict_interval(self, horizon: int, alpha: float = 0.05) -> pd.DataFrame | None:
        """Khoang du bao. Mac dinh: uoc luong tu do lech chuan phan du."""
        mean = self.predict(horizon)
        if self.fitted_ is None or self.y_ is None:
            return None
        resid = (self.y_ - self.fitted_).dropna()
        if resid.empty:
            return None
        from scipy.stats import norm
        z = norm.ppf(1 - alpha / 2)
        # sai so tich luy theo can bac hai cua tam nhin du bao
        widen = z * resid.std() * np.sqrt(np.arange(1, horizon + 1))
        return pd.DataFrame({"lower": (mean - widen).clip(lower=0),
                             "mean": mean, "upper": mean + widen}, index=mean.index)

    def _store(self, y: pd.Series) -> None:
        self.y_ = y.astype("float64")

    def __repr__(self) -> str:
        return f"<{self.name}>"


# ================================================================ 1. BASELINE
class NaiveForecaster(Forecaster):
    """y_hat(t+h) = y(t). Lap lai gia tri quan sat cuoi cung."""
    name = "Naive (gia tri hom qua)"

    def fit(self, y):
        self._store(y)
        self.last_ = float(y.iloc[-1])
        self.fitted_ = y.shift(1)
        return self

    def predict(self, horizon):
        return pd.Series(self.last_, index=future_index(self.y_.index[-1], horizon),
                         name=self.name)


class SeasonalNaiveForecaster(Forecaster):
    """y_hat(t+h) = y(t+h-m). Lap lai chu ky mua vu gan nhat (m = 7 ngay)."""
    name = "Seasonal Naive (tuan truoc)"

    def __init__(self, period: int = SEASONAL_PERIOD):
        super().__init__()
        self.period = period

    def fit(self, y):
        self._store(y)
        self.season_ = y.iloc[-self.period:].to_numpy(dtype="float64")
        self.fitted_ = y.shift(self.period)
        return self

    def predict(self, horizon):
        vals = [self.season_[i % self.period] for i in range(horizon)]
        return pd.Series(vals, index=future_index(self.y_.index[-1], horizon),
                         name=self.name)


class MeanForecaster(Forecaster):
    """Trung binh toan bo (window=None) hoac trung binh truot window ngay cuoi."""
    name = "Trung binh truot"

    def __init__(self, window: int | None = 7):
        super().__init__()
        self.window = window
        if window:
            self.name = f"Trung binh truot {window} ngay"
        else:
            self.name = "Trung binh toan chuoi"

    def fit(self, y):
        self._store(y)
        tail = y if self.window is None else y.iloc[-self.window:]
        self.value_ = float(tail.mean())
        self.fitted_ = (y.expanding().mean().shift(1) if self.window is None
                        else y.rolling(self.window).mean().shift(1))
        return self

    def predict(self, horizon):
        return pd.Series(self.value_, index=future_index(self.y_.index[-1], horizon),
                         name=self.name)


class DriftForecaster(Forecaster):
    """Noi dai duong thang tu diem dau den diem cuoi cua chuoi."""
    name = "Drift (ngoai suy tuyen tinh)"

    def fit(self, y):
        self._store(y)
        n = len(y)
        self.slope_ = (float(y.iloc[-1]) - float(y.iloc[0])) / max(n - 1, 1)
        self.last_ = float(y.iloc[-1])
        self.fitted_ = y.shift(1) + self.slope_
        return self

    def predict(self, horizon):
        vals = [self.last_ + self.slope_ * (h + 1) for h in range(horizon)]
        return pd.Series(np.clip(vals, 0, None),
                         index=future_index(self.y_.index[-1], horizon), name=self.name)


# =============================================================== 2. THONG KE
class ExpSmoothingForecaster(Forecaster):
    """
    Ho lam tron ham mu (Exponential Smoothing / ETS).

    variant="ses"   - chi co muc (level): phu hop chuoi khong xu the, khong mua vu
    variant="holt"  - muc + xu the
    variant="hw"    - muc + xu the + mua vu (Holt-Winters)
    """
    supports_interval = True

    def __init__(self, variant: str = "hw", period: int = SEASONAL_PERIOD,
                 damped: bool = True, seasonal: str = "add"):
        super().__init__()
        self.variant, self.period, self.damped, self.seasonal = (
            variant, period, damped, seasonal)
        self.name = {"ses": "SES (lam tron ham mu don gian)",
                     "holt": "Holt (xu the tuyen tinh)",
                     "hw": f"Holt-Winters (mua vu {period} ngay)"}[variant]

    def fit(self, y):
        self._store(y)
        from statsmodels.tsa.holtwinters import (ExponentialSmoothing,
                                                 SimpleExpSmoothing)
        if self.variant == "ses":
            self.res_ = SimpleExpSmoothing(self.y_, initialization_method="estimated").fit()
        elif self.variant == "holt":
            self.res_ = ExponentialSmoothing(
                self.y_, trend="add", damped_trend=self.damped,
                initialization_method="estimated").fit()
        else:
            self.res_ = ExponentialSmoothing(
                self.y_, trend="add", damped_trend=self.damped,
                seasonal=self.seasonal, seasonal_periods=self.period,
                initialization_method="estimated").fit()
        self.fitted_ = self.res_.fittedvalues
        return self

    def predict(self, horizon):
        out = self.res_.forecast(horizon).clip(lower=0)
        out.index = future_index(self.y_.index[-1], horizon)
        out.name = self.name
        return out


class ThetaForecaster(Forecaster):
    """Phuong phap Theta - vo dich cuoc thi M3, don gian ma rat manh."""
    name = "Theta"
    supports_interval = True

    def __init__(self, period: int = SEASONAL_PERIOD):
        super().__init__()
        self.period = period

    def fit(self, y):
        self._store(y)
        from statsmodels.tsa.forecasting.theta import ThetaModel
        self.res_ = ThetaModel(self.y_, period=self.period,
                               deseasonalize=True).fit()
        self.fitted_ = None
        return self

    def predict(self, horizon):
        out = self.res_.forecast(horizon).clip(lower=0)
        out.index = future_index(self.y_.index[-1], horizon)
        out.name = self.name
        return out

    def predict_interval(self, horizon, alpha=0.05):
        ci = self.res_.prediction_intervals(horizon, alpha=alpha)
        ci.index = future_index(self.y_.index[-1], horizon)
        ci.columns = ["lower", "upper"]
        ci["mean"] = self.predict(horizon)
        return ci[["lower", "mean", "upper"]].clip(lower=0)


class ARIMAForecaster(Forecaster):
    """
    ARIMA(p, d, q) - AR (tu hoi quy) + I (sai phan) + MA (trung binh truot).
    Truyen them seasonal_order=(P,D,Q,m) de thanh SARIMA.
    """
    supports_interval = True

    def __init__(self, order=(2, 1, 2), seasonal_order=(0, 0, 0, 0),
                 trend=None, label: str | None = None):
        super().__init__()
        self.order, self.seasonal_order, self.trend = order, seasonal_order, trend
        if label:
            self.name = label
        elif seasonal_order[-1]:
            self.name = f"SARIMA{order}x{seasonal_order}"
        else:
            self.name = f"ARIMA{order}"

    def fit(self, y):
        self._store(y)
        from statsmodels.tsa.arima.model import ARIMA
        self.res_ = ARIMA(self.y_, order=self.order,
                          seasonal_order=self.seasonal_order,
                          trend=self.trend,
                          enforce_stationarity=False,
                          enforce_invertibility=False).fit()
        self.fitted_ = self.res_.fittedvalues
        self.aic_ = float(self.res_.aic)
        return self

    def predict(self, horizon):
        out = self.res_.forecast(horizon).clip(lower=0)
        out.index = future_index(self.y_.index[-1], horizon)
        out.name = self.name
        return out

    def predict_interval(self, horizon, alpha=0.05):
        pr = self.res_.get_forecast(horizon)
        ci = pr.conf_int(alpha=alpha)
        idx = future_index(self.y_.index[-1], horizon)
        return pd.DataFrame({"lower": ci.iloc[:, 0].to_numpy(),
                             "mean": pr.predicted_mean.to_numpy(),
                             "upper": ci.iloc[:, 1].to_numpy()},
                            index=idx).clip(lower=0)


class AutoARIMAForecaster(ARIMAForecaster):
    """
    Tu dong chon (p, d, q) bang tim kiem luoi, toi thieu hoa AIC.

    AIC = -2*log-likelihood + 2k: can bang giua do khop va do phuc tap,
    tranh mo hinh qua khop (overfitting).
    """
    name = "Auto-ARIMA"

    def __init__(self, max_p: int = 3, max_q: int = 3, d: int | None = None,
                 seasonal_period: int = 0, max_P: int = 1, max_Q: int = 1):
        Forecaster.__init__(self)
        self.max_p, self.max_q, self.d = max_p, max_q, d
        self.seasonal_period, self.max_P, self.max_Q = seasonal_period, max_P, max_Q
        self.trend = None

    def fit(self, y):
        from statsmodels.tsa.arima.model import ARIMA

        from .analysis import suggest_differencing
        self._store(y)
        d = suggest_differencing(self.y_) if self.d is None else self.d

        seasonal_grid = [(0, 0, 0, 0)]
        if self.seasonal_period:
            seasonal_grid = [(P, 0, Q, self.seasonal_period)
                             for P in range(self.max_P + 1)
                             for Q in range(self.max_Q + 1)]

        best = (np.inf, None, None)
        for p in range(self.max_p + 1):
            for q in range(self.max_q + 1):
                if p == 0 and q == 0:
                    continue
                for so in seasonal_grid:
                    try:
                        res = ARIMA(self.y_, order=(p, d, q), seasonal_order=so,
                                    enforce_stationarity=False,
                                    enforce_invertibility=False).fit()
                        if np.isfinite(res.aic) and res.aic < best[0]:
                            best = (float(res.aic), (p, d, q), so)
                    except Exception:
                        continue
        if best[1] is None:
            best = (np.nan, (1, d, 0), (0, 0, 0, 0))

        self.order, self.seasonal_order, self.aic_ = best[1], best[2], best[0]
        label = (f"Auto-ARIMA{self.order}" if not self.seasonal_order[-1]
                 else f"Auto-SARIMA{self.order}x{self.seasonal_order}")
        self.name = label
        return ARIMAForecaster.fit(self, y)


class ProphetForecaster(Forecaster):
    """
    Prophet (Meta): mo hinh cong tinh y(t) = xu the + mua vu + ngay le + nhieu.
    Manh o cho tu dong bat diem gay xu the (changepoints) - rat hop voi
    cac "lan song" dich.
    """
    name = "Prophet"
    supports_interval = True

    def __init__(self, weekly: bool = True, yearly: bool = False,
                 changepoint_prior_scale: float = 0.1,
                 growth: str = "linear", interval_width: float = 0.95):
        super().__init__()
        self.weekly, self.yearly = weekly, yearly
        self.cps, self.growth, self.interval_width = (
            changepoint_prior_scale, growth, interval_width)

    def fit(self, y):
        from prophet import Prophet
        self._store(y)
        df = pd.DataFrame({"ds": self.y_.index, "y": self.y_.to_numpy()})
        self.model_ = Prophet(weekly_seasonality=self.weekly,
                              yearly_seasonality=self.yearly,
                              daily_seasonality=False,
                              growth=self.growth,
                              changepoint_prior_scale=self.cps,
                              interval_width=self.interval_width)
        self.model_.fit(df)
        self.fitted_ = None
        return self

    def _forecast_frame(self, horizon):
        fut = self.model_.make_future_dataframe(periods=horizon, freq="D")
        return self.model_.predict(fut).set_index("ds").iloc[-horizon:]

    def predict(self, horizon):
        out = self._forecast_frame(horizon)["yhat"].clip(lower=0)
        out.name = self.name
        return out

    def predict_interval(self, horizon, alpha=0.05):
        fc = self._forecast_frame(horizon)
        return pd.DataFrame({"lower": fc["yhat_lower"], "mean": fc["yhat"],
                             "upper": fc["yhat_upper"]}).clip(lower=0)


# ================================================================ 3. HOC MAY
class MLForecaster(Forecaster):
    """
    Bien du bao thanh hoi quy co giam sat tren dac trung tre, sau do du bao
    DE QUY: du doan t+1 -> noi vao lich su -> du doan t+2 -> ...

    Uu diem: dung duoc moi thuat toan hoi quy cua scikit-learn/XGBoost.
    Nhuoc diem: sai so tich luy dan qua tung buoc de quy.
    """
    def __init__(self, estimator, name: str, scale: bool = False,
                 lags=F.DEFAULT_LAGS, rolling_windows=F.DEFAULT_ROLL,
                 log_target: bool = False):
        super().__init__()
        self.estimator, self.name = estimator, name
        self.scale, self.lags, self.rolling_windows = scale, lags, rolling_windows
        self.log_target = log_target

    def fit(self, y):
        self._store(y)
        target = np.log1p(self.y_.clip(lower=0)) if self.log_target else self.y_
        feat = F.make_lag_features(target, self.lags, self.rolling_windows)
        X, y_sup = F.supervised_split(feat)
        self.columns_ = list(X.columns)
        self.t_offset_ = float(feat["t"].iloc[-1])

        if self.scale:
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            self.model_ = make_pipeline(StandardScaler(), self.estimator)
        else:
            self.model_ = self.estimator
        self.model_.fit(X, y_sup)

        pred_in = pd.Series(self.model_.predict(X), index=X.index)
        self.fitted_ = (np.expm1(pred_in) if self.log_target else pred_in).reindex(self.y_.index)
        return self

    def predict(self, horizon):
        hist = (np.log1p(self.y_.clip(lower=0)) if self.log_target else self.y_).copy()
        idx = future_index(self.y_.index[-1], horizon)
        preds = []
        for step, date in enumerate(idx, start=1):
            row = F.build_row(hist, self.columns_, self.lags,
                              self.rolling_windows, date, self.t_offset_ + step)
            yhat = float(self.model_.predict(row)[0])
            preds.append(yhat)
            hist = pd.concat([hist, pd.Series([yhat], index=[date])])
        out = np.expm1(np.asarray(preds)) if self.log_target else np.asarray(preds)
        return pd.Series(np.clip(out, 0, None), index=idx, name=self.name)


def ridge(alpha: float = 1.0, **kw) -> MLForecaster:
    from sklearn.linear_model import Ridge
    return MLForecaster(Ridge(alpha=alpha), "Ridge (hoi quy tuyen tinh)",
                        scale=True, **kw)


def random_forest(n_estimators: int = 300, **kw) -> MLForecaster:
    from sklearn.ensemble import RandomForestRegressor
    return MLForecaster(
        RandomForestRegressor(n_estimators=n_estimators, max_depth=12,
                              min_samples_leaf=2, n_jobs=-1,
                              random_state=RANDOM_STATE),
        "Random Forest", **kw)


def gradient_boosting(**kw) -> MLForecaster:
    from sklearn.ensemble import GradientBoostingRegressor
    return MLForecaster(
        GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                  max_depth=3, random_state=RANDOM_STATE),
        "Gradient Boosting", **kw)


def knn(n_neighbors: int = 5, **kw) -> MLForecaster:
    from sklearn.neighbors import KNeighborsRegressor
    return MLForecaster(KNeighborsRegressor(n_neighbors=n_neighbors),
                        "K-Nearest Neighbors", scale=True, **kw)


def svr(**kw) -> MLForecaster:
    from sklearn.svm import SVR
    return MLForecaster(SVR(kernel="rbf", C=100.0, gamma="scale", epsilon=0.1),
                        "SVR (RBF kernel)", scale=True, log_target=True, **kw)


def xgboost(n_estimators: int = 400, **kw) -> MLForecaster:
    from xgboost import XGBRegressor
    return MLForecaster(
        XGBRegressor(n_estimators=n_estimators, learning_rate=0.05, max_depth=5,
                     subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                     random_state=RANDOM_STATE, n_jobs=-1, verbosity=0),
        "XGBoost", **kw)


# ==================================================================== 4. DEEP
class LSTMForecaster(Forecaster):
    """
    Mang LSTM (Long Short-Term Memory) mot lop, cai bang PyTorch.

    Dau vao la cua so `lookback` ngay gan nhat (da chuan hoa MinMax),
    dau ra la gia tri ngay ke tiep; du bao nhieu buoc theo kieu de quy.
    """
    name = "LSTM (PyTorch)"

    def __init__(self, lookback: int = 28, hidden: int = 64, layers: int = 1,
                 epochs: int = 120, lr: float = 1e-2, batch_size: int = 32,
                 log_target: bool = True, seed: int = RANDOM_STATE):
        super().__init__()
        self.lookback, self.hidden, self.layers = lookback, hidden, layers
        self.epochs, self.lr, self.batch_size = epochs, lr, batch_size
        self.log_target, self.seed = log_target, seed

    def _build(self):
        import torch
        from torch import nn

        class Net(nn.Module):
            def __init__(self, hidden, layers):
                super().__init__()
                self.lstm = nn.LSTM(1, hidden, layers, batch_first=True)
                self.head = nn.Linear(hidden, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.head(out[:, -1, :])

        torch.manual_seed(self.seed)
        return Net(self.hidden, self.layers)

    def fit(self, y):
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset

        self._store(y)
        raw = np.log1p(self.y_.clip(lower=0)) if self.log_target else self.y_
        self.lo_, self.hi_ = float(raw.min()), float(raw.max())
        rng = max(self.hi_ - self.lo_, 1e-8)
        scaled = (raw - self.lo_) / rng

        X, Y = F.make_windows(scaled, self.lookback, horizon=1)
        if len(X) < 10:
            raise ValueError("Chuoi qua ngan so voi lookback cua LSTM")

        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        self.net_ = self._build()
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=8)
        crit = nn.MSELoss()

        self.net_.train()
        for _ in range(self.epochs):
            total = 0.0
            for xb, yb in dl:
                opt.zero_grad()
                loss = crit(self.net_(xb), yb)
                loss.backward()
                opt.step()
                total += float(loss) * len(xb)
            sched.step(total / len(ds))

        self.net_.eval()
        with torch.no_grad():
            pred = self.net_(torch.from_numpy(X)).numpy().ravel()
        pred = pred * rng + self.lo_
        if self.log_target:
            pred = np.expm1(pred)
        self.fitted_ = pd.Series(pred, index=scaled.index[self.lookback:]).reindex(self.y_.index)
        self._scaled_tail = scaled.to_numpy(dtype="float32")
        self._rng = rng
        return self

    def predict(self, horizon):
        import torch
        window = list(self._scaled_tail[-self.lookback:])
        preds = []
        self.net_.eval()
        with torch.no_grad():
            for _ in range(horizon):
                x = torch.tensor(np.asarray(window[-self.lookback:], dtype="float32")
                                 ).view(1, self.lookback, 1)
                nxt = float(self.net_(x).item())
                preds.append(nxt)
                window.append(nxt)
        vals = np.asarray(preds) * self._rng + self.lo_
        if self.log_target:
            vals = np.expm1(vals)
        return pd.Series(np.clip(vals, 0, None),
                         index=future_index(self.y_.index[-1], horizon), name=self.name)


# ================================================================== ensemble
class EnsembleForecaster(Forecaster):
    """Trung binh (co trong so) du bao cua nhieu mo hinh - thuong on dinh hon."""

    def __init__(self, models: list[Forecaster], weights=None,
                 name: str = "Ensemble"):
        super().__init__()
        self.models, self.name = models, name
        self.weights = None if weights is None else np.asarray(weights, dtype="float64")

    def fit(self, y):
        self._store(y)
        for m in self.models:
            m.fit(y)
        return self

    def predict(self, horizon):
        preds = np.column_stack([m.predict(horizon).to_numpy() for m in self.models])
        w = (np.ones(len(self.models)) if self.weights is None else self.weights)
        w = w / w.sum()
        return pd.Series(preds @ w, index=future_index(self.y_.index[-1], horizon),
                         name=self.name)


# ============================================================== danh muc chung
def default_models(period: int = SEASONAL_PERIOD, fast: bool = False,
                   use_prophet: bool = True, use_lstm: bool = True) -> dict[str, Forecaster]:
    """Danh muc mo hinh mac dinh dung trong thu nghiem so sanh."""
    models: dict[str, Forecaster] = {}

    def add(m: Forecaster):
        models[m.name] = m

    # 1. baseline
    add(NaiveForecaster())
    add(SeasonalNaiveForecaster(period))
    add(MeanForecaster(7))
    add(DriftForecaster())

    # 2. thong ke
    add(ExpSmoothingForecaster("ses"))
    add(ExpSmoothingForecaster("holt"))
    add(ExpSmoothingForecaster("hw", period))
    add(ThetaForecaster(period))
    add(ARIMAForecaster((2, 1, 2)))
    add(ARIMAForecaster((1, 1, 1), (1, 0, 1, period), label=f"SARIMA(1,1,1)x(1,0,1,{period})"))
    if not fast:
        add(AutoARIMAForecaster(max_p=3, max_q=3))
    if use_prophet:
        try:
            import prophet  # noqa: F401
            add(ProphetForecaster())
        except Exception:
            pass

    # 3. hoc may
    add(ridge())
    add(knn())
    add(random_forest(150 if fast else 300))
    add(gradient_boosting())
    add(svr())
    try:
        import xgboost as _xgb  # noqa: F401
        add(xgboost(200 if fast else 400))
    except Exception:
        pass

    # 4. deep learning
    if use_lstm:
        try:
            import torch  # noqa: F401
            add(LSTMForecaster(epochs=40 if fast else 120))
        except Exception:
            pass

    return models
