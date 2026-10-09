"""
Ve bieu do cho toan bo quy trinh phan tich - du bao.

Nguyen tac trinh bay dung xuyen suot module (de cac hinh nhin nhu MOT bo):
  * net ve MANH, luoi va truc MO  -> mat doc nhin vao du lieu truoc
  * bang mau phan loai co thu tu CO DINH, da kiem tra voi nguoi mu mau
    (khoang cach mau toi thieu deltaE >= 8 giua hai mau canh nhau)
  * mau GAN VOI MO HINH, khong gan voi thu hang -> loc bot mo hinh thi
    nhung mo hinh con lai van giu nguyen mau
  * toi da 5 duong mau tren mot hinh, phan con lai gom thanh mot nhom xam
  * khong bao gio dung hai truc y tren cung mot hinh
  * moi hinh tu 2 chuoi tro len deu co chu thich (legend)

Tat ca ham deu tra ve `matplotlib.figure.Figure` de nguoi dung tu quyet dinh
hien thi (notebook) hay luu file (`save`).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import OUT_FIG, SEASONAL_PERIOD

# ----------------------------------------------------------------- bang mau
SURFACE = "#fcfcfb"      # nen
INK = "#0b0b0b"          # chu chinh
INK_2 = "#52514e"        # chu phu
GRID = "#e6e5e1"         # luoi (mo hon nen mot bac)
HIST = "#8a8984"         # duong lich su / chuoi nen
BAND = "#c9c8c3"         # cot/vung khong duoc nhan manh

# thu tu mau CO DINH - da kiem dinh doi mau canh nhau voi 3 dang mu mau
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

POS, NEG = "#2a78d6", "#e34948"   # cap mau doi lap (duong / am)

_STYLE = {
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "figure.dpi": 110, "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 10.5, "axes.titlesize": 12.5, "axes.titleweight": "bold",
    "axes.labelsize": 10, "axes.labelcolor": INK_2,
    "text.color": INK, "axes.titlecolor": INK,
    "axes.edgecolor": GRID, "axes.linewidth": 0.8,
    "axes.grid": True, "axes.axisbelow": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "xtick.color": INK_2, "ytick.color": INK_2,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "xtick.direction": "out", "ytick.direction": "out",
    "legend.frameon": False, "legend.fontsize": 9.5,
    "lines.linewidth": 2.0, "lines.solid_capstyle": "round",
    "figure.autolayout": False,
}


def use_style() -> None:
    """Ap dung style chung (goi mot lan o dau notebook)."""
    mpl.rcParams.update(_STYLE)


use_style()


def _clean(ax, spine_left: bool = True) -> None:
    """Bo khung thua, chi giu luoi ngang - hinh 'nhe' hon nhieu."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_visible(spine_left)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)


def _thousands(ax) -> None:
    ax.yaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}".replace(",", ".")))


def _label_end(ax, s: pd.Series, color: str, text: str | None = None) -> None:
    """Dan nhan ngay tai diem cuoi duong - doc khong can do mat sang legend."""
    s = s.dropna()
    if s.empty:
        return
    ax.annotate(text or str(s.name), xy=(s.index[-1], s.iloc[-1]),
                xytext=(6, 0), textcoords="offset points",
                va="center", fontsize=9, color=color, fontweight="bold",
                clip_on=False)


def save(fig, filename: str, folder: Path = OUT_FIG) -> Path:
    """Luu hinh vao outputs/figures va tra ve duong dan."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (filename if filename.endswith(".png") else f"{filename}.png")
    fig.savefig(path)
    return path


# ============================================================ 1. kham pha chuoi
def plot_series(series: pd.Series | dict[str, pd.Series], title: str = "",
                ylabel: str = "So ca", highlight: str | None = None,
                figsize=(11, 4.2)):
    """
    Ve mot hoac vai chuoi theo thoi gian.

    highlight: ten chuoi can nhan manh - cac chuoi con lai chuyen sang xam
               (ky thuat "emphasis", tranh hinh 5 mau ma khong co trong tam).
    """
    data = {series.name or "series": series} if isinstance(series, pd.Series) else series
    fig, ax = plt.subplots(figsize=figsize)

    for i, (label, s) in enumerate(data.items()):
        if highlight is not None and label != highlight:
            ax.plot(s.index, s.to_numpy(), color=BAND, lw=1.4, zorder=2, label=label)
        else:
            c = SERIES[i % len(SERIES)] if highlight is None else SERIES[0]
            ax.plot(s.index, s.to_numpy(), color=c, lw=2.0, zorder=3, label=label)
            if len(data) <= 4:
                _label_end(ax, s, c, label)

    ax.set_title(title or "Dien bien theo thoi gian")
    ax.set_ylabel(ylabel)
    _clean(ax)
    _thousands(ax)
    if len(data) >= 2:
        ax.legend(loc="upper left", ncols=min(len(data), 4))
    fig.tight_layout()
    return fig


def plot_decomposition(decomp: pd.DataFrame, title: str = "Phan ra chuoi (STL)",
                       figsize=(11, 8.2)):
    """
    Bon panel: chuoi goc / xu the / mua vu / phan du.
    Moi panel mot chuoi duy nhat -> khong can legend, tieu de panel da du.
    """
    parts = [("observed", "Chuoi goc", HIST), ("trend", "Xu the (trend)", SERIES[0]),
             ("seasonal", f"Mua vu ({SEASONAL_PERIOD} ngay)", SERIES[2]),
             ("resid", "Phan du (resid)", SERIES[1])]
    parts = [p for p in parts if p[0] in decomp.columns]

    fig, axes = plt.subplots(len(parts), 1, figsize=figsize, sharex=True)
    for ax, (col, name, color) in zip(np.atleast_1d(axes), parts):
        s = decomp[col].dropna()
        if col == "resid":
            ax.axhline(0, color=GRID, lw=1)
            ax.plot(s.index, s.to_numpy(), color=color, lw=0, marker="o",
                    ms=3, alpha=0.65)
        else:
            ax.plot(s.index, s.to_numpy(), color=color, lw=1.8)
        ax.set_title(name, loc="left", fontsize=11)
        _clean(ax)
        _thousands(ax)
    fig.suptitle(title, x=0.008, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def plot_rolling_stats(series: pd.Series, window: int = 30,
                       title: str | None = None, figsize=(11, 4.2)):
    """
    Trung binh & do lech chuan truot - cach nhin nhanh tinh DUNG:
    neu hai duong nay troi theo thoi gian thi chuoi chua dung.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(series.index, series.to_numpy(), color=BAND, lw=1.2,
            label="Chuoi goc", zorder=2)
    m = series.rolling(window).mean()
    sd = series.rolling(window).std()
    ax.plot(m.index, m.to_numpy(), color=SERIES[0], lw=2.2,
            label=f"Trung binh truot {window} ngay", zorder=4)
    ax.plot(sd.index, sd.to_numpy(), color=SERIES[1], lw=2.0,
            label=f"Do lech chuan truot {window} ngay", zorder=3)
    ax.set_title(title or "Kiem tra tinh dung bang thong ke truot")
    _clean(ax)
    _thousands(ax)
    ax.legend(loc="upper left", ncols=3)
    fig.tight_layout()
    return fig


def plot_acf_pacf(table: pd.DataFrame, title: str = "ACF va PACF",
                  figsize=(11, 4.6)):
    """
    Bieu do cot tu tuong quan. Cot vuot ra ngoai vung xam la co y nghia
    thong ke -> goi y bac p (PACF) va q (ACF) cho ARIMA.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    for ax, col, name, color in ((axes[0], "acf", "ACF - goi y bac q (MA)", SERIES[0]),
                                 (axes[1], "pacf", "PACF - goi y bac p (AR)", SERIES[2])):
        d = table.dropna(subset=[col])
        ax.axhspan(d["nguong_duoi"].iloc[0], d["nguong_tren"].iloc[0],
                   color=GRID, alpha=0.9, zorder=1, lw=0)
        ax.axhline(0, color=INK_2, lw=0.8, zorder=2)
        ax.vlines(d["lag"], 0, d[col], color=color, lw=2.4, zorder=3)
        ax.plot(d["lag"], d[col], "o", ms=4, color=color, zorder=4,
                mec=SURFACE, mew=1.2)
        ax.set_title(name, loc="left", fontsize=11)
        ax.set_xlabel("Do tre (ngay)")
        _clean(ax)
    axes[0].set_ylabel("He so tuong quan")
    fig.suptitle(title, x=0.008, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


# ============================================================== 2. du bao
def plot_forecast(train: pd.Series, test: pd.Series | None, pred: pd.Series,
                  interval: pd.DataFrame | None = None, title: str | None = None,
                  ylabel: str = "So ca moi", context: int = 120,
                  figsize=(11, 4.6)):
    """
    Hinh "kinh dien" cua bai toan du bao:
    lich su (xam) | thuc te tap kiem tra (den) | du bao (xanh) | khoang tin cay.
    context: chi ve `context` ngay cuoi cua tap huan luyen cho de nhin.
    """
    fig, ax = plt.subplots(figsize=figsize)
    hist = train.iloc[-context:] if context else train
    ax.plot(hist.index, hist.to_numpy(), color=HIST, lw=1.6,
            label="Lich su (huan luyen)", zorder=2)

    if interval is not None:
        ax.fill_between(interval.index, interval["lower"], interval["upper"],
                        color=SERIES[0], alpha=0.15, lw=0, zorder=2,
                        label="Khoang tin cay 95%")
    if test is not None:
        joined_t = pd.concat([hist.iloc[-1:], test])
        ax.plot(joined_t.index, joined_t.to_numpy(), color=INK, lw=2.2,
                label="Thuc te", zorder=4)
    joined_p = pd.concat([hist.iloc[-1:], pred])
    ax.plot(joined_p.index, joined_p.to_numpy(), color=SERIES[0], lw=2.2,
            ls=(0, (5, 2)), label="Du bao", zorder=5)
    ax.plot(pred.index, pred.to_numpy(), "o", ms=4.5, color=SERIES[0],
            mec=SURFACE, mew=1.2, zorder=6)

    ax.axvline(hist.index[-1], color=INK_2, lw=1, ls=":", zorder=3)
    ax.annotate("bat dau du bao", xy=(hist.index[-1], ax.get_ylim()[1]),
                xytext=(5, -10), textcoords="offset points",
                fontsize=8.5, color=INK_2, va="top")

    ax.set_title(title or f"Du bao {len(pred)} ngay - {pred.name}")
    ax.set_ylabel(ylabel)
    _clean(ax)
    _thousands(ax)
    ax.legend(loc="upper left", ncols=2)
    fig.tight_layout()
    return fig


def plot_model_comparison(actual: pd.Series, preds: pd.DataFrame,
                          top: int = 5, order: list[str] | None = None,
                          title: str | None = None, figsize=(11, 5),
                          context: pd.Series | None = None):
    """
    Dat du bao cua nhieu mo hinh canh nhau tren cung mot truc.

    Chi to mau `top` mo hinh dau tien (theo `order` - thuong la thu hang tu
    bang xep hang); so con lai ve xam nhat thanh mot nhom "cac mo hinh khac"
    - qua 8 mau thi mat khong con phan biet duoc nua.
    """
    cols = [c for c in (order or list(preds.columns)) if c in preds.columns]
    shown, rest = cols[:top], cols[top:]

    fig, ax = plt.subplots(figsize=figsize)
    if context is not None and len(context):
        ax.plot(context.index, context.to_numpy(), color=HIST, lw=1.4,
                label="Lich su", zorder=2)
    for i, c in enumerate(rest):
        ax.plot(preds.index, preds[c].to_numpy(), color=BAND, lw=1.2, alpha=0.8,
                zorder=3, label=f"{len(rest)} mo hinh khac" if i == 0 else None)
    for i, c in enumerate(shown):
        ax.plot(preds.index, preds[c].to_numpy(), color=SERIES[i % len(SERIES)],
                lw=2.0, zorder=4, label=c)
    ax.plot(actual.index, actual.to_numpy(), color=INK, lw=2.6, zorder=6,
            label="Thuc te")

    ax.set_title(title or f"So sanh du bao {len(actual)} ngay cua cac mo hinh")
    ax.set_ylabel("So ca moi")
    _clean(ax)
    _thousands(ax)
    ax.legend(loc="upper left", ncols=2, fontsize=9)
    fig.tight_layout()
    return fig


def plot_fan(history: pd.Series, forecast: pd.DataFrame, context: int = 150,
             title: str | None = None, figsize=(11, 4.6)):
    """Du bao tuong lai that su (khong co thuc te de doi chieu) + dai tin cay."""
    fig, ax = plt.subplots(figsize=figsize)
    hist = history.iloc[-context:] if context else history
    ax.plot(hist.index, hist.to_numpy(), color=HIST, lw=1.8, label="Da quan sat")

    if {"can_duoi", "can_tren"} <= set(forecast.columns):
        ax.fill_between(forecast.index, forecast["can_duoi"], forecast["can_tren"],
                        color=SERIES[0], alpha=0.15, lw=0, label="Khoang tin cay 95%")
    line = pd.concat([hist.iloc[-1:], forecast["du_bao"]])
    ax.plot(line.index, line.to_numpy(), color=SERIES[0], lw=2.4,
            ls=(0, (5, 2)), label="Du bao")
    ax.axvline(hist.index[-1], color=INK_2, lw=1, ls=":")

    ax.set_title(title or f"Du bao {len(forecast)} ngay toi")
    ax.set_ylabel("So ca moi")
    _clean(ax)
    _thousands(ax)
    ax.legend(loc="upper left", ncols=3)
    fig.tight_layout()
    return fig


# ========================================================== 3. danh gia
def plot_leaderboard(board: pd.DataFrame, metric: str = "RMSE", top: int = 12,
                     title: str | None = None, figsize=(10, 5.6)):
    """
    Bang xep hang dang cot ngang.

    Tat ca cot cung MOT mau (danh muc khong co thu tu tu nhien), rieng mo hinh
    tot nhat duoc to dam de mat bat ngay - khong dung dai mau theo gia tri.
    """
    asc = metric != "R2"
    d = board[metric].dropna().sort_values(ascending=asc).head(top)
    d = d.iloc[::-1]                                   # tot nhat nam tren cung
    best = d.index[-1]

    fig, ax = plt.subplots(figsize=figsize)
    colors = [SERIES[0] if n == best else BAND for n in d.index]
    bars = ax.barh(range(len(d)), d.to_numpy(), color=colors, height=0.68)
    ax.set_yticks(range(len(d)), [str(i) for i in d.index])

    span = float(d.max()) or 1.0
    for rect, val in zip(bars, d.to_numpy()):
        ax.annotate(f"{val:,.2f}".replace(",", "."),
                    xy=(rect.get_width(), rect.get_y() + rect.get_height() / 2),
                    xytext=(5, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK_2)

    ax.set_xlim(0, span * 1.16)
    ax.set_title(title or f"Xep hang mo hinh theo {metric} (cang thap cang tot)")
    ax.set_xlabel(metric)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    return fig


def plot_metric_pair(board: pd.DataFrame, left: str = "RMSE", right: str = "MASE",
                     top: int = 10, figsize=(11, 5.4)):
    """
    Hai chi so canh nhau bang HAI panel rieng - khong bao gio chong hai truc y
    len cung mot hinh (ty le giua hai truc la tuy tien -> gay hieu nham).
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for ax, metric in zip(axes, (left, right)):
        d = board[metric].dropna().sort_values().head(top).iloc[::-1]
        best = d.index[-1]
        ax.barh(range(len(d)), d.to_numpy(), height=0.66,
                color=[SERIES[0] if n == best else BAND for n in d.index])
        ax.set_yticks(range(len(d)), [str(i)[:26] for i in d.index], fontsize=8.5)
        ax.set_title(metric, loc="left", fontsize=11)
        if metric == "MASE":
            ax.axvline(1.0, color=NEG, lw=1.2, ls="--")
            ax.annotate("=1: ngang mo hinh naive", xy=(1.0, len(d) - 0.4),
                        xytext=(4, 0), textcoords="offset points",
                        fontsize=8.5, color=NEG)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
    fig.suptitle("Hai goc nhin ve cung mot ket qua", x=0.008, ha="left",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def plot_residuals(actual: pd.Series, pred: pd.Series, nlags: int = 24,
                   title: str = "Chan doan phan du", figsize=(11, 7)):
    """
    Bon kiem tra bat buoc voi phan du e = thuc te - du bao.
    Mo hinh tot thi phan du phai: quanh 0, khong xu the, khong tu tuong quan,
    va gan phan phoi chuan.
    """
    from scipy import stats
    from statsmodels.tsa.stattools import acf

    resid = (actual - pred).dropna()
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    ax = axes[0, 0]
    ax.axhline(0, color=INK_2, lw=1)
    ax.vlines(resid.index, 0, resid.to_numpy(),
              color=[POS if v >= 0 else NEG for v in resid], lw=2.2)
    ax.set_title("Phan du theo thoi gian", loc="left", fontsize=11)
    _clean(ax)

    ax = axes[0, 1]
    ax.hist(resid, bins=min(20, max(6, len(resid) // 3)), color=SERIES[0],
            alpha=0.85, edgecolor=SURFACE, linewidth=1.5)
    ax.set_title("Phan bo phan du", loc="left", fontsize=11)
    _clean(ax)

    ax = axes[1, 0]
    n = len(resid)
    a = acf(resid, nlags=min(nlags, n // 2), fft=True)
    ci = 1.96 / np.sqrt(n)
    ax.axhspan(-ci, ci, color=GRID, alpha=0.9, lw=0)
    ax.axhline(0, color=INK_2, lw=0.8)
    ax.vlines(range(len(a)), 0, a, color=SERIES[2], lw=2.4)
    ax.set_title("ACF phan du (nam trong vung xam = tot)", loc="left", fontsize=11)
    ax.set_xlabel("Do tre")
    _clean(ax)

    ax = axes[1, 1]
    (osm, osr), (slope, icept, _) = stats.probplot(resid, dist="norm")
    ax.plot(osm, osr, "o", ms=4.5, color=SERIES[0], mec=SURFACE, mew=1.1)
    ax.plot(osm, slope * osm + icept, color=INK_2, lw=1.4, ls="--")
    ax.set_title("Q-Q plot (thang hang = phan phoi chuan)", loc="left", fontsize=11)
    ax.set_xlabel("Phan vi ly thuyet")
    _clean(ax)

    fig.suptitle(f"{title} - {pred.name}", x=0.008, ha="left",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def plot_backtest_folds(y: pd.Series, folds: pd.DataFrame, metric: str = "RMSE",
                        title: str | None = None, figsize=(11, 4.4)):
    """
    Sai so cua tung fold trong rolling-origin: cho thay mo hinh on dinh hay
    chi tot o mot giai doan (vi du chi tot khi dich dang phang).
    """
    d = folds.dropna(subset=[metric]) if metric in folds.columns else folds
    fig, ax = plt.subplots(figsize=figsize)
    x = [str(v) for v in d["cat_tai"]]
    ax.bar(x, d[metric].to_numpy(), color=SERIES[0], width=0.55)
    mean = float(d[metric].mean())
    ax.axhline(mean, color=INK_2, lw=1.2, ls="--")
    ax.annotate(f"trung binh {mean:,.1f}".replace(",", "."),
                xy=(len(x) - 0.5, mean), xytext=(0, 5),
                textcoords="offset points", ha="right", fontsize=9, color=INK_2)
    ax.set_title(title or f"{metric} theo tung fold (rolling-origin)")
    ax.set_xlabel("Ngay bat dau doan kiem tra")
    ax.set_ylabel(metric)
    _clean(ax)
    fig.tight_layout()
    return fig
