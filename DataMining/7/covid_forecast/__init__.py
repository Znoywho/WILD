"""
covid_forecast
==============
Thu vien minh hoa phan tich & du bao chuoi thoi gian (time series)
tren bo du lieu COVID-19 cua JHU CSSE.

Cac module:
    config    - tham so & duong dan chung
    data      - tai, lam sach, bien doi du lieu JHU ve dang chuoi thoi gian
    analysis  - phan tich chuoi: decomposition, stationarity, ACF/PACF
    metrics   - cac do do sai so du bao
    features  - sinh dac trung (lag / rolling / lich) cho mo hinh ML
    models    - cac mo hinh du bao (baseline, thong ke, ML, deep learning)
    backtest  - chia du lieu, danh gia holdout va rolling-origin
    plots     - ve bieu do ket qua
"""

__version__ = "1.0.0"
