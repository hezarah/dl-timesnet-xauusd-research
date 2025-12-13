# -*- coding: utf-8 -*-
# اجرای استریم‌لیت (فقط برای یادآوری):
#   streamlit run app_GPU.py

import os
import time
import gc
import json
import re
from pathlib import Path
import io
import pandas as pd
import numpy as np
import cupy as cp
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype
from cupyx.scipy.ndimage import maximum_filter1d, minimum_filter1d
import streamlit as st
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

# ---- MetaTrader5 for live trading ----
try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None



def normalize_volume(symbol: str, volume: float) -> float:
    """
    حجم را طوری تنظیم می‌کند که:
      - بین volume_min و volume_max باشد
      - مضرب volume_step باشد
    و همیشه به سمت پایین گرد می‌کند.
    """
    info = mt5.symbol_info(symbol)
    if info is None:
        return 0.0

    vol_min = info.volume_min
    vol_max = info.volume_max
    step = info.volume_step or 0.01

    # اول بین مین و مکس کلیپ کنیم
    vol = max(vol_min, min(float(volume), vol_max))

    if vol < vol_min:
        return 0.0

    # گرد به پایین روی استپ
    n_steps = int(vol / step)  # floor
    vol = n_steps * step

    if vol < vol_min:
        return 0.0

    return float(vol)



def tf_str_to_mt5(tf_str: str):
    """
    تبدیل رشته‌ی تایم‌فریم (مثل '5T','15T','30T','1H','4H','1D')
    به کانستنت‌های TIMEFRAME_* مربوط به MetaTrader5.

    اگر mt5 در دسترس نباشد یا تایم‌فریم ناشناخته باشد، None برمی‌گرداند.
    """
    if mt5 is None:
        return None

    s = str(tf_str).upper().replace(" ", "")

    mapping = {
        "5T":  mt5.TIMEFRAME_M5,
        "M5":  mt5.TIMEFRAME_M5,

        "15T": mt5.TIMEFRAME_M15,
        "M15": mt5.TIMEFRAME_M15,

        "30T": mt5.TIMEFRAME_M30,
        "M30": mt5.TIMEFRAME_M30,

        "1H":  mt5.TIMEFRAME_H1,
        "H1":  mt5.TIMEFRAME_H1,

        "4H":  mt5.TIMEFRAME_H4,
        "H4":  mt5.TIMEFRAME_H4,

        "1D":  mt5.TIMEFRAME_D1,
        "D1":  mt5.TIMEFRAME_D1,
    }

    return mapping.get(s, None)


# ---------------------------------------------------------------------
#  تنظیمات عمومی GPU / PyTorch
# ---------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TORCH_DTYPE = torch.float32

torch.set_num_threads(1)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# ---------------------------------------------------------------------
#  مدیریت نام دیتاست و مسیر فایل‌ها (پایداری بین اجراها)
# ---------------------------------------------------------------------

# می‌تونی اگر خواستی پوشه‌ی جدا برای دیتا تعریف کنی
DATA_ROOT = Path(".").resolve()
LAST_DATASET_FILE = DATA_ROOT / "last_dataset.json"


# -----------------------------
# توابع کمکی GPU برای کار با CuPy
# -----------------------------

def to_cp(x):
    """
    تبدیل Series/ndarray به آرایه‌ی CuPy با dtype=float32
    """
    if isinstance(x, (pd.Series, pd.Index)):
        arr = x.to_numpy(copy=False)
    else:
        arr = np.asarray(x)
    return cp.asarray(arr, dtype=cp.float32)


def cp_to_np32(x_cp):
    """
    تبدیل آرایه‌ی CuPy به numpy.float32 (برای ذخیره در DataFrame)
    """
    arr = cp.asnumpy(x_cp)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    return arr


def gpu_shift(x, n: int, fill_value=cp.nan):
    """
    مشابه تابع shift در pandas اما روی آرایه‌ی CuPy.
    x: آرایه CuPy
    n: تعداد شیفت (مثبت = به جلو، منفی = به عقب)
    fill_value: مقدار پرکردن خانه‌های خالی
    """
    x = cp.asarray(x)
    res = cp.empty_like(x)

    if n > 0:
        res[:n] = fill_value
        res[n:] = x[:-n]
    elif n < 0:
        res[n:] = fill_value
        res[:n] = x[-n:]
    else:
        res[...] = x
    return res


# ---------------------------------------
# Rolling functions with CuPy (GPU-based)
# ---------------------------------------

def rolling_mean_cp(x, window: int):
    """
    محاسبه میانگین متحرک روی GPU با CuPy
    """
    x = cp.asarray(x, dtype=cp.float32)
    n = x.shape[0]

    if window <= 1 or window > n:
        # اگر پنجره ۱ یا بزرگتر از طول سری بود، همان x یا همه NaN برمی‌گردانیم
        return x.copy()

    # cumsum معمولی (طول N)
    cumsum = cp.cumsum(x, dtype=cp.float32)

    # cumsum با یک صفر در ابتدای آن → طول N+1
    cumsum0 = cp.concatenate((cp.zeros(1, dtype=cumsum.dtype), cumsum))

    res = cp.empty_like(x)
    # تا قبل از کامل شدن پنجره → NaN
    res[:window-1] = cp.nan

    # از ایندکس window-1 به بعد: میانگین روی پنجره‌ی طول window
    # sum[i] = cumsum0[i+1] - cumsum0[i+1-window]
    # i از window-1 تا n-1 → cumsum0[window: ] و cumsum0[:n+1-window]
    res[window-1:] = (cumsum0[window:] - cumsum0[:n+1-window]) / window

    return res


def rolling_sum_cp(x, window: int):
    """
    جمع متحرک روی GPU با CuPy
    """
    x = cp.asarray(x, dtype=cp.float32)
    n = x.shape[0]

    if window <= 1 or window > n:
        return x.copy()

    cumsum = cp.cumsum(x, dtype=cp.float32)
    cumsum0 = cp.concatenate((cp.zeros(1, dtype=cumsum.dtype), cumsum))

    res = cp.empty_like(x)
    res[:window-1] = cp.nan
    res[window-1:] = cumsum0[window:] - cumsum0[:n+1-window]

    return res


def rolling_std_cp(x, window: int):
    """
    انحراف معیار متحرک روی GPU
    """
    x = cp.asarray(x, dtype=cp.float32)
    n = x.shape[0]

    if window <= 1 or window > n:
        return cp.zeros_like(x)

    mean = rolling_mean_cp(x, window)
    mean_sq = rolling_mean_cp(x * x, window)
    var = mean_sq - mean * mean
    return cp.sqrt(cp.maximum(var, 0.0))

# ---------------------------------------
# اندیکاتورهای GPU (CuPy) برای تب "افزودن فیچرها"
# ---------------------------------------

def ema_cp(x, period: int):
    """
    EMA کلاسیک روی آرایه CuPy
    """
    x = cp.asarray(x, dtype=cp.float32)
    n = x.shape[0]
    if n == 0:
        return x.copy()
    if period <= 1:
        return x.copy()

    alpha = cp.float32(2.0) / cp.float32(period + 1)
    ema = cp.empty_like(x)
    ema[0] = x[0]
    for i in range(1, n):
        ema[i] = alpha * x[i] + (cp.float32(1.0) - alpha) * ema[i - 1]
    return ema


def _wilder_smooth_cp(x, period: int):
    """
    هموارسازی Wilder (برای ATR, +DM, -DM, gain, loss ...)
    """
    x = cp.asarray(x, dtype=cp.float32)
    n = x.shape[0]
    out = cp.empty_like(x)
    out[:] = cp.nan

    if n == 0 or period <= 0:
        return out

    if n <= period:
        return out

    # اولین مقدار: میانگین ساده روی دورهٔ اول (از اندیس 1 تا period)
    first = cp.mean(x[1:period+1])
    out[:period] = cp.nan
    out[period] = first

    for i in range(period + 1, n):
        out[i] = out[i-1] - out[i-1] / period + x[i] / period

    return out


def rsi_cp(c, period: int):
    """
    RSI به روش Wilder روی GPU
    """
    c = cp.asarray(c, dtype=cp.float32)
    n = c.shape[0]
    if n == 0 or period <= 0:
        return cp.zeros_like(c)

    delta = c - gpu_shift(c, 1)
    delta = cp.where(cp.isnan(delta), 0.0, delta)

    gain = cp.where(delta > 0, delta, 0.0)
    loss = cp.where(delta < 0, -delta, 0.0)

    avg_gain = _wilder_smooth_cp(gain, period)
    avg_loss = _wilder_smooth_cp(loss, period)

    rs = avg_gain / (avg_loss + 1e-8)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:period] = cp.nan
    return rsi


def adx_full_cp(h, l, c, period: int):
    """
    ADX کامل به‌همراه +DI, -DI و ATR
    خروجی: (adx_vals, plus_di, minus_di, atr_vals)

    این نسخه طوری اصلاح شده که:
      - NaN های اولیه در +DI/-DI و DX باعث NaN شدن کل سری ADX نشوند.
      - با این‌حال برای 2*period اول، ADX عمداً NaN نگه داشته می‌شود (warmup منطقی).
    """
    h = cp.asarray(h, dtype=cp.float32)
    l = cp.asarray(l, dtype=cp.float32)
    c = cp.asarray(c, dtype=cp.float32)

    n = c.shape[0]
    if n == 0 or period <= 0:
        z = cp.zeros_like(c)
        return z, z, z, z

    # قیمت‌های قبلی
    h_prev = gpu_shift(h, 1)
    l_prev = gpu_shift(l, 1)
    c_prev = gpu_shift(c, 1)

    # --- True Range ---
    tr1 = h - l
    tr2 = cp.abs(h - c_prev)
    tr3 = cp.abs(l - c_prev)
    tr = cp.maximum(tr1, cp.maximum(tr2, tr3))
    # اگر به هر دلیلی NaN شد، صفرش می‌کنیم
    tr = cp.where(cp.isfinite(tr), tr, 0.0)

    # --- +DM و -DM ---
    up_move   = h - h_prev
    down_move = l_prev - l

    plus_dm = cp.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = cp.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = cp.where(cp.isfinite(plus_dm), plus_dm, 0.0)
    minus_dm = cp.where(cp.isfinite(minus_dm), minus_dm, 0.0)

    # --- Wilder smoothing روی TR و DM ---
    atr         = _wilder_smooth_cp(tr, period)
    plus_dm_s   = _wilder_smooth_cp(plus_dm, period)
    minus_dm_s  = _wilder_smooth_cp(minus_dm, period)

    # --- +DI و -DI ---
    plus_di = 100.0 * (plus_dm_s / (atr + 1e-8))
    minus_di = 100.0 * (minus_dm_s / (atr + 1e-8))

    # 🔧 فیکس مهم: جلوگیری از انتشار NaN/inf به داخل DX
    plus_di = cp.where(cp.isfinite(plus_di), plus_di, 0.0)
    minus_di = cp.where(cp.isfinite(minus_di), minus_di, 0.0)

    # --- DX ---
    dx = 100.0 * cp.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
    dx = cp.where(cp.isfinite(dx), dx, 0.0)

    # --- ADX با Wilder smoothing روی DX ---
    adx = _wilder_smooth_cp(dx, period)

    # طبق تعریف کلاسیک، معمولاً 2*period اول ADX را NaN در نظر می‌گیرند
    adx[: 2 * period] = cp.nan

    return adx, plus_di, minus_di, atr


def cci_cp(h, l, c, period: int):
    """
    CCI (Commodity Channel Index) روی GPU

    CCI = (TP - SMA(TP)) / (0.015 * MeanDeviation(TP))
    که در آن:
        TP = (High + Low + Close) / 3
        MeanDeviation = میانگین |TP - SMA(TP)| در پنجره‌ی period
    """
    h = cp.asarray(h, dtype=cp.float32)
    l = cp.asarray(l, dtype=cp.float32)
    c = cp.asarray(c, dtype=cp.float32)

    n = c.shape[0]

    # گارد روی period و طول داده
    try:
        period = int(period)
    except Exception:
        return cp.zeros_like(c)

    if period <= 1 or n == 0:
        return cp.zeros_like(c)

    if period > n:
        # اگر داده کمتر از period باشد، برای جلوگیری از NaN روی کندل پایه، صفر برمی‌گردانیم
        return cp.zeros_like(c)

    # --- محاسبه‌ی تیپیکال پرایس ---
    tp = (h + l + c) / 3.0  # typical price

    # میانگین متحرک TP
    sma_tp = rolling_mean_cp(tp, period)

    # انحراف از میانگین
    dev = cp.abs(tp - sma_tp)

    # مهم: NaN های dev را صفر می‌کنیم تا وارد cumsum در rolling_mean_cp نشوند
    dev = cp.where(cp.isnan(dev), cp.float32(0.0), dev)

    # میانگین انحراف
    mean_dev = rolling_mean_cp(dev, period)

    denom = 0.015 * (mean_dev + 1e-8)
    cci = (tp - sma_tp) / denom

    # warm-up اولیه: CCI در اولین period کندل معتبر نیست
    cci[:period] = cp.nan

    # اگر جایی به هر دلیل inf یا NaN شد، آن‌جا را NaN نگه می‌داریم
    cci = cp.where(cp.isfinite(cci), cci, cp.nan)

    return cci


def williams_r_cp(h, l, c, period: int):
    """
    Williams %R روی GPU
    %R = -100 * (HH - Close) / (HH - LL)
    """
    h = cp.asarray(h, dtype=cp.float32)
    l = cp.asarray(l, dtype=cp.float32)
    c = cp.asarray(c, dtype=cp.float32)
    n = c.shape[0]
    if period <= 1 or period > n:
        return cp.zeros_like(c)

    # رولینگ بیشینه/کمینه با cupyx.scipy.ndimage
    highest = maximum_filter1d(h, size=period, mode="nearest")
    lowest  = minimum_filter1d(l, size=period, mode="nearest")

    wr = -100.0 * (highest - c) / (highest - lowest + 1e-8)
    wr[:period-1] = cp.nan
    return wr


def hv_cp(log_returns, period: int, annualization: float = 252.0):
    """
    Historical Volatility: std(log_return) * sqrt(annualization)
    """
    lr = cp.asarray(log_returns, dtype=cp.float32)
    n = lr.shape[0]
    if period <= 1 or period > n:
        return cp.zeros_like(lr)

    rolling_std = rolling_std_cp(lr, period)
    hv = cp.sqrt(annualization) * rolling_std
    hv[:period] = cp.nan
    return hv



# ---------------------------------------
# الگوهای کندلی روی GPU (ساده‌شده)
# ---------------------------------------

def patterns1_gpu(o, h, l, c):
    """
    برچسب‌گذاری تک‌کندلی با جزئیات بالا (خروجی: کُد عددی در یک ستون)

    0  = بدون الگو / کندل عادی (یا رِنج بسیار کوچک)
    ---- Doji و مشتقات ----
    1  = Doji رِنج کوچک (بدنه خیلی کوچک، رِنج کوچک)
    2  = Doji رِنج متوسط
    3  = Doji رِنج بزرگ
    4  = Long-Legged Doji (سایه‌های بالا و پایین هر دو بزرگ)
    5  = Dragonfly Doji  (سایه‌ی پایینی خیلی بزرگ، بالایی خیلی کوچک)
    6  = Gravestone Doji (سایه‌ی بالایی خیلی بزرگ، پایینی خیلی کوچک)

    ---- Spinning Top ----
    7  = Spinning Top صعودی  (بدنه کوچک، سایه‌ها نسبتاً بزرگ، close > open)
    8  = Spinning Top نزولی  (بدنه کوچک، سایه‌ها نسبتاً بزرگ, open > close)
    9  = Spinning Top خنثی   (بدنه کوچک، سایه‌ها بزرگ، close ≈ open ولی نه به کوچکی Doji)

    ---- کندل‌های معمولی بر اساس اندازه بدنه (نه دوجی و نه چکش و ...) ----
    10 = Bull Small Body      (body_rel ∈ [0.10, 0.30), close > open)
    11 = Bear Small Body      (body_rel ∈ [0.10, 0.30), open  > close)
    12 = Bull Medium Body     (body_rel ∈ [0.30, 0.60), close > open)
    13 = Bear Medium Body     (body_rel ∈ [0.30, 0.60), open  > close)
    14 = Bull Long Body       (body_rel ∈ [0.60, 0.85), close > open)
    15 = Bear Long Body       (body_rel ∈ [0.60, 0.85), open  > close)
    16 = Bull Very Long Body  (body_rel ≥ 0.85, close > open)
    17 = Bear Very Long Body  (body_rel ≥ 0.85, open  > close)

    ---- خانواده Hammer / Hanging Man (سایه پایینی بزرگ، بالایی کوچک) ----
    18 = Hammer ضعیف (lower_rel >= 0.5, upper_rel <= 0.25, body_rel بین 0.1 تا 0.4)
    19 = Hammer قوی  (lower_rel >= 0.7, upper_rel <= 0.15, body_rel بین 0.1 تا 0.4)
    20 = Hanging Man ضعیف (هم شکل hammer، ترجیحاً کندل نزولی یا close نزدیک high)
    21 = Hanging Man قوی  (مثل بالا اما شدیدتر)

    ---- خانواده Inverted Hammer / Shooting Star (سایه بالایی بزرگ، پایینی کوچک) ----
    22 = Inverted Hammer ضعیف (upper_rel >= 0.5, lower_rel <= 0.25, body_rel بین 0.1 تا 0.4)
    23 = Inverted Hammer قوی  (upper_rel >= 0.7, lower_rel <= 0.15, body_rel بین 0.1 تا 0.4)
    24 = Shooting Star ضعیف (شبیه inverted hammer ولی با تمایل نزولی)
    25 = Shooting Star قوی  (همان، با شرایط قوی‌تر)

    ---- Marubozu (بدنه تقریباً تمام رِنج) ----
    26 = Bullish Marubozu قوی   (body_rel ≥ 0.90, upper_rel <= 0.05, lower_rel <= 0.05)
    27 = Bullish Marubozu ضعیف  (body_rel ∈ [0.80, 0.90), upper_rel <= 0.10, lower_rel <= 0.10)
    28 = Bearish Marubozu قوی   (body_rel ≥ 0.90, upper_rel <= 0.05, lower_rel <= 0.05)
    29 = Bearish Marubozu ضعیف  (body_rel ∈ [0.80, 0.90), upper_rel <= 0.10, lower_rel <= 0.10)

    نکته:
    - بر اساس ترتیب اعمال، الگوهای خاص (Doji, Hammer, Marubozu, ...) می‌توانند
      کُدهای عمومی‌تر (small/medium/long body) را Override کنند.
    """

    o = cp.asarray(o, dtype=cp.float32)
    h = cp.asarray(h, dtype=cp.float32)
    l = cp.asarray(l, dtype=cp.float32)
    c = cp.asarray(c, dtype=cp.float32)

    n = c.shape[0]
    codes = cp.zeros(n, dtype=cp.int16)

    # اجزای کندل
    body = cp.abs(c - o)
    rng = h - l
    eps = cp.float32(1e-8)
    rng_safe = cp.where(rng <= eps, eps, rng)

    upper = h - cp.maximum(o, c)
    lower = cp.minimum(o, c) - l

    upper_rel = upper / rng_safe
    lower_rel = lower / rng_safe
    body_rel = body / rng_safe

    is_bull = c > o
    is_bear = o > c

    # برای تشخیص کوچک/متوسط/بزرگ بودن رِنج، از مدین استفاده می‌کنیم
    rng_med = cp.median(rng_safe)
    small_range = rng_safe <= 0.5 * rng_med
    large_range = rng_safe >= 1.5 * rng_med
    mid_range = ~(small_range | large_range)

    # -----------------------
    # ۱) Doji و مشتقات
    # -----------------------
    very_small_body = body_rel <= 0.05
    small_body = (body_rel > 0.05) & (body_rel <= 0.15)

    # Doji اصلی (بدنه خیلی کوچک)
    is_doji = very_small_body & (rng > eps)

    # Doji بر اساس رِنج
    codes = cp.where(is_doji & small_range, 1, codes)
    codes = cp.where(is_doji & mid_range,   2, codes)
    codes = cp.where(is_doji & large_range, 3, codes)

    # Long-Legged Doji
    long_leg = is_doji & (upper_rel >= 0.3) & (lower_rel >= 0.3)
    codes = cp.where(long_leg, 4, codes)

    # Dragonfly Doji
    dragonfly = is_doji & (lower_rel >= 0.6) & (upper_rel <= 0.1)
    codes = cp.where(dragonfly, 5, codes)

    # Gravestone Doji
    gravestone = is_doji & (upper_rel >= 0.6) & (lower_rel <= 0.1)
    codes = cp.where(gravestone, 6, codes)

    # -----------------------
    # ۲) Spinning Top
    # -----------------------
    # بدنه کوچک، سایه‌ها نسبتاً بزرگ، ولی بدنه کمی بزرگ‌تر از دوجی
    spinning_core = small_body & ((upper_rel + lower_rel) >= 0.4)

    spin_bull = spinning_core & is_bull
    spin_bear = spinning_core & is_bear
    spin_neutral = spinning_core & (~is_bull & ~is_bear)

    codes = cp.where(spin_bull,    7, codes)
    codes = cp.where(spin_bear,    8, codes)
    codes = cp.where(spin_neutral, 9, codes)

    # -----------------------
    # ۳) طبقه‌بندی عمومی بر اساس بدنه (اگر هنوز کُدی نخورده باشد)
    # -----------------------
    # tiny: body_rel <= 0.05 (قبلاً دوجی‌ها را پوشش داده‌ایم)
    small_body2 = (body_rel > 0.05) & (body_rel <= 0.30)
    med_body = (body_rel > 0.30) & (body_rel <= 0.60)
    long_body = (body_rel > 0.60) & (body_rel <= 0.85)
    very_long_body = body_rel > 0.85

    # فقط جایی که هنوز کُد صفر است، کُد عمومی می‌دهیم
    no_code = codes == 0

    codes = cp.where(no_code & small_body2 & is_bull, 10, codes)
    codes = cp.where(no_code & small_body2 & is_bear, 11, codes)

    codes = cp.where(no_code & med_body & is_bull,   12, codes)
    codes = cp.where(no_code & med_body & is_bear,   13, codes)

    codes = cp.where(no_code & long_body & is_bull,  14, codes)
    codes = cp.where(no_code & long_body & is_bear,  15, codes)

    codes = cp.where(no_code & very_long_body & is_bull, 16, codes)
    codes = cp.where(no_code & very_long_body & is_bear, 17, codes)

    # در صورت رِنج خیلی نزدیک به صفر، همان ۰ بماند
    # (نیازی به کاری نیست؛ چون کُد اولیه ۰ است)

    # -----------------------
    # ۴) Hammer / Hanging Man
    # -----------------------
    hammer_core = (body_rel >= 0.10) & (body_rel <= 0.40) & (lower_rel >= 0.5) & (upper_rel <= 0.25)
    hammer_strong = hammer_core & (lower_rel >= 0.7) & (upper_rel <= 0.15)
    hammer_weak = hammer_core & ~hammer_strong

    # Hammer (تمایل صعودی یا بدنه نزدیک کف)
    hammer_like_bullish = hammer_core & (is_bull | ((c >= o) & (upper_rel <= 0.2)))

    # Hanging Man (همان شکل ولی تمایل نزولی / نزدیک سقف)
    hanging_like_bearish = hammer_core & (~hammer_like_bullish)

    codes = cp.where(hammer_weak & hammer_like_bullish, 18, codes)
    codes = cp.where(hammer_strong & hammer_like_bullish, 19, codes)

    codes = cp.where(hammer_weak & hanging_like_bearish, 20, codes)
    codes = cp.where(hammer_strong & hanging_like_bearish, 21, codes)

    # -----------------------
    # ۵) Inverted Hammer / Shooting Star
    # -----------------------
    inv_core = (body_rel >= 0.10) & (body_rel <= 0.40) & (upper_rel >= 0.5) & (lower_rel <= 0.25)
    inv_strong = inv_core & (upper_rel >= 0.7) & (lower_rel <= 0.15)
    inv_weak = inv_core & ~inv_strong

    inv_bullish = inv_core & (is_bull | ((c >= o) & (lower_rel <= 0.2)))
    inv_bearish = inv_core & (~inv_bullish)

    codes = cp.where(inv_weak & inv_bullish, 22, codes)
    codes = cp.where(inv_strong & inv_bullish, 23, codes)

    codes = cp.where(inv_weak & inv_bearish, 24, codes)
    codes = cp.where(inv_strong & inv_bearish, 25, codes)

    # -----------------------
    # ۶) Marubozu (در انتها، چون خاص‌ترین شکل Long Body است)
    # -----------------------
    bull_mar_strong = is_bull & (body_rel >= 0.90) & (upper_rel <= 0.05) & (lower_rel <= 0.05)
    bull_mar_weak = is_bull & (body_rel >= 0.80) & (body_rel < 0.90) & (upper_rel <= 0.10) & (lower_rel <= 0.10)

    bear_mar_strong = is_bear & (body_rel >= 0.90) & (upper_rel <= 0.05) & (lower_rel <= 0.05)
    bear_mar_weak = is_bear & (body_rel >= 0.80) & (body_rel < 0.90) & (upper_rel <= 0.10) & (lower_rel <= 0.10)

    codes = cp.where(bull_mar_strong, 26, codes)
    codes = cp.where(bull_mar_weak,   27, codes)
    codes = cp.where(bear_mar_strong, 28, codes)
    codes = cp.where(bear_mar_weak,   29, codes)

    return codes



def patterns2_gpu(o, h, l, c):
    """
    الگوهای دوکندلی با کدگذاری پشت سر هم (خروجی: کُد عددی روی کندل دوم هر جفت)

    تعریف جفت:
        برای i >= 1، جفت (i-1, i) بررسی می‌شود و نتیجه روی ایندکس i ثبت می‌شود.
        ایندکس 0 همیشه 0 می‌ماند (چون کندل قبلی ندارد).

    0  = بدون الگو / تشخیص نشده

    ---- Engulfing ----
    1  = Bullish Engulfing ضعیف
    2  = Bullish Engulfing قوی
    3  = Bearish Engulfing ضعیف
    4  = Bearish Engulfing قوی

    ---- Harami / Harami Cross ----
    5  = Bullish Harami ضعیف
    6  = Bullish Harami قوی
    7  = Bearish Harami ضعیف
    8  = Bearish Harami قوی
    9  = Bullish Harami Cross (کندل دوم تقریباً دوجی)
    10 = Bearish Harami Cross

    ---- Piercing Line / Dark Cloud Cover ----
    11 = Piercing Line ضعیف
    12 = Piercing Line قوی
    13 = Dark Cloud Cover ضعیف
    14 = Dark Cloud Cover قوی

    ---- Tweezer Tops / Bottoms ----
    15 = Tweezer Bottom ضعیف
    16 = Tweezer Bottom قوی
    17 = Tweezer Top ضعیف
    18 = Tweezer Top قوی

    ---- Inside / Outside Bars ----
    19 = Bullish Inside Bar
    20 = Bearish Inside Bar
    21 = Inside Bar خنثی
    22 = Bullish Outside Bar
    23 = Bearish Outside Bar
    """

    o = cp.asarray(o, dtype=cp.float32)
    h = cp.asarray(h, dtype=cp.float32)
    l = cp.asarray(l, dtype=cp.float32)
    c = cp.asarray(c, dtype=cp.float32)

    n = c.shape[0]
    codes = cp.zeros(n, dtype=cp.int16)
    if n < 2:
        return codes

    # جفت‌ها: کندل قبلی (1) و کندل جاری (2)
    o1 = o[:-1]
    h1 = h[:-1]
    l1 = l[:-1]
    c1 = c[:-1]

    o2 = o[1:]
    h2 = h[1:]
    l2 = l[1:]
    c2 = c[1:]

    rng1 = h1 - l1
    rng2 = h2 - l2
    eps = cp.float32(1e-8)
    rng1_safe = cp.where(rng1 <= eps, eps, rng1)
    rng2_safe = cp.where(rng2 <= eps, eps, rng2)

    body1 = cp.abs(c1 - o1)
    body2 = cp.abs(c2 - o2)
    body_rel1 = body1 / rng1_safe
    body_rel2 = body2 / rng2_safe

    upper1 = h1 - cp.maximum(o1, c1)
    lower1 = cp.minimum(o1, c1) - l1
    upper2 = h2 - cp.maximum(o2, c2)
    lower2 = cp.minimum(o2, c2) - l2

    upper_rel1 = upper1 / rng1_safe
    lower_rel1 = lower1 / rng1_safe
    upper_rel2 = upper2 / rng2_safe
    lower_rel2 = lower2 / rng2_safe

    is_bull1 = c1 > o1
    is_bear1 = o1 > c1
    is_bull2 = c2 > o2
    is_bear2 = o2 > c2

    mid1 = (o1 + c1) * 0.5
    avg_rng = (rng1_safe + rng2_safe) * 0.5

    doji2 = body_rel2 <= 0.05
    small_body2 = (body_rel2 > 0.05) & (body_rel2 <= 0.30)
    big_body1 = body_rel1 >= 0.40

    # کدها روی آرایه-length-1 و در نهایت روی codes[1:] می‌نشیند
    pair_codes = cp.zeros(n - 1, dtype=cp.int16)

    # ---------------------------------------------------------
    # ۱) Inside / Outside Bars (پایه)
    # ---------------------------------------------------------
    inside = (h2 <= h1) & (l2 >= l1) & (body_rel2 > 0.05)
    outside = (h2 >= h1) & (l2 <= l1) & (body_rel2 > 0.05)

    pair_codes = cp.where(inside & is_bull2, 19, pair_codes)
    pair_codes = cp.where(inside & is_bear2, 20, pair_codes)
    pair_codes = cp.where(inside & (~is_bull2 & ~is_bear2), 21, pair_codes)

    pair_codes = cp.where(outside & is_bull2, 22, pair_codes)
    pair_codes = cp.where(outside & is_bear2, 23, pair_codes)

    # ---------------------------------------------------------
    # ۲) Tweezer Tops / Bottoms
    # ---------------------------------------------------------
    thr = 0.1 * avg_rng  # آستانه نزدیکی high/low

    eq_high = cp.abs(h1 - h2) <= thr
    eq_low = cp.abs(l1 - l2) <= thr

    # Bottom: لوها تقریباً یکسان، تمایل تغییر جهت به بالا
    tweezer_bottom_core = eq_low & (
        (is_bear1 & is_bull2) |  # نزولی → صعودی
        ((c2 > c1) & (c2 > mid1))
    )
    # Top: های تقریباً یکسان، تمایل تغییر جهت به پایین
    tweezer_top_core = eq_high & (
        (is_bull1 & is_bear2) |  # صعودی → نزولی
        ((c2 < c1) & (c2 < mid1))
    )

    strong_tb = tweezer_bottom_core & (body_rel2 >= 0.4)
    weak_tb = tweezer_bottom_core & ~strong_tb

    strong_tt = tweezer_top_core & (body_rel2 >= 0.4)
    weak_tt = tweezer_top_core & ~strong_tt

    pair_codes = cp.where(weak_tb, 15, pair_codes)
    pair_codes = cp.where(strong_tb, 16, pair_codes)
    pair_codes = cp.where(weak_tt, 17, pair_codes)
    pair_codes = cp.where(strong_tt, 18, pair_codes)

    # ---------------------------------------------------------
    # ۳) Engulfing (بلعنده)
    # ---------------------------------------------------------
    real_low1 = cp.minimum(o1, c1)
    real_high1 = cp.maximum(o1, c1)
    real_low2 = cp.minimum(o2, c2)
    real_high2 = cp.maximum(o2, c2)

    engulf_bull_core = (
        is_bear1 & is_bull2 &
        (real_low2 <= real_low1) &
        (real_high2 >= real_high1) &
        (body_rel2 >= 0.4)
    )
    engulf_bull_strong = engulf_bull_core & (body_rel2 >= 0.6) & (body2 > body1)
    engulf_bull_weak = engulf_bull_core & ~engulf_bull_strong

    engulf_bear_core = (
        is_bull1 & is_bear2 &
        (real_low2 <= real_low1) &
        (real_high2 >= real_high1) &
        (body_rel2 >= 0.4)
    )
    engulf_bear_strong = engulf_bear_core & (body_rel2 >= 0.6) & (body2 > body1)
    engulf_bear_weak = engulf_bear_core & ~engulf_bear_strong

    pair_codes = cp.where(engulf_bull_weak, 1, pair_codes)
    pair_codes = cp.where(engulf_bull_strong, 2, pair_codes)
    pair_codes = cp.where(engulf_bear_weak, 3, pair_codes)
    pair_codes = cp.where(engulf_bear_strong, 4, pair_codes)

    # ---------------------------------------------------------
    # ۴) Harami / Harami Cross
    # ---------------------------------------------------------
    harami_core = (
        big_body1 &
        (body_rel2 <= 0.30) &
        (h2 <= h1) &
        (l2 >= l1)
    )

    bull_harami = harami_core & is_bear1 & is_bull2
    bull_harami_strong = bull_harami & (body_rel2 <= 0.15)
    bull_harami_weak = bull_harami & ~bull_harami_strong

    bear_harami = harami_core & is_bull1 & is_bear2
    bear_harami_strong = bear_harami & (body_rel2 <= 0.15)
    bear_harami_weak = bear_harami & ~bear_harami_strong

    pair_codes = cp.where(bull_harami_weak, 5, pair_codes)
    pair_codes = cp.where(bull_harami_strong, 6, pair_codes)
    pair_codes = cp.where(bear_harami_weak, 7, pair_codes)
    pair_codes = cp.where(bear_harami_strong, 8, pair_codes)

    bull_harami_cross = harami_core & doji2 & is_bear1
    bear_harami_cross = harami_core & doji2 & is_bull1

    pair_codes = cp.where(bull_harami_cross, 9, pair_codes)
    pair_codes = cp.where(bear_harami_cross, 10, pair_codes)

    # ---------------------------------------------------------
    # ۵) Piercing Line / Dark Cloud Cover
    # ---------------------------------------------------------
    piercing_core = (
        is_bear1 & is_bull2 &
        (o2 < l1) &           # گپ رو به پایین
        (c2 > mid1) & (c2 < o1)
    )
    piercing_strong = piercing_core & (c2 > (mid1 + o1) * 0.5)
    piercing_weak = piercing_core & ~piercing_strong

    pair_codes = cp.where(piercing_weak, 11, pair_codes)
    pair_codes = cp.where(piercing_strong, 12, pair_codes)

    dark_core = (
        is_bull1 & is_bear2 &
        (o2 > h1) &           # گپ رو به بالا
        (c2 < mid1) & (c2 > o1)
    )
    dark_strong = dark_core & (c2 < (mid1 + o1) * 0.5)
    dark_weak = dark_core & ~dark_strong

    pair_codes = cp.where(dark_weak, 13, pair_codes)
    pair_codes = cp.where(dark_strong, 14, pair_codes)

    # ---------------------------------------------------------
    # ۶) ریختن روی آرایه‌ی اصلی
    # ---------------------------------------------------------
    codes[1:] = pair_codes
    return codes



def patterns3_gpu(o, h, l, c):
    """
    الگوهای سه‌کندلی (خروجی: کُد عددی روی کندل سوم هر سه‌تایی)

    برای i >= 2، روی سه‌تایی (i-2, i-1, i) کار می‌کنیم و نتیجه را روی ایندکس i می‌نویسیم.
    ایندکس‌های 0 و 1 همیشه 0 می‌مانند.

    0  = بدون الگو / تشخیص نشده

    ---- Morning / Evening Star ----
    1  = Morning Star ضعیف
    2  = Morning Star قوی
    3  = Evening Star ضعیف
    4  = Evening Star قوی
    5  = Morning Doji Star
    6  = Evening Doji Star

    ---- Three White Soldiers / Three Black Crows ----
    7  = Three White Soldiers معمولی
    8  = Three White Soldiers قوی
    9  = Three Black Crows معمولی
    10 = Three Black Crows قوی

    ---- Three Inside / Outside Up / Down ----
    11 = Three Inside Up
    12 = Three Inside Down
    13 = Three Outside Up
    14 = Three Outside Down
    """

    o = cp.asarray(o, dtype=cp.float32)
    h = cp.asarray(h, dtype=cp.float32)
    l = cp.asarray(l, dtype=cp.float32)
    c = cp.asarray(c, dtype=cp.float32)

    n = c.shape[0]
    codes = cp.zeros(n, dtype=cp.int16)
    if n < 3:
        return codes

    # سه‌تایی‌ها: کندل 1، 2، 3
    o1 = o[:-2]
    h1 = h[:-2]
    l1 = l[:-2]
    c1 = c[:-2]

    o2 = o[1:-1]
    h2 = h[1:-1]
    l2 = l[1:-1]
    c2 = c[1:-1]

    o3 = o[2:]
    h3 = h[2:]
    l3 = l[2:]
    c3 = c[2:]

    rng1 = h1 - l1
    rng2 = h2 - l2
    rng3 = h3 - l3

    eps = cp.float32(1e-8)
    rng1_safe = cp.where(rng1 <= eps, eps, rng1)
    rng2_safe = cp.where(rng2 <= eps, eps, rng2)
    rng3_safe = cp.where(rng3 <= eps, eps, rng3)

    body1 = cp.abs(c1 - o1)
    body2 = cp.abs(c2 - o2)
    body3 = cp.abs(c3 - o3)

    body_rel1 = body1 / rng1_safe
    body_rel2 = body2 / rng2_safe
    body_rel3 = body3 / rng3_safe

    is_bull1 = c1 > o1
    is_bear1 = o1 > c1
    is_bull2 = c2 > o2
    is_bear2 = o2 > c2
    is_bull3 = c3 > o3
    is_bear3 = o3 > c3

    # تعریف بدنه بزرگ/کوچک
    big_body1 = body_rel1 >= 0.6
    big_body3 = body_rel3 >= 0.6
    small_body2 = body_rel2 <= 0.3

    mid1 = (o1 + c1) * 0.5

    # کندل دوم دوجی/خیلی ریز
    doji2 = body_rel2 <= 0.05

    # کدها برای سه‌تایی‌ها (روی کندل سوم)
    triple_codes = cp.zeros(n - 2, dtype=cp.int16)

    # ---------------------------------------------------------
    # ۱) Morning / Evening Star و Doji-Star
    # ---------------------------------------------------------
    # Morning Star: نزولی قوی → کندل کوچک → صعودی که حداقل تا نصف بدنه اول برگردد
    morning_core = (
        is_bear1 & big_body1 &
        small_body2 &
        is_bull3 & big_body3 &
        (c3 > mid1)
    )
    morning_strong = morning_core & (c3 > (mid1 + cp.maximum(o1, c1)) * 0.5)
    morning_weak = morning_core & ~morning_strong

    # Evening Star: صعودی قوی → کندل کوچک → نزولی که حداقل تا نصف بدنه اول برگردد
    evening_core = (
        is_bull1 & big_body1 &
        small_body2 &
        is_bear3 & big_body3 &
        (c3 < mid1)
    )
    evening_strong = evening_core & (c3 < (mid1 + cp.minimum(o1, c1)) * 0.5)
    evening_weak = evening_core & ~evening_strong

    triple_codes = cp.where(morning_weak, 1, triple_codes)
    triple_codes = cp.where(morning_strong, 2, triple_codes)
    triple_codes = cp.where(evening_weak, 3, triple_codes)
    triple_codes = cp.where(evening_strong, 4, triple_codes)

    # Morning Doji Star
    morning_doji = (
        is_bear1 & big_body1 &
        doji2 &
        is_bull3 & big_body3 &
        (c3 > mid1)
    )
    triple_codes = cp.where(morning_doji, 5, triple_codes)

    # Evening Doji Star
    evening_doji = (
        is_bull1 & big_body1 &
        doji2 &
        is_bear3 & big_body3 &
        (c3 < mid1)
    )
    triple_codes = cp.where(evening_doji, 6, triple_codes)

    # ---------------------------------------------------------
    # ۲) Three White Soldiers / Three Black Crows
    # ---------------------------------------------------------
    big_bull1 = is_bull1 & (body_rel1 >= 0.5)
    big_bull2 = is_bull2 & (body_rel2 >= 0.5)
    big_bull3 = is_bull3 & (body_rel3 >= 0.5)

    big_bear1 = is_bear1 & (body_rel1 >= 0.5)
    big_bear2 = is_bear2 & (body_rel2 >= 0.5)
    big_bear3 = is_bear3 & (body_rel3 >= 0.5)

    real_low1 = cp.minimum(o1, c1)
    real_high1 = cp.maximum(o1, c1)
    real_low2 = cp.minimum(o2, c2)
    real_high2 = cp.maximum(o2, c2)
    real_low3 = cp.minimum(o3, c3)
    real_high3 = cp.maximum(o3, c3)

    # Three White Soldiers
    three_white_core = (
        big_bull1 & big_bull2 & big_bull3 &
        (o2 >= real_low1) & (o2 <= real_high1) & (c2 > c1) &
        (o3 >= real_low2) & (o3 <= real_high2) & (c3 > c2)
    )
    three_white_strong = three_white_core & (body_rel2 >= 0.7) & (body_rel3 >= 0.7)
    three_white_weak = three_white_core & ~three_white_strong

    triple_codes = cp.where(three_white_weak, 7, triple_codes)
    triple_codes = cp.where(three_white_strong, 8, triple_codes)

    # Three Black Crows
    three_black_core = (
        big_bear1 & big_bear2 & big_bear3 &
        (o2 <= real_high1) & (o2 >= real_low1) & (c2 < c1) &
        (o3 <= real_high2) & (o3 >= real_low2) & (c3 < c2)
    )
    three_black_strong = three_black_core & (body_rel2 >= 0.7) & (body_rel3 >= 0.7)
    three_black_weak = three_black_core & ~three_black_strong

    triple_codes = cp.where(three_black_weak, 9, triple_codes)
    triple_codes = cp.where(three_black_strong, 10, triple_codes)

    # ---------------------------------------------------------
    # ۳) Three Inside / Outside Up / Down
    # ---------------------------------------------------------
    # Harami بین کندل 1 و 2
    harami_core = (
        (body_rel1 >= 0.4) &
        (body_rel2 <= 0.3) &
        (h2 <= h1) &
        (l2 >= l1)
    )
    bull_harami = harami_core & is_bear1 & is_bull2
    bear_harami = harami_core & is_bull1 & is_bear2

    # Engulfing بین کندل 1 و 2
    engulf_core = (
        (body_rel1 >= 0.4) &
        (body_rel2 >= 0.4)
    )
    engulf_bull = engulf_core & is_bear1 & is_bull2 & (real_low2 <= real_low1) & (real_high2 >= real_high1)
    engulf_bear = engulf_core & is_bull1 & is_bear2 & (real_low2 <= real_low1) & (real_high2 >= real_high1)

    # Three Inside Up: Bullish Harami + کندل سوم صعودی بالاتر از کندل دوم
    three_inside_up = bull_harami & is_bull3 & (c3 > c2)
    triple_codes = cp.where(three_inside_up, 11, triple_codes)

    # Three Inside Down: Bearish Harami + کندل سوم نزولی پایین‌تر از کندل دوم
    three_inside_down = bear_harami & is_bear3 & (c3 < c2)
    triple_codes = cp.where(three_inside_down, 12, triple_codes)

    # Three Outside Up: Bullish Engulfing + کندل سوم صعودی ادامه‌دهنده
    three_outside_up = engulf_bull & is_bull3 & (c3 > c2)
    triple_codes = cp.where(three_outside_up, 13, triple_codes)

    # Three Outside Down: Bearish Engulfing + کندل سوم نزولی ادامه‌دهنده
    three_outside_down = engulf_bear & is_bear3 & (c3 < c2)
    triple_codes = cp.where(three_outside_down, 14, triple_codes)

    # ---------------------------------------------------------
    # ۴) اعمال روی آرایه اصلی
    # ---------------------------------------------------------
    codes[2:] = triple_codes
    return codes


def patterns4_gpu(o, h, l, c):
    """
    الگوهای ۴‌کندلی (خروجی: کُد عددی روی کندل چهارم هر چهار‌تایی)

    برای i >= 3، روی چهار‌تایی (i-3, i-2, i-1, i) کار می‌کنیم
    و نتیجه روی ایندکس i نوشته می‌شود.
    ایندکس‌های 0, 1, 2 همیشه 0 می‌مانند.

    0  = بدون الگو / تشخیص نشده

    ---- روندهای خالص چهارکندلی ----
    1  = 4 Bull Small Run   (۴ کندل صعودی، میانگین بدنه کوچک)
    2  = 4 Bull Medium Run
    3  = 4 Bull Long Run
    4  = 4 Bear Small Run
    5  = 4 Bear Medium Run
    6  = 4 Bear Long Run

    ---- 3+1 اصلاحی ----
    7  = 3 Bulls + 1 Bear Correction (net صعودی)
    8  = 3 Bears + 1 Bull Correction (net نزولی)

    ---- خوشه نوسان ----
    9  = Volatility Spike (میانگین رنج پنجره ≥ 1.5 * مدین)
    10 = Volatility Squeeze (میانگین رنج پنجره ≤ 0.5 * مدین)

    ---- الگوی رفت و برگشتی ----
    11 = Bullish Alternating (Bull, Bear, Bull, Bear) و net صعودی
    12 = Bearish Alternating (Bear, Bull, Bear, Bull) و net نزولی

    ---- الگوهای چند گپی ----
    13 = Multi Gap Up Sequence  (حداقل دو گپ رو به بالا + net صعودی)
    14 = Multi Gap Down Sequence (حداقل دو گپ رو به پایین + net نزولی)
    """

    o = cp.asarray(o, dtype=cp.float32)
    h = cp.asarray(h, dtype=cp.float32)
    l = cp.asarray(l, dtype=cp.float32)
    c = cp.asarray(c, dtype=cp.float32)

    n = c.shape[0]
    codes = cp.zeros(n, dtype=cp.int16)
    if n < 4:
        return codes

    # چهار‌تایی‌ها: کندل 1، 2، 3، 4
    o1 = o[:-3]
    h1 = h[:-3]
    l1 = l[:-3]
    c1 = c[:-3]

    o2 = o[1:-2]
    h2 = h[1:-2]
    l2 = l[1:-2]
    c2 = c[1:-2]

    o3 = o[2:-1]
    h3 = h[2:-1]
    l3 = l[2:-1]
    c3 = c[2:-1]

    o4 = o[3:]
    h4 = h[3:]
    l4 = l[3:]
    c4 = c[3:]

    rng1 = h1 - l1
    rng2 = h2 - l2
    rng3 = h3 - l3
    rng4 = h4 - l4

    eps = cp.float32(1e-8)
    rng1_safe = cp.where(rng1 <= eps, eps, rng1)
    rng2_safe = cp.where(rng2 <= eps, eps, rng2)
    rng3_safe = cp.where(rng3 <= eps, eps, rng3)
    rng4_safe = cp.where(rng4 <= eps, eps, rng4)

    body1 = cp.abs(c1 - o1)
    body2 = cp.abs(c2 - o2)
    body3 = cp.abs(c3 - o3)
    body4 = cp.abs(c4 - o4)

    body_rel1 = body1 / rng1_safe
    body_rel2 = body2 / rng2_safe
    body_rel3 = body3 / rng3_safe
    body_rel4 = body4 / rng4_safe

    is_bull1 = c1 > o1
    is_bear1 = o1 > c1
    is_bull2 = c2 > o2
    is_bear2 = o2 > c2
    is_bull3 = c3 > o3
    is_bear3 = o3 > c3
    is_bull4 = c4 > o4
    is_bear4 = o4 > c4

    # برای تشخیص نوسان، مدین رنج کل سری
    rng_all = h - l
    rng_med = cp.median(cp.where(rng_all <= eps, eps, rng_all))

    avg_body_rel = (body_rel1 + body_rel2 + body_rel3 + body_rel4) * 0.25
    avg_rng = (rng1_safe + rng2_safe + rng3_safe + rng4_safe) * 0.25

    # شمارش جهت‌ها در پنجره
    bull_count = (is_bull1.astype(cp.int32) +
                  is_bull2.astype(cp.int32) +
                  is_bull3.astype(cp.int32) +
                  is_bull4.astype(cp.int32))
    bear_count = (is_bear1.astype(cp.int32) +
                  is_bear2.astype(cp.int32) +
                  is_bear3.astype(cp.int32) +
                  is_bear4.astype(cp.int32))

    real_low1 = cp.minimum(o1, c1)
    real_high1 = cp.maximum(o1, c1)

    # کدها برای چهار‌تایی‌ها (روی کندل چهارم)
    quad_codes = cp.zeros(n - 3, dtype=cp.int16)

    # ---------------------------------------------------------
    # ۱) روندهای خالص چهارکندلی
    # ---------------------------------------------------------
    pure_bull = is_bull1 & is_bull2 & is_bull3 & is_bull4
    pure_bear = is_bear1 & is_bear2 & is_bear3 & is_bear4

    small_run = avg_body_rel <= 0.30
    med_run = (avg_body_rel > 0.30) & (avg_body_rel <= 0.60)
    long_run = avg_body_rel > 0.60

    quad_codes = cp.where(pure_bull & small_run, 1, quad_codes)
    quad_codes = cp.where(pure_bull & med_run,   2, quad_codes)
    quad_codes = cp.where(pure_bull & long_run,  3, quad_codes)

    quad_codes = cp.where(pure_bear & small_run, 4, quad_codes)
    quad_codes = cp.where(pure_bear & med_run,   5, quad_codes)
    quad_codes = cp.where(pure_bear & long_run,  6, quad_codes)

    # ---------------------------------------------------------
    # ۲) 3+1 اصلاحی
    # ---------------------------------------------------------
    # سه کندل در یک جهت، یک کندل خلاف، و جهت خالص مشخص
    three_bull_one_bear = (bull_count >= 3) & (bear_count == 1)
    three_bear_one_bull = (bear_count >= 3) & (bull_count == 1)

    # جهت خالص (از open کندل اول تا close کندل چهارم)
    net_bull = c4 > o1
    net_bear = c4 < o1

    # فقط جاهایی که هنوز کدی نگرفته‌اند را override می‌کنیم؟
    # اینجا اجازه می‌دهیم 3+1 روی pure_run override کند (چون شکل خاص‌تری است)
    bear_mask_4 = is_bear1 | is_bear2 | is_bear3 | is_bear4
    bull_mask_4 = is_bull1 | is_bull2 | is_bull3 | is_bull4

    quad_codes = cp.where(three_bull_one_bear & net_bull, 7, quad_codes)
    quad_codes = cp.where(three_bear_one_bull & net_bear, 8, quad_codes)

    # ---------------------------------------------------------
    # ۳) خوشه نوسان (Spike / Squeeze) روی جاهایی که هنوز 0 هستند
    # ---------------------------------------------------------
    no_code = quad_codes == 0

    high_vol = avg_rng >= 1.5 * rng_med
    low_vol = avg_rng <= 0.5 * rng_med

    quad_codes = cp.where(no_code & high_vol, 9, quad_codes)
    quad_codes = cp.where(no_code & low_vol,  10, quad_codes)

    # ---------------------------------------------------------
    # ۴) الگوی رفت‌وبرگشتی (Alternating)
    # ---------------------------------------------------------
    alt_bull = (
        is_bull1 & is_bear2 & is_bull3 & is_bear4 &
        (c4 > o1)
    )
    alt_bear = (
        is_bear1 & is_bull2 & is_bear3 & is_bull4 &
        (c4 < o1)
    )

    quad_codes = cp.where(alt_bull, 11, quad_codes)
    quad_codes = cp.where(alt_bear, 12, quad_codes)

    # ---------------------------------------------------------
    # ۵) Multi-gap sequence (در انتها، خاص‌ترین حالت)
    # ---------------------------------------------------------
    # گپ‌ها روی سه فاصله:
    gap_up_12 = l2 > h1
    gap_down_12 = h2 < l1

    gap_up_23 = l3 > h2
    gap_down_23 = h3 < l2

    gap_up_34 = l4 > h3
    gap_down_34 = h4 < l3

    gap_up_count = (
        gap_up_12.astype(cp.int32) +
        gap_up_23.astype(cp.int32) +
        gap_up_34.astype(cp.int32)
    )
    gap_down_count = (
        gap_down_12.astype(cp.int32) +
        gap_down_23.astype(cp.int32) +
        gap_down_34.astype(cp.int32)
    )

    multi_gap_up = (gap_up_count >= 2) & net_bull
    multi_gap_down = (gap_down_count >= 2) & net_bear

    quad_codes = cp.where(multi_gap_up, 13, quad_codes)
    quad_codes = cp.where(multi_gap_down, 14, quad_codes)

    # ---------------------------------------------------------
    # ۶) اعمال روی آرایه اصلی
    # ---------------------------------------------------------
    codes[3:] = quad_codes
    return codes


def patterns5_gpu(o, h, l, c):
    """
    الگوهای ۵‌کندلی (خروجی: کُد عددی روی کندل پنجم هر پنجره)

    برای i >= 4، روی پنجره (i-4, i-3, i-2, i-1, i) کار می‌کنیم
    و نتیجه روی ایندکس i نوشته می‌شود.
    ایندکس‌های 0,1,2,3 همیشه 0 می‌مانند.

    0  = بدون الگو / تشخیص نشده

    ---- روندهای خالص پنج‌کندلی ----
    1  = 5 Bull Small Run   (۵ کندل صعودی، میانگین بدنه کوچک)
    2  = 5 Bull Medium Run
    3  = 5 Bull Long Run
    4  = 5 Bear Small Run
    5  = 5 Bear Medium Run
    6  = 5 Bear Long Run

    ---- الگوهای 3+2 و 2+3 ----
    7  = 3 Bulls + 2 Bears با جهت نهایی نزولی
    8  = 3 Bears + 2 Bulls با جهت نهایی صعودی
    9  = 2 Bulls + 3 Bears با جهت نهایی نزولی
    10 = 2 Bears + 3 Bulls با جهت نهایی صعودی

    ---- خوشه نوسان / فشردگی ----
    11 = Volatility Spike (میانگین رنج پنجره ≥ 1.8 * مدین)
    12 = Volatility Squeeze (میانگین رنج پنجره ≤ 0.4 * مدین)

    ---- Squeeze + Breakout ----
    13 = Squeeze + Bull Break
    14 = Squeeze + Bear Break

    ---- الگوی رفت و برگشتی ----
    15 = Bullish Alternating (Bull, Bear, Bull, Bear, Bull) + جهت نهایی صعودی
    16 = Bearish Alternating (Bear, Bull, Bear, Bull, Bear) + جهت نهایی نزولی

    ---- رِنج و ترند پله‌ای ----
    17 = Sideways Range (رنج کوچک و close آخر نزدیک وسط)
    18 = Step-up Bull Trend (۵ کندل صعودی، closes صعودی، بدنه‌ها متوسط/بزرگ)
    19 = Step-down Bear Trend (۵ کندل نزولی، closes نزولی، بدنه‌ها متوسط/بزرگ)
    """

    o = cp.asarray(o, dtype=cp.float32)
    h = cp.asarray(h, dtype=cp.float32)
    l = cp.asarray(l, dtype=cp.float32)
    c = cp.asarray(c, dtype=cp.float32)

    n = c.shape[0]
    codes = cp.zeros(n, dtype=cp.int16)
    if n < 5:
        return codes

    # پنجره‌های ۵‌تایی: کندل 1..5
    o1 = o[:-4]
    h1 = h[:-4]
    l1 = l[:-4]
    c1 = c[:-4]

    o2 = o[1:-3]
    h2 = h[1:-3]
    l2 = l[1:-3]
    c2 = c[1:-3]

    o3 = o[2:-2]
    h3 = h[2:-2]
    l3 = l[2:-2]
    c3 = c[2:-2]

    o4 = o[3:-1]
    h4 = h[3:-1]
    l4 = l[3:-1]
    c4 = c[3:-1]

    o5 = o[4:]
    h5 = h[4:]
    l5 = l[4:]
    c5 = c[4:]

    rng1 = h1 - l1
    rng2 = h2 - l2
    rng3 = h3 - l3
    rng4 = h4 - l4
    rng5 = h5 - l5

    eps = cp.float32(1e-8)
    rng1_safe = cp.where(rng1 <= eps, eps, rng1)
    rng2_safe = cp.where(rng2 <= eps, eps, rng2)
    rng3_safe = cp.where(rng3 <= eps, eps, rng3)
    rng4_safe = cp.where(rng4 <= eps, eps, rng4)
    rng5_safe = cp.where(rng5 <= eps, eps, rng5)

    body1 = cp.abs(c1 - o1)
    body2 = cp.abs(c2 - o2)
    body3 = cp.abs(c3 - o3)
    body4 = cp.abs(c4 - o4)
    body5 = cp.abs(c5 - o5)

    body_rel1 = body1 / rng1_safe
    body_rel2 = body2 / rng2_safe
    body_rel3 = body3 / rng3_safe
    body_rel4 = body4 / rng4_safe
    body_rel5 = body5 / rng5_safe

    is_bull1 = c1 > o1
    is_bear1 = o1 > c1
    is_bull2 = c2 > o2
    is_bear2 = o2 > c2
    is_bull3 = c3 > o3
    is_bear3 = o3 > c3
    is_bull4 = c4 > o4
    is_bear4 = o4 > c4
    is_bull5 = c5 > o5
    is_bear5 = o5 > c5

    # رنج کل سری برای مدین
    rng_all = h - l
    rng_med = cp.median(cp.where(rng_all <= eps, eps, rng_all))

    avg_body_rel = (body_rel1 + body_rel2 + body_rel3 + body_rel4 + body_rel5) / 5.0
    avg_rng = (rng1_safe + rng2_safe + rng3_safe + rng4_safe + rng5_safe) / 5.0

    # جهت نهایی (از open کندل اول تا close کندل پنجم)
    net_bull = c5 > o1
    net_bear = c5 < o1

    # تعداد کندل‌های صعودی/نزولی
    bull_count = (
        is_bull1.astype(cp.int32)
        + is_bull2.astype(cp.int32)
        + is_bull3.astype(cp.int32)
        + is_bull4.astype(cp.int32)
        + is_bull5.astype(cp.int32)
    )
    bear_count = (
        is_bear1.astype(cp.int32)
        + is_bear2.astype(cp.int32)
        + is_bear3.astype(cp.int32)
        + is_bear4.astype(cp.int32)
        + is_bear5.astype(cp.int32)
    )

    # برای رنج کلی پنجره (Sideways)
    highs_stack = cp.stack([h1, h2, h3, h4, h5], axis=1)
    lows_stack = cp.stack([l1, l2, l3, l4, l5], axis=1)
    window_high = highs_stack.max(axis=1)
    window_low = lows_stack.min(axis=1)
    window_range = window_high - window_low

    # کدها برای پنجره‌ها (روی کندل پنجم)
    win_codes = cp.zeros(n - 4, dtype=cp.int16)

    # ---------------------------------------------------------
    # ۱) روندهای خالص پنج‌کندلی
    # ---------------------------------------------------------
    pure_bull = is_bull1 & is_bull2 & is_bull3 & is_bull4 & is_bull5
    pure_bear = is_bear1 & is_bear2 & is_bear3 & is_bear4 & is_bear5

    small_run = avg_body_rel <= 0.30
    med_run = (avg_body_rel > 0.30) & (avg_body_rel <= 0.60)
    long_run = avg_body_rel > 0.60

    win_codes = cp.where(pure_bull & small_run, 1, win_codes)
    win_codes = cp.where(pure_bull & med_run,   2, win_codes)
    win_codes = cp.where(pure_bull & long_run,  3, win_codes)

    win_codes = cp.where(pure_bear & small_run, 4, win_codes)
    win_codes = cp.where(pure_bear & med_run,   5, win_codes)
    win_codes = cp.where(pure_bear & long_run,  6, win_codes)

    # ---------------------------------------------------------
    # ۲) الگوهای 3+2 و 2+3
    # ---------------------------------------------------------
    # 3 Bulls + 2 Bears (با جهت نهایی نزولی)
    pattern_3b_2s = (
        is_bull1 & is_bull2 & is_bull3 & is_bear4 & is_bear5 & net_bear
    )
    # 3 Bears + 2 Bulls (با جهت نهایی صعودی)
    pattern_3s_2b = (
        is_bear1 & is_bear2 & is_bear3 & is_bull4 & is_bull5 & net_bull
    )
    # 2 Bulls + 3 Bears
    pattern_2b_3s = (
        is_bull1 & is_bull2 & is_bear3 & is_bear4 & is_bear5 & net_bear
    )
    # 2 Bears + 3 Bulls
    pattern_2s_3b = (
        is_bear1 & is_bear2 & is_bull3 & is_bull4 & is_bull5 & net_bull
    )

    win_codes = cp.where(pattern_3b_2s, 7, win_codes)
    win_codes = cp.where(pattern_3s_2b, 8, win_codes)
    win_codes = cp.where(pattern_2b_3s, 9, win_codes)
    win_codes = cp.where(pattern_2s_3b, 10, win_codes)

    # ---------------------------------------------------------
    # ۳) خوشه نوسان / فشردگی روی جاهایی که هنوز 0 هستند
    # ---------------------------------------------------------
    no_code = win_codes == 0

    high_vol = avg_rng >= 1.8 * rng_med
    low_vol = avg_rng <= 0.4 * rng_med

    win_codes = cp.where(no_code & high_vol, 11, win_codes)
    win_codes = cp.where(no_code & low_vol,  12, win_codes)

    # ---------------------------------------------------------
    # ۴) Squeeze + Breakout (روی low_vol، override روی 12)
    # ---------------------------------------------------------
    low_vol_window = avg_rng <= 0.5 * rng_med

    bull_break = low_vol_window & net_bull & (body_rel5 >= 0.6) & is_bull5
    bear_break = low_vol_window & net_bear & (body_rel5 >= 0.6) & is_bear5

    win_codes = cp.where(bull_break, 13, win_codes)
    win_codes = cp.where(bear_break, 14, win_codes)

    # ---------------------------------------------------------
    # ۵) Alternating (Bull/Bear/Bull/Bear/Bull و بالعکس)
    # ---------------------------------------------------------
    alt_bull = (
        is_bull1 & is_bear2 & is_bull3 & is_bear4 & is_bull5 & net_bull
    )
    alt_bear = (
        is_bear1 & is_bull2 & is_bear3 & is_bull4 & is_bear5 & net_bear
    )

    win_codes = cp.where(alt_bull, 15, win_codes)
    win_codes = cp.where(alt_bear, 16, win_codes)

    # ---------------------------------------------------------
    # ۶) Sideways Range (فقط روی جاهایی که هنوز 0 هستند)
    # ---------------------------------------------------------
    no_code = win_codes == 0

    # رنج کل کوچک و close آخر نزدیک وسط
    small_range = window_range <= 0.7 * rng_med
    mid_price = (window_high + window_low) * 0.5
    close_near_mid = cp.abs(c5 - mid_price) <= 0.25 * window_range

    sideways = small_range & close_near_mid
    win_codes = cp.where(no_code & sideways, 17, win_codes)

    # ---------------------------------------------------------
    # ۷) Step-up / Step-down Trend (خاص‌ترین، override)
    # ---------------------------------------------------------
    # Step-up Bull Trend: همه صعودی، هر close بالاتر از قبلی، بدنه‌ها متوسط/بزرگ، net_bull
    step_bull = (
        is_bull1 & is_bull2 & is_bull3 & is_bull4 & is_bull5 &
        (c2 > c1) & (c3 > c2) & (c4 > c3) & (c5 > c4) &
        (avg_body_rel >= 0.4) &
        net_bull
    )

    # Step-down Bear Trend: همه نزولی، هر close پایین‌تر از قبلی، بدنه‌ها متوسط/بزرگ، net_bear
    step_bear = (
        is_bear1 & is_bear2 & is_bear3 & is_bear4 & is_bear5 &
        (c2 < c1) & (c3 < c2) & (c4 < c3) & (c5 < c4) &
        (avg_body_rel >= 0.4) &
        net_bear
    )

    win_codes = cp.where(step_bull, 18, win_codes)
    win_codes = cp.where(step_bear, 19, win_codes)

    # ---------------------------------------------------------
    # ۸) اعمال روی آرایه اصلی
    # ---------------------------------------------------------
    codes[4:] = win_codes
    return codes



def build_paths_from_prefix(prefix: str) -> dict:
    """
    بر اساس نام دیتاست (بدون پسوند) همه‌ی مسیرهای استاندارد را می‌سازد.
    مثال:
        prefix = 'XAUUSD_5_Minutes_historical_data_type_m'
    خروجی:
        {
            "csv":        ./XAUUSD_5_Minutes_historical_data_type_m.csv
            "parquet":    ./XAUUSD_5_Minutes_historical_data_type_m.parquet
            "features":   ./XAUUSD_5_Minutes_historical_data_type_m_feat.parquet
            "labels":     ./XAUUSD_5_Minutes_historical_data_type_m_labels.parquet
            "train":      ./XAUUSD_5_Minutes_historical_data_type_m_train.parquet
            "test":       ./XAUUSD_5_Minutes_historical_data_type_m_test.parquet
            "models_dir": ./models/XAUUSD_5_Minutes_historical_data_type_m
        }
    """
    base = Path(prefix).stem
    return {
        "csv":        DATA_ROOT / f"{base}.csv",
        "parquet":    DATA_ROOT / f"{base}.parquet",
        "features":   DATA_ROOT / f"{base}_feat.parquet",
        "labels":     DATA_ROOT / f"{base}_labels.parquet",
        "train":      DATA_ROOT / f"{base}_train.parquet",
        "test":       DATA_ROOT / f"{base}_test.parquet",
        "models_dir": DATA_ROOT / "models" / base,
    }


def save_active_prefix(prefix: str) -> None:
    """
    نام دیتاست فعلی را روی دیسک ذخیره می‌کند تا دفعه‌ی بعد
    بدون نیاز به آپلود CSV دوباره، همان دیتاست لود شود.
    """
    try:
        LAST_DATASET_FILE.write_text(
            json.dumps({"prefix": prefix}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        # اگر مجوز نوشتن نباشد، نگران نباشیم؛ فقط پایداری از بین می‌رود
        pass


def load_active_prefix() -> str | None:
    """
    اگر قبلاً دیتاستی انتخاب شده، نامش را از last_dataset.json می‌خواند.
    اگر وجود نداشت، None برمی‌گرداند.
    """
    try:
        if LAST_DATASET_FILE.exists():
            data = json.loads(LAST_DATASET_FILE.read_text(encoding="utf-8"))
            return data.get("prefix")
    except Exception:
        return None
    return None


def set_project_paths(prefix: str | None = None) -> dict:
    """
    قبلاً احتمالاً در کدت set_project_paths داشتی؛
    این نسخه‌ی جدیدش است که با prefix کار می‌کند.
    اگر prefix None بود، از session_state یا last_dataset.json کمک می‌گیرد.
    """
    ss = st.session_state

    # اگر prefix صراحتاً داده شده
    if prefix is not None:
        ss["dataset_prefix"] = prefix
        save_active_prefix(prefix)
    else:
        # اگر در همین جلسه prefix در session است
        if "dataset_prefix" in ss:
            prefix = ss["dataset_prefix"]
        else:
            # سعی کن از فایل last_dataset.json بخوانی
            prefix = load_active_prefix()
            if prefix is not None:
                ss["dataset_prefix"] = prefix

    # اگر هنوز prefix نداریم، paths خالی برگردان
    if not prefix:
        return {}

    paths = build_paths_from_prefix(prefix)

    # اطمینان از اینکه پوشه‌ی مدل‌ها وجود دارد
    models_dir = paths["models_dir"]
    try:
        models_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    return {k: str(v) for k, v in paths.items()}


def ensure_paths_and_autoload() -> dict:
    """
    این تابع:
      ۱) اگر در session_state نام دیتاست هست، از همان استفاده می‌کند.
      ۲) اگر نیست، سعی می‌کند از last_dataset.json بخواند.
      ۳) اگر آن هم نبود، جدیدترین فایل parquet موجود را حدس می‌زند.
      ۴) سپس همه‌ی مسیرها را در st.session_state ست می‌کند:
         parquet_path, features_path, labels_path, train_path, test_path, models_dir
    """
    ss = st.session_state

    # مرحله‌ی ۱ و ۲: prefix را پیدا کن
    prefix = ss.get("dataset_prefix", None)
    if not prefix:
        prefix = load_active_prefix()
        if prefix:
            ss["dataset_prefix"] = prefix

    # مرحله‌ی ۳: اگر هنوز prefix نداریم، جدیدترین parquet را حدس بزن
    if not prefix:
        parqs = sorted(
            DATA_ROOT.glob("*.parquet"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if parqs:
            prefix = parqs[0].stem
            ss["dataset_prefix"] = prefix
            save_active_prefix(prefix)

    paths: dict[str, str] = {}
    if prefix:
        paths = set_project_paths(prefix)

        # این مسیرها را به صورت کلیدهای استاندارد در session_state ذخیره کن
        # تا تب‌های مختلف از آن استفاده کنند.
        ss.setdefault("parquet_path", paths.get("parquet", ""))
        ss.setdefault("features_path", paths.get("features", ""))
        ss.setdefault("labels_path",  paths.get("labels", ""))
        ss.setdefault("train_path",   paths.get("train", ""))
        ss.setdefault("test_path",    paths.get("test", ""))
        ss.setdefault("models_dir",   paths.get("models_dir", ""))

        # برای سازگاری با کد قبلی‌ات اگر قبلاً st.session_state["paths"] استفاده می‌شد:
        ss["paths"] = paths

    return paths


def detect_base_freq_minutes(time_series: pd.Series) -> float:
    """
    تخمین بازه‌ی زمانی پایه (به دقیقه) از ستون time.
    از میانه‌ی اختلاف زمانی (median) استفاده می‌کند تا به نویز حساس نباشد.
    """
    t = pd.to_datetime(time_series, utc=True, errors="coerce")
    t = t.sort_values()
    dt = t.diff().dropna()
    if len(dt) == 0:
        return 1.0
    # تبدیل به دقیقه
    return float(dt.median().total_seconds() / 60.0)



# ===================== تنظیمات برچسب‌ها (SL/TP) =====================

def save_label_config(
    parquet_path: str,
    pip_value: float,
    spread: float,
    m1_tp_pips: float,
    m1_sl_pips: float,
    m1_lookahead: int,
    m2_lookahead: int,
    m2_rr: float,
    m3_min_sl_pips: float,
    m3_max_sl_pips: float,
    m3_rr: float,
    m3_future_window: int,
):
    """
    تنظیمات برچسب‌گذاری (مخصوص SL/TP) را در کنار فایل Parquet ذخیره می‌کند.
    این فایل بعداً در تب معامله خودکار خوانده می‌شود تا همان منطق SL/TP اعمال شود.
    """
    try:
        p = Path(parquet_path)
        cfg = {
            "pip_value": float(pip_value),   # اندازه هر pip روی قیمت (مثلاً 0.01 یا 0.1)
            "spread": float(spread),
            "m1": {
                "tp_pips": float(m1_tp_pips),
                "sl_pips": float(m1_sl_pips),
                "lookahead": int(m1_lookahead),
            },
            "m2": {
                "lookahead": int(m2_lookahead),
                "rr": float(m2_rr),
            },
            "m3": {
                "min_sl_pips": float(m3_min_sl_pips),
                "max_sl_pips": float(m3_max_sl_pips),
                "rr": float(m3_rr),
                "future_window": int(m3_future_window),
            },
        }
        cfg_path = p.with_suffix(".labelcfg.json")
        with cfg_path.open("w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        st.info(f"تنظیمات برچسب‌ها در فایل `{cfg_path.name}` ذخیره شد.")
        # برای استفاده‌ی سریع
        st.session_state["label_cfg_path"] = str(cfg_path)
        st.session_state["label_cfg"] = cfg
    except Exception as e:
        st.warning(f"خطا در ذخیره تنظیمات برچسب‌گذاری (SL/TP): {e}")


def load_label_config(parquet_path: str) -> dict | None:
    """
    تنظیمات برچسب‌گذاری (SL/TP) را از کنار فایل Parquet می‌خواند.
    اگر فایل نباشد، None برمی‌گرداند.
    """
    try:
        if "label_cfg" in st.session_state:
            return st.session_state["label_cfg"]

        p = Path(parquet_path)
        cfg_path = p.with_suffix(".labelcfg.json")
        if not cfg_path.exists():
            return None

        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        st.session_state["label_cfg_path"] = str(cfg_path)
        st.session_state["label_cfg"] = cfg
        return cfg
    except Exception as e:
        st.warning(f"خطا در بارگذاری تنظیمات برچسب‌گذاری: {e}")
        return None


def compute_m3_sl_tp_for_index(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    time_col: pd.Series,
    idx: int,
    pip_value: float,
    m3_min_sl_pips: float,
    m3_max_sl_pips: float,
    m3_rr: float,
):
    """
    منطق تعیین SL و TP برای label_m3 برای *یک کندل* (index = idx)
    این تابع منطق پیدا کردن سقف/کف pivot در گذشته و تعیین SL/TP را
    دقیقاً مشابه تب «افزودن برچسب‌ها» اجرا می‌کند، فقط بدون نگاه به آینده.

    خروجی:
        sl_price, tp_price, sl_is_high, sl_ref_time
    اگر SL معتبر پیدا نشود، (None, None, None, NaT) برمی‌گرداند.
    """

    N = len(close)
    if idx < 0 or idx >= N:
        return None, None, None, pd.NaT

    H = high
    L = low

    min_sl = float(m3_min_sl_pips * pip_value)
    max_sl = float(m3_max_sl_pips * pip_value)
    RR3 = float(m3_rr)

    # تشخیص pivotها
    piv_high = np.zeros(N, dtype=bool)
    piv_low  = np.zeros(N, dtype=bool)

    for i in range(3, N - 3):
        if (
            H[i] > H[i - 1] and H[i] > H[i - 2] and H[i] > H[i - 3]
            and H[i] > H[i + 1] and H[i] > H[i + 2] and H[i] > H[i + 3]
        ):
            piv_high[i] = True
        if (
            L[i] < L[i - 1] and L[i] < L[i - 2] and L[i] < L[i - 3]
            and L[i] < L[i + 1] and L[i] < L[i + 2] and L[i] < L[i + 3]
        ):
            piv_low[i] = True

    c0 = close[idx]
    sl_price = None
    sl_is_high = None
    violated_max = False
    sl_ref_time = pd.NaT

    # جستجوی SL در گذشته (مثل تب برچسب‌ها)
    for j in range(idx - 1, -1, -1):
        if piv_high[j] and H[j] > c0:
            dist = H[j] - c0
            if dist < min_sl:
                continue
            if dist > max_sl:
                violated_max = True
                break
            sl_price = H[j]
            sl_is_high = True
            sl_ref_time = time_col.iloc[j]
            break
        if piv_low[j] and L[j] < c0:
            dist = c0 - L[j]
            if dist < min_sl:
                continue
            if dist > max_sl:
                violated_max = True
                break
            sl_price = L[j]
            sl_is_high = False
            sl_ref_time = time_col.iloc[j]
            break

    if sl_price is None or violated_max:
        return None, None, None, pd.NaT

    dist = abs(c0 - sl_price)
    tp_offset = dist * RR3

    if sl_is_high:
        # SL بالاست → معامله‌ی SELL (TP پایین)
        tp_price = c0 - tp_offset
    else:
        # SL پایین است → معامله‌ی BUY (TP بالا)
        tp_price = c0 + tp_offset

    return float(sl_price), float(tp_price), bool(sl_is_high), sl_ref_time

# ---------------------------------------------------------------------
#  (در اینجا بقیه‌ی توابع اصلی خودت می‌آید: محاسبه فیچرها، برچسب‌ها، مدل‌ها و ...)
#  آن‌ها را دست نزن؛ فقط مطمئن شو که این بلاک قبل از tabs = st.tabs(...) باشد.
# ---------------------------------------------------------------------


# ---------------------------
# Helper گلوبال: ساخت مدل از meta
# در همه‌جا (تست مدل‌ها + معامله خودکار) از همین استفاده می‌کنیم
# ---------------------------
def build_model_from_meta(
    meta: dict,
    device: torch.device | None = None,
    num_classes: int | None = None,
) -> nn.Module:
    """
    ساخت مدل از روی meta که در فایل *_meta.json ذخیره شده است.
    """
    # تعداد فیچرها
    in_dim = int(meta["n_features"])

    # تعداد کلاس‌ها
    if num_classes is not None:
        nclass = int(num_classes)
    elif "n_classes" in meta:
        nclass = int(meta["n_classes"])
    else:
        lbl2idx = meta.get("label2idx")
        if isinstance(lbl2idx, dict):
            nclass = len(lbl2idx)
        else:
            raise ValueError("n_classes در meta نیست و label2idx هم مشخص نیست.")

    # نوع مدل
    model_name = meta.get("model_name", "MLP")

    # هایپرپارامترها (اگر در meta نبودند، مقدار پیش‌فرض)
    hidden_dim = int(meta.get("hidden_dim", 128))
    n_heads = int(meta.get("n_heads", 4))
    n_layers  = int(meta.get("n_layers", 2))
    # ساخت مدل بر اساس نوع
    if model_name == "MLP":
        model = MLP(in_dim, hidden_dim, nclass)
    elif model_name == "AutoencoderMLP":
        model = AutoencoderMLPModel(in_dim, hidden_dim, nclass)
    elif model_name == "CNN1D":
        model = CNN1DModel(in_dim, hidden_dim, nclass)
    elif model_name == "LSTM":
        model = LSTMModel(in_dim, hidden_dim, nclass)
    elif model_name == "GRU":
        model = GRUModel(in_dim, hidden_dim, nclass)
    elif model_name == "TCN":
        model = TCNModel(in_dim, hidden_dim, nclass)
    elif model_name == "Transformer":
        # نسخه قدیمی (ساده‌تر) که قبلاً داشتی
        model = TransformerModel(in_dim, hidden_dim, n_heads, n_layers=2, nclass=nclass)

    elif model_name == "TransformerV2":
        # نسخه جدیدتر و قوی‌تر با چند لایه و multi-head
        model = TransformerV2Model(in_dim, hidden_dim, n_heads, n_layers=2, nclass=nclass)
    elif model_name == "CNNLSTM":
        model = CNNLSTMModel(in_dim, hidden_dim, nclass)
    elif model_name == "CNNTransformer":
        model = CNNTransformerModel(in_dim, hidden_dim, n_heads, n_layers=2, nclass=nclass)
    elif model_name == "WaveNet":
        model = WaveNetModel(in_dim, hidden_dim, nclass)
    elif model_name == "DeepAR":
        # 🔹 DeepAR: از hidden_dim و n_classes داخل meta استفاده می‌کنیم
        n_layers = int(meta.get("n_layers", 2))
        model = DeepARModel(in_dim, hidden_dim, nclass, n_layers=n_layers)
    elif model_name == "NBEATS":
        # ⚠️ N-BEATS به window نیاز دارد (در meta ذخیره شده)
        window = int(meta.get("window") or 64)
        model = NBeatsModel(
            in_dim=in_dim,
            window=window,
            hidden_dim=hidden_dim,
            n_blocks=4,
            nclass=nclass,
        )
    elif model_name == "TimesNet":
        # TimesNet با همان hidden_dim و n_blocks ثابت (۴)
        model = TimesNetModel(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            n_blocks=4,
            nclass=nclass,
        )
    elif model_name == "PatchTST":
        # مثل Train: patch_len و stride از window
        window    = int(meta.get("window") or 64)
        patch_len = min(window, 16)
        stride    = max(1, patch_len // 2)

        model = PatchTST(
            input_dim=in_dim,
            patch_len=patch_len,
            stride=stride,
            d_model=hidden_dim,   # ❗ همان hidden_dim ذخیره شده در meta
            n_heads=n_heads,      # ❗ همان n_heads ذخیره شده در meta
            n_layers=n_layers,    # الان 2 است؛ با Train هم‌خوان
            n_classes=nclass,
        )

    elif model_name == "TFT":
        model = TFTModel(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            nclass=nclass,
        )
    else:
        raise ValueError(f"مدل ناشناخته در meta: {model_name}")

    if device is not None:
        model = model.to(device)

    return model



def log_step(status_obj, text: str, pbar, step_inc: int = 5):
    """
    تابع ساده برای گزارش پیشرفت در استریم‌لیت.
    status_obj: همان status داخل st.status(...)
    text: توضیح مرحله
    pbar: progress bar
    step_inc: مقدار افزایش درصد
    """
    status_obj.write(text)
    if "progress_val" not in st.session_state:
        st.session_state["progress_val"] = 0
    st.session_state["progress_val"] += step_inc
    pbar.progress(min(st.session_state["progress_val"], 100))


def is_categorical_feature_for_leveling(df: pd.DataFrame, col: str, max_unique: int = 30):
    """
    تشخیص این‌که یک فیچر برای سطح‌بندی به صورت دسته‌ای (کَتگوریکال) در نظر گرفته شود یا نه.
    - اگر نام ستون حاوی 'pattern' باشد یا به '_id' / '_rank' ختم شود → کَتگوریکال
    - یا اگر عددیِ صحیح با تعداد یکتای کم (<= max_unique) باشد → کَتگوریکال
    """
    if col not in df.columns:
        return False, np.array([])

    vals = df[col].to_numpy()
    vals = vals[~pd.isna(vals)]
    if vals.size == 0:
        return False, np.array([])

    uniq = np.unique(vals)

    name_l = col.lower()
    is_name_pattern_like = (
        ("pattern" in name_l) or
        name_l.endswith("_id") or
        name_l.endswith("_rank")
    )

    # آیا مقادیر عددیِ تقریباً صحیح هستند؟
    if np.issubdtype(uniq.dtype, np.number):
        is_integerish = np.all(np.isfinite(uniq)) and np.all(np.abs(uniq - np.round(uniq)) < 1e-6)
    else:
        is_integerish = False

    is_small_cardinality = len(uniq) <= max_unique

    is_cat = bool(is_name_pattern_like or (is_integerish and is_small_cardinality))
    return is_cat, uniq


# -------------------------------------------------------------
#  تنظیم صفحه و ساخت تب‌ها
# -------------------------------------------------------------
st.set_page_config(
    page_title="GPU Feature/Label/Model Pipeline",
    layout="wide",
)


def apply_feature_pipeline_gpu(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    کل منطق ساخت فیچرها روی GPU (CuPy) بر اساس تنظیمات cfg.
    این همان چیزی است که قبلاً در تب «افزودن فیچرها (GPU)» انجام می‌دادی.
    df ورودی تغییر داده می‌شود و همان df (با ستون‌های فیچر جدید) برگردانده می‌شود.
    """

    df = df.copy()

    # --------- باز کردن تنظیمات از cfg ---------
    b_return1      = cfg.get("b_return1", True)
    b_returnn      = cfg.get("b_returnn", True)
    n_return       = int(cfg.get("n_return", 5))
    b_price_change = cfg.get("b_price_change", True)
    b_hl_range     = cfg.get("b_hl_range", True)
    b_body         = cfg.get("b_body", True)
    b_upper        = cfg.get("b_upper", True)
    b_lower        = cfg.get("b_lower", True)
    b_candle_ratio = cfg.get("b_candle_ratio", True)

    b_sma   = cfg.get("b_sma", True)
    n_sma   = int(cfg.get("n_sma", 20))
    b_ema   = cfg.get("b_ema", True)
    n_ema   = int(cfg.get("n_ema", 20))
    b_macd      = cfg.get("b_macd", True)
    b_macd_sig  = cfg.get("b_macd_sig", True)
    b_rsi   = cfg.get("b_rsi", True)
    n_rsi   = int(cfg.get("n_rsi", 14))
    b_mom   = cfg.get("b_mom", True)
    n_mom   = int(cfg.get("n_mom", 10))
    b_roc   = cfg.get("b_roc", True)
    n_roc   = int(cfg.get("n_roc", 10))
    b_adx   = cfg.get("b_adx", True)
    n_adx   = int(cfg.get("n_adx", 14))
    b_cci   = cfg.get("b_cci", True)
    n_cci   = int(cfg.get("n_cci", 20))
    b_wr    = cfg.get("b_wr", True)
    n_wr    = int(cfg.get("n_wr", 14))

    b_rstd  = cfg.get("b_rstd", True)
    n_rstd  = int(cfg.get("n_rstd", 20))
    b_atr   = cfg.get("b_atr", True)
    b_rs    = cfg.get("b_rs", True)
    b_hv    = cfg.get("b_hv", True)
    n_hv    = int(cfg.get("n_hv", 30))

    b_hcr  = cfg.get("b_hcr", True)

    b_lcr  = cfg.get("b_lcr", True)
    b_ocr  = cfg.get("b_ocr", True)
    b_hloc = cfg.get("b_hloc", True)

    b_skew = cfg.get("b_skew", True)
    b_kurt = cfg.get("b_kurt", True)
    b_acorr = cfg.get("b_acorr", True)
    n_stat  = int(cfg.get("n_stat", 30))

    b_dow         = cfg.get("b_dow", True)
    b_hod         = cfg.get("b_hod", True)
    b_sincos      = cfg.get("b_sincos", True)
    b_month_flags = cfg.get("b_month_flags", True)
    b_days_since_hi = cfg.get("b_days_since_hi", True)
    b_days_since_lo = cfg.get("b_days_since_lo", True)

    ma_cfg = cfg.get("ma_cfg", [])       # لیست ۱۰ تایی (typ, per)
    tfs    = cfg.get("tfs", [])          # لیست TFهای بالاتر مثل ["5T","15T",...]

    enable_single = cfg.get("enable_single", True)
    enable_2      = cfg.get("enable_2", True)
    enable_3      = cfg.get("enable_3", True)
    enable_4      = cfg.get("enable_4", True)
    enable_5      = cfg.get("enable_5", True)

    # --------- کنترل ستون‌های ضروری ---------
    for col in ["open", "high", "low", "close"]:
        if col not in df.columns:
            raise ValueError(f"ستون ضروری `{col}` در df موجود نیست.")

    # --------- انتقال به GPU ---------
    o = to_cp(df["open"])
    h = to_cp(df["high"])
    l = to_cp(df["low"])
    c = to_cp(df["close"])

    # --- انتخاب منبع حجم: real_volume اگر معنی‌دار باشد، وگرنه tick_volume ---
    if "real_volume" in df.columns:
        rv_np = pd.to_numeric(df["real_volume"], errors="coerce").astype(np.float32)

        # اگر عملاً همه‌چیز صفر است (بروکر real_volume واقعی نمی‌دهد)،
        # و ستون tick_volume داریم، سوییچ کن روی tick_volume
        if (np.nanmax(np.abs(rv_np)) < 1e-6) and ("tick_volume" in df.columns):
            rv_np = pd.to_numeric(df["tick_volume"], errors="coerce").astype(np.float32)

    elif "tick_volume" in df.columns:
        # اگر real_volume نداریم، ولی tick_volume داریم
        rv_np = pd.to_numeric(df["tick_volume"], errors="coerce").astype(np.float32)
    else:
        # هیچ ستونی از حجم نداریم → همه‌چیز صفر
        rv_np = np.zeros(len(df), dtype=np.float32)

    # جایگزینی NaN/Inf با صفر برای جلوگیری از خراب شدن اندیکاتورها
    rv_np = np.nan_to_num(rv_np, nan=0.0, posinf=0.0, neginf=0.0)
    rv = to_cp(rv_np)


    # ---------- Basic ----------
    lr = cp.log(c / gpu_shift(c, 1))
    lr = cp.where(cp.isfinite(lr), lr, 0.0)

    if b_return1:
        df["return_1"] = cp_to_np32(lr)
    if b_returnn:
        nret = int(n_return)
        df[f"return_{nret}"] = cp_to_np32(cp.log(c / gpu_shift(c, nret)))
    if b_price_change:
        df["price_change"] = cp_to_np32(c - gpu_shift(c, 1))
    if b_hl_range:
        df["high_low_range"] = cp_to_np32((h - l) / (gpu_shift(c, 1) + 1e-8))
    if b_body:
        df["body_size"] = cp_to_np32(cp.abs(c - o))
    if b_upper:
        df["upper_shadow"] = cp_to_np32(h - cp.maximum(o, c))
    if b_lower:
        df["lower_shadow"] = cp_to_np32(cp.minimum(o, c) - l)
    if b_candle_ratio:
        df["candle_ratio"] = cp_to_np32(cp.abs(c - o) / (h - l + 1e-8))

    # ---------- Trend / Momentum ----------
    if b_sma:
        sma_vals = rolling_mean_cp(c, int(n_sma))
        if b_sma:
            df[f"SMA_{int(n_sma)}"] = cp_to_np32(sma_vals)
    if b_ema:
        df[f"EMA_{int(n_ema)}"] = cp_to_np32(ema_cp(c, int(n_ema)))
    if b_macd or b_macd_sig:
        ema12 = ema_cp(c, 12)
        ema26 = ema_cp(c, 26)
        macd_vals = ema12 - ema26
        if b_macd:
            df["MACD"] = cp_to_np32(macd_vals)
        if b_macd_sig:
            df["MACD_signal"] = cp_to_np32(ema_cp(macd_vals, 9))
    if b_rsi:
        df[f"RSI_{int(n_rsi)}"] = cp_to_np32(rsi_cp(c, int(n_rsi)))
    if b_mom:
        df[f"Momentum_{int(n_mom)}"] = cp_to_np32(c - gpu_shift(c, int(n_mom)))
    if b_roc:
        nrc = int(n_roc)
        df[f"ROC_{nrc}"] = cp_to_np32(100.0 * (c - gpu_shift(c, nrc)) / (gpu_shift(c, nrc) + 1e-8))
    if b_adx or b_atr:
        adx_vals, plus_di, minus_di, atr_vals = adx_full_cp(h, l, c, int(n_adx))
        if b_adx:
            df[f"ADX_{int(n_adx)}"] = cp_to_np32(adx_vals)
            df[f"+DI_{int(n_adx)}"] = cp_to_np32(plus_di)
            df[f"-DI_{int(n_adx)}"] = cp_to_np32(minus_di)
        if b_atr:
            df[f"ATR_{int(n_adx)}"] = cp_to_np32(atr_vals)
    if b_cci:
        df[f"CCI_{int(n_cci)}"] = cp_to_np32(cci_cp(h, l, c, int(n_cci)))
    if b_wr:
        df[f"Williams_%R_{int(n_wr)}"] = cp_to_np32(williams_r_cp(h, l, c, int(n_wr)))

    # ---------- Volatility ----------
    if b_rstd:
        df[f"rolling_std_{int(n_rstd)}"] = cp_to_np32(rolling_std_cp(lr, int(n_rstd)))
    
    if b_rs:
        df["RS"] = cp_to_np32((h - l) / (cp.nanmean(c) + 1e-8))
    if b_hv:
        df[f"HV_{int(n_hv)}"] = cp_to_np32(hv_cp(lr, int(n_hv)))

    # ---------- Volume ----------

        
    
   
    # ---------- Ratios ----------
    if b_hcr:
        df["high_close_ratio"] = cp_to_np32((h - c) / (c + 1e-8))
    if b_lcr:
        df["low_close_ratio"] = cp_to_np32((c - l) / (c + 1e-8))
    if b_ocr:
        df["open_close_ratio"] = cp_to_np32((c - o) / (o + 1e-8))
    if b_hloc:
        df["HL_to_OC_ratio"] = cp_to_np32((h - l) / (cp.abs(c - o) + 1e-8))

    # ---------- Statistical ----------
    if b_skew or b_kurt or b_acorr:
        w = int(n_stat)
        x = lr.astype(cp.float64)
        valid = cp.isfinite(x)
        x0 = cp.where(valid, x, 0.0)
        csum = cp.cumsum(x0)
        c2 = cp.cumsum(x0 * x0)
        c3 = cp.cumsum(x0 * x0 * x0)
        c4 = cp.cumsum(x0 * x0 * x0 * x0)
        cnt = cp.cumsum(valid.astype(cp.int64))

        sum_w = csum - gpu_shift(csum, w, 0.0)
        sum2 = c2 - gpu_shift(c2, w, 0.0)
        sum3 = c3 - gpu_shift(c3, w, 0.0)
        sum4 = c4 - gpu_shift(c4, w, 0.0)
        n_w = cnt - gpu_shift(cnt, w, 0)

        m = sum_w / (n_w + 1e-8)
        m2 = sum2 / (n_w + 1e-8) - m * m
        m3 = sum3 / (n_w + 1e-8) - 3 * m * sum2 / (n_w + 1e-8) + 2 * m * m * m
        m4 = sum4 / (n_w + 1e-8) - 4 * m * sum3 / (n_w + 1e-8) + 6 * m * m * sum2 / (n_w + 1e-8) - 3 * m * m * m * m
        var = m2 / (n_w + 1e-8)
        std = cp.sqrt(cp.maximum(var, 0.0))

        if b_skew:
            skew = (m3 / (n_w + 1e-8)) / (cp.power(std + 1e-8, 3))
            skew[:w] = cp.nan
            df[f"skew_{w}"] = cp_to_np32(skew)
        if b_kurt:
            kurt = (m4 / (n_w + 1e-8)) / (cp.power(std + 1e-8, 4))
            kurt[:w] = cp.nan
            df[f"kurtosis_{w}"] = cp_to_np32(kurt)

        if b_acorr:
            x_lag = gpu_shift(x0, 1, 0.0)
            csum_l = cp.cumsum(x_lag)
            c2_l = cp.cumsum(x_lag * x_lag)
            sum_wl = csum_l - gpu_shift(csum_l, w, 0.0)
            sum2_l = c2_l - gpu_shift(c2_l, w, 0.0)
            mx = sum_w / (n_w + 1e-8)
            mxl = sum_wl / (n_w + 1e-8)
            cov = (sum2_l / (n_w + 1e-8)) - mx * mxl
            varx = (sum2 / (n_w + 1e-8)) - mx * mx
            varxl = (sum2_l / (n_w + 1e-8)) - mxl * mxl
            ac = cov / cp.sqrt(cp.maximum(varx * varxl, 1e-12))
            ac[:w] = cp.nan
            df[f"auto_corr_{w}"] = cp_to_np32(ac)

    # ---------- Time-based (CPU) ----------
    if "time" in df.columns:
        if b_dow:
            df["day_of_week"] = pd.to_datetime(df["time"], utc=True, errors="coerce").dt.dayofweek
        if b_hod:
            df["hour_of_day"] = pd.to_datetime(df["time"], utc=True, errors="coerce").dt.hour
        if b_sincos:
            hnum = pd.to_datetime(df["time"], utc=True, errors="coerce").dt.hour.to_numpy()
            df["sin_time"] = np.sin(2 * np.pi * hnum / 24.0).astype(np.float32)
            df["cos_time"] = np.cos(2 * np.pi * hnum / 24.0).astype(np.float32)
        if b_month_flags:
            tdt = pd.to_datetime(df["time"], utc=True, errors="coerce")
            df["is_month_start"] = tdt.dt.is_month_start.astype(np.int8)
            df["is_month_end"] = tdt.dt.is_month_end.astype(np.int8)
        if b_days_since_hi:
            df["days_since_high"] = (
                df["close"]
                .rolling(60)
                .apply(lambda x: np.argmax(x), raw=True)
                .astype(np.float32)
            )
        if b_days_since_lo:
            df["days_since_low"] = (
                df["close"]
                .rolling(60)
                .apply(lambda x: np.argmin(x), raw=True)
                .astype(np.float32)
            )

    # ---------- Custom 10 MAs + ranking (Base TF) ----------
    if ma_cfg:
        c_cp = c
        ma_matrix = []
        for idx, (typ, per) in enumerate(ma_cfg, start=1):
            per = int(per)
            if typ == "SMA":
                ma = rolling_mean_cp(c_cp, per)
                df[f"MA{idx}_{typ}_{per}"] = cp_to_np32(ma)
            else:
                ma = ema_cp(c_cp, per)
                df[f"MA{idx}_{typ}_{per}"] = cp_to_np32(ma)
            ma_matrix.append(ma)
        if ma_matrix:
            MA = cp.stack(ma_matrix, axis=1)
            dist = cp.abs(MA - c_cp[:, None])
            order = cp.argsort(dist, axis=1)
            n, k = dist.shape
            ranks = cp.empty_like(order)
            row_idx = cp.arange(n)[:, None]
            ranks[row_idx, order] = cp.arange(k)[None, :] + 1
            for j in range(k):
                df[f"MA{j+1}_rank"] = cp_to_np32(ranks[:, j])

    # ---------- همان ۱۰ MA روی TFهای بالاتر ----------
    if ma_cfg and tfs and "time" in df.columns:
        df_tmp = df[["time", "open", "high", "low", "close"]].copy()
        df_tmp["time"] = pd.to_datetime(df_tmp["time"], utc=True, errors="coerce")
        df_tmp = df_tmp.set_index("time").sort_index()
        base_idx = df_tmp.index

        for tf in tfs:
            tf_str = str(tf)
            if "T" not in tf_str and "H" not in tf_str and "D" not in tf_str:
                continue

            if "real_volume" in df.columns:
                agg_dict = {"open": "first", "high": "max", "low": "min", "close": "last", "real_volume": "sum"}
                df_tmp2 = df[["time", "open", "high", "low", "close", "real_volume"]].copy()
                df_tmp2["time"] = pd.to_datetime(df_tmp2["time"], utc=True, errors="coerce")
                df_tmp2 = df_tmp2.set_index("time").sort_index()
            else:
                agg_dict = {"open": "first", "high": "max", "low": "min", "close": "last"}
                df_tmp2 = df_tmp.copy()

            htf = df_tmp2.resample(tf_str).agg(agg_dict).dropna()
            if htf.empty:
                continue

            c_htf = to_cp(htf["close"])
            ma_list_htf = []
            for jdx, (typ, per) in enumerate(ma_cfg, start=1):
                per = int(per)
                if typ == "SMA":
                    ma = rolling_mean_cp(c_htf, per)
                else:
                    ma = ema_cp(c_htf, per)
                htf[f"MA{jdx}_{typ}_{per}"] = cp_to_np32(ma)
                ma_list_htf.append(ma)

            if ma_list_htf:
                MAh = cp.stack(ma_list_htf, axis=1)
                dist_h = cp.abs(MAh - c_htf[:, None])
                order_h = cp.argsort(dist_h, axis=1)
                n_h, k_h = dist_h.shape
                ranks_h = cp.empty_like(order_h)
                row_idx_h = cp.arange(n_h)[:, None]
                ranks_h[row_idx_h, order_h] = cp.arange(k_h)[None, :] + 1
                for j in range(k_h):
                    htf[f"MA{j+1}_rank"] = cp_to_np32(ranks_h[:, j])

            htf_aligned = htf.reindex(base_idx, method="ffill")
            skip_cols = ["open", "high", "low", "close", "real_volume"]
            new_cols = [cname for cname in htf.columns if cname not in skip_cols]
            suffix = tf_str.replace("T", "m")
            for cname in new_cols:
                df[f"{cname}_TF_{suffix}"] = (
                    htf_aligned[cname].astype(np.float32).to_numpy(copy=False)
                )

            del htf_aligned, ma_list_htf
            gc.collect()

    # ---------- Candlestick patterns ----------
    o_cp = o
    h_cp = h
    l_cp = l
    c_cp = c
    if enable_single:
        single_codes = patterns1_gpu(o_cp, h_cp, l_cp, c_cp)
        df["candle_pattern_1"] = cp_to_np32(single_codes)
    if enable_2:
        pat2 = patterns2_gpu(o_cp, h_cp, l_cp, c_cp)
        df["candle_pattern_2"] = cp_to_np32(pat2)
    if enable_3:
        pat3 = patterns3_gpu(o_cp, h_cp, l_cp, c_cp)
        df["candle_pattern_3"] = cp_to_np32(pat3)
    if enable_4:
        pat4 = patterns4_gpu(o_cp, h_cp, l_cp, c_cp)
        df["candle_pattern_4"] = cp_to_np32(pat4)
    if enable_5:
        pat5 = patterns5_gpu(o_cp, h_cp, l_cp, c_cp)
        df["candle_pattern_5"] = cp_to_np32(pat5)

    return df






# اول paths را (در صورت وجود دیتاست قبلی) پر کنیم
paths = ensure_paths_and_autoload()

tabs = st.tabs(
    [
        "📁 تبدیل فایل (CSV → Parquet)",
        "➕ افزودن فیچرها (GPU)",
        "🏷 افزودن برچسب‌ها (۳ روش)",
        "🧠 آموزش مدل‌ها (Training Only)",
        "🧪 تست مدل‌ها",
        " دیتاست مناسب مدلسازیه؟ (بعد از تقسیم به test/train چک شود!)",
        "🤖 معامله خودکار",
        
    ]
)

# =============================================================
#  تب ۱ — تبدیل فایل و انتخاب دیتاست فعال
# =============================================================
with tabs[0]:
    st.header("📁 تبدیل فایل CSV به Parquet و انتخاب دیتاست فعال")

    ss = st.session_state

    # --- ۱) نمایش دیتاست فعال فعلی (اگر هست) ---
    current_prefix = ss.get("dataset_prefix", "")
    current_parquet = ss.get("parquet_path", "")

    if current_parquet and os.path.exists(current_parquet):
        st.success(
            f"دیتاست فعال فعلی: **{current_prefix or Path(current_parquet).stem}**\n\n"
            f"فایل Parquet: `{current_parquet}`"
        )
    else:
        st.info("هنوز دیتاست فعالی انتخاب نشده یا فایل Parquet مربوطه پیدا نشد.")

    # --- ۲) انتخاب از بین فایل‌های Parquet موجود (ادامه‌ی کار با دیتاست قبلی) ---
    st.subheader("📂 انتخاب از بین دیتاست‌های موجود (اختیاری)")
    parq_files = sorted(Path(".").glob("*.parquet"))

    if parq_files:
        parq_options = ["(بدون تغییر)"] + [p.stem for p in parq_files]
        default_idx = 0
        if current_prefix and current_prefix in parq_options:
            default_idx = parq_options.index(current_prefix)

        chosen = st.selectbox(
            "اگر می‌خواهی با یکی از دیتاست‌های ذخیره‌شده قبلی کار را ادامه دهی، انتخابش کن:",
            options=parq_options,
            index=default_idx,
        )

        if chosen != "(بدون تغییر)":
            new_paths = set_project_paths(chosen)
            # بروزرسانی session_state
            ss["dataset_prefix"] = chosen
            save_active_prefix(chosen)

            ss["parquet_path"] = new_paths.get("parquet", "")
            ss["features_path"] = new_paths.get("features", "")
            ss["labels_path"] = new_paths.get("labels", "")
            ss["train_path"]   = new_paths.get("train", "")
            ss["test_path"]    = new_paths.get("test", "")
            ss["models_dir"]   = new_paths.get("models_dir", "")

            paths = new_paths
            current_prefix = chosen
            current_parquet = ss["parquet_path"]

            st.success(
                f"✅ دیتاست فعال تغییر کرد به: **{current_prefix}**\n\n"
                f"فایل Parquet: `{current_parquet}`"
            )
    else:
        st.info("فعلاً هیچ فایل Parquet در پوشه‌ی فعلی پیدا نشد.")

    st.markdown("---")

    # --- ۳) ساخت دیتاست جدید از روی CSV (اختیاری) ---
    st.subheader("📤 بارگذاری CSV جدید و ساخت Parquet (اختیاری)")

    uploaded = st.file_uploader(
        "فایل CSV خام (ستون‌ها: time, open, high, low, close, real_volume, tick_volume)",
        type=["csv"],
    )

    # نام پیش‌فرض دیتاست: اگر قبلاً prefix داریم، همان؛ وگرنه نام فایل CSV
    default_prefix = current_prefix or (Path(uploaded.name).stem if uploaded is not None else "")
    prefix = st.text_input("نام دیتاست (برای نام‌گذاری فایل‌ها)", value=default_prefix)

    overwrite = st.checkbox(
        "اگر فایل Parquet با این نام وجود دارد، آن را بازنویسی کن.",
        value=False,
    )

    if uploaded is not None and prefix.strip():
        prefix = prefix.strip()
        out_paths = build_paths_from_prefix(prefix)
        parquet_out = out_paths["parquet"]

        st.write(f"📦 فایل Parquet خروجی: `{parquet_out}`")

        if os.path.exists(parquet_out) and not overwrite:
            st.warning(
                "⚠️ فایل Parquet با این نام از قبل وجود دارد.\n"
                "اگر می‌خواهی دوباره از روی CSV بسازی، تیک «بازنویسی» را فعال کن."
            )

        if st.button("⚙️ تبدیل CSV به Parquet و تنظیم به‌عنوان دیتاست فعال"):
            try:
                # خواندن CSV خام
                df = pd.read_csv(uploaded)

                # اگر لازم بود، اینجا می‌توانی تبدیل نوع ستون time را انجام دهی:
                # df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)

                df.to_parquet(parquet_out, engine="pyarrow", index=False)

                # تنظیم به‌عنوان دیتاست فعال
                ss["dataset_prefix"] = prefix
                save_active_prefix(prefix)

                new_paths = set_project_paths(prefix)
                ss["parquet_path"] = new_paths.get("parquet", str(parquet_out))
                ss["features_path"] = new_paths.get("features", "")
                ss["labels_path"]   = new_paths.get("labels", "")
                ss["train_path"]    = new_paths.get("train", "")
                ss["test_path"]     = new_paths.get("test", "")
                ss["models_dir"]    = new_paths.get("models_dir", "")

                paths = new_paths

                st.success(
                    f"✅ CSV با موفقیت به Parquet تبدیل و ذخیره شد:\n`{parquet_out}`\n\n"
                    f"و به‌عنوان دیتاست فعال تنظیم شد."
                )
            except Exception as e:
                st.error(f"❌ خطا در خواندن CSV یا ذخیره‌ی Parquet: {e}")
    else:
        st.info(
            "اگر می‌خواهی دیتاست جدید بسازی، یک CSV انتخاب کن و نام دیتاست را وارد کن.\n"
            "اگر نمی‌خواهی دیتاست جدید بسازی، از بخش بالا یک دیتاست موجود را برای ادامه انتخاب کن."
        )

with tabs[1]:
    st.subheader("افزودن فیچرها (GPU)")

    paths = ensure_paths_and_autoload()
    parquet_path = st.session_state.get("parquet_path") or paths.get("parquet", "")

    if not parquet_path or not os.path.exists(parquet_path):
        st.warning(
            "هیچ فایل Parquet فعالی پیدا نشد. اگر بار اول است، در تب «📁 تبدیل فایل (CSV → Parquet)» یک CSV را تبدیل کن."
        )
        st.stop()

    # بارگذاری df از پارکت
    if "df_parquet" not in st.session_state:
        df = pd.read_parquet(parquet_path)
        st.session_state["df_parquet"] = df
    else:
        df = st.session_state["df_parquet"]

    st.info(f"دیتافریم فعال از فایل Parquet: `{parquet_path}`  |  شکل: {df.shape}")
    st.dataframe(df.head(50), use_container_width=True)

    # ====== انتخاب زیرمجموعه (optional) ======
    st.markdown("### ✂️ انتخاب زیرمجموعه‌ای از ردیف‌ها (اختیاری)")
    with st.expander("تنظیم ردیف‌های مورد استفاده برای ساخت فیچرها", expanded=False):
        n_total = len(df)
        start_idx = st.number_input("شروع از ایندکس", 0, max(0, n_total - 1), 0)
        end_idx = st.number_input("تا ایندکس (انحصاری)", 1, n_total, n_total)
        if start_idx < end_idx:
            df = df.iloc[start_idx:end_idx].copy()
            st.info(f"زیرمجموعه انتخاب شده: {df.shape}")
        else:
            st.warning("بازه انتخابی معتبر نیست (start >= end). از کل df استفاده می‌شود.")

    # ========== تنظیمات فیچر ==========
    st.markdown("### 🧩 فیچرهای پایه‌ای")
    b_return1 = st.checkbox("return_1", True)
    b_returnn = st.checkbox("return_n", True); n_return = st.number_input("n برای return_n", 2, 500, 5)
    b_price_change = st.checkbox("price_change", True)
    b_hl_range = st.checkbox("high_low_range", True)
    b_body = st.checkbox("body_size", True)
    b_upper = st.checkbox("upper_shadow", True)
    b_lower = st.checkbox("lower_shadow", True)
    b_candle_ratio = st.checkbox("candle_ratio", True)

    st.markdown("### 📈 حرکتی/مومنتوم")
    b_sma = st.checkbox("SMA_n", True); n_sma = st.number_input("n برای SMA", 2, 300, 20)
    b_ema = st.checkbox("EMA_n", True); n_ema = st.number_input("n برای EMA", 2, 300, 20)
    b_macd = st.checkbox("MACD (EMA12-EMA26)", True)
    b_macd_sig = st.checkbox("MACD_signal (EMA9 روی MACD)", True)
    b_rsi = st.checkbox("RSI", True); n_rsi = st.number_input("period RSI", 2, 500, 14)
    b_mom = st.checkbox("Momentum_n", True); n_mom = st.number_input("n برای Momentum", 1, 2000, 10)
    b_roc = st.checkbox("ROC_n", True); n_roc = st.number_input("n برای ROC", 1, 2000, 10)
    b_adx = st.checkbox("ADX (+DI/-DI/ATR)", True); n_adx = st.number_input("period ADX/ATR", 2, 500, 14)
    b_cci = st.checkbox("CCI", True); n_cci = st.number_input("period CCI", 2, 500, 20)
    b_wr = st.checkbox("Williams %R", True); n_wr = st.number_input("period Williams %R", 2, 500, 14)

    st.markdown("### ⚡ نوسان")
    b_rstd = st.checkbox("rolling_std_n روی log-returns", True); n_rstd = st.number_input("n برای rolling std", 2, 2000, 20)
    b_atr = st.checkbox("ATR", True)
    
    b_rs = st.checkbox("RS نسبت دامنه high-low به میانگین close", True)
    b_hv = st.checkbox("HV_n (Historical Volatility)", True); n_hv = st.number_input("period HV", 2, 2000, 30)



    st.markdown("### 🔄 نسبتی")
    b_hcr = st.checkbox("high_close_ratio", True)
    b_lcr = st.checkbox("low_close_ratio", True)
    b_ocr = st.checkbox("open_close_ratio", True)
    b_hloc = st.checkbox("HL_to_OC_ratio", True)

    st.markdown("### 🧮 آماری (GPU روی log-returns)")
    b_skew = st.checkbox("skew_n", True)
    b_kurt = st.checkbox("kurtosis_n", True)
    b_acorr = st.checkbox("auto_corr lag-1", True)
    n_stat = st.number_input("window آماری", 2, 2000, 30)

    st.markdown("### ⏱ زمانی (CPU)")
    b_dow = st.checkbox("day_of_week", True)
    b_hod = st.checkbox("hour_of_day", True)
    b_sincos = st.checkbox("sin/cos_time (دوره روزانه)", True)
    b_month_flags = st.checkbox("is_month_start/end", True)
    b_days_since_hi = st.checkbox("days_since_high (rolling 60)", True)
    b_days_since_lo = st.checkbox("days_since_low (rolling 60)", True)

    # ====== Custom MAs (10) & higher TF ======
    st.markdown("### 📏 ۱۰ MA سفارشی + رتبه‌بندی")
    ma_cfg = []
    for i in range(10):
        cols = st.columns(2)
        with cols[0]:
            typ = st.selectbox(
                f"نوع MA{i+1}",
                ["SMA", "EMA"],
                index=1,
                key=f"ma_type_{i}",
            )
        with cols[1]:
            per = st.number_input(
                f"دوره MA{i+1}",
                min_value=1,
                max_value=5000,
                value=[15,30,60,90,120,150,200,220,250,300][i] if i < 10 else 20,
                key=f"ma_per_{i}",
            )
        ma_cfg.append((typ, per))

    st.markdown("### ⬆️ همان ۱۰ MA و رتبه‌بندی در تایم‌فریم‌های بالاتر")
    base_tf_minutes = None
    if "time" in df.columns:
        base_tf_minutes = detect_base_freq_minutes(df["time"])
        st.caption(f"TF پایه حدوداً ≈ {base_tf_minutes:.2f} دقیقه")

    tf_options = []
    if base_tf_minutes:
        tf_map = {"5T": 5,"15T": 15,"30T": 30,"1H": 60,"4H": 240,"1D": 1440}
        for k, mins in tf_map.items():
            if mins >= base_tf_minutes:
                tf_options.append(k)
    tfs = st.multiselect("انتخاب تایم‌فریم‌های هدف", options=tf_options, default=["5T", "15T", "30T", "1H", "4H", "1D"])

    # ====== Candle patterns ======
    st.markdown("### 🕯️ برچسب الگوهای کندلی (تک/۲/۳/۴/۵)")
    enable_single = st.checkbox("تک کندلی", True)
    enable_2 = st.checkbox("دوکندلی", True)
    enable_3 = st.checkbox("سه‌کندلی", True)
    enable_4 = st.checkbox("چهارکندلی", True)
    enable_5 = st.checkbox("پنج‌کندلی", True)

    # ====== ساخت cfg و ذخیره در session_state ======
    feature_cfg = {
        "b_return1": b_return1,
        "b_returnn": b_returnn,
        "n_return": int(n_return),
        "b_price_change": b_price_change,
        "b_hl_range": b_hl_range,
        "b_body": b_body,
        "b_upper": b_upper,
        "b_lower": b_lower,
        "b_candle_ratio": b_candle_ratio,

        "b_sma": b_sma,
        "n_sma": int(n_sma),
        "b_ema": b_ema,
        "n_ema": int(n_ema),
        "b_macd": b_macd,
        "b_macd_sig": b_macd_sig,
        "b_rsi": b_rsi,
        "n_rsi": int(n_rsi),
        "b_mom": b_mom,
        "n_mom": int(n_mom),
        "b_roc": b_roc,
        "n_roc": int(n_roc),
        "b_adx": b_adx,
        "n_adx": int(n_adx),
        "b_cci": b_cci,
        "n_cci": int(n_cci),
        "b_wr": b_wr,
        "n_wr": int(n_wr),

        "b_rstd": b_rstd,
        "n_rstd": int(n_rstd),
        "b_atr": b_atr,
        "b_rs": b_rs,
        "b_hv": b_hv,
        "n_hv": int(n_hv),

        "b_hcr": b_hcr,

        "b_lcr": b_lcr,
        "b_ocr": b_ocr,
        "b_hloc": b_hloc,

        "b_skew": b_skew,
        "b_kurt": b_kurt,
        "b_acorr": b_acorr,
        "n_stat": int(n_stat),

        "b_dow": b_dow,
        "b_hod": b_hod,
        "b_sincos": b_sincos,
        "b_month_flags": b_month_flags,
        "b_days_since_hi": b_days_since_hi,
        "b_days_since_lo": b_days_since_lo,

        "ma_cfg": ma_cfg,
        "tfs": tfs,

        "enable_single": enable_single,
        "enable_2": enable_2,
        "enable_3": enable_3,
        "enable_4": enable_4,
        "enable_5": enable_5,
    }
    st.session_state["feature_cfg_gpu"] = feature_cfg

    # ====== Compute & Save ======
    if st.button("محاسبه و ذخیره روی همان Parquet (همهٔ فیچرها + موارد جدید) — حافظه‌دوست"):
        with st.spinner("در حال محاسبه فیچرها روی GPU ..."):
            try:
                df_feat = apply_feature_pipeline_gpu(df, feature_cfg)
                base_path = parquet_path  # روی همان فایل ذخیره شود
                df_feat.to_parquet(base_path, engine="pyarrow", index=False)
                # ذخیره تنظیمات فیچرها کنار همان فایل پارکت
                cfg_path = Path(base_path).with_suffix(".featurecfg.json")
                try:
                    with open(cfg_path, "w", encoding="utf-8") as f:
                        json.dump(feature_cfg, f, ensure_ascii=False, indent=2)
                    st.info(f"تنظیمات فیچرها در فایل {cfg_path.name} ذخیره شد.")
                except Exception as e_cfg:
                    st.warning(f"خطا در ذخیره فایل تنظیمات فیچر: {e_cfg}")
                st.session_state["df_parquet"] = df_feat
                st.success(f"✅ محاسبات انجام و در فایل Parquet ذخیره شد: {base_path}")
                st.dataframe(df_feat.tail(50), use_container_width=True)
            except Exception as e:
                st.error(f"❌ خطا در محاسبه فیچرها: {e}")

# =============================================================
#  تب ۳ — افزودن برچسب‌ها (۳ روش) با درنظر گرفتن اسپرد
# =============================================================
with tabs[2]:
    st.subheader("🏷 افزودن برچسب‌ها (۳ روش) — با ذخیره روی همان Parquet")

    paths = ensure_paths_and_autoload()
    parquet_path = st.session_state.get("parquet_path") or paths.get("parquet", "")

    if not parquet_path or not os.path.exists(parquet_path):
        st.warning("هیچ فایل Parquet فعالی پیدا نشد. اگر بار اول است، در تب «📁 تبدیل فایل» یک CSV را به Parquet تبدیل کن.")
        st.stop()

    # لود دیتافریم
    if "df_parquet" in st.session_state and st.session_state.get("parquet_path") == parquet_path:
        df = st.session_state["df_parquet"].copy()
    else:
        df = pd.read_parquet(parquet_path)
        st.session_state["df_parquet"] = df.copy()
        st.session_state["parquet_path"] = parquet_path

    st.caption(f"مسیر دیتاست: {parquet_path}")

    required_cols = ["time", "open", "high", "low", "close"]
    missing_req = [c for c in required_cols if c not in df.columns]
    if missing_req:
        st.error(f"ستون‌های ضروری برای برچسب‌گذاری موجود نیستند: {missing_req}")
        st.stop()

    time_col = pd.to_datetime(df["time"], utc=True, errors="coerce")
    close = df["close"].to_numpy(dtype=float)
    high  = df["high"].to_numpy(dtype=float)
    low   = df["low"].to_numpy(dtype=float)
    N = len(df)

    # ---------------- تنظیمات مشترک ----------------
    st.markdown("### ⚙️ تنظیمات مشترک")
    pip_value = st.number_input(
        "ارزش هر پیپ (pip_value) — برای XAUUSD معمولا 0.1 یا 0.01",
        min_value=1e-5, max_value=10.0, value=0.1, step=0.01, format="%.5f"
    )
    spread = st.number_input(
        "اسپرد (به واحد قیمت، مثلاً 0.2 برای XAUUSD)",
        min_value=0.0, max_value=10.0, value=0.2, step=0.01, format="%.3f"
    )

    # آرایه‌های HIGH_SPREAD و LOW_SPREAD
    HIGH_SPREAD = high - spread
    LOW_SPREAD  = low  + spread

    # ---------------- روش اول ----------------
    st.markdown("## روش اول — label_m1 (ترکیب BUY و SELL با TP/SL ثابت)")

    enable_m1 = st.checkbox("فعال بودن روش اول (label_m1)", value=True)
    col_m1 = st.columns(4)
    with col_m1[0]:
        m1_tp_pips = st.number_input("حد سود (پیپ) برای روش ۱", min_value=1.0, max_value=10_000.0, value=100.0, step=10.0)
    with col_m1[1]:
        m1_sl_pips = st.number_input("حد ضرر (پیپ) برای روش ۱", min_value=1.0, max_value=10_000.0, value=100.0, step=10.0)
    with col_m1[2]:
        m1_lookahead = st.number_input("پنجره بررسی آینده (تعداد کندل) برای روش ۱", min_value=1, max_value=500000, value=200, step=10)
    with col_m1[3]:
        st.markdown(
            """
    برچسب‌ها:
    - 1: BUY موفق
    - 2: SELL موفق
    - 3: سناریوی خاص شکست BUY
    - 4: سناریوی خاص شکست SELL
    - 5: سایر حالات / بدون نتیجه واضح
            """
        )

    # ---------------- روش دوم ----------------
    st.markdown("## روش دوم — label_m2 (بیشترین High/Low آینده + نسبت سود به ضرر)")

    enable_m2 = st.checkbox("فعال بودن روش دوم (label_m2)", value=True)
    col_m2 = st.columns(3)
    with col_m2[0]:
        m2_lookahead = st.number_input("پنجره نگاه به آینده (کندل)", min_value=1, max_value=500000, value=100, step=10)
    with col_m2[1]:
        m2_rr = st.number_input("حداقل نسبت سود به ضرر (R)", min_value=1.0, max_value=100.0, value=2.0, step=0.1)


    # ---------------- روش سوم ----------------
    st.markdown("## روش سوم — label_m3 (SL از سقف/کف گذشته + TP با R)")

    enable_m3 = st.checkbox("فعال بودن روش سوم (label_m3)", value=True)
    col_m3 = st.columns(4)
    with col_m3[0]:
        m3_min_sl_pips = st.number_input("حداقل پیپ مجاز برای SL", min_value=1.0, max_value=10_000.0, value=50.0, step=1.0)
    with col_m3[1]:
        m3_max_sl_pips = st.number_input("حداکثر پیپ مجاز برای SL", min_value=1.0, max_value=10_000.0, value=60.0, step=1.0)
    with col_m3[2]:
        m3_rr = st.number_input("نسبت TP به SL (R)", min_value=0.5, max_value=20.0, value=2.0, step=0.1)
    with col_m3[3]:
        m3_future_window = st.number_input("پنجره بررسی آینده برای لمس TP/SL", min_value=1, max_value=500000, value=300, step=10)

        st.markdown("---")

    if st.button("⚙️ محاسبه و ذخیره برچسب‌ها روی فایل Parquet"):
        with st.status("در حال محاسبه برچسب‌ها...", expanded=True) as status:
            try:
                N = len(df)
                status.write(f"تعداد ردیف‌ها: {N}")

                # ---------------- روش اول: BUY/SELL با اسپرد ----------------
                if enable_m1:
                    status.write("▶️ روش اول (label_m1) ...")

                    tp_dist = m1_tp_pips * pip_value
                    sl_dist = m1_sl_pips * pip_value
                    max_ahead = int(m1_lookahead)

                    label_m1 = np.full(N, 5, dtype=np.int8)
                    m1_tp_price = np.full(N, np.nan, dtype=float)
                    m1_sl_price = np.full(N, np.nan, dtype=float)
                    m1_tp_time = pd.Series([pd.NaT] * N, dtype="datetime64[ns, UTC]")
                    m1_sl_time = pd.Series([pd.NaT] * N, dtype="datetime64[ns, UTC]")

                    for i in range(N):
                        c0 = close[i]

                        # BUY
                        tp_up_buy = c0 + tp_dist
                        sl_dn_buy = c0 - sl_dist

                        # SELL
                        tp_up_sell = c0 - sl_dist
                        sl_dn_sell = c0 + tp_dist

                        end_i = min(N, i + 1 + max_ahead)

                        lab = 5
                        tp_price = np.nan
                        sl_price = np.nan
                        tp_time = pd.NaT
                        sl_time = pd.NaT

                        # وضعیت BUY
                        hit_tp_buy_idx = None
                        min_low_sp_until_tp = None

                        for j in range(i, end_i):
                            hsp = HIGH_SPREAD[j]
                            lsp = LOW_SPREAD[j]

                            if hit_tp_buy_idx is None and hsp >= tp_up_buy:
                                hit_tp_buy_idx = j
                            if (min_low_sp_until_tp is None) or (lsp < min_low_sp_until_tp):
                                min_low_sp_until_tp = lsp

                        buy_go_sell = False

                        if hit_tp_buy_idx is not None:
                            if min_low_sp_until_tp is None or min_low_sp_until_tp >= sl_dn_buy:
                                lab = 1
                                tp_price = tp_up_buy
                                sl_price = sl_dn_buy
                                tp_time = time_col.iloc[hit_tp_buy_idx]
                            else:
                                if (min_low_sp_until_tp < sl_dn_buy) and (min_low_sp_until_tp > tp_up_sell):
                                    lab = 3
                                elif (min_low_sp_until_tp < sl_dn_buy) and (min_low_sp_until_tp <= tp_up_sell):
                                    buy_go_sell = True
                                else:
                                    buy_go_sell = True
                        else:
                            buy_go_sell = True

                        # وضعیت SELL
                        if buy_go_sell and lab not in (1, 3):
                            hit_tp_sell_idx = None
                            max_high_sp_until_tp = None

                            for j in range(i, end_i):
                                hsp = HIGH_SPREAD[j]
                                lsp = LOW_SPREAD[j]

                                if hit_tp_sell_idx is None and lsp <= tp_up_sell:
                                    hit_tp_sell_idx = j
                                if (max_high_sp_until_tp is None) or (hsp > max_high_sp_until_tp):
                                    max_high_sp_until_tp = hsp

                            if hit_tp_sell_idx is not None:
                                if max_high_sp_until_tp is None or max_high_sp_until_tp <= sl_dn_sell:
                                    lab = 2
                                    tp_price = tp_up_sell
                                    sl_price = sl_dn_sell
                                    tp_time = time_col.iloc[hit_tp_sell_idx]
                                else:
                                    if (max_high_sp_until_tp > sl_dn_sell) and (max_high_sp_until_tp < tp_up_buy):
                                        if lab != 3:
                                            lab = 4
                                    elif (max_high_sp_until_tp > sl_dn_sell) and (max_high_sp_until_tp >= tp_up_buy):
                                        if lab != 3:
                                            lab = 5
                                    else:
                                        if lab not in (1, 3):
                                            lab = 5

                        label_m1[i] = lab
                        m1_tp_price[i] = tp_price
                        m1_sl_price[i] = sl_price
                        m1_tp_time.iloc[i] = tp_time
                        m1_sl_time.iloc[i] = sl_time

                    df["label_m1"] = label_m1
                    df["m1_tp_price"] = m1_tp_price
                    df["m1_sl_price"] = m1_sl_price
                    df["m1_tp_time"] = m1_tp_time
                    df["m1_sl_time"] = m1_sl_time

                    status.write("✅ روش اول با منطق جدید BUY/SELL و اسپرد تکمیل شد.")

                # ---------------- روش دوم: label_m2 (منطق جدید) ----------------
                if enable_m2:
                    status.write("▶️ محاسبه روش دوم (label_m2) با منطق جدید ...")

                    L2 = int(m2_lookahead)
                    R2 = float(m2_rr)

                    label_m2 = np.full(N, 0, dtype=np.int8)
                    m2_tp_price = np.full(N, np.nan, dtype=float)
                    m2_sl_price = np.full(N, np.nan, dtype=float)
                    m2_tp_time = pd.Series([pd.NaT] * N, dtype="datetime64[ns, UTC]")
                    m2_sl_time = pd.Series([pd.NaT] * N, dtype="datetime64[ns, UTC]")

                    for i in range(N):
                        c0 = close[i]
                        end_i = min(N, i + 1 + L2)
                        if end_i <= i + 1:
                            label_m2[i] = 0
                            continue

                        local_high = close[i+1:end_i].max()
                        local_low  = close[i+1:end_i].min()

                        up_move = local_high - c0
                        dn_move = c0 - local_low

                        if up_move <= 0 and dn_move <= 0:
                            label_m2[i] = 0
                            continue

                        if up_move > 0 and dn_move <= 0:
                            label_m2[i] = 1
                            tp_price = local_high
                            sl_price = c0
                        elif dn_move > 0 and up_move <= 0:
                            label_m2[i] = 2
                            tp_price = local_low
                            sl_price = c0
                        else:
                            rr1 = up_move / (dn_move + 1e-8)
                            rr2 = dn_move / (up_move + 1e-8)

                            if rr1 >= R2 and rr1 > rr2:
                                label_m2[i] = 1
                                tp_price = local_high
                                sl_price = local_low
                            elif rr2 >= R2 and rr2 > rr1:
                                label_m2[i] = 2
                                tp_price = local_low
                                sl_price = local_high
                            else:
                                label_m2[i] = 0
                                tp_price = np.nan
                                sl_price = np.nan

                        m2_tp_price[i] = tp_price
                        m2_sl_price[i] = sl_price

                        tp_index = None
                        sl_index = None
                        for k in range(i+1, end_i):
                            if tp_index is None:
                                if close[k] >= tp_price and label_m2[i] == 1:
                                    tp_index = k
                                if close[k] <= tp_price and label_m2[i] == 2:
                                    tp_index = k
                            if sl_index is None:
                                if close[k] <= sl_price and label_m2[i] == 1:
                                    sl_index = k
                                if close[k] >= sl_price and label_m2[i] == 2:
                                    sl_index = k
                            if tp_index is not None and sl_index is not None:
                                break

                        if tp_index is not None:
                            m2_tp_time.iloc[i] = time_col.iloc[tp_index]
                        if sl_index is not None:
                            m2_sl_time.iloc[i] = time_col.iloc[sl_index]

                    df["label_m2"] = label_m2
                    df["m2_tp_price"] = m2_tp_price
                    df["m2_sl_price"] = m2_sl_price
                    df["m2_tp_time"] = m2_tp_time
                    df["m2_sl_time"] = m2_sl_time

                    status.write("✅ روش دوم (label_m2) با منطق جدید highest/lowest و اسپرد محاسبه شد.")

                # ---------------- روش سوم: لمس TP/SL با HIGH_SPREAD/LOW_SPREAD ----------------
                if enable_m3:
                    status.write("▶️ روش سوم (label_m3) ...")

                    min_sl = m3_min_sl_pips * pip_value
                    max_sl = m3_max_sl_pips * pip_value
                    RR3 = float(m3_rr)
                    Wf = int(m3_future_window)

                    label_m3 = np.full(N, 4, dtype=np.int8)
                    m3_tp_price = np.full(N, np.nan, dtype=float)
                    m3_sl_price = np.full(N, np.nan, dtype=float)
                    m3_tp_time = pd.Series([pd.NaT] * N, dtype="datetime64[ns, UTC]")
                    m3_sl_time = pd.Series([pd.NaT] * N, dtype="datetime64[ns, UTC]")
                    m3_sl_ref_time = pd.Series([pd.NaT] * N, dtype="datetime64[ns, UTC]")

                    # pivotها
                    H = high
                    L = low
                    piv_high = np.zeros(N, dtype=bool)
                    piv_low  = np.zeros(N, dtype=bool)

                    for i in range(3, N - 3):
                        if (
                            H[i] > H[i-1] and H[i] > H[i-2] and H[i] > H[i-3]
                            and H[i] > H[i+1] and H[i] > H[i+2] and H[i] > H[i+3]
                        ):
                            piv_high[i] = True
                        if (
                            L[i] < L[i-1] and L[i] < L[i-2] and L[i] < L[i-3]
                            and L[i] < L[i+1] and L[i] < L[i+2] and L[i] < L[i+3]
                        ):
                            piv_low[i] = True

                    for i in range(N):
                        c0 = close[i]
                        sl_price = None
                        sl_is_high = None
                        violated_max = False

                        # جستجوی SL در گذشته
                        for j in range(i - 1, -1, -1):
                            if piv_high[j] and H[j] > c0:
                                dist = H[j] - c0
                                if dist < min_sl:
                                    continue
                                if dist > max_sl:
                                    violated_max = True
                                    break
                                sl_price = H[j]
                                sl_is_high = True
                                m3_sl_ref_time.iloc[i] = time_col.iloc[j]
                                break
                            if piv_low[j] and L[j] < c0:
                                dist = c0 - L[j]
                                if dist < min_sl:
                                    continue
                                if dist > max_sl:
                                    violated_max = True
                                    break
                                sl_price = L[j]
                                sl_is_high = False
                                m3_sl_ref_time.iloc[i] = time_col.iloc[j]
                                break

                        if sl_price is None or violated_max:
                            label_m3[i] = 4
                            continue

                        dist = abs(c0 - sl_price)
                        tp_offset = dist * RR3

                        if sl_is_high:
                            tp_price = c0 - tp_offset
                        else:
                            tp_price = c0 + tp_offset

                        hit_tp_idx = None
                        hit_sl_idx = None

                        end_i = min(N, i + 1 + Wf)
                        for k in range(i + 1, end_i):
                            hsp = HIGH_SPREAD[k]
                            lsp = LOW_SPREAD[k]

                            if sl_is_high:
                                if hit_tp_idx is None and lsp <= tp_price:
                                    hit_tp_idx = k
                                if hit_sl_idx is None and hsp >= sl_price:
                                    hit_sl_idx = k
                            else:
                                if hit_tp_idx is None and hsp >= tp_price:
                                    hit_tp_idx = k
                                if hit_sl_idx is None and lsp <= sl_price:
                                    hit_sl_idx = k

                            if hit_tp_idx is not None and hit_sl_idx is not None:
                                break

                        if hit_tp_idx is None and hit_sl_idx is None:
                            label_m3[i] = 4
                        elif hit_tp_idx is not None and hit_sl_idx is None:
                            label_m3[i] = 2 if sl_is_high else 1
                        elif hit_sl_idx is not None and hit_tp_idx is None:
                            label_m3[i] = 3
                        else:
                            if hit_tp_idx < hit_sl_idx:
                                label_m3[i] = 2 if sl_is_high else 1
                            elif hit_sl_idx < hit_tp_idx:
                                label_m3[i] = 3
                            else:
                                label_m3[i] = 3

                        m3_tp_price[i] = tp_price
                        m3_sl_price[i] = sl_price
                        if hit_tp_idx is not None:
                            m3_tp_time.iloc[i] = time_col.iloc[hit_tp_idx]
                        if hit_sl_idx is not None:
                            m3_sl_time.iloc[i] = time_col.iloc[hit_sl_idx]

                    df["label_m3"] = label_m3
                    df["m3_tp_price"] = m3_tp_price
                    df["m3_sl_price"] = m3_sl_price
                    df["m3_tp_time"] = m3_tp_time
                    df["m3_sl_time"] = m3_sl_time
                    df["m3_sl_ref_time"] = m3_sl_ref_time

                    status.write("✅ روش سوم با استفاده از HIGH_SPREAD/LOW_SPREAD در لمس آینده تکمیل شد.")

                # 🔴 ذخیره تنظیمات برچسب‌ها (SL/TP) کنار فایل Parquet
                save_label_config(
                    parquet_path=parquet_path,
                    pip_value=pip_value,
                    spread=spread,
                    m1_tp_pips=m1_tp_pips,
                    m1_sl_pips=m1_sl_pips,
                    m1_lookahead=m1_lookahead,
                    m2_lookahead=m2_lookahead,
                    m2_rr=m2_rr,
                    m3_min_sl_pips=m3_min_sl_pips,
                    m3_max_sl_pips=m3_max_sl_pips,
                    m3_rr=m3_rr,
                    m3_future_window=m3_future_window,
                )

                # ذخیره نهایی دیتافریم
                df.to_parquet(parquet_path, engine="pyarrow", index=False)
                st.session_state["df_parquet"] = df.copy()

                status.update(label="✅ برچسب‌ها با موفقیت محاسبه و ذخیره شدند.", state="complete")
                st.success("همهٔ برچسب‌های فعال محاسبه و در Parquet ذخیره شدند.")
                safe_cols = [c for c in ["time", "label_m1", "label_m2", "label_m3"] if c in df.columns]
                st.dataframe(df[safe_cols].tail(50), use_container_width=True)

            except Exception as e:
                status.update(label=f"❌ خطا در برچسب‌گذاری: {e}", state="error")
                st.exception(e)

# =============================================================
#  تب ۴ — آموزش مدل‌ها (Training Only)
# =============================================================
with tabs[3]:
    st.header("🧠 آموزش مدل‌ها (Training Only)")

    paths = ensure_paths_and_autoload()
    parquet_path = st.session_state.get("parquet_path") or paths.get("parquet", "")
    train_path = st.session_state.get("train_path") or paths.get("train", "")
    test_path  = st.session_state.get("test_path")  or paths.get("test", "")

    if not parquet_path or not os.path.exists(parquet_path):
        st.warning("هیچ فایل Parquet فعالی پیدا نشد. اگر بار اول است، در تب «📁 تبدیل فایل (CSV → Parquet)» یک CSV را تبدیل کن.")
        st.stop()

    st.caption(f"دیتاست فعال: `{parquet_path}`")

    # ---------------------------------------------------------
    # گام ۱: ساخت / استفاده از Train و Test (۸۰/۲۰ زمانی)
    # ---------------------------------------------------------
    st.subheader("گام ۱: ساخت / استفاده از Train/Test Split")

    col_split = st.columns(3)
    with col_split[0]:
        exists_train = bool(train_path and os.path.exists(train_path))
        st.write(f"Train path: `{train_path or '(تعریف نشده)'}`")
        st.write("✅ موجود است" if exists_train else "❌ هنوز ساخته نشده")

    with col_split[1]:
        exists_test = bool(test_path and os.path.exists(test_path))
        st.write(f"Test path: `{test_path or '(تعریف نشده)'}`")
        st.write("✅ موجود است" if exists_test else "❌ هنوز ساخته نشده")

    with col_split[2]:
        overwrite_split = st.checkbox("بازنویسی Train/Test در صورت وجود", value=False)

    if st.button("📐 ساخت/بازنویسی Train/Test (۸۰/۲۰ بر اساس زمان)"):
        try:
            # لود دیتافریم اصلی
            if "df_parquet" in st.session_state and st.session_state.get("parquet_path") == parquet_path:
                df_all = st.session_state["df_parquet"].copy()
            else:
                df_all = pd.read_parquet(parquet_path)
                st.session_state["df_parquet"] = df_all.copy()
                st.session_state["parquet_path"] = parquet_path

            if "time" not in df_all.columns:
                st.error("ستون time برای تقسیم زمانی لازم است.")
                st.stop()

            df_all["time"] = pd.to_datetime(df_all["time"], utc=True, errors="coerce")
            df_all = df_all.sort_values("time").reset_index(drop=True)

            N = len(df_all)
            if N < 100:
                st.error("داده برای Train/Test خیلی کم است (کمتر از 100 ردیف).")
                st.stop()

            split_idx = int(N * 0.8)
            df_train = df_all.iloc[:split_idx].reset_index(drop=True)
            df_test  = df_all.iloc[split_idx:].reset_index(drop=True)

            # مسیرهای جدید بر اساس prefix
            prefix = st.session_state.get("dataset_prefix") or Path(parquet_path).stem
            split_paths = build_paths_from_prefix(prefix)
            train_path = str(split_paths["train"])
            test_path  = str(split_paths["test"])

            # ذخیره
            df_train.to_parquet(train_path, engine="pyarrow", index=False)
            df_test.to_parquet(test_path,  engine="pyarrow", index=False)

            # به‌روزرسانی session_state
            st.session_state["train_path"] = train_path
            st.session_state["test_path"]  = test_path

            st.success(
                f"✅ Train/Test ساخته شد.\n\n"
                f"- Train: `{train_path}` (ردیف‌ها: {len(df_train)})\n"
                f"- Test: `{test_path}` (ردیف‌ها: {len(df_test)})"
            )
        except Exception as e:
            st.error(f"❌ خطا در ساخت Train/Test: {e}")
            st.stop()

    # اگر هنوز Train وجود ندارد، ادامه‌ی تب را متوقف کن
    train_path = st.session_state.get("train_path") or paths.get("train", "")
    if not train_path or not os.path.exists(train_path):
        st.info("برای ادامه، ابتدا Train/Test بساز.")
        st.stop()

    # ---------------------------------------------------------
    # گام ۲: تنظیمات برچسب و فیچرها
    # ---------------------------------------------------------
        # ---------------------------------------------------------
    # گام ۲: انتخاب برچسب‌ها و فیچرها (با تفکیک ۱۰ دسته)
    # ---------------------------------------------------------
    st.subheader("گام ۲: انتخاب برچسب‌ها و فیچرها")

    try:
        df_train = pd.read_parquet(train_path)
    except Exception as e:
        st.error(f"خطا در خواندن فایل Train: {e}")
        st.stop()

        # ---------------------------
    # 2-1) انتخاب ستون‌های برچسب (y) – چندتایی
    # ---------------------------
    label_cols_all = [
        c
        for c in df_train.columns
        if c.startswith("label_") and pd.api.types.is_integer_dtype(df_train[c])
    ]
    if not label_cols_all:
        st.error("هیچ ستونی با نام label_ در Train پیدا نشد. ابتدا در تب «🏷 افزودن برچسب‌ها» برچسب‌گذاری کن.")
        st.stop()

    # حالا به‌جای انتخاب یک برچسب، چندتایی انتخاب می‌کنیم
    label_cols = st.multiselect(
        "ستون‌های برچسب (y) برای آموزش را انتخاب کن (می‌توانی چندتا را با هم انتخاب کنی):",
        options=label_cols_all,
        default=label_cols_all,  # پیش‌فرض: همه
    )

    if not label_cols:
        st.warning("هیچ برچسبی انتخاب نشده است.")
        st.stop()

    # برای نمایش صرفاً نمونه‌ای از مقادیر یکتا، از اولین برچسب انتخاب‌شده استفاده می‌کنیم
    sample_label = label_cols[0]
    uniq_vals = sorted(df_train[sample_label].dropna().unique().tolist())
    st.caption(f"نمونه مقادیر یکتا در `{sample_label}`: {uniq_vals}")


    # ---------------------------
    # 2-2) پیدا کردن ستون‌های عددی کاندید X
    # ---------------------------
    import pandas as pd

    exclude_labels = set(label_cols_all)

    numeric_all = []
    for c in df_train.columns:
        if c in exclude_labels:
            continue
        if not pd.api.types.is_numeric_dtype(df_train[c]):
            continue
        numeric_all.append(c)

    def is_price_level_col(name: str) -> bool:
        """
        ستون‌هایی که به‌عنوان «سطح قیمت خام» حذف می‌کنیم:
        ❌ OHLC
        ❌ قیمت‌های TP/SL
        ❌ SMA/EMA کلاسیک
        ❌ MAهای سفارشی (ma1_..., ma2_...) که rank نیستند
        ❌ Bollinger_up/down
        ⚠️ هیچ‌کدام از ستون‌های حجم (real_volume, tick_volume, volume_...) اینجا حذف نمی‌شوند.
        """
        n = name.lower()

        # OHLC اصلی
        if n in {"open", "high", "low", "close"}:
            return True

        # قیمت TP/SL (مثلا m1_tp_price, m2_sl_price, ...)
        if n.endswith("_price") or "tp_price" in n or "sl_price" in n:
            return True

        # میانگین‌ها: فقط SMA/EMA/MA حذف شوند (شرط ۲️⃣)
        if n.startswith("sma_") or n.startswith("ema_"):
            return True

        # MAهای سفارشی؛ فقط خود MA (اعداد)، نه رتبه‌ها
        if n.startswith("ma") and "rank" not in n:
            # مثال: ma1_ema_15, ma2_sma_30, ma1_ema_15_tf_15m
            return True

        # فقط Bollinger_up / Bollinger_down حذف شوند (شرط ۳️⃣)
        if "bollinger_up" in n or "bollinger_down" in n:
            return True

        # ❗ هیچ چیز مربوط به volume اینجا حذف نمی‌شود (شرط ۴️⃣)
        return False

    numeric_filtered = [c for c in numeric_all if not is_price_level_col(c)]

    if not numeric_filtered:
        st.error("بعد از حذف ستون‌های قیمتی خام و MAهای عددی، هیچ ستون عددی برای فیچر باقی نمانده!")
        st.stop()

    # ---------------------------
    # 2-3) گروه‌بندی ستون‌ها در ۱۰ دسته
    # ---------------------------
        # ---------------------------
    # 2-3) گروه‌بندی ستون‌ها در ۱۰ دسته
    # ---------------------------
    basic_cols = []
    momentum_cols = []
    vol_cols = []
    vol_act_cols = []
    ratio_cols = []
    stat_cols = []
    time_cols = []
    ma_base_rank_cols = []
    ma_htf_rank_cols = []
    pattern_cols = []
    other_cols = []

    for c in numeric_filtered:
        cl = c.lower()

        # 1) فیچرهای پایه‌ای (Basic) – همه بماند
        if (
            cl.startswith("return_")
            or c in [
                "price_change",
                "high_low_range",
                "body_size",
                "upper_shadow",
                "lower_shadow",
                "candle_ratio",
            ]
        ):
            basic_cols.append(c)

        # 2) فیچرهای حرکتی / مومنتوم – همه به‌جز میانگین‌ها (که قبلاً در is_price_level_col حذف شده‌اند)
        elif (
            cl.startswith("macd")          # MACD, MACD_signal, ...
            or cl.startswith("rsi")        # RSI, rsi_14, ...
            or cl.startswith("momentum")   # momentum_*
            or cl.startswith("roc")        # ROC_*
            or cl.startswith("adx")        # ADX_*
            or cl.startswith("+di")        # +DI_*
            or cl.startswith("-di")        # -DI_*
            or cl.startswith("cci")        # CCI_*
            or "williams" in cl           # Williams %R
        ):
            momentum_cols.append(c)

        # 3) فیچرهای نوسان (Volatility) – همه به‌جز Bollinger_up/down (که قبلاً حذف شده‌اند)
        elif (
            cl.startswith("rolling_std")   # rolling_std_*
            or cl.startswith("atr")        # ATR, atr_14, ...
            or cl.startswith("hv_")        # HV_*
            or "hist_vol" in cl            # اگر اسم دیگه‌ای برای HV داشته باشی
            or cl == "rs"                  # RS (نسبت دامنه به میانگین)
        ):
            vol_cols.append(c)

        

        # 5) فیچرهای نسبتی (Ratios) – همه بماند
        elif cl in {
            "high_close_ratio",
            "low_close_ratio",
            "open_close_ratio",
            "hl_to_oc_ratio",
        }:
            ratio_cols.append(c)

        # 6) فیچرهای آماری روی log-returns – همه بماند
        elif cl.startswith("skew_") or cl.startswith("kurtosis_") or cl.startswith("auto_corr_"):
            stat_cols.append(c)

        # 7) فیچرهای زمانی (Time-based) – همه بماند
        elif cl in {
            "day_of_week",
            "hour_of_day",
            "is_month_start",
            "is_month_end",
            "days_since_high",
            "days_since_low",
            "sin_time",
            "cos_time",
        }:
            time_cols.append(c)

        # 8) ده مووینگ اوریج سفارشی – فقط رتبه‌ها بمانند
        elif "rank" in cl and "tf_" not in cl:
            # مثال: MA0_rank, MA1_rank, ...
            ma_base_rank_cols.append(c)

        # 9) رتبه‌های MA روی تایم‌فریم‌های بالاتر – فقط رتبه‌ها
        elif "rank" in cl and "tf_" in cl:
            # مثال: MA0_rank_TF_15m, ...
            ma_htf_rank_cols.append(c)

        # 10) برچسب الگوهای کندلی – همه بماند
        elif cl.startswith("candle_pattern_"):
            pattern_cols.append(c)

        else:
            other_cols.append(c)

    # ---------------------------
    # 2-4) UI انتخاب فیچرها به‌صورت ۱۰ دسته چک‌لیستی
    # ---------------------------
    st.markdown("### انتخاب گروه‌های فیچر (X)")

    st.caption(
        "ستون‌های قیمتیِ خام (OHLC)، قیمت‌های TP/SL، خطوط SMA/EMA و MAهای عددی و Bollinger_up/down "
        "حذف شده‌اند. بقیه‌ی ستون‌ها در ۱۰ گروه زیر قابل انتخاب‌اند؛ به‌صورت پیش‌فرض همه‌ی گروه‌های اصلی تیک خورده‌اند."
    )

    def group_selector(title: str, cols: list, key_prefix: str, default_on: bool = True):
        """یک چک‌باکس برای فعال/غیرفعال کردن گروه + یک multiselect برای ستون‌های داخل گروه."""
        if not cols:
            return []

        with st.expander(f"{title}  ({len(cols)} ستون)", expanded=True):
            enabled = st.checkbox(
                "فعال باشد؟",
                value=default_on,
                key=f"{key_prefix}_enabled",
            )
            if not enabled:
                return []

            selected = st.multiselect(
                "ستون‌های این گروه:",
                options=sorted(cols),
                default=sorted(cols),
                key=f"{key_prefix}_cols",
            )
            return selected

    selected_features = []

    # 1️⃣ Basic
    selected_features += group_selector("۱️⃣ فیچرهای پایه‌ای (Basic)", basic_cols, "basic", True)

    # 2️⃣ Momentum (فقط غیر از میانگین‌ها)
    selected_features += group_selector("۲️⃣ فیچرهای حرکتی / مومنتوم", momentum_cols, "momentum", True)

    # 3️⃣ Volatility (بدون Bollinger_up/down)
    selected_features += group_selector("۳️⃣ فیچرهای نوسان (Volatility)", vol_cols, "vol", True)

    # 4️⃣ Volume / Activity (همه باقی می‌مانند)
    selected_features += group_selector("۴️⃣ فیچرهای حجم / فعالیت (Volume / Activity)", vol_act_cols, "volact", True)

    # 5️⃣ Ratios
    selected_features += group_selector("۵️⃣ فیچرهای نسبتی (Ratios)", ratio_cols, "ratios", True)

    # 6️⃣ Statistical
    selected_features += group_selector("۶️⃣ فیچرهای آماری روی log-returns (GPU)", stat_cols, "stats", True)

    # 7️⃣ Time-based
    selected_features += group_selector("۷️⃣ فیچرهای زمانی (Time-Based)", time_cols, "timefeat", True)

    # 8️⃣ MA Base TF Ranks
    selected_features += group_selector("۸️⃣ رتبه‌بندی ۱۰ مووینگ اوریج سفارشی (Base TF)", ma_base_rank_cols, "mabase", True)

    # 9️⃣ MA Higher TF Ranks
    selected_features += group_selector("۹️⃣ رتبه‌بندی مووینگ اوریج‌ها روی تایم‌فریم‌های بالاتر", ma_htf_rank_cols, "mahtf", True)

    # 🔟 Candle Patterns
    selected_features += group_selector("🔟 برچسب الگوهای کندلی (Candle Patterns)", pattern_cols, "patterns", True)

    # سایر ستون‌های عددی که در هیچ گروهی قرار نگرفتند
    if other_cols:
        selected_features += group_selector("➕ سایر ستون‌های عددی (Other)", other_cols, "others", False)

    # حذف تکراری‌ها و مرتب‌سازی
    feature_cols = sorted(set(selected_features))

    if not feature_cols:
        st.error("هیچ فیچری انتخاب نشده است. حداقل یک ستون را در یکی از گروه‌ها فعال کن.")
        st.stop()

    st.success(f"تعداد فیچرهای انتخاب‌شده: {len(feature_cols)}")


    # ---------------------------------------------------------
    # گام ۳: تنظیمات سکانسی و هایپرفرمت‌ها
    # ---------------------------------------------------------
    st.subheader("گام ۳: تنظیمات توالی و هایپرفرمت‌ها")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        window = st.number_input("طول پنجره سکانسی (window)", min_value=4, max_value=1024, value=64, step=1)
    with c2:
        stride = st.number_input("گام پنجره (stride)", min_value=1, max_value=256, value=1, step=1)
    with c3:
        target_shift = st.number_input("فاصله پیش‌بینی (target_shift)", min_value=0, max_value=512, value=0, step=1)
    with c4:
        batch_size = st.number_input("Batch size", min_value=16, max_value=4096, value=256, step=16)

    d1, d2, d3, d4 = st.columns(4)
    with d1:
        epochs = st.number_input("Epochs", min_value=1, max_value=500, value=10, step=1)
    with d2:
        lr = st.number_input("Learning rate", min_value=1e-5, max_value=1e-1, value=1e-3, step=1e-5, format="%.5f")
    with d3:
        hidden_dim = st.number_input("Dimension پنهان (برای اکثر مدل‌ها)", min_value=8, max_value=1024, value=128, step=8)
    with d4:
        n_heads = st.number_input("تعداد سرهای Transformer", min_value=1, max_value=16, value=4, step=1)

    st.subheader("انتخاب مدل‌ها (Training Only)")
    model_names = [
        "MLP",
        "AutoencoderMLP",
        "CNN1D",
        "LSTM",
        "GRU",
        "DeepAR",          # 🔹 مدل جدید
        "TCN",
        "Transformer",      # مدل قدیمی خودت
        "TransformerV2",    # مدل قوی‌تر جدید
        "CNNLSTM",
        "CNNTransformer",
        "WaveNet",
        "NBEATS",
        "TimesNet",        # 🔥 مدل جدید TimesNet
        "PatchTST",
        "TFT",
    ]

    selected_models = st.multiselect("مدل‌ها", options=model_names, default=model_names)



    # ---------- Torch device ----------
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    st.info(f"دستگاه اجرا: **{device_str.upper()}**")
    device = torch.device(device_str)

    # ---------------------------------------------------------
    # Dataset classes
    # ---------------------------------------------------------
    class TabularDataset(torch.utils.data.Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.long)

        def __len__(self):
            return self.X.shape[0]

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]


    class SequenceDataset(torch.utils.data.Dataset):
        """
        تولید سکانس‌ها روی پرواز:
        - X: (N, D)
        - window: طول پنجره
        - stride: گام حرکت پنجره
        - target_shift: برچسب از index: i + window - 1 + target_shift
        """
        def __init__(self, X, y, window: int, stride: int = 1, target_shift: int = 0):
            self.X = torch.tensor(X, dtype=torch.float32)  # روی CPU
            self.y = torch.tensor(y, dtype=torch.long)
            self.window = int(window)
            self.stride = max(1, int(stride))
            self.target_shift = int(target_shift)

            self.N = self.X.shape[0]
            max_start = self.N - self.window - self.target_shift
            self.nseq = 0 if max_start < 0 else (max_start // self.stride + 1)

        def __len__(self):
            return self.nseq

        def __getitem__(self, idx):
            i = idx * self.stride
            x = self.X[i : i + self.window]              # (W, D)
            y_idx = i + self.window - 1 + self.target_shift
            if y_idx >= self.N:
                y_idx = self.N - 1
            y = self.y[y_idx]
            return x, y

    # ---------------------------------------------------------
    # مدل‌ها
    # ---------------------------------------------------------
    class MLP(nn.Module):
        def __init__(self, in_dim, hidden_dim, nclass):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, nclass),
            )

        def forward(self, x):
            return self.net(x)


    class CNN1DModel(nn.Module):
        def __init__(self, in_dim, hidden_dim, nclass):
            super().__init__()
            c1 = hidden_dim
            self.conv1 = nn.Conv1d(in_dim, c1, kernel_size=5, padding=2)
            self.conv2 = nn.Conv1d(c1, c1, kernel_size=5, padding=2)
            self.pool  = nn.AdaptiveMaxPool1d(1)
            self.fc    = nn.Linear(c1, nclass)

        def forward(self, x):
            # x: (B, W, D) → (B, D, W)
            x = x.transpose(1, 2)
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = self.pool(x).squeeze(-1)
            return self.fc(x)


    class RNNBase(nn.Module):
        def __init__(self, in_dim, hidden_dim, nclass, kind="LSTM"):
            super().__init__()
            if kind == "LSTM":
                self.rnn = nn.LSTM(in_dim, hidden_dim, batch_first=True)
            elif kind == "GRU":
                self.rnn = nn.GRU(in_dim, hidden_dim, batch_first=True)
            else:
                raise ValueError("kind should be LSTM or GRU")
            self.fc = nn.Linear(hidden_dim, nclass)

        def forward(self, x):
            # x: (B, W, D)
            out, _ = self.rnn(x)
            h_last = out[:, -1, :]
            return self.fc(h_last)


    class LSTMModel(RNNBase):
        def __init__(self, in_dim, hidden_dim, nclass):
            super().__init__(in_dim, hidden_dim, nclass, kind="LSTM")


    class GRUModel(RNNBase):
        def __init__(self, in_dim, hidden_dim, nclass):
            super().__init__(in_dim, hidden_dim, nclass, kind="GRU")


    class TCNBlock(nn.Module):
        def __init__(self, ch_in, ch_out, kernel_size=3, dilation=1):
            super().__init__()
            padding = (kernel_size - 1) * dilation
            self.conv1 = nn.Conv1d(ch_in, ch_out, kernel_size, padding=padding, dilation=dilation)
            self.conv2 = nn.Conv1d(ch_out, ch_out, kernel_size, padding=padding, dilation=dilation)
            self.down  = nn.Conv1d(ch_in, ch_out, 1) if ch_in != ch_out else nn.Identity()
            self.relu  = nn.ReLU()

        def forward(self, x):
            # x: (B, C, L)
            out = self.conv1(x)
            out = self.relu(out)
            out = self.conv2(out)
            # برش به طول ورودی (برای درست شدن ابعاد residual)
            if out.size(2) != x.size(2):
                diff = out.size(2) - x.size(2)
                if diff > 0:
                    out = out[:, :, :-diff]
                else:
                    out = F.pad(out, (0, -diff))
            res = self.down(x)
            if res.size(2) != out.size(2):
                diff = res.size(2) - out.size(2)
                if diff > 0:
                    res = res[:, :, :-diff]
                else:
                    res = F.pad(res, (0, -diff))
            return self.relu(out + res)


    class TCNModel(nn.Module):
        def __init__(self, in_dim, hidden_dim, nclass, levels=3, kernel_size=3):
            super().__init__()
            chs = [in_dim] + [hidden_dim] * levels
            blocks = []
            for i in range(levels):
                blocks.append(
                    TCNBlock(
                        ch_in=chs[i],
                        ch_out=chs[i+1],
                        kernel_size=kernel_size,
                        dilation=2**i,
                    )
                )
            self.net = nn.Sequential(*blocks)
            self.pool = nn.AdaptiveMaxPool1d(1)
            self.fc   = nn.Linear(hidden_dim, nclass)

        def forward(self, x):
            # x: (B, W, D) → (B, D, W)
            x = x.transpose(1, 2)
            x = self.net(x)
            x = self.pool(x).squeeze(-1)
            return self.fc(x)


    class TransformerModel(nn.Module):
        def __init__(self, in_dim, hidden_dim, n_heads, n_layers, nclass):
            super().__init__()
            self.proj = nn.Linear(in_dim, hidden_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=n_heads,
                dim_feedforward=hidden_dim * 4,
                batch_first=True,
            )
            self.enc = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.fc  = nn.Linear(hidden_dim, nclass)

        def forward(self, x):
            # x: (B, W, D)
            x = self.proj(x)
            x = self.enc(x)
            x = x[:, -1, :]
            return self.fc(x)

    class CNNLSTMModel(nn.Module):
        """
        ترکیب CNN یک‌بعدی روی توالی + LSTM
        ورودی: (B, W, D)
        """
        def __init__(self, in_dim, hidden_dim, nclass):
            super().__init__()
            # ابتدا کانولوشن روی بعد زمان با کانال‌های in_dim
            self.conv1 = nn.Conv1d(in_dim, hidden_dim, kernel_size=5, padding=2)
            self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
            # سپس LSTM روی ویژگی‌های استخراج‌شده
            self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
            self.fc   = nn.Linear(hidden_dim, nclass)

        def forward(self, x):
            # x: (B, W, D)
            # CNN روی (B, D, W)
            x = x.transpose(1, 2)  # (B, D, W)
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = x.transpose(1, 2)  # (B, W, hidden_dim)
            # LSTM روی توالی
            out, _ = self.lstm(x)
            h_last = out[:, -1, :]
            return self.fc(h_last)


    class CNNTransformerModel(nn.Module):
        """
        ترکیب CNN یک‌بعدی + Transformer
        ورودی: (B, W, D)
        """
        def __init__(self, in_dim, hidden_dim, n_heads, n_layers, nclass):
            super().__init__()
            self.conv1 = nn.Conv1d(in_dim, hidden_dim, kernel_size=5, padding=2)
            self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=n_heads,
                dim_feedforward=hidden_dim * 4,
                batch_first=True,
            )
            self.enc = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.fc  = nn.Linear(hidden_dim, nclass)

        def forward(self, x):
            # x: (B, W, D)
            x = x.transpose(1, 2)          # (B, D, W)
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = x.transpose(1, 2)          # (B, W, hidden_dim)
            x = self.enc(x)                # (B, W, hidden_dim)
            x = x[:, -1, :]                # آخرین تایم‌استپ
            return self.fc(x)


    class AutoencoderMLPModel(nn.Module):
        """
        MLP با bottleneck شبیه اتوانکودر (برای داده‌ی جدولی / non-sequence)
        ورودی: (B, D)
        """
        def __init__(self, in_dim, hidden_dim, nclass):
            super().__init__()
            bottleneck = max(hidden_dim // 2, 8)

            self.encoder = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, bottleneck),
                nn.ReLU(),
            )
            # اینجا فقط از کد فشرده‌شده برای کلاسیفیکیشن استفاده می‌کنیم
            self.classifier = nn.Linear(bottleneck, nclass)

        def forward(self, x):
            # x: (B, D)
            z = self.encoder(x)
            logits = self.classifier(z)
            return logits

    class WaveNetBlock(nn.Module):
        """
        یک بلوک ساده WaveNet با طول خروجی برابر طول ورودی (same length)
        از kernel_size=3 و padding=dilation استفاده می‌کنیم تا طول ثابت بماند.
        ورودی و خروجی: (B, C, L)
        """
        def __init__(self, channels: int, dilation: int):
            super().__init__()
            kernel_size = 3
            padding = dilation  # برای k=3 → out_len = L + 2*d - d*2 = L

            self.conv_filter = nn.Conv1d(
                channels, channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
            )
            self.conv_gate = nn.Conv1d(
                channels, channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
            )
            self.conv_res = nn.Conv1d(channels, channels, kernel_size=1)
            self.conv_skip = nn.Conv1d(channels, channels, kernel_size=1)

        def forward(self, x):
            # x: (B, C, L)
            f = torch.tanh(self.conv_filter(x))
            g = torch.sigmoid(self.conv_gate(x))
            z = f * g
            res = self.conv_res(z)
            skip = self.conv_skip(z)

            # طول خروجی convها دقیقاً برابر طول ورودی است → نیازی به crop نیست
            out = x + res   # residual connection
            return out, skip


    class WaveNetModel(nn.Module):
        """
        مدل WaveNet ساده برای کلاسیفیکیشن روی توالی
        ورودی: (B, W, D)
        """
        def __init__(self, in_dim, hidden_dim, nclass, n_layers: int = 6):
            super().__init__()
            self.input_conv = nn.Conv1d(in_dim, hidden_dim, kernel_size=1)

            blocks = []
            for i in range(n_layers):
                dilation = 2 ** i
                blocks.append(WaveNetBlock(hidden_dim, dilation=dilation))
            self.blocks = nn.ModuleList(blocks)

            self.out_conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
            self.out_conv2 = nn.Conv1d(hidden_dim, nclass, kernel_size=1)

        def forward(self, x):
            # x: (B, W, D)
            x = x.transpose(1, 2)        # (B, D, W) → (B, C, L)
            x = self.input_conv(x)       # (B, hidden_dim, L)

            skip_total = 0
            for block in self.blocks:
                x, skip = block(x)       # هر دو شکل (B, C, L) دارند
                skip_total = skip_total + skip  # broadcast اولین بار از 0 → بدون مشکل

            out = F.relu(skip_total)
            out = F.relu(self.out_conv1(out))
            out = self.out_conv2(out)    # (B, nclass, L)
            out = out.mean(dim=-1)       # global average over time → (B, nclass)
            return out


    class TransformerV2Model(nn.Module):
        """
        نسخه قوی‌تر ترنسفورمر:
        - ابتدا نگاشت خطی از in_dim → hidden_dim
        - سپس چند لایه TransformerEncoder
        - خروجی از آخرین تایم‌استپ
        ورودی: (B, W, D)
        """
        def __init__(self, in_dim, hidden_dim, n_heads, n_layers, nclass):
            super().__init__()
            self.proj = nn.Linear(in_dim, hidden_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=n_heads,
                dim_feedforward=hidden_dim * 4,
                batch_first=True,
            )
            self.enc = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.fc  = nn.Linear(hidden_dim, nclass)

        def forward(self, x):
            # x: (B, W, D)
            x = self.proj(x)      # (B, W, hidden_dim)
            x = self.enc(x)       # (B, W, hidden_dim)
            x = x[:, -1, :]       # آخرین تایم‌استپ
            return self.fc(x)     # (B, nclass)

    class NBeatsBlock(nn.Module):
        """
        یک بلوک ساده N-BEATS:
        - ورودی: (B, input_size)
        - خروجی: backcast (هم‌اندازه ورودی) و نمایه‌ای به نام theta (برای ساخت ویژگی نهایی)
        """
        def __init__(self, input_size: int, hidden_dim: int, theta_size: int):
            super().__init__()
            layers = []
            for i in range(4):
                in_features = input_size if i == 0 else hidden_dim
                layers.append(nn.Linear(in_features, hidden_dim))
                layers.append(nn.ReLU())
            self.fc = nn.Sequential(*layers)

            self.backcast_fc = nn.Linear(hidden_dim, input_size)
            self.theta_fc    = nn.Linear(hidden_dim, theta_size)

        def forward(self, x):
            # x: (B, input_size)
            h = self.fc(x)
            backcast = self.backcast_fc(h)   # بخش بازسازی‌شده ورودی
            theta    = self.theta_fc(h)      # بردار ویژگی برای پیش‌بینی
            x_res = x - backcast             # شبیه N-BEATS: باقیمانده
            return x_res, theta

    class NBeatsModel(nn.Module):
        """
        نسخه ساده‌شده N-BEATS برای کلاسیفیکیشن:
        - ورودی: سکانس (B, W, D)
        - ابتدا سکانس را به بردار (B, W*D) فلت می‌کنیم
        - چند بلوک NBeatsBlock و در نهایت کلاسیفیکیشن
        """
        def __init__(self, in_dim: int, window: int, hidden_dim: int, n_blocks: int, nclass: int):
            super().__init__()
            self.window = int(window)
            self.in_dim = int(in_dim)
            self.input_size = self.window * self.in_dim

            self.blocks = nn.ModuleList(
                [NBeatsBlock(self.input_size, hidden_dim, hidden_dim) for _ in range(n_blocks)]
            )
            self.fc_out = nn.Linear(hidden_dim, nclass)

        def forward(self, x):
            # x: (B, W, D)
            B, W, D = x.shape
            # اگر به هر دلیل W کمی فرق داشته باشد، آخرین window را برمی‌داریم
            if W != self.window:
                if W < self.window:
                    # اگر توالی کوتاه‌تر بود، از جلو صفرپد می‌کنیم
                    pad = self.window - W
                    x = F.pad(x, (0, 0, pad, 0))  # روی محور زمان پد می‌کنیم
                else:
                    x = x[:, -self.window :, :]
                B, W, D = x.shape

            # فلت کردن توالی
            flat = x.reshape(B, -1)  # (B, window * in_dim)

            backcast = flat
            theta_sum = 0
            for block in self.blocks:
                backcast, theta = block(backcast)
                theta_sum = theta_sum + theta  # جمع ویژگی‌ها

            h = theta_sum  # (B, hidden_dim)
            logits = self.fc_out(h)  # (B, nclass)
            return logits   

    class DeepARModel(nn.Module):
        """
        نسخه ساده‌شده از ایده DeepAR برای کلاسیفیکیشن:
        - چند لایه LSTM روی توالی
        - استفاده از آخرین حالت برای کلاسیفیکیشن
        """
        def __init__(self, in_dim: int, hidden_dim: int, nclass: int, n_layers: int = 2, dropout: float = 0.1):
            super().__init__()
            self.rnn = nn.LSTM(
                input_size=in_dim,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                batch_first=True,
                dropout=dropout if n_layers > 1 else 0.0,
            )
            self.fc = nn.Linear(hidden_dim, nclass)

        def forward(self, x):
            # x: (B, W, D)
            out, _ = self.rnn(x)          # out: (B, W, H)
            h_last = out[:, -1, :]        # (B, H)
            logits = self.fc(h_last)      # (B, C)
            return logits

    class TFTModel(nn.Module):
        """
        نسخهٔ ساده‌شده‌ی Temporal Fusion Transformer
        ورودی: x با شکل (batch, seq_len, n_features)
        خروجی: logits با شکل (batch, nclass)
        """
        def __init__(
            self,
            in_dim: int,
            hidden_dim: int,
            n_heads: int,
            n_layers: int,
            nclass: int,
            dropout: float = 0.1,
        ):
            super().__init__()

            self.input_proj = nn.Linear(in_dim, hidden_dim)

            # LSTM encoder
            self.lstm = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                batch_first=True,
                bidirectional=False,
            )

            # Multi-head attention روی خروجی LSTM
            self.attn = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=n_heads,
                dropout=dropout,
                batch_first=True,
            )

            # GRN ساده (FFN + GLU) شبیه TFT
            self.ffn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GLU(),                      # gate
                nn.Linear(hidden_dim, hidden_dim),
            )

            self.dropout = nn.Dropout(dropout)
            self.ln1 = nn.LayerNorm(hidden_dim)
            self.ln2 = nn.LayerNorm(hidden_dim)

            self.out = nn.Linear(hidden_dim, nclass)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            x: (B, T, F)
            """
            # 1) پروجکشن اولیه
            h = self.input_proj(x)           # (B, T, H)

            # 2) LSTM encoder
            h, _ = self.lstm(h)              # (B, T, H)

            # 3) self-attention
            attn_out, _ = self.attn(h, h, h) # (B, T, H)

            # 4) residual + layernorm
            h = self.ln1(h + self.dropout(attn_out))

            # 5) GRN / FFN + residual
            f = self.ffn(h)                  # (B, T, H)
            h = self.ln2(h + self.dropout(f))

            # 6) استفاده از آخرین تایم‌استپ برای کلاس‌بندی
            last = h[:, -1, :]               # (B, H)

            logits = self.out(last)          # (B, nclass)
            return logits


    # --------------------------- TimesNet ساده ---------------------------
    class TimesBlock(nn.Module):
        """
        یک بلوک ساده TimesNet-style:
        - کانولوشن روی محور زمان با dilation (multi-scale)
        - residual + normalization
        ورودی/خروجی: (B, C, L)
        """
        def __init__(self, channels: int, expansion: int = 2, kernel_size: int = 3,
                     dilation: int = 1, dropout: float = 0.1):
            super().__init__()
            padding = (kernel_size - 1) // 2 * dilation
            self.conv = nn.Conv1d(
                channels,
                channels * expansion,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
            )
            self.proj = nn.Conv1d(channels * expansion, channels, kernel_size=1)
            self.norm = nn.BatchNorm1d(channels)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            # x: (B, C, L)
            z = self.conv(x)
            z = F.gelu(z)
            z = self.proj(z)
            z = self.dropout(z)
            out = x + z
            out = self.norm(out)
            return out

    class TimesNetModel(nn.Module):
        """
        TimesNet ساده برای کلاسیفیکیشن روی توالی
        ورودی: (B, W, D)
        - نگاشت خطی D → hidden_dim
        - چند TimesBlock با dilation متفاوت
        - گرفتن آخرین تایم‌استپ برای کلاسیفیکیشن
        """
        def __init__(self, in_dim: int, hidden_dim: int, n_blocks: int, nclass: int, dropout: float = 0.1):
            super().__init__()
            self.proj = nn.Linear(in_dim, hidden_dim)

            blocks = []
            for i in range(n_blocks):
                dilation = 2 ** i
                blocks.append(
                    TimesBlock(
                        channels=hidden_dim,
                        expansion=2,
                        kernel_size=3,
                        dilation=dilation,
                        dropout=dropout,
                    )
                )
            self.blocks = nn.ModuleList(blocks)
            self.ln = nn.LayerNorm(hidden_dim)
            self.fc = nn.Linear(hidden_dim, nclass)

        def forward(self, x):
            # x: (B, W, D)
            B, W, D = x.shape
            h = self.proj(x)         # (B, W, H)
            h = h.transpose(1, 2)    # (B, H, W) برای Conv1d

            for blk in self.blocks:
                h = blk(h)           # (B, H, W)

            h = h.transpose(1, 2)    # (B, W, H)
            h_last = h[:, -1, :]     # آخرین تایم‌استپ
            h_last = self.ln(h_last)
            logits = self.fc(h_last) # (B, nclass)
            return logits

    # ---------------------------------------------
    # PatchTST - Patch-based Transformer for Time Series
    # ---------------------------------------------
    class PatchTST(nn.Module):
        def __init__(self, input_dim, patch_len=16, stride=8, d_model=64,
                    n_heads=4, n_layers=2, n_classes=3):
            super().__init__()

            self.patch_len = patch_len
            self.stride = stride

            # تبدیل توالی به پچ های پشت سر هم
            self.proj = nn.Linear(input_dim * patch_len, d_model)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                batch_first=True
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

            self.head = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(d_model, n_classes)
            )

        def create_patches(self, x):
            """
            x: (B, seq_len, features)
            تبدیل به پچ‌های (B, num_patches, features*patch_len)
            """
            B, T, F = x.size()
            patches = []

            for start in range(0, T - self.patch_len + 1, self.stride):
                end = start + self.patch_len
                patch = x[:, start:end, :].reshape(B, -1)
                patches.append(patch)

            patches = torch.stack(patches, dim=1)  # (B, P, patch_len*F)
            return patches

        def forward(self, x):
            # پچ‌سازی
            patches = self.create_patches(x)

            # پروجکشن
            z = self.proj(patches)

            # ترنسفورمر
            z = self.encoder(z)

            # خروجی
            z = z.transpose(1, 2)  # برای AdaptiveAvgPool1d
            out = self.head(z)
            return out


    # ---------------------------------------------------------
    # Standardization helpers
    # ---------------------------------------------------------
    def standardize_fit(X: np.ndarray):
        mean = X.mean(axis=0, keepdims=True)
        std  = X.std(axis=0, keepdims=True) + 1e-8
        return mean.astype("float32"), std.astype("float32")

    def standardize_apply(X: np.ndarray, mean: np.ndarray, std: np.ndarray):
        return (X - mean) / std

    # ---------------------------------------------------------
    # Loop آموزش
    # ---------------------------------------------------------
    def train_loop(model, loader, epochs: int, device, lr: float, label_name: str, model_name: str):
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for ep in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            total_n = 0
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)

                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * yb.size(0)
                total_n += yb.size(0)

            mean_loss = total_loss / max(total_n, 1)
            st.write(f"{label_name} — {model_name} | Epoch {ep}/{epochs} | loss={mean_loss:.4f}")

    # ---------------------------------------------------------
    # دکمه آموزش
    # ---------------------------------------------------------
    models_root = Path(st.session_state.get("models_dir") or paths.get("models_dir", "models"))
    models_root.mkdir(parents=True, exist_ok=True)

    if st.button("🚀 آموزش و ذخیره مدل‌ها روی Train"):
        with st.status("در حال آموزش مدل‌ها ...", expanded=True) as status:
            try:
                # آماده‌سازی X کامل (Train)
                df_all = df_train.copy()
                X_full = df_all[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(np.float32)

                # اسکالر کلی برای همین Train
                mean_all, std_all = standardize_fit(X_full)
                Xz_full = standardize_apply(X_full, mean_all, std_all)

                for lab in label_cols:
                    status.write(f"▶️ برچسب: {lab}")
                    y_raw = df_all[lab].to_numpy()

                    # نگاشت برچسب‌ها به 0..C-1
                    vals = np.unique(y_raw[~pd.isna(y_raw)])
                    vals = vals[~np.isnan(vals)]
                    vals = np.sort(vals.astype(int))
                    if len(vals) < 2:
                        status.write(f"⚠️ {lab}: تعداد کلاس‌ها کمتر از 2 است، رد شد.")
                        continue

                    label2idx = {int(v): i for i, v in enumerate(vals)}
                    idx2label = {i: int(v) for i, v in enumerate(vals)}
                    y_idx = np.array([label2idx.get(int(v), -1) for v in y_raw], dtype=int)
                    mask = y_idx >= 0

                    Xz = Xz_full[mask]
                    y_used = y_idx[mask]

                    if Xz.shape[0] < 100:
                        status.write(f"⚠️ {lab}: دادهٔ کافی برای آموزش نیست (کمتر از 100 رکورد).")
                        continue

                    nclass = len(vals)
                    in_dim = Xz.shape[1]

                    # DataLoaders
                    num_workers = 0  # برای ویندوز و جلوگیری از خطای worker
                    pin_mem = (device_str == "cuda")

                    for mname in selected_models:
                        status.write(f"🚀 {lab} — {mname}")

                        # ---- مدل‌های غیرتوالی (جدولی) ----
                        if mname in ("MLP", "AutoencoderMLP"):
                            ds = TabularDataset(Xz, y_used)
                            dl = torch.utils.data.DataLoader(
                                ds,
                                batch_size=int(batch_size),
                                shuffle=True,
                                num_workers=num_workers,
                                pin_memory=pin_mem,
                                drop_last=False,
                            )

                            if mname == "MLP":
                                model = MLP(in_dim, int(hidden_dim), nclass)
                            else:  # AutoencoderMLP
                                model = AutoencoderMLPModel(in_dim, int(hidden_dim), nclass)

                            is_seq = False

                        # ---- مدل‌های سکانسی (sequence) ----
                        else:
                            ds = SequenceDataset(
                                Xz, y_used,
                                window=int(window),
                                stride=int(stride),
                                target_shift=int(target_shift),
                            )
                            if len(ds) == 0:
                                status.write(
                                    f"⚠️ {lab} — {mname}: با تنظیمات window/stride/target_shift هیچ سکانسی ساخته نشد."
                                )
                                continue

                            dl = torch.utils.data.DataLoader(
                                ds,
                                batch_size=int(batch_size),
                                shuffle=True,
                                num_workers=num_workers,
                                pin_memory=pin_mem,
                                drop_last=False,
                            )

                            if mname == "CNN1D":
                                model = CNN1DModel(in_dim, int(hidden_dim), nclass)
                            elif mname == "LSTM":
                                model = LSTMModel(in_dim, int(hidden_dim), nclass)
                            elif mname == "GRU":
                                model = GRUModel(in_dim, int(hidden_dim), nclass)
                            elif mname == "DeepAR":
                                # 🔹 DeepAR روی همان داده‌ی سکانسی
                                model = DeepARModel(in_dim, int(hidden_dim), nclass)
                            elif mname == "TCN":
                                model = TCNModel(in_dim, int(hidden_dim), nclass)
                            elif mname == "Transformer":
                                # نسخه قدیمی خودت، بدون n_heads / n_layers
                                model = TransformerModel(in_dim, int(hidden_dim), int(n_heads), 2, nclass)
                            elif mname == "TransformerV2":
                                # نسخه جدید و قوی‌تر با head و لایه
                                model = TransformerV2Model(in_dim, int(hidden_dim), int(n_heads), n_layers=2, nclass=nclass)
                            elif mname == "CNNLSTM":
                                model = CNNLSTMModel(in_dim, int(hidden_dim), nclass)
                            elif mname == "CNNTransformer":
                                model = CNNTransformerModel(in_dim, int(hidden_dim), int(n_heads), n_layers=2, nclass=nclass)
                            elif mname == "WaveNet":
                                model = WaveNetModel(in_dim, int(hidden_dim), nclass)
                            elif mname == "NBEATS":
                                # ✅ N-BEATS روی توالی با window فعلی
                                model = NBeatsModel(
                                    in_dim=in_dim,
                                    window=int(window),
                                    hidden_dim=int(hidden_dim),
                                    n_blocks=4,
                                    nclass=nclass,
                                )
                            elif mname == "TimesNet":
                                # 🔥 TimesNet روی همان داده‌ی سکانسی
                                model = TimesNetModel(
                                    in_dim=int(in_dim),
                                    hidden_dim=int(hidden_dim),
                                    n_blocks=4,
                                    nclass=nclass,
                                )
                            elif mname == "PatchTST":
                                # فرض: window و n_heads و hidden_dim از UI گرفته شده‌اند
                                patch_len = min(int(window), 16)       # پچ حداکثر ۱۶ یا خود window
                                stride    = max(1, patch_len // 2)     # هم‌پوشانی نصف پچ

                                model = PatchTST(
                                    input_dim=in_dim,
                                    patch_len=patch_len,
                                    stride=stride,
                                    d_model=int(hidden_dim),
                                    n_heads=int(n_heads),
                                    n_layers=2,
                                    n_classes=nclass,
                                )
                            elif mname == "TFT":
                                # Temporal Fusion Transformer ساده
                                model = TFTModel(
                                    in_dim=in_dim,
                                    hidden_dim=int(hidden_dim),
                                    n_heads=int(n_heads),
                                    n_layers=2,      # اگر خواستی بعداً از UI قابل تنظیمش کنیم
                                    nclass=nclass,
                                )
                            else:
                                status.write(f"مدل ناشناخته: {mname}")
                                continue

                            is_seq = True

                        # -------- آموزش مشترک --------
                        train_loop(model, dl, int(epochs), device, float(lr), lab, mname)

                        # -------- ذخیره‌ی مدل و متادیتا --------
                        label_dir = models_root / lab
                        label_dir.mkdir(parents=True, exist_ok=True)
                        weight_pt = label_dir / f"{mname}.pt"
                        torch.save(model.state_dict(), weight_pt)

                        meta = {
                            "label_col": lab,
                            "model_name": mname,
                            "n_features": int(in_dim),
                            "n_classes": int(nclass),
                            "feature_cols": feature_cols,
                            "label2idx": label2idx,
                            "idx2label": idx2label,
                            "is_sequence": bool(is_seq),
                            "window": int(window) if is_seq else None,
                            "stride": int(stride) if is_seq else None,
                            "target_shift": int(target_shift) if is_seq else None,
                            "mean": mean_all.flatten().tolist(),
                            "std": std_all.flatten().tolist(),
                            "hidden_dim": int(hidden_dim),
                            "n_heads": int(n_heads),
                        }
                        meta_path = label_dir / f"{mname}_meta.json"
                        with meta_path.open("w", encoding="utf-8") as f:
                            json.dump(meta, f, ensure_ascii=False, indent=2)

                        status.write(f"💾 ذخیره شد: {weight_pt.name}")

                status.update(label="✅ آموزش مدل‌ها به پایان رسید و همه‌ی مدل‌های انتخاب‌شده ذخیره شدند.", state="complete")
            except Exception as e:
                status.update(label=f"❌ خطا در آموزش: {e}", state="error")
                st.exception(e)


def render_tab4_test_models():
    st.header("🧪 تست مدل‌ها، تحلیل و شبیه‌سازی سرمایه روی Test (GPU)")

    # ========================
    # ۱) مسیرها و دیتاست Test
    # ========================
    paths = ensure_paths_and_autoload()
    parquet_path = st.session_state.get("parquet_path") or paths.get("parquet", "")
    test_path = st.session_state.get("test_path") or paths.get("test", "")
    models_root = Path(st.session_state.get("models_dir") or paths.get("models_dir", "models"))
    models_root.mkdir(parents=True, exist_ok=True)

    if not test_path or not os.path.exists(test_path):
        st.warning("⚠️ هیچ فایل Test پیدا نشد. یک‌بار در تب «🧠 آموزش مدل‌ها» Train/Test را بساز.")
        return

    try:
        df_test = pd.read_parquet(test_path)
    except Exception as e:
        st.error(f"❌ خطا در خواندن فایل Test:\n{e}")
        return

    if df_test.empty:
        st.warning("⚠️ فایل Test خالی است.")
        return

    if "time" in df_test.columns:
        df_test["time"] = pd.to_datetime(df_test["time"], errors="coerce")

    # ========================
    # ۲) انتخاب برچسب‌ها و مدل‌ها
    # ========================
    label_cols = [c for c in df_test.columns if c.startswith("label_")]
    if not label_cols:
        st.error("❌ هیچ ستونی با پیشوند label_ در Test پیدا نشد.")
        return

    st.subheader("گام ۱: انتخاب برچسب‌ها و مدل‌ها")

    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        selected_labels = st.multiselect(
            "برچسب‌هایی که می‌خواهی تست کنی:",
            options=sorted(label_cols),
            default=sorted(label_cols),
            key="tab4_labels_multiselect",
        )

    # مدل‌های موجود روی دیسک برای هر label
    available_models_all = set()
    label_to_models_on_disk: dict[str, list[str]] = {}
    for lab in selected_labels:
        lab_dir = models_root / lab
        models_for_lab = []
        if lab_dir.exists():
            for fname in os.listdir(lab_dir):
                if fname.endswith("_meta.json"):
                    mname = fname.replace("_meta.json", "")
                    models_for_lab.append(mname)
                    available_models_all.add(mname)
        label_to_models_on_disk[lab] = sorted(models_for_lab)

    with col_sel2:
        selected_models = st.multiselect(
            "کدام مدل‌ها تست شوند؟ (فقط مدل‌هایی که واقعاً ذخیره شده‌اند)",
            options=sorted(available_models_all),
            default=sorted(available_models_all),
            key="tab4_models_multiselect",
        )

    if not selected_labels or not selected_models:
        st.info("حداقل یک برچسب و یک مدل انتخاب کن.")
        return

    # ========================
    # ۳) تنظیمات ریسک / حجم
    # ========================
    st.subheader("گام ۲: تنظیمات ریسک و مدیریت سرمایه (برای شبیه‌سازی بالانس)")

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        init_equity = st.number_input(
            "سرمایه اولیه (Equity Start)",
            min_value=10.0,
            max_value=1_000_000.0,
            value=1_000.0,
            step=100.0,
            key="tab4_init_equity",
        )

    with col_r2:
        risk_mode = st.radio(
            "روش مدیریت حجم:",
            options=["percent", "equity_based_lot"],
            format_func=lambda x: "درصد ریسک ثابت از بالانس" if x == "percent" else "لات پله‌ای بر اساس Equity",
            key="tab4_risk_mode",
        )

    col_r3, col_r4, col_r5 = st.columns(3)
    if risk_mode == "percent":
        with col_r3:
            risk_perc = st.number_input(
                "درصد ریسک هر معامله (%)",
                min_value=0.01,
                max_value=100.0,
                value=1.0,
                step=0.1,
                key="tab4_risk_perc",
            )
        base_equity = None
        base_lot = None
        with col_r4:
            pip_size = st.number_input(
                "pip size (مثلاً 0.1 برای XAUUSD)",
                min_value=0.00001,
                max_value=1.0,
                value=0.1,
                step=0.00001,
                key="tab4_pip_size_percent",
                format="%.5f",
            )
        with col_r5:
            pip_value_per_lot = st.number_input(
                "ارزش هر pip برای 1 lot",
                min_value=0.0001,
                max_value=10_000.0,
                value=10.0,
                step=0.1,
                key="tab4_pip_value_per_lot_percent",
            )
    else:
        risk_perc = None
        with col_r3:
            base_equity = st.number_input(
                "Equity مبنا برای افزایش حجم (مثلاً 1000$)",
                min_value=10.0,
                max_value=100_000.0,
                value=1_000.0,
                step=10.0,
                key="tab4_base_equity",
            )
        with col_r4:
            base_lot = st.number_input(
                "لات مبنا برای هر پله",
                min_value=0.001,
                max_value=100.0,
                value=0.01,
                step=0.001,
                key="tab4_base_lot",
            )
        with col_r5:
            pip_size = st.number_input(
                "pip size (مثلاً 0.1 برای XAUUSD)",
                min_value=0.00001,
                max_value=1.0,
                value=0.1,
                step=0.00001,
                key="tab4_pip_size_equity",
                format="%.5f",
            )
        pip_value_per_lot = st.number_input(
            "ارزش هر pip برای 1 lot",
            min_value=0.0001,
            max_value=10_000.0,
            value=10.0,
            step=0.1,
            key="tab4_pip_value_per_lot_equity",
        )

    # ========================
    # ۴) فیلتر under_allow
    # ========================
    st.subheader("گام ۳: فیلتر under_allow بر اساس confidence")

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        use_conf_filter = st.checkbox(
            "فعال‌کردن under_allow برای برچسب‌های 1 و 2 (BUY/SELL)",
            value=True,
            key="tab4_use_conf_filter",
        )
    with col_c2:
        conf_threshold = st.slider(
            "حداقل confidence برای اجازه معامله",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.01,
            key="tab4_conf_threshold",
        )

    st.caption(
        "- برای label_m3:\n"
        "  • اگر SL بالاتر از close باشد → فقط SELL(2) یا 3 مجاز؛ هر 1 به 3 تبدیل می‌شود.\n"
        "  • اگر SL پایین‌تر از close باشد → فقط BUY(1) یا 3 مجاز؛ هر 2 به 3 تبدیل می‌شود.\n"
        "- under_allow: اگر pred ∈ {1,2} و confidence < آستانه باشد، آن pred به 3 تبدیل می‌شود (no trade).\n"
        "- شبیه‌سازی بالانس با خود ستون‌های TP/SL دیتاست انجام می‌شود."
    )

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    st.info(f"مدل‌ها روی **{device_str.upper()}** اجرا می‌شوند.")

    # ========================
    # ۵) Datasetهای Test
    # ========================
    class TabularDatasetTest(torch.utils.data.Dataset):
        def __init__(self, X: np.ndarray, y_idx: np.ndarray, base_indices: np.ndarray):
            self.X = torch.from_numpy(X.astype("float32", copy=False))
            self.y = torch.from_numpy(y_idx.astype("int64", copy=False))
            self.base_indices = base_indices.astype("int64", copy=False)

        def __len__(self):
            return self.X.shape[0]

        def __getitem__(self, idx: int):
            return self.X[idx], self.y[idx], int(self.base_indices[idx])

    class SequenceDatasetTest(torch.utils.data.Dataset):
        def __init__(self, X: np.ndarray, y_idx: np.ndarray, window: int, stride: int, target_shift: int = 0):
            self.X = torch.from_numpy(X.astype("float32", copy=False))
            self.y = torch.from_numpy(y_idx.astype("int64", copy=False))
            self.window = int(window)
            self.stride = max(1, int(stride))
            self.target_shift = int(target_shift)

            self.N = self.X.shape[0]
            max_start = self.N - self.window - self.target_shift
            if max_start < 0:
                self.nseq = 0
            else:
                self.nseq = max_start // self.stride + 1

        def __len__(self):
            return self.nseq

        def __getitem__(self, idx: int):
            i = idx * self.stride
            x = self.X[i : i + self.window]
            y_index = i + self.window - 1 + self.target_shift
            if y_index >= self.N:
                y_index = self.N - 1
            y = self.y[y_index]
            base_index = int(y_index)
            return x, y, base_index

    # ========================
    # ۶) آمار برچسب‌ها
    # ========================
    def accumulate_label_logic(label_name: str, preds: np.ndarray, trues: np.ndarray) -> dict:
        stats = {
            "trade_correct": 0,
            "trade_wrong": 0,
            "avoid_loss_correct_buy": 0,
            "avoid_loss_correct_sell": 0,
            "avoid_loss_correct_buyOrsell": 0,
            "avoid_loss_correct_total": 0,
            "missed_buy": 0,
            "missed_sell": 0,
            "missed_total": 0,
            "buy_win": 0,
            "buy_loss": 0,
            "sell_win": 0,
            "sell_loss": 0,
            "overall_correct": 0,
            "overall_wrong": 0,
        }

        preds = preds.astype(int)
        trues = trues.astype(int)

        for p, t in zip(preds, trues):
            if p == t:
                stats["overall_correct"] += 1
            else:
                stats["overall_wrong"] += 1

            if p == 1:
                if t == 1:
                    stats["trade_correct"] += 1
                    stats["buy_win"] += 1
                else:
                    stats["trade_wrong"] += 1
                    stats["buy_loss"] += 1
            elif p == 2:
                if t == 2:
                    stats["trade_correct"] += 1
                    stats["sell_win"] += 1
                else:
                    stats["trade_wrong"] += 1
                    stats["sell_loss"] += 1
            elif p == 3:
                if t == 1:
                    stats["missed_buy"] += 1
                    stats["missed_total"] += 1
                elif t == 2:
                    stats["missed_sell"] += 1
                    stats["missed_total"] += 1
                else:
                    stats["avoid_loss_correct_total"] += 1

        return stats

    # ========================
    # ۷) شبیه‌سازی بالانس با TP/SL
    # ========================
    def simulate_equity_gpu(
        label_name: str,
        df_base: pd.DataFrame,
        preds_label: np.ndarray,
        trues_label: np.ndarray,
        base_indices_all: np.ndarray,
        init_equity: float,
        risk_mode: str,
        risk_perc: float | None,
        base_equity: float | None,
        base_lot: float | None,
        pip_size: float,
        pip_value_per_lot: float,
        device: torch.device,
    ):
        # انتخاب ستون‌های TP/SL بر اساس نوع برچسب
        if label_name == "label_m1":
            tp_col = "m1_tp_price"
            sl_col = "m1_sl_price"
        elif label_name == "label_m2":
            tp_col = "m2_tp_price"
            sl_col = "m2_sl_price"
        elif label_name == "label_m3":
            tp_col = "m3_tp_price"
            sl_col = "m3_sl_price"
        else:
            return float(init_equity), [], pd.DataFrame()

        # اگر ستون‌های لازم موجود نباشند، شبیه‌سازی نمی‌کنیم
        if tp_col not in df_base.columns or sl_col not in df_base.columns or "close" not in df_base.columns:
            return float(init_equity), [], pd.DataFrame()

        preds = preds_label.astype(int)
        trues = trues_label.astype(int)
        base_indices_all = base_indices_all.astype(int)

        # فقط جاهایی که مدل گفته BUY یا SELL
        trade_mask = (preds == 1) | (preds == 2)
        if not np.any(trade_mask):
            return float(init_equity), [], pd.DataFrame()

        trade_preds_np = preds[trade_mask]
        trade_trues_np = trues[trade_mask]
        trade_base_idx = base_indices_all[trade_mask]

        close_arr = df_base["close"].to_numpy(dtype="float32")
        tp_arr = df_base[tp_col].to_numpy(dtype="float32")
        sl_arr = df_base[sl_col].to_numpy(dtype="float32")

        trade_close = close_arr[trade_base_idx]
        trade_tp = tp_arr[trade_base_idx]
        trade_sl = sl_arr[trade_base_idx]

        # اگر ستون time هم وجود دارد، آن را هم نگه می‌داریم
        has_time = "time" in df_base.columns
        if has_time:
            time_arr = df_base["time"].to_numpy()
            trade_time = time_arr[trade_base_idx]
        else:
            trade_time = None

        # حذف ردیف‌هایی که قیمت‌هایشان NaN است
        valid_price_mask = ~(
            np.isnan(trade_close) |
            np.isnan(trade_tp) |
            np.isnan(trade_sl)
        )
        if not np.any(valid_price_mask):
            return float(init_equity), [], pd.DataFrame()

        trade_preds_np = trade_preds_np[valid_price_mask]
        trade_trues_np = trade_trues_np[valid_price_mask]
        trade_base_idx = trade_base_idx[valid_price_mask]
        trade_close = trade_close[valid_price_mask]
        trade_tp = trade_tp[valid_price_mask]
        trade_sl = trade_sl[valid_price_mask]
        if has_time:
            trade_time = trade_time[valid_price_mask]

        # تبدیل به Tensor روی GPU/CPU
        t_close = torch.tensor(trade_close, dtype=torch.float32, device=device)
        t_tp = torch.tensor(trade_tp, dtype=torch.float32, device=device)
        t_sl = torch.tensor(trade_sl, dtype=torch.float32, device=device)

        dist_tp = torch.abs(t_tp - t_close)
        dist_sl = torch.abs(t_sl - t_close)

        # فقط جایی که SL معنا دارد
        valid_sl_mask = dist_sl > 1e-8
        if not torch.any(valid_sl_mask):
            return float(init_equity), [], pd.DataFrame()

        dist_tp = dist_tp[valid_sl_mask]
        dist_sl = dist_sl[valid_sl_mask]
        trade_close = trade_close[valid_sl_mask.cpu().numpy()]
        trade_tp = trade_tp[valid_sl_mask.cpu().numpy()]
        trade_sl = trade_sl[valid_sl_mask.cpu().numpy()]
        trade_preds_np = trade_preds_np[valid_sl_mask.cpu().numpy()]
        trade_trues_np = trade_trues_np[valid_sl_mask.cpu().numpy()]
        trade_base_idx = trade_base_idx[valid_sl_mask.cpu().numpy()]
        if has_time:
            trade_time = trade_time[valid_sl_mask.cpu().numpy()]

        # برد/باخت بر اساس درست بودن جهت
        mask_win = (torch.tensor(trade_preds_np, device=device) ==
                    torch.tensor(trade_trues_np, device=device)) & (
                        (torch.tensor(trade_trues_np, device=device) == 1) |
                        (torch.tensor(trade_trues_np, device=device) == 2)
                    )

        # --- حالت ۱: ریسک درصدی ثابت از بالانس ---
        if risk_mode == "percent":
            if risk_perc is None:
                risk_perc = 1.0
            risk = float(risk_perc) / 100.0
            one = torch.tensor(1.0, device=device)
            profit_factor = dist_tp / dist_sl

            mult = torch.where(
                mask_win,
                one + risk * profit_factor,
                one - risk,
            )

            init_e = float(init_equity)
            equity_after = init_e * torch.cumprod(mult, dim=0)
            equity_after_np = equity_after.detach().cpu().numpy()

            prev_equity = np.concatenate([[init_e], equity_after_np[:-1]])
            pnl_np = equity_after_np - prev_equity
            lot_np = np.ones_like(equity_after_np)

            dist_tp_np = dist_tp.detach().cpu().numpy()
            dist_sl_np = dist_sl.detach().cpu().numpy()
            mask_win_np = mask_win.detach().cpu().numpy()

        # --- حالت ۲: لات پله‌ای بر اساس Equity ---
        else:
            if base_equity is None or base_lot is None:
                base_equity = 1000.0
                base_lot = 0.01

            dist_tp_np = dist_tp.detach().cpu().numpy()
            dist_sl_np = dist_sl.detach().cpu().numpy()
            trade_pips = dist_tp_np / pip_size
            trade_sl_pips = dist_sl_np / pip_size
            mask_win_np = mask_win.detach().cpu().numpy()

            equity_curve = [float(init_equity)]
            pnl_list = []
            lot_list = []

            for i in range(len(trade_pips)):
                equity_now = equity_curve[-1]
                k = int(equity_now / base_equity) + 1
                lot = k * base_lot

                pnl = lot * (trade_pips[i] if mask_win_np[i] else -trade_sl_pips[i]) * pip_value_per_lot
                new_eq = equity_now + pnl

                equity_curve.append(new_eq)
                pnl_list.append(pnl)
                lot_list.append(lot)

            equity_after_np = np.array(equity_curve[1:], dtype="float64")
            pnl_np = np.array(pnl_list, dtype="float64")
            lot_np = np.array(lot_list, dtype="float64")

        # منحنی کامل equity (با نقطه شروع)
        equity_curve_full = [float(init_equity)] + equity_after_np.tolist()

        # ساخت DataFrame تریدها (اینجا time را هم اضافه می‌کنیم)
        rows = []
        for i in range(len(trade_preds_np)):
            row = {
                "trade_index": int(trade_base_idx[i]),
                "direction": int(trade_preds_np[i]),
                "true": int(trade_trues_np[i]),
                "close": float(trade_close[i]),
                "tp_price": float(trade_tp[i]),
                "sl_price": float(trade_sl[i]),
                "dist_tp": float(dist_tp_np[i]),
                "dist_sl": float(dist_sl_np[i]),
                "is_win": bool(mask_win_np[i]),
                "lot": float(lot_np[i]),
                "pnl": float(pnl_np[i]),
                "equity_after": float(equity_after_np[i]),
            }
            if has_time:
                # مقدار time همان سطر دیتاست که ترید از آن گرفته شده
                row["time"] = trade_time[i]
            rows.append(row)

        trade_df = pd.DataFrame(rows)
        return float(equity_curve_full[-1]), equity_curve_full, trade_df

    # ========================
    # ۸) اجرای تست (یکبار) + ذخیره در session_state
    # ========================
    if "tab4_results_summary" not in st.session_state:
        st.session_state["tab4_results_summary"] = None
    if "tab4_detailed_results" not in st.session_state:
        st.session_state["tab4_detailed_results"] = None

    run_test = st.button("🚀 اجرای تست مدل‌ها روی Test (GPU)", key="tab4_run_test_btn")

    if run_test:
        results_summary: list[dict] = []
        detailed_results: dict[tuple[str, str], dict] = {}

        with st.spinner("در حال اجرای تست مدل‌ها روی دیتاست Test..."):
            for lab in selected_labels:
                lab_dir = models_root / lab
                models_for_lab = label_to_models_on_disk.get(lab, [])
                if not models_for_lab:
                    continue

                for mname in models_for_lab:
                    if mname not in selected_models:
                        continue

                    meta_path = lab_dir / f"{mname}_meta.json"
                    model_path = lab_dir / f"{mname}.pt"
                    if not meta_path.exists() or not model_path.exists():
                        continue

                    with meta_path.open("r", encoding="utf-8") as f:
                        meta = json.load(f)

                    feature_cols = meta["feature_cols"]
                    mean = np.array(meta["mean"], dtype="float32").reshape(1, -1)
                    std = np.array(meta["std"], dtype="float32").reshape(1, -1)
                    label2idx = {int(k): int(v) for k, v in meta["label2idx"].items()}
                    idx2label = {int(k): int(v) for k, v in meta["idx2label"].items()}
                    is_seq = bool(meta.get("is_sequence", False))
                    window = int(meta.get("window") or 0)
                    stride = int(meta.get("stride") or 1)
                    target_shift = int(meta.get("target_shift") or 0)

                    df_lab = df_test.dropna(subset=[lab]).reset_index(drop=True)
                    if df_lab.empty:
                        continue

                    y_true_raw = df_lab[lab].astype(int).to_numpy()
                    X_test = (
                        df_lab[feature_cols]
                        .replace([np.inf, -np.inf], np.nan)
                        .fillna(0.0)
                        .to_numpy(np.float32)
                    )
                    Xz = (X_test - mean) / std

                    y_idx = np.array(
                        [label2idx.get(int(v), -1) if not pd.isna(v) else -1 for v in y_true_raw],
                        dtype="int64",
                    )

                    if is_seq:
                        if window <= 0:
                            continue
                        ds_test = SequenceDatasetTest(Xz, y_idx, window=window, stride=stride, target_shift=target_shift)
                    else:
                        base_indices = np.arange(len(Xz), dtype="int64")
                        ds_test = TabularDatasetTest(Xz, y_idx, base_indices)

                    if len(ds_test) == 0:
                        continue

                    dl_test = torch.utils.data.DataLoader(
                        ds_test,
                        batch_size=1024,
                        shuffle=False,
                        num_workers=0,
                        pin_memory=(device_str == "cuda"),
                    )

                    model_test = build_model_from_meta(meta).to(device)
                    model_test.load_state_dict(torch.load(model_path, map_location=device))
                    model_test.eval()

                    all_preds_idx = []
                    all_probs_max = []
                    all_trues_idx = []
                    all_base_idx = []

                    with torch.inference_mode():
                        for xb, yb, baseb in dl_test:
                            xb = xb.to(device)
                            yb = yb.to(device)
                            logits = model_test(xb)
                            prob = torch.softmax(logits, dim=1)
                            max_prob, pred_idx = torch.max(prob, dim=1)
                            all_preds_idx.append(pred_idx.cpu().numpy())
                            all_probs_max.append(max_prob.cpu().numpy())
                            all_trues_idx.append(yb.cpu().numpy())
                            all_base_idx.append(baseb.cpu().numpy())

                    preds_idx = np.concatenate(all_preds_idx, axis=0)
                    probs_max = np.concatenate(all_probs_max, axis=0)
                    trues_idx = np.concatenate(all_trues_idx, axis=0)
                    base_indices_all = np.concatenate(all_base_idx, axis=0)

                    mask_valid = trues_idx >= 0
                    preds_idx = preds_idx[mask_valid]
                    trues_idx = trues_idx[mask_valid]
                    base_indices_all = base_indices_all[mask_valid]
                    probs_max = probs_max[mask_valid]

                    if len(trues_idx) == 0:
                        continue

                    preds_label = np.array([idx2label[int(i)] for i in preds_idx], dtype="int32")
                    trues_label = np.array([idx2label[int(i)] for i in trues_idx], dtype="int32")

                    # منطق SL برای label_m3
                    if lab == "label_m3" and "m3_sl_price" in df_lab.columns and "close" in df_lab.columns:
                        sl_all = df_lab["m3_sl_price"].to_numpy(dtype="float64")
                        close_all = df_lab["close"].to_numpy(dtype="float64")
                        sl_samples = sl_all[base_indices_all]
                        close_samples = close_all[base_indices_all]
                        valid_sl = ~np.isnan(sl_samples) & ~np.isnan(close_samples)

                        bad_buy_mask = (sl_samples > close_samples) & (preds_label == 1) & valid_sl
                        preds_label[bad_buy_mask] = 3
                        bad_sell_mask = (sl_samples < close_samples) & (preds_label == 2) & valid_sl
                        preds_label[bad_sell_mask] = 3

                    preds_for_trades = preds_label.copy()

                    if use_conf_filter:
                        under_allow_conf_mask = ((preds_label == 1) | (preds_label == 2)) & (probs_max < conf_threshold)
                    else:
                        under_allow_conf_mask = np.zeros_like(preds_label, dtype=bool)

                    preds_for_trades[under_allow_conf_mask & np.isin(preds_for_trades, [1, 2])] = 3

                    correct_mask = (preds_label == trues_label)
                    accuracy = float(correct_mask.mean())

                    uniq_cls = np.unique(trues_label)
                    cls_accs = []
                    for c in uniq_cls:
                        mask_c = trues_label == c
                        if mask_c.sum() > 0:
                            cls_accs.append((preds_label[mask_c] == c).mean())
                    balanced_acc = float(np.mean(cls_accs)) if cls_accs else np.nan

                                        # آمار طبقه‌بندی (برای missed / overall و ...). این دیگه برای Winrate استفاده نمی‌شه
                    stats = accumulate_label_logic(lab, preds_for_trades, trues_label)

                    # شبیه‌سازی تریدها و ساختن trade_df
                    final_equity, equity_curve, trade_df = simulate_equity_gpu(
                        lab,
                        df_lab,
                        preds_for_trades,
                        trues_label,
                        base_indices_all,
                        init_equity=init_equity,
                        risk_mode=risk_mode,
                        risk_perc=risk_perc,
                        base_equity=base_equity,
                        base_lot=base_lot,
                        pip_size=pip_size,
                        pip_value_per_lot=pip_value_per_lot,
                        device=device,
                    )

                    # ✅ از این‌جا به بعد: تعداد و Winrate فقط از خود trade_df
                    if isinstance(trade_df, pd.DataFrame) and not trade_df.empty:
                        executed_trades = int(len(trade_df))
                        trade_correct = int((trade_df["is_win"] == True).sum())
                        trade_wrong = executed_trades - trade_correct
                        trade_winrate = trade_correct / executed_trades
                    else:
                        executed_trades = 0
                        trade_correct = 0
                        trade_wrong = 0
                        trade_winrate = float("nan")


                    if equity_curve:
                        eq_arr = np.array(equity_curve, dtype="float64")
                        running_max = np.maximum.accumulate(eq_arr)
                        drawdown = (running_max - eq_arr) / running_max
                        max_dd = float(drawdown.max())
                    else:
                        max_dd = np.nan

                    base_cols = [c for c in df_lab.columns if c not in meta["feature_cols"]]
                    base_subset = df_lab.iloc[base_indices_all][base_cols].reset_index(drop=True)
                    pred_df_full = base_subset.copy()
                    pred_df_full["label_true"] = trues_label
                    pred_df_full["label_pred"] = preds_label
                    pred_df_full["is_correct"] = (pred_df_full["label_true"] == pred_df_full["label_pred"]).astype(int)
                    pred_df_full["pred_confidence"] = probs_max
                    # در فایل تو اسم ستون under_allow اینه:
                    pred_df_full["is_under_allow_conf"] = under_allow_conf_mask.astype(int)

                    # تعداد under_allow برای این مدل/برچسب
                    under_allow_count = int(under_allow_conf_mask.sum())

                    detailed_results[(lab, mname)] = {
                        "lab": lab,
                        "model": mname,
                        "n_samples": int(len(trues_label)),
                        "accuracy": accuracy,
                        "balanced_acc": balanced_acc,
                        "trade_winrate": trade_winrate,
                        # ✅ این سه تا مبنا را از trade_df گرفتیم
                        "trade_correct": trade_correct,
                        "trade_wrong": trade_wrong,
                        "executed_trades": executed_trades,
                        # بقیه مثل قبل
                        "stats": stats,
                        "equity_start": float(init_equity),
                        "equity_final": float(final_equity),
                        "equity_curve": equity_curve,
                        "max_drawdown": max_dd,
                        "under_allow_count": under_allow_count,
                        "trade_df": trade_df,
                        "pred_df_full": pred_df_full,
                    }


                    results_summary.append(
                        {
                            "label": lab,
                            "model": mname,
                            "n_samples": int(len(trues_label)),
                            "accuracy": accuracy,
                            "balanced_acc": balanced_acc,
                            # ✅ الان این ستون هم تعداد تریدهای اجرا شده است
                            "total_trades(1&2)": int(executed_trades),
                            "trade_winrate": trade_winrate,
                            "equity_start": float(init_equity),
                            "equity_final": float(final_equity),
                            "max_drawdown": max_dd,
                            "under_allow_count": under_allow_count,
                            "overall_correct": int(stats.get("overall_correct", 0)),
                            "overall_wrong": int(stats.get("overall_wrong", 0)),
                            "missed_buy": int(stats.get("missed_buy", 0)),
                            "missed_sell": int(stats.get("missed_sell", 0)),
                            "missed_total": int(stats.get("missed_total", 0)),
                        }
                    )



        st.session_state["tab4_results_summary"] = results_summary
        st.session_state["tab4_detailed_results"] = detailed_results

    # ========================
    # ۹) نمایش نتایج ذخیره‌شده (بدون نیاز به زدن دوباره دکمه)
    # ========================
    results_summary = st.session_state.get("tab4_results_summary")
    detailed_results = st.session_state.get("tab4_detailed_results")

    if not results_summary:
        st.info("هنوز تستی اجرا نشده یا نتیجه‌ای ذخیره نشده است.")
        return

    st.subheader("📊 خلاصه نتایج مدل‌ها")
    df_summary = pd.DataFrame(results_summary)
    st.dataframe(df_summary, use_container_width=True)

    st.subheader("🔍 جزئیات یک مدل")

    labels_in_results = sorted(set(r["label"] for r in results_summary))
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        sel_lab = st.selectbox("برچسب", labels_in_results, key="tab4_detail_label")
    with col_d2:
        models_for_sel_lab = sorted(
            set(r["model"] for r in results_summary if r["label"] == sel_lab)
        )
        sel_model = st.selectbox("مدل", models_for_sel_lab, key="tab4_detail_model")

    key = (sel_lab, sel_model)
    if key not in detailed_results:
        st.warning("برای این ترکیب برچسب/مدل، نتیجه‌ای پیدا نشد.")
        return

    res = detailed_results[key]
    stats = res["stats"]
    trade_df = res["trade_df"]

    # اگر به هر دلیل مقادیر در detailed_results نبود، از خود trade_df دوباره حساب می‌کنیم
    executed_trades = int(res.get("executed_trades", 0))
    trade_correct = res.get("trade_correct", None)
    trade_wrong = res.get("trade_wrong", None)

    if (trade_correct is None) or (trade_wrong is None):
        if isinstance(trade_df, pd.DataFrame) and not trade_df.empty:
            executed_trades = int(len(trade_df))
            trade_correct = int((trade_df["is_win"] == True).sum())
            trade_wrong = executed_trades - trade_correct
        else:
            executed_trades = 0
            trade_correct = 0
            trade_wrong = 0

    st.markdown(f"### جزئیات: `{sel_lab}` — `{sel_model}`")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Accuracy (دقت کلی)", f"{res['accuracy']:.3f}")
        st.metric(
            "Balanced Accuracy (میانگین دقت کلاس‌ها)",
            f"{res['balanced_acc']:.3f}" if not np.isnan(res["balanced_acc"]) else "NaN",
        )
    with col_b:
        st.metric(
            "Trade Winrate (فقط تریدهای اجرا شده)",
            f"{res['trade_winrate']:.3f}" if res["trade_winrate"] == res["trade_winrate"] else "NaN",
        )
        st.metric("Total Trades (اجرا شده)", executed_trades)
    with col_c:
        st.metric("Equity Start", f"{res.get('equity_start', 0.0):.2f}")
        st.metric("Equity Final", f"{res.get('equity_final', 0.0):.2f}")
        max_dd = res.get("max_drawdown", float("nan"))
        st.metric("Max Drawdown", f"{max_dd:.3f}" if max_dd == max_dd else "NaN")

    # کمی توضیح شفاف برای خودت/کاربر:
    st.markdown("#### تفکیک تریدهای اجرا شده")
    col_t1, col_t2, col_t3 = st.columns(3)
    with col_t1:
        st.metric("تعداد تریدهای اجرا شده (در trade_df)", executed_trades)
    with col_t2:
        st.metric("تریدهای سودده (correct)", trade_correct)
    with col_t3:
        st.metric("تریدهای ضررده (wrong)", trade_wrong)

    st.caption("براساس trade_df: تعداد ردیف‌های CSV = executed_trades = correct + wrong")



    under_allow_count = res.get("under_allow_count", 0)
    st.metric(
        "تعداد پردیکت‌های under_allow (1 یا 2 با اطمینان کم)",
        under_allow_count,
    )

    st.markdown("""
    - **under_allow** یعنی مدل گفته BUY/SELL ولی اطمینانش زیر آستانه بوده؛ در نتیجه ترید اجرا نشده.
    - این پردیکت‌ها در Accuracy و Confusion Matrix دیده می‌شن، ولی در بالانس و Winrate تریدها لحاظ نمی‌شن.
    """)

    # ---- Confusion Matrix + توزیع کلاس‌ها ----
    st.markdown("#### ماتریس اشتباه (Confusion Matrix – روی label_pred واقعی)")
    pred_df_full = res["pred_df_full"]

    if pred_df_full is not None and not pred_df_full.empty:
        cm = pd.crosstab(
            pred_df_full["label_true"],
            pred_df_full["label_pred"],
            rownames=["واقعی"],
            colnames=["پیش‌بینی"],
        )
        st.dataframe(cm, use_container_width=True)

        st.markdown("#### توزیع کلاس‌های واقعی و پیش‌بینی‌شده")
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            dist_true = pred_df_full["label_true"].value_counts().sort_index()
            st.bar_chart(dist_true)
            st.caption("تعداد هر کلاس در داده‌ی واقعی")
        with col_chart2:
            dist_pred = pred_df_full["label_pred"].value_counts().sort_index()
            st.bar_chart(dist_pred)
            st.caption("تعداد هر کلاس در پیش‌بینی مدل")

        st.markdown("#### تعداد under_allow بر اساس برچسب پیش‌بینی‌شده")
        # در فایل خودت نام ستون، is_under_allow_conf است
        ua_col = "is_under_allow_conf"
        if ua_col in pred_df_full.columns:
            ua_counts = pred_df_full[pred_df_full[ua_col] == 1]["label_pred"].value_counts().sort_index()
            if not ua_counts.empty:
                st.bar_chart(ua_counts)
    else:
        st.info("برای این مدل دیتاست کامل پردیکت ساخته نشده یا خالی است.")

    # ---- نمودار Equity ----
    st.markdown("#### نمودار روند Equity (روی تریدهای اجرا شده)")
    equity_curve = res["equity_curve"]
    if equity_curve:
        eq_df = pd.DataFrame({"step": range(len(equity_curve)), "equity": equity_curve})
        st.line_chart(eq_df.set_index("step"))
    else:
        st.info("هیچ تریدی برای این مدل/برچسب اجرا نشده (بعد از منطق SL و فیلتر اطمینان).")

    # ---- تریدها + دانلود CSV (همان چیزی که در فایل جدید داری) ----
        # ---- تریدها + دانلود Excel ----
        # ---- تریدها + دانلود Excel ----
    st.markdown("#### جزئیات تریدها (فقط اجرا شده‌ها)")
    trade_df = res["trade_df"]

    if trade_df is None or trade_df.empty:
        st.info("هیچ ترید ثبت نشده یا ذخیره‌ی تریدها غیرفعال شده است.")
    else:
        st.dataframe(trade_df.head(500), use_container_width=True)

        # یک کپی می‌گیریم که روی دیتافریم اصلی دست نزنیم
        df_xlsx = trade_df.copy()

        # ✅ حذف timezone از همه ستون‌های datetime (مشکل اصلی اینجاست)
        datetime_tz_cols = df_xlsx.select_dtypes(include=["datetimetz"]).columns
        for c in datetime_tz_cols:
            df_xlsx[c] = df_xlsx[c].dt.tz_localize(None)

        # ساخت فایل اکسل در حافظه
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df_xlsx.to_excel(writer, index=False, sheet_name="trades")
        xlsx_data = output.getvalue()

        st.download_button(
            "⬇️ دانلود Excel تریدها (trade_df)",
            data=xlsx_data,
            file_name=f"trades_{sel_lab}_{sel_model}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"tab4_download_trades_{sel_lab}_{sel_model}",
        )



        # ---- دیتاست کامل پردیکت + دانلود Excel ----
    st.markdown("### 🔬 دیتاست کامل پردیکت روی Test")
    if pred_df_full is not None and not pred_df_full.empty:
        st.dataframe(pred_df_full.head(500), use_container_width=True)

        # یک کپی برای خروجی اکسل می‌گیریم
        df_pred_xlsx = pred_df_full.copy()

        # ✅ حذف timezone از ستون‌های datetime (برای سازگاری با Excel)
        datetime_tz_cols_pred = df_pred_xlsx.select_dtypes(include=["datetimetz"]).columns
        for c in datetime_tz_cols_pred:
            df_pred_xlsx[c] = df_pred_xlsx[c].dt.tz_localize(None)

        # ساخت فایل اکسل در حافظه
        output_pred = io.BytesIO()
        with pd.ExcelWriter(output_pred, engine="openpyxl") as writer:
            df_pred_xlsx.to_excel(writer, index=False, sheet_name="predictions")
        xlsx_pred_data = output_pred.getvalue()

        st.download_button(
            "⬇️ دانلود Excel کامل پردیکت‌ها",
            data=xlsx_pred_data,
            file_name=f"predictions_{sel_lab}_{sel_model}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"tab4_download_pred_{sel_lab}_{sel_model}",
        )
    else:
        st.info("دیتاست پردیکت خالی است.")




with tabs[4]:
    # فقط همین تابع را صدا می‌زنیم؛ داخلش هرجا لازم باشد return می‌زنیم
    render_tab4_test_models()


def render_tab_label_overview():
    """
    تب کامل تحلیل برچسب‌ها:
    - توزیع کلاس‌ها
    - کفایت حجم داده برای دیپ‌لرنینگ
    - بررسی تعادل / عدم تعادل
    - نتیجه‌گیری حرفه‌ای و قابل‌فهم
    """

    st.header("📊 تحلیل حرفه‌ای برچسب‌ها در دیتاست Train")

    paths = ensure_paths_and_autoload()
    train_path = (
        st.session_state.get("train_path")
        or paths.get("train")
        or paths.get("train_parquet")
        or ""
    )

    if not train_path or not os.path.exists(train_path):
        st.warning("⚠️ ابتدا باید در تب ساخت دیتاست، train/test را ایجاد کنید.")
        return

    try:
        df_train = pd.read_parquet(train_path)
    except Exception as e:
        st.error(f"❌ خطا در خواندن Train:\n{e}")
        return

    if df_train.empty:
        st.warning("⚠️ دیتاست Train خالی است.")
        return

    st.info(f"🔢 تعداد کل رکوردهای Train: **{len(df_train):,}**")

    # استخراج ستون‌های برچسب
    label_cols = [c for c in df_train.columns if c.startswith("label_")]
    if not label_cols:
        st.error("❌ هیچ ستون label_ پیدا نشد.")
        return

    # تابع ارزیابی حجم داده
    def interpret_total_samples(n):
        if n < 1000:
            return ("🔴 **حجم داده بسیار کم است.**\n"
                    "برای مدل‌های دیپ‌لرنینگ این مقدار کافی نیست. احتمالاً مدل یاد نمی‌گیرد.")
        elif n < 5000:
            return ("🟠 **حجم داده مرزی است.**\n"
                    "ممکن است مدل‌های کم‌عمق جواب دهند اما خطر overfitting وجود دارد.")
        elif n < 20000:
            return ("🟡 **حجم داده قابل‌قبول است.**\n"
                    "مدل‌های ساده و متوسط می‌توانند خوب عمل کنند.")
        elif n < 80000:
            return ("🟢 **حجم داده خوب است.**\n"
                    "برای مدل‌های LSTM/TCN/MLP متوسط کاملاً مناسب است.")
        else:
            return ("🟢 **حجم داده عالی است.**\n"
                    "برای مدل‌های جدی دیپ‌لرنینگ هم بسیار مناسب است.")

    # تابع ارزیابی تعادل کلاس‌ها
    def interpret_balance(counts, n_total):
        major = counts.max()
        minor = counts.min()
        major_p = major / n_total
        minor_p = minor / n_total

        text = []
        text.append(f"- بیشترین کلاس: **{major:,}** نمونه ({major_p*100:.1f}%)")
        text.append(f"- کم‌ترین کلاس: **{minor:,}** نمونه ({minor_p*100:.1f}%)")

        # درجه عدم تعادل
        if major_p > 0.90:
            text.append("🔴 **شدیداً نامتوازن.** مدل تقریباً فقط یک کلاس را یاد می‌گیرد.")
        elif major_p > 0.80:
            text.append("🟠 **نامتوازن قابل توجه.** مدل بایاس می‌شود مگر این‌که وزن‌دهی استفاده شود.")
        elif minor < 200:
            text.append("🟠 **کلاس‌های کم‌نمونه وجود دارد.** دقت روی این کلاس‌ها پایین خواهد بود.")
        else:
            text.append("🟢 **تعادل کلاس‌ها مناسب یا قابل‌قبول است.**")

        return "\n".join(text)

    # حلقه روی تک تک برچسب‌ها
    for lab in sorted(label_cols):
        st.markdown("---")
        st.subheader(f"🔖 برچسب: **{lab}**")

        s = df_train[lab].dropna()
        try:
            s = s.astype(int)
        except:
            pass

        n_total = len(s)
        counts = s.value_counts().sort_index()
        perc = (counts / n_total * 100).round(2)

        # جدول توزیع
        dist_df = pd.DataFrame({
            "label_value": counts.index,
            "count": counts.values,
            "percent": perc.values
        })

        st.markdown("### 📌 توزیع کلاس‌ها")
        st.dataframe(dist_df, use_container_width=True)

        # نمودار
        if len(dist_df) <= 30:
            st.bar_chart(dist_df.set_index("label_value")["count"])

        # خلاصه عددی
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📌 کل نمونه‌ها", f"{n_total:,}")
        with col2:
            st.metric("📌 تعداد کلاس‌ها", f"{len(counts)}")
        with col3:
            st.metric("📌 سهم کلاس غالب", f"{(counts.max()/n_total*100):.1f}%")

        # ---------- تحلیل حرفه‌ای ----------
        st.markdown("### 📈 ارزیابی حجم داده")
        st.info(interpret_total_samples(n_total))

        st.markdown("### ⚖️ ارزیابی تعادل برچسب‌ها")
        st.info(interpret_balance(counts, n_total))

        # ---------- نتیجه‌گیری نهایی ----------
        st.markdown("### 🧠 **نتیجه‌گیری نهایی برای این برچسب**")

        maj_ratio = counts.max() / n_total
        minor_ratio = counts.min() / n_total

        if maj_ratio < 0.6 and minor_ratio > 0.03:
            # حالت خوب
            st.success(
                "✔️ **این برچسب برای دیپ‌لرنینگ مناسب است.**\n"
                "- حجم داده کافی یا عالی\n"
                "- توزیع کلاس‌ها قابل قبول\n"
                "- مدل می‌تواند به خوبی تفاوت کلاس‌ها را یاد بگیرد"
            )
        elif maj_ratio < 0.8 and minor_ratio > 0.01:
            st.warning(
                "🟡 **قابل قبول است اما کلاس‌های کم‌نمونه دقت پایین‌تری خواهند داشت.**\n"
                "پیشنهاد می‌شود بعداً نتایج کلاس‌ها را جداگانه بررسی کنید."
            )
        else:
            st.error(
                "🔴 **این برچسب شدیداً نامتوازن یا کم‌داده است.**\n"
                "مدل احتمالاً روی برخی کلاس‌ها بی‌دقت عمل می‌کند."
            )

        # نکته ویژه label_m3
        if lab == "label_m3":
            st.info(
                "💡 برچسب `label_m3` ترکیبی است (مثلاً جهت + منطق SL/TP). "
                "بنابراین طبیعی است کلاس‌های 3 و 4 بزرگ‌تر باشند. "
                "مادامی که کلاس‌های 1 و 2 حداقل چند ده هزار نمونه داشته باشند، مدل دچار مشکل نمی‌شود."
            )

    # ------------- جمع‌بندی کلی -------------
    st.markdown("---")
    st.subheader("📚 جمع‌بندی کلی")

    st.write(
        "این تحلیل نشان می‌دهد آیا داده‌های برچسب‌گذاری‌شده برای مدل‌های دیپ‌لرنینگ مناسب هستند یا خیر. "
        "توزیع کلاس‌ها، حجم داده، و میزان تعادل اهمیت زیادی دارند.\n\n"
        "در کل:\n"
        "- اگر هر برچسب **بالای ۱۰–۲۰ هزار** نمونه داشته باشد → معماری‌های متوسط DL جواب می‌دهند.\n"
        "- اگر **کلاس غالب کمتر از ۷۰٪** باشد → توزیع مناسب است.\n"
        "- اگر کلاس‌های کوچک **بیش از ۱٪** از داده باشند → مدل آن‌ها را یاد می‌گیرد.\n"
        "- اگر کلاس کوچک **کمتر از ۰.۵٪** باشد → دقت روی آن ضعیف می‌شود."
    )

with tabs[5]:
    # فقط همین تابع را صدا می‌زنیم؛ داخلش هرجا لازم باشد return می‌زنیم
    render_tab_label_overview()

with tabs[6]:
    try:
        st.header("🤖 معامله خودکار با متاتریدر (روی مدل‌های آموزش‌داده‌شده)")

        # ---------------------------
        # بررسی دسترسی به MetaTrader5
        # ---------------------------
        if mt5 is None:
            st.error("کتابخانه MetaTrader5 نصب نیست. اول با دستور زیر نصبش کن:\n\n`pip install MetaTrader5`")
            st.stop()

        # ---------------------------
        # اتصال/قطع اتصال به متاتریدر
        # ---------------------------
        if "mt5_initialized" not in st.session_state:
            st.session_state["mt5_initialized"] = False

        c_conn1, c_conn2 = st.columns(2)
        with c_conn1:
            if not st.session_state["mt5_initialized"]:
                if st.button("اتصال به متاتریدر", key="tab5_mt5_connect"):
                    if mt5.initialize():
                        st.session_state["mt5_initialized"] = True
                        st.success("✅ اتصال به متاتریدر برقرار شد.")
                    else:
                        st.error("❌ اتصال به متاتریدر ناموفق بود.")
                else:
                    st.info("برای شروع، روی دکمهٔ «اتصال به متاتریدر» کلیک کن.")
            else:
                st.success("✅ متاتریدر هم‌اکنون متصل است.")

        with c_conn2:
            if st.session_state["mt5_initialized"]:
                if st.button("قطع اتصال از متاتریدر", key="tab5_mt5_shutdown"):
                    try:
                        mt5.shutdown()
                    except Exception:
                        pass
                    st.session_state["mt5_initialized"] = False
                    st.warning("ارتباط با متاتریدر قطع شد.")

        if not st.session_state["mt5_initialized"]:
            st.stop()

        # ---------------------------
        # اطلاعات حساب و انتخاب سمبل / تایم‌فریم
        # ---------------------------
        account_info = mt5.account_info()
        if account_info is None:
            st.error("اطلاعات حساب در دسترس نیست. مطمئن شو متاتریدر باز است و لاگین شده‌ای.")
            st.stop()

        st.subheader("اطلاعات حساب")
        c_acc1, c_acc2, c_acc3 = st.columns(3)
        with c_acc1:
            st.metric("شماره حساب", account_info.login)
        with c_acc2:
            st.metric("اکوئیتی", f"{account_info.equity:.2f}")
        with c_acc3:
            st.metric("بالانس", f"{account_info.balance:.2f}")

        st.markdown("---")
        st.subheader("انتخاب سمبل و تایم‌فریم")

        all_symbols = mt5.symbols_get()
        sym_names = [s.name for s in all_symbols] if all_symbols else []
        if not sym_names:
            st.error("هیچ سمبلی از متاتریدر خوانده نشد.")
            st.stop()

        default_symbol = "XAUUSD" if "XAUUSD" in sym_names else sym_names[0]
        symbol = st.selectbox("سمبل برای معامله خودکار", sym_names, index=sym_names.index(default_symbol))

        timeframe_dict = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
        }
        tf_label = st.selectbox("تایم‌فریم پایه (همان تایم‌فریم دیتاست ۵ دقیقه‌ای)", list(timeframe_dict.keys()), index=1)
        timeframe = timeframe_dict[tf_label]

        # ---------------------------
        # مسیر دیتاست و تنظیمات برچسب‌ها (SL/TP)
        # ---------------------------
        st.markdown("---")
        st.subheader("تنظیمات برچسب‌ها (SL/TP) و فیچرها")

        paths = ensure_paths_and_autoload()
        parquet_path = paths.get("parquet") or st.session_state.get("parquet_path")

        if not parquet_path or not os.path.exists(parquet_path):
            st.error("فایل Parquet دیتاست پیدا نشد. ابتدا در تب‌های قبل دیتاست را بساز و ذخیره کن.")
            st.stop()

        label_cfg = load_label_config(parquet_path)
        if label_cfg is None:
            st.error("فایل تنظیمات برچسب‌ها (.labelcfg.json) پیدا نشد. اول در تب «🏷 افزودن برچسب‌ها» برچسب‌گذاری کن.")
            st.stop()

        pip_value_cfg = float(label_cfg.get("pip_value", 0.01))
        spread_cfg = float(label_cfg.get("spread", 0.0))

        m1_cfg = label_cfg.get("m1", {})
        m1_sl_pips_cfg = float(m1_cfg.get("sl_pips", 100))
        m1_tp_pips_cfg = float(m1_cfg.get("tp_pips", 100))

        m3_cfg = label_cfg.get("m3", {})
        m3_min_sl_pips_cfg = float(m3_cfg.get("min_sl_pips", 50))
        m3_max_sl_pips_cfg = float(m3_cfg.get("max_sl_pips", 500))
        m3_rr_cfg = float(m3_cfg.get("rr", 1.5))
        m3_future_window_cfg = int(m3_cfg.get("future_window", 50))

        with st.expander("خلاصه تنظیمات SL/TP از تب برچسب‌گذاری"):
            st.write(
                f"- pip_value: **{pip_value_cfg}**\n"
                f"- spread: **{spread_cfg}**\n"
                f"- روش ۱: SL={m1_sl_pips_cfg} pip, TP={m1_tp_pips_cfg} pip\n"
                f"- روش ۳ (label_m3): min SL={m3_min_sl_pips_cfg} pip, max SL={m3_max_sl_pips_cfg} pip, RR={m3_rr_cfg}, "
                f"future_window={m3_future_window_cfg}"
            )

        # ---------------------------
        # لود feature_cfg (همان تب افزودن فیچرها)
        # ---------------------------
        def _load_feature_cfg_for_autotrade(pq_path: str):
            p = Path(pq_path)
            cfg_path = p.with_suffix(".featurecfg.json")
            if not cfg_path.exists():
                st.error(
                    f"فایل تنظیمات فیچرها `{cfg_path.name}` پیدا نشد. "
                    "لطفاً یک‌بار در تب «📈 افزودن فیچرها» فیچرها را محاسبه و ذخیره کن."
                )
                st.stop()
            with cfg_path.open("r", encoding="utf-8") as f:
                cfg = json.load(f)
            return cfg, cfg_path

        feature_cfg, feature_cfg_path = _load_feature_cfg_for_autotrade(parquet_path)

        with st.expander("خلاصه تنظیمات فیچرها (feature_cfg)"):
            st.json(feature_cfg)

        # ---------------------------
        # لود meta و مدل
        # ---------------------------
        st.subheader("انتخاب مدل آموزش‌داده‌شده")

        models_root = Path(paths.get("models_dir", "models"))
        if not models_root.exists():
            st.error("دایرکتوری مدل‌ها پیدا نشد.")
            st.stop()

        # همه‌ی metaها را بخوان
        meta_entries: list[tuple[Path, dict]] = []
        for mf in models_root.glob("*/*_meta.json"):
            try:
                with mf.open("r", encoding="utf-8") as f:
                    meta_tmp = json.load(f)
            except Exception:
                continue

            # label_name را مثل تب ۴ مشخص می‌کنیم
            label_name_tmp = (
                meta_tmp.get("label_name")
                or meta_tmp.get("label_col")
                or mf.parent.name
            )

            model_name_tmp = meta_tmp.get("model_name") or mf.stem.replace("_meta", "")

            # توی خود meta هم ذخیره‌اش می‌کنیم که بعداً استفاده شود
            meta_tmp["label_name"] = label_name_tmp
            meta_tmp["model_name"] = model_name_tmp

            meta_entries.append((mf, meta_tmp))

        if not meta_entries:
            st.error("هیچ مدل آموزش‌داده‌شده‌ای پیدا نشد.")
            st.stop()

        # متن نمایش در selectbox
        options = [
            f"{meta['label_name']} | {meta['model_name']} | {mf.parent.name}"
            for (mf, meta) in meta_entries
        ]

        selected_idx = st.selectbox(
            "فایل مدل (meta) را انتخاب کن",
            range(len(meta_entries)),
            format_func=lambda i: options[i],
        )

        selected_meta_file, meta = meta_entries[selected_idx]
        label_name = meta["label_name"]
        model_name = meta["model_name"]

        st.info(f"مدل انتخاب‌شده: **{model_name}** روی برچسب **{label_name}**")


        num_classes = int(
            meta.get("n_classes")
            or (len(meta.get("label2idx", {})) if meta.get("label2idx") else 0)
        )

        feature_cols = meta.get("feature_cols", None)
        if feature_cols is None:
            st.error("در meta ستون feature_cols پیدا نشد. مدل دوباره باید ذخیره شود.")
            st.stop()

        mean = np.array(meta.get("mean", [0.0] * len(feature_cols)), dtype=np.float32)
        std = np.array(meta.get("std", [1.0] * len(feature_cols)), dtype=np.float32)
        std[std == 0] = 1.0

        is_seq = bool(meta.get("is_sequence", False))
        window = int(meta.get("window", 1))
        target_shift = int(meta.get("target_shift", 0))

        label2idx = meta.get("label2idx", None)
        if label2idx is None:
            idx2label = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
        else:
            # label2idx در meta معمولاً شبیه {"1":0,"2":1,...} است
            tmp = {int(k): int(v) for k, v in label2idx.items()}
            idx2label = {v: k for k, v in tmp.items()}

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            model_live = build_model_from_meta(meta, device=device, num_classes=num_classes)
        except Exception as e:
            st.error(f"خطا در ساخت مدل از روی meta: {e}")
            st.stop()

        model_path = selected_meta_file.with_name(f"{model_name}.pt")
        if not model_path.exists():
            st.error(f"فایل مدل `{model_path.name}` در مسیر `{model_path.parent}` پیدا نشد.")
            st.stop()

        try:
            state = torch.load(model_path, map_location=device)
            model_live.load_state_dict(state)
            model_live.to(device)
            model_live.eval()
        except Exception as e:
            st.error(f"خطا در لود state_dict مدل: {e}")
            st.stop()

        st.success("✅ مدل برای معامله خودکار آماده شد.")

        # اگر مدل روی label_m3 نیست، اخطار بده
        if label_name != "label_m3":
            st.warning(
                "مدل انتخاب‌شده روی برچسبی غیر از `label_m3` آموزش داده شده است. "
                "منطق SL/TP مخصوص label_m3 است؛ برای برچسب‌های دیگر، SL/TP ساده با pip ثابت استفاده می‌شود."
            )

        # ---------------------------
        # تنظیمات ریسک و حجم معامله
        # ---------------------------
        st.markdown("---")
        st.subheader("تنظیمات ریسک و حجم معاملات")

        col_risk1, col_risk2 = st.columns(2)
        with col_risk1:
            risk_mode = st.radio(
                "روش مدیریت ریسک",
                options=["percent", "fixed_lot"],
                format_func=lambda x: "بر اساس درصد اکوئیتی" if x == "percent" else "لات ثابت (پله‌ای با رشد اکوئیتی)",
                key="tab5_risk_mode",
            )

        with col_risk2:
            base_equity = st.number_input(
                "سرمایه مبنا (برای حالت لات ثابت پله‌ای)",
                min_value=0.0,
                value=float(account_info.equity),
                step=100.0,
                key="tab5_base_equity",
            )
            base_lot = st.number_input(
                "حجم لات پایه (برای سرمایه مبنا)",
                min_value=0.01,
                max_value=100.0,
                value=0.1,
                step=0.01,
                key="tab5_base_lot",
            )

        if risk_mode == "percent":
            risk_perc = st.number_input(
                "درصد ریسک هر معامله (مثلاً ۱٪ = 0.01)",
                min_value=0.0001,
                max_value=0.1,
                value=0.01,
                step=0.0001,
                key="tab5_risk_perc",
            )
        else:
            risk_perc = None

        # ---------------------------
        # محاسبه ارزش تقریبی هر pip بر اساس متاتریدر
        # ---------------------------
        def get_pip_value_per_lot() -> float:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return 0.0

            tick_val = symbol_info.trade_tick_value
            tick_size = symbol_info.trade_tick_size
            if tick_size == 0:
                return 10.0

            pip_in_ticks = pip_value_cfg / tick_size
            return float(tick_val * pip_in_ticks)

        pip_auto = get_pip_value_per_lot()
        st.info(
            f"ارزش تقریبی هر pip برای 1 لات روی {symbol}: **{pip_auto:.4f}** "
            f"(بر اساس trade_tick_value و pip_value تنظیم‌شده در تب برچسب‌ها)."
        )

        pip_value_per_lot = st.number_input(
            "ارزش هر 1 pip برای 1 لات (می‌توانی مقدار خودکار را تغییر دهی)",
            min_value=0.0001,
            max_value=100000.0,
            value=float(pip_auto),
            step=0.1,
            key="tab5_pip_value_per_lot",
        )

        # ---------------------------
        # کمک‌تابع محاسبه lot
        # ---------------------------
        def calc_lot(sl_pips_effective: float, equity_now: float) -> float:
            """
            sl_pips_effective: فاصله‌ی واقعی SL تا قیمت ورود، بر حسب pip
            """
            if sl_pips_effective <= 0:
                return 0.0

            if risk_mode == "percent":
                risk_amount = equity_now * float(risk_perc or 0.0)
                if risk_amount <= 0 or pip_value_per_lot <= 0:
                    return 0.0
                raw_lot = risk_amount / (sl_pips_effective * pip_value_per_lot)
                return raw_lot
            else:
                if base_equity is None or base_equity <= 0 or base_lot is None or base_lot <= 0:
                    return 0.0
                ratio = equity_now / base_equity
                k = int(ratio) + 1
                if k < 1:
                    k = 1
                raw_lot = base_lot * k
                return raw_lot

        # ⚠️ دقت کن: اینجا دیگر calc_lot را قبل از محاسبه SL صدا نمی‌زنیم.
        # مقدار lot فقط بعد از محاسبه‌ی دقیق sl_pips_effective محاسبه می‌شود.

        # ---------------------------
        # تنظیمات اجرای خودکار (polling)
        # ---------------------------
        st.subheader("اجرای خودکار روی هر کندل جدید")

        poll_interval = st.number_input(
            "فاصله زمانی بین هر چک (ثانیه)",
            min_value=1,
            max_value=600,
            value=10,
            step=1,
            key="tab5_poll_interval",
        )

        # کلید وضعیت آخرین کندل بسته‌شده
        last_bar_key = f"tab5_last_bar_time_{symbol}_{tf_label}"
        if last_bar_key not in st.session_state:
            st.session_state[last_bar_key] = None

        # کلید فعال/غیرفعال بودن اتوترید
        auto_flag_key = f"tab5_auto_flag_{symbol}_{tf_label}"
        if auto_flag_key not in st.session_state:
            st.session_state[auto_flag_key] = False

        # مسیر لاگ
        log_dir = Path("autotrade_logs")
        log_dir.mkdir(exist_ok=True)
        log_path = log_dir / f"log_autotrade_{Path(parquet_path).stem}_{symbol}_{tf_label}.parquet"

        # ---------------------------
        # کمک‌تابع: ساخت MAهای HTF از متاتریدر
        # ---------------------------
        def build_htf_ma_features_from_mt5(
            base_times: pd.Series,
            symbol: str,
            tfs_list: list,
            ma_cfg_list: list,
        ) -> pd.DataFrame:
            """
            ساخت MA و rank برای تایم‌فریم‌های بالاتر از روی داده‌های واقعی MT5

            - برای هر TF در tfs_list (مثلاً "5T", "15T", "30T", "1H", "4H", "1D")
            از متاتریدر حدود 5000 کندل همان TF را می‌گیرد
            - روی close آن TF، برای هر (typ, per) در ma_cfg_list:
                MA1_TYP_PER, ..., MA10_TYP_PER
            را می‌سازد و بعد rank آن‌ها نسبت به close را حساب می‌کند:
                MA1_rank, ..., MA10_rank
            - بعد این ستون‌ها را روی base_times (کندل‌های 5 دقیقه‌ای) align می‌کند
            و با نام‌هایی دقیقاً مطابق تب «افزودن فیچر» برمی‌گرداند:
                MA1_EMA_400_TF_1D, MA1_rank_TF_1D, ...
            """

            # اگر تنظیمات خالی بود، فقط یک DF خالی با همان ایندکس برگردان
            base_times = pd.to_datetime(base_times, utc=True, errors="coerce")
            if not tfs_list or not ma_cfg_list:
                return pd.DataFrame(index=base_times)

            df_all = pd.DataFrame(index=base_times)

            # ستون‌هایی که فیچر نیستند و بعداً حذف می‌کنیم
            skip_cols = {"open", "high", "low", "close", "real_volume", "tick_volume", "spread"}

            for tf_raw in tfs_list:
                tf_str = str(tf_raw)
                tf_mt5 = tf_str_to_mt5(tf_str)
                if tf_mt5 is None:
                    continue

                # حدود 5000 کندل از TF بالاتر
                rates = mt5.copy_rates_from_pos(symbol, tf_mt5, 0, 5000)
                if rates is None or len(rates) == 0:
                    continue

                htf = pd.DataFrame(rates)
                if "time" not in htf.columns or "close" not in htf.columns:
                    continue

                htf["time"] = pd.to_datetime(htf["time"], unit="s", utc=True, errors="coerce")
                htf = htf.sort_values("time").set_index("time")
                if htf.empty:
                    continue

                c_htf = to_cp(htf["close"])
                ma_list_cp = []
                ma_cols = []

                # ---------- خود MAها روی TF بالاتر ----------
                for jdx, (typ, per) in enumerate(ma_cfg_list, start=1):
                    typ_u = str(typ).upper()
                    try:
                        p = int(per)
                    except Exception:
                        # اگر پریود خراب بود، به جای اسکپ، مثل پریود 1 رفتار کن
                        p = 1

                    if p <= 1:
                        # برای اینکه MAx و rankx حذف نشوند، در این حالت
                        # خود close را به عنوان MA در نظر می‌گیریم
                        ma_cp = c_htf.copy()
                    else:
                        if typ_u == "SMA":
                            ma_cp = rolling_mean_cp(c_htf, p)
                        else:
                            ma_cp = ema_cp(c_htf, p)

                    col_ma = f"MA{jdx}_{typ_u}_{p}"
                    htf[col_ma] = cp_to_np32(ma_cp)
                    ma_list_cp.append(ma_cp)
                    ma_cols.append(col_ma)

                # ---------- rankها روی TF بالاتر ----------
                rank_cols = []
                if ma_list_cp:
                    MA = cp.stack(ma_list_cp, axis=1)           # (n, k)
                    dist = cp.abs(MA - c_htf[:, None])          # فاصله تا close
                    order = cp.argsort(dist, axis=1)
                    n, k = dist.shape
                    ranks = cp.empty_like(order)
                    row_idx = cp.arange(n)[:, None]
                    ranks[row_idx, order] = cp.arange(k)[None, :] + 1

                    for j in range(k):
                        rname = f"MA{j+1}_rank"
                        htf[rname] = cp_to_np32(ranks[:, j])
                        rank_cols.append(rname)


                # ---------- align روی زمان‌های TF پایه ----------
                htf_aligned = htf.reindex(base_times, method="ffill")

                new_cols = [c for c in htf_aligned.columns if c not in skip_cols]
                if not new_cols:
                    continue

                # دقیقاً همان suffix تب فیچر: "5T" -> "5m", "15T" -> "15m"
                suffix = tf_str.replace("T", "m")

                for cname in new_cols:
                    full_name = f"{cname}_TF_{suffix}"
                    df_all[full_name] = (
                        htf_aligned[cname].astype(np.float32).to_numpy(copy=False)
                    )

            return df_all

        # ---------------------------
        # کمک‌تابع: ساخت فیچر لایو + اجرای مدل
        # ---------------------------
        def build_live_features_with_htf(bars_df: pd.DataFrame) -> tuple[str, dict]:
            """
            ساخت فیچرهای لایو برای تب معامله خودکار.

            - دقیقا از همان feature_cfg ذخیره‌شده استفاده می‌کند.
            - فیچرهای تایم‌فریم پایه با apply_feature_pipeline_gpu ساخته می‌شوند.
            - MA و rank تایم‌فریم‌های بالاتر از متاتریدر گرفته می‌شوند و
            با همان نام‌های تب «افزودن فیچر» به df_feat اضافه می‌شوند.
            - هیچ ردیفی به خاطر NaN حذف نمی‌شود؛ فقط روی کندل پایه چک NaN/صفر انجام می‌شود.
            """

            if bars_df is None or len(bars_df) == 0:
                return "too_short", {"msg": "bars_df خالی است."}

            # ۱) تنظیمات فیچر
            cfg_all = feature_cfg
            if cfg_all is None:
                return "no_feature_cfg", {"msg": "feature_cfg_gpu برای این مدل لود نشده است."}

            ma_cfg_list = list(cfg_all.get("ma_cfg", []))
            tfs_list = list(cfg_all.get("tfs", []))

            # real_volume را مثل تب فیچر تنظیم می‌کنیم
            df_b = bars_df.copy()
            if "real_volume" not in df_b.columns and "tick_volume" in df_b.columns:
                df_b["real_volume"] = df_b["tick_volume"].astype(np.float32)

            # ۲) اجرای pipeline روی TF پایه (بدون HTF)
            cfg_base = dict(cfg_all)
            cfg_base["tfs"] = []   # HTFها را خودمان از MT5 می‌سازیم

            df_feat = apply_feature_pipeline_gpu(df_b, cfg_base)

            if "time" not in df_feat.columns:
                return "missing_features", {"missing": ["time"]}

            # ۳) افزودن MAهای HTF از MT5 با نام‌های مطابق تب فیچر
            # ۳) افزودن MAهای HTF از MT5 با نام‌های مطابق تب فیچر
            if ma_cfg_list and tfs_list:
                try:
                    df_htf = build_htf_ma_features_from_mt5(
                        base_times=df_feat["time"],
                        symbol=symbol,
                        tfs_list=tfs_list,
                        ma_cfg_list=ma_cfg_list,
                    )

                    # ⚠️ خیلی مهم: اندیس‌ها را عددی می‌کنیم که با هم هم‌تراز شوند
                    df_feat = df_feat.reset_index(drop=True)
                    df_htf = df_htf.reset_index(drop=True)

                    # بر اساس ترتیب ردیف‌ها کنار هم می‌چینیم
                    df_feat = pd.concat([df_feat, df_htf], axis=1)

                except Exception as e:
                    return "htf_error", {"msg": f"خطا در ساخت فیچر HTF از MT5: {e}"}


            df_feat = df_feat.sort_values("time").reset_index(drop=True)

            # ۴) اطمینان از وجود همهٔ ستون‌های موردنیاز مدل
            if not feature_cols:
                return "missing_features", {"missing": ["<feature_cols در meta خالی است>"]}

            missing_cols = [c for c in feature_cols if c not in df_feat.columns]
            if missing_cols:
                return "missing_features", {"missing": missing_cols}

            # ۵) آرایهٔ عددی همهٔ فیچرها (بدون dropna روی ردیف‌ها)
            X_all = df_feat[feature_cols].to_numpy(np.float32)
            N, D = X_all.shape

            # ۶) حداقل تعداد ردیف لازم را از روی بزرگ‌ترین دوره در feature_cfg محاسبه می‌کنیم
            max_param = 0
            for k, v in cfg_all.items():
                if isinstance(v, (int, float)):
                    try:
                        iv = int(v)
                    except Exception:
                        continue
                    if iv > 0:
                        max_param = max(max_param, iv)

            ma_cfg_list_local = cfg_all.get("ma_cfg") or []
            for item in ma_cfg_list_local:
                if not isinstance(item, (list, tuple)) or len(item) < 2:
                    continue
                try:
                    iv = int(item[1])
                except Exception:
                    continue
                if iv > 0:
                    max_param = max(max_param, iv)

            # این همون چیزی است که خواستی:
            # حداقل N = (بزرگ‌ترین دوره) + 100 ؛ اگر چیزی پیدا نشد، 3
            min_required = max_param + 100 if max_param > 0 else 3

            if N < min_required:
                return "too_short", {
                    "msg": f"N={N} < min_required={min_required} بر اساس بزرگ‌ترین دوره تنظیمات ({max_param})."
                }

                        # ۷) کندل پایه = کندل بسته‌شدهٔ آخر (سطر قبل از آخرین سطر)
                        # ۷) کندل پایه = کندل بسته‌شدهٔ آخر (سطر قبل از آخرین سطر)
            base_idx = N - 2
            if base_idx < 0 or base_idx >= N:
                return "index_error", {"msg": f"base_idx={base_idx} خارج از بازه [0,{N}) است."}

            row_base = df_feat.iloc[base_idx]

                        # ۸) چک NaN فقط روی کندل پایه برای فیچرهای مدل
            nan_cols = [c for c in feature_cols if pd.isna(row_base[c])]
            if nan_cols:
                # اگر برای NaN قبلاً ساختار خروجی را عوض کرده‌ای (base_idx, row_base و ...),
                # همان را اینجا نگه دار؛ من فقط منطق صفر را عوض می‌کنم.
                return "nan_features", {"nan_feats": nan_cols}

            # ۹) چک صفر بودن فیچرها روی کندل پایه
            # فعلاً صفر بودن هیچ فیچری را خطا در نظر نمی‌گیریم،
            # چون برای candle_pattern_* ، is_month_* ، real_volume و خیلی فیچرهای دیگه
            # مقدار ۰ کاملاً طبیعی است. اگر بعداً خواستیم، می‌توانیم
            # فقط برای چند فیچر خاص حساسیت به صفر اضافه کنیم.
            zero_cols: list[str] = []
            # zero_cols را خالی می‌گذاریم و هیچ status = "zero_features" برنمی‌گردانیم.




                        # 🔟 نرمال‌سازی با mean/std مدل

            # از mean و std که بیرون تابع (از meta) تعریف شده‌اند فقط *می‌خوانیم*
            # و نسخه‌ی محلی‌شان را برای نرمال‌سازی می‌سازیم تا مشکل scope پیش نیاید.
            mean_local = np.asarray(mean, dtype=np.float32)
            std_local  = np.asarray(std, dtype=np.float32)

            if mean_local.ndim == 1:
                mean_local = mean_local.reshape(1, -1)
            if std_local.ndim == 1:
                std_local = std_local.reshape(1, -1)

            if mean_local.shape[1] != D or std_local.shape[1] != D:
                return "feature_dim_mismatch", {
                    "X_shape": (N, D),
                    "mean_shape": mean_local.shape,
                    "std_shape": std_local.shape,
                }

            # احتیاط: std صفر نشود
            std_local = std_local.copy()
            std_local[std_local == 0] = 1.0

            Xz_all = (X_all - mean_local) / std_local  # (N, D)



            # ۱۱) ساخت ورودی مدل (ساده یا sequence)
            if (not is_seq) or window <= 1:
                x_in_np = Xz_all[base_idx: base_idx + 1]   # شکل (1, D)
            else:
                i_start = base_idx - window + 1 - target_shift
                if i_start < 0:
                    return "index_error", {"msg": "i_start < 0 برای sequence."}
                i_end = i_start + window
                if i_end > N:
                    return "index_error", {"msg": "i_end > N برای sequence."}

                seq_arr = Xz_all[i_start:i_end]
                if seq_arr.shape[0] != window:
                    return "index_error", {"msg": "طول سکانس با window برابر نیست."}
                x_in_np = seq_arr[None, :, :]  # (1, window, D)

            # ۱۲) اجرای مدل
            x_tensor = torch.from_numpy(x_in_np).to(device)
            with torch.inference_mode():
                logits = model_live(x_tensor)
                prob = torch.softmax(logits, dim=1)
                probs_np = prob.detach().cpu().numpy()[0]
                pred_idx = int(np.argmax(probs_np))
                conf_val = float(np.max(probs_np))

            pred_label_raw = idx2label.get(pred_idx, pred_idx)

            return "ok", {
                "x_np": x_in_np,
                "pred_idx": pred_idx,
                "pred_label_raw": pred_label_raw,
                "conf": conf_val,
                "base_idx": base_idx,
                "df_feat": df_feat,
            }

        # ---------------------------
                # ---------------------------
        # کنترل شروع/توقف اتوترید
        # ---------------------------
        st.markdown("---")
        st.subheader("کنترل اجرای اتوترید")

        c_run1, c_run2 = st.columns(2)
        status_placeholder = st.empty()
        table_placeholder = st.empty()

        # دکمه شروع/توقف
        with c_run1:
            if not st.session_state[auto_flag_key]:
                if st.button("▶️ شروع معامله خودکار", key="tab5_start_auto"):
                    st.session_state[auto_flag_key] = True
                    status_placeholder.success("معامله خودکار شروع شد.")
            else:
                if st.button("⏹ توقف معامله خودکار", key="tab5_stop_auto"):
                    st.session_state[auto_flag_key] = False
                    status_placeholder.warning("معامله خودکار متوقف شد.")

        # دکمه نمایش لاگ
        with c_run2:
            if st.button("📄 مشاهده/بروزرسانی لاگ", key="tab5_show_log"):
                if os.path.exists(log_path):
                    df_log_existing = pd.read_parquet(log_path)
                    with table_placeholder.container():
                        st.markdown("### چند سطر آخر لاگ معامله خودکار")
                        st.dataframe(df_log_existing.tail(20), use_container_width=True)
                else:
                    st.info("هنوز هیچ لاگی ثبت نشده است.")

        # ---------------------------
        # حلقهٔ اصلی — فقط وقتی flag روشن است
        # ---------------------------
        if st.session_state[auto_flag_key]:
            status_placeholder.info("در حال پایش کندل‌های جدید و اجرای اتوترید ...")

            # ۱) دریافت داده از متاتریدر
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 2000)
            if rates is None or len(rates) < 50:
                status_placeholder.error("ناتوان در دریافت داده‌های کندل از متاتریدر.")
            else:
                df_bars = pd.DataFrame(rates)
                df_bars["time"] = pd.to_datetime(df_bars["time"], unit="s", errors="coerce", utc=True)
                df_bars = df_bars.sort_values("time").reset_index(drop=True)

                # ۲) زمان فعلی سرور متاتریدر برای سنجش سن کندل جاری
                tick = mt5.symbol_info_tick(symbol)
                if tick is None:
                    status_placeholder.error("ناتوان در دریافت تیک برای محاسبه زمان فعلی سرور.")
                else:
                    now_server = pd.to_datetime(tick.time, unit="s", utc=True)

                    # کندل جاری (هنوز بسته نشده) = آخرین ردیف
                    current_bar_open = df_bars["time"].iloc[-1]
                    elapsed_sec = (now_server - current_bar_open).total_seconds()
                    if elapsed_sec < 0:
                        elapsed_sec = 0.0

                    # اگر بیشتر از ۱۵ ثانیه از شروع کندل جاری گذشته → در این چرخه پردیکتی انجام نمی‌شود
                    if elapsed_sec >= 30:
                        status_placeholder.info(
                            f"از شروع کندل جاری حدود {elapsed_sec:.1f} ثانیه گذشته؛ در این چرخه پردیکتی انجام نمی‌شود."
                        )
                    else:
                        # کندل پایه = کندل بسته‌شدهٔ قبلی (سطر قبل از آخرین سطر)
                        base_candle_time = df_bars["time"].iloc[-2]

                        # اگر قبلاً برای این کندل پردیکت کرده‌ایم، دوباره انجام نده
                        if (
                            st.session_state[last_bar_key] is not None
                            and st.session_state[last_bar_key] == base_candle_time
                        ):
                            status_placeholder.info("برای این کندل قبلاً پردیکت انجام شده؛ منتظر کندل بعدی هستیم.")
                        else:
                            # این اولین بار است که برای این کندل پردیکت می‌کنیم
                            st.session_state[last_bar_key] = base_candle_time

                            # ۳) ساخت فیچرهای لایو
                            status, out = build_live_features_with_htf(df_bars)

                            # ۴) هندل وضعیت‌های مختلف
                            if status != "ok":
                                if status == "too_short":
                                    status_placeholder.warning(
                                        "عدم امکان ساخت فیچر لایو (too_short) – احتمالاً به دلیل حذف ردیف‌های دارای NaN و کم شدن داده."
                                    )
                                elif status == "missing_features":
                                    missing = out.get("missing", [])
                                    status_placeholder.error(
                                        "عدم امکان ساخت فیچر لایو: برخی فیچرهای مورد نیاز مدل موجود نیستند:\n"
                                        + ", ".join(missing)
                                    )
                                elif status in ("nan_features", "zero_features"):
                                    # اگر build_live_features_with_htf این اطلاعات را برگردانده باشد، جدول تشخیصی بساز
                                    bad_feats = out.get("nan_feats") or out.get("zero_feats") or []
                                    problem_text = "مقدار NaN دارند" if status == "nan_features" else "مقدار ۰ دارند (که نباید باشد)"

                                    base_idx_local = int(out.get("base_idx", len(df_bars) - 2))
                                    row_base = out.get("row_base", {}) or {}

                                    status_placeholder.error(
                                        "عدم امکان ساخت فیچر لایو: این فیچرها روی کندل پایه "
                                        + problem_text
                                        + ":\n"
                                        + (", ".join(bad_feats) if bad_feats else "(هیچ فیچری گزارش نشده)")
                                        + "\n\nجدول زیر وضعیت همهٔ فیچرها و اطلاعات کندل پایه را نشان می‌دهد."
                                    )

                                    try:
                                        try:
                                            base_bar = df_bars.iloc[base_idx_local]
                                        except Exception:
                                            base_bar = df_bars.iloc[-2]

                                        rows = []

                                        candle_cols = [
                                            ("candle.time", "time"),
                                            ("candle.open", "open"),
                                            ("candle.high", "high"),
                                            ("candle.low", "low"),
                                            ("candle.close", "close"),
                                        ]
                                        for name_display, col_name in candle_cols:
                                            if col_name in base_bar.index:
                                                val = base_bar[col_name]
                                                missing = pd.isna(val)
                                                rows.append(
                                                    {
                                                        "نام": name_display,
                                                        "مقدار": "" if missing else str(val),
                                                        "وضعیت": "❌" if missing else "✅",
                                                    }
                                                )

                                        for feat in feature_cols:
                                            val = row_base.get(feat, None)
                                            try:
                                                missing = pd.isna(val)
                                            except Exception:
                                                missing = val is None

                                            rows.append(
                                                {
                                                    "نام": feat,
                                                    "مقدار": "" if missing else str(val),
                                                    "وضعیت": "❌" if missing else "✅",
                                                }
                                            )

                                        df_info = pd.DataFrame(rows)
                                        with table_placeholder.container():
                                            st.markdown("### جدول وضعیت فیچرها روی کندل پایه")
                                            st.dataframe(df_info, use_container_width=True)

                                    except Exception as e_diag:
                                        status_placeholder.warning(f"خطا در ساخت جدول تشخیصی فیچرها: {e_diag}")

                                elif status == "feature_dim_mismatch":
                                    status_placeholder.error(
                                        f"عدم تطابق ابعاد فیچرها با mean/std مدل: {out}"
                                    )
                                elif status == "htf_error":
                                    status_placeholder.error(
                                        f"خطا در ساخت فیچر HTF از MT5: {out.get('msg', '')}"
                                    )
                                else:
                                    status_placeholder.warning(f"عدم امکان ساخت فیچر لایو ({status})")

                            else:
                                # ۵) مدل با موفقیت اجرا شده است
                                pred_label_raw = out["pred_label_raw"]
                                conf_val = out["conf"]
                                df_feat = out["df_feat"]
                                base_idx_local = out["base_idx"]

                                # جهت معامله بر اساس برچسب
                                direction = "none"
                                if label_name == "label_m3":
                                    # در label_m3:
                                    # 1 = TP-BUY, 2 = TP-SELL, 3 = SL, 4 = NoTrade, 5 = Other/NoTrade
                                    if pred_label_raw == 1:
                                        direction = "buy"
                                    elif pred_label_raw == 2:
                                        direction = "sell"
                                    else:
                                        direction = "none"
                                else:
                                    # برای سایر برچسب‌ها
                                    if pred_label_raw in (1, "BUY", "buy"):
                                        direction = "buy"
                                    elif pred_label_raw in (2, "SELL", "sell"):
                                        direction = "sell"
                                    else:
                                        direction = "none"

                                executed_trade = False
                                entry_price = None
                                sl_price = None
                                tp_price = None
                                lot_val = 0.0

                                if direction in ("buy", "sell"):
                                    last_close = float(df_bars["close"].iloc[-2])
                                    last_spread = float(df_bars["spread"].iloc[-2]) if "spread" in df_bars.columns else spread_cfg

                                    sl_price_for_order = None
                                    tp_price_for_order = None

                                    if label_name == "label_m3":
                                        # آرایه‌ها برای تابع compute_m3_sl_tp_for_index
                                        high_arr = df_bars["high"].to_numpy(np.float64)
                                        low_arr = df_bars["low"].to_numpy(np.float64)
                                        close_arr = df_bars["close"].to_numpy(np.float64)
                                        time_col = df_bars["time"]

                                        idx_bar = len(df_bars) - 2  # همان کندلی که مدل برایش پیش‌بینی کرده

                                        sl_pivot_price, tp_from_pivot, sl_is_high_live, sl_ref_time = compute_m3_sl_tp_for_index(
                                            high_arr,
                                            low_arr,
                                            close_arr,
                                            time_col,
                                            idx_bar,
                                            pip_value_cfg,
                                            m3_min_sl_pips_cfg,
                                            m3_max_sl_pips_cfg,
                                            m3_rr_cfg,
                                        )

                                        if (
                                            sl_pivot_price is None
                                            or tp_from_pivot is None
                                            or np.isnan(sl_pivot_price)
                                            or np.isnan(tp_from_pivot)
                                        ):
                                            direction = "none"
                                        else:
                                            # BUY: SL باید زیر close باشد (pivot کف)
                                            # SELL: SL باید بالای close باشد (pivot سقف)
                                            if direction == "buy":
                                                if sl_is_high_live:
                                                    # مدل گفته BUY ولی SL روی سقف است → معامله کنسل
                                                    direction = "none"
                                                else:
                                                    sl_price_for_order = sl_pivot_price
                                                    tp_price_for_order = tp_from_pivot
                                            elif direction == "sell":
                                                if not sl_is_high_live:
                                                    # مدل گفته SELL ولی SL روی کف است → معامله کنسل
                                                    direction = "none"
                                                else:
                                                    sl_price_for_order = sl_pivot_price
                                                    tp_price_for_order = tp_from_pivot
                                    else:
                                        # منطق ساده SL/TP برای برچسب‌های غیر از label_m3 (مثلاً label_m1)
                                        if direction == "buy":
                                            sl_price_for_order = last_close - m1_sl_pips_cfg * pip_value_cfg
                                            tp_price_for_order = last_close + m1_tp_pips_cfg * pip_value_cfg
                                        elif direction == "sell":
                                            sl_price_for_order = last_close + m1_sl_pips_cfg * pip_value_cfg
                                            tp_price_for_order = last_close - m1_tp_pips_cfg * pip_value_cfg

                                    if direction in ("buy", "sell") and sl_price_for_order is not None and tp_price_for_order is not None:
                                        # محاسبه فاصله SL تا قیمت ورود بر حسب pip
                                        sl_pips_effective = abs(last_close - sl_price_for_order) / pip_value_cfg

                                        lot_val_raw = calc_lot(sl_pips_effective, float(account_info.equity))
                                        lot_val = normalize_volume(symbol, lot_val_raw)

                                        if lot_val > 0:
                                            # --- ارسال سفارش واقعی به متاتریدر بر اساس پردیکت این کندل ---
                                            tick = mt5.symbol_info_tick(symbol)

                                            if tick is not None:
                                                # تعیین نوع سفارش و قیمت ورود بر اساس جهت معامله
                                                if direction == "buy":
                                                    order_type = mt5.ORDER_TYPE_BUY
                                                    price = float(getattr(tick, "ask", tick.last))
                                                elif direction == "sell":
                                                    order_type = mt5.ORDER_TYPE_SELL
                                                    price = float(getattr(tick, "bid", tick.last))
                                                else:
                                                    order_type = None
                                                    price = None

                                                if order_type is not None and price is not None:
                                                    request = {
                                                        "action": mt5.TRADE_ACTION_DEAL,
                                                        "symbol": symbol,
                                                        "volume": float(lot_val),
                                                        "type": order_type,
                                                        "price": price,
                                                        "sl": float(sl_price_for_order),
                                                        "tp": float(tp_price_for_order),
                                                        "deviation": 20,
                                                        "magic": 987654,  # هر عدد دلخواهی می‌تونی بذاری برای علامت‌گذاری ربات
                                                        "comment": f"auto_trade_{label_name}",
                                                        "type_time": mt5.ORDER_TIME_GTC,
                                                        "type_filling": mt5.ORDER_FILLING_IOC,
                                                    }

                                                    result = mt5.order_send(request)

                                                    if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
                                                        # سفارش با موفقیت اجرا شد
                                                        executed_trade = True
                                                        entry_price = float(getattr(result, "price", price))
                                                        sl_price = sl_price_for_order
                                                        tp_price = tp_price_for_order
                                                    else:
                                                        # سفارش ارسال شده ولی موفق نبوده
                                                        executed_trade = False
                                                        status_placeholder.error(
                                                            f"ارسال سفارش ناموفق بود. retcode = {getattr(result, 'retcode', 'N/A')}"
                                                        )
                                                else:
                                                    # جهت نامعتبر (نباید برسیم اینجا، مگر direction = 'none')
                                                    executed_trade = False
                                            else:
                                                # اگر به هر دلیل تیک در دسترس نباشد
                                                executed_trade = False
                                                status_placeholder.error("تیک سمبل در دسترس نیست؛ سفارش ارسال نشد.")


                                # ۶) ثبت در لاگ (حتی اگر معامله‌ای انجام نشود)
                                log_rec = {
                                    "time": base_candle_time,  # زمان کندلی که برایش پردیکت شده
                                    "symbol": symbol,
                                    "timeframe": tf_label,
                                    "label_name": label_name,
                                    "model_name": model_name,
                                    "pred_label_raw": int(pred_label_raw),
                                    "pred_conf": float(conf_val),
                                    "direction": direction,
                                    "executed": bool(executed_trade),
                                    "entry_price": float(entry_price) if entry_price is not None else np.nan,
                                    "sl_price": float(sl_price) if sl_price is not None else np.nan,
                                    "tp_price": float(tp_price) if tp_price is not None else np.nan,
                                    "lot": float(lot_val),
                                    "equity": float(account_info.equity),
                                    "pip_value_cfg": float(pip_value_cfg),
                                    "spread_cfg": float(spread_cfg),
                                    "m1_sl_pips_cfg": float(m1_sl_pips_cfg),
                                    "m1_tp_pips_cfg": float(m1_tp_pips_cfg),
                                    "m3_min_sl_pips_cfg": float(m3_min_sl_pips_cfg),
                                    "m3_max_sl_pips_cfg": float(m3_max_sl_pips_cfg),
                                    "m3_rr_cfg": float(m3_rr_cfg),
                                }

                                if os.path.exists(log_path):
                                    df_log = pd.read_parquet(log_path)
                                    df_log = pd.concat([df_log, pd.DataFrame([log_rec])], ignore_index=True)
                                else:
                                    df_log = pd.DataFrame([log_rec])

                                df_log.to_parquet(log_path, index=False)

            # ✅ همیشه جدول لاگ را آپدیت و نمایش بده
            if os.path.exists(log_path):
                df_log_existing = pd.read_parquet(log_path)
                with table_placeholder.container():
                    st.markdown("### چند سطر آخر لاگ معامله خودکار")
                    st.dataframe(df_log_existing.tail(5), use_container_width=True)
                    

            # در پایان هر چرخه، کمی صبر کن و بعد دوباره اسکریپت را اجرا کن
            time.sleep(poll_interval)
            st.rerun()


        else:
            # اگر اتوترید غیرفعال است، آخرین لاگ (در صورت وجود) را فقط یک‌بار نشان بده
            if os.path.exists(log_path):
                df_log_existing = pd.read_parquet(log_path)
                with table_placeholder.container():
                    st.markdown("### چند سطر آخر لاگ معامله خودکار")
                    st.dataframe(df_log_existing.tail(5), use_container_width=True)

    except Exception as e:
        st.error(f"❌ خطا در تب معامله خودکار: {e}")
        st.exception(e)
