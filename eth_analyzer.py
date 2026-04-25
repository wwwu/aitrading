#!/usr/bin/env python3
"""
ETH 超短线策略 — 本地技术分析引擎
从 OKX 公开 API 获取 K 线数据，在本地计算全部技术指标。
输出紧凑 JSON，供 AI 直接用于评分和决策，无需传输原始 K 线数据。

用法: python3 eth_analyzer.py
输出: JSON 格式的指标结果到 stdout
"""

import json
import sys
import urllib.request
import urllib.error

# ═══════════════════════════════════════
#  OKX API (公开，无需认证)
# ═══════════════════════════════════════

API_BASE = "https://www.okx.com"
INST_ID  = "ETH-USDT-SWAP"


def _get_json(url):
    """带 User-Agent 的 GET 请求，返回解析后的 JSON。"""
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
    })
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode())


def fetch_candles(inst_id, bar, limit):
    """从 OKX 获取 K 线数据（最新在前），返回按时间升序的列表。"""
    url = f"{API_BASE}/api/v5/market/candles?instId={inst_id}&bar={bar}&limit={limit}"
    try:
        data = _get_json(url)
    except Exception as e:
        print(f"ERROR fetching {bar} candles: {e}", file=sys.stderr)
        sys.exit(1)

    if data.get("code") != "0":
        print(f"ERROR API {bar}: {data.get('msg', 'unknown')}", file=sys.stderr)
        sys.exit(1)

    candles = []
    for c in reversed(data["data"]):  # 反转为时间升序
        candles.append({
            "ts": int(c[0]),
            "o":  float(c[1]),
            "h":  float(c[2]),
            "l":  float(c[3]),
            "c":  float(c[4]),
            "v":  float(c[5]),
        })
    return candles


def fetch_ticker(inst_id):
    """获取最新行情。"""
    url = f"{API_BASE}/api/v5/market/ticker?instId={inst_id}"
    try:
        data = _get_json(url)
    except Exception as e:
        print(f"ERROR fetching ticker: {e}", file=sys.stderr)
        sys.exit(1)

    if data.get("code") != "0":
        print(f"ERROR ticker: {data.get('msg', 'unknown')}", file=sys.stderr)
        sys.exit(1)

    t = data["data"][0]
    return {
        "last":  float(t["last"]),
        "bid":   float(t["bidPx"]),
        "ask":   float(t["askPx"]),
        "vol24": float(t["vol24h"]),
    }


# ═══════════════════════════════════════
#  技术指标计算
# ═══════════════════════════════════════

def ema(data, period):
    """指数移动平均。"""
    if len(data) < period:
        return [None] * len(data)
    k = 2 / (period + 1)
    result = [None] * (period - 1)
    result.append(sum(data[:period]) / period)
    for i in range(period, len(data)):
        result.append(data[i] * k + result[-1] * (1 - k))
    return result


def sma(data, period):
    """简单移动平均。"""
    if len(data) < period:
        return [None] * len(data)
    result = [None] * (period - 1)
    for i in range(period - 1, len(data)):
        result.append(sum(data[i - period + 1 : i + 1]) / period)
    return result


def rsi(closes, period=14):
    """RSI（Wilder 平滑法）。"""
    if len(closes) < period + 1:
        return [None] * len(closes)
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [max(d, 0) for d in deltas]
    losses = [abs(min(d, 0)) for d in deltas]
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    result = [None] * period
    rs = avg_gain / avg_loss if avg_loss != 0 else 100
    result.append(100 - 100 / (1 + rs))
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        rs = avg_gain / avg_loss if avg_loss != 0 else 100
        result.append(100 - 100 / (1 + rs))
    return result


def macd(closes, fast=12, slow=26, signal=9):
    """MACD。返回 (macd_line, signal_line, histogram)，均为列表。"""
    ema_fast = ema(closes, fast)
    ema_slow = ema(closes, slow)
    macd_line = []
    for i in range(len(closes)):
        if ema_fast[i] is not None and ema_slow[i] is not None:
            macd_line.append(ema_fast[i] - ema_slow[i])
        else:
            macd_line.append(None)
    valid_macd = [v for v in macd_line if v is not None]
    sig = ema(valid_macd, signal) if len(valid_macd) >= signal else [None] * len(valid_macd)
    signal_line = [None] * (len(macd_line) - len(sig)) + sig
    histogram = []
    for i in range(len(macd_line)):
        if macd_line[i] is not None and signal_line[i] is not None:
            histogram.append(macd_line[i] - signal_line[i])
        else:
            histogram.append(None)
    return macd_line, signal_line, histogram


def bollinger(closes, period=20, mult=2):
    """布林带。返回 (upper, middle, lower, bb_pos)。"""
    mid = sma(closes, period)
    upper, lower, bb_pos = [], [], []
    for i in range(len(closes)):
        if mid[i] is None:
            upper.append(None)
            lower.append(None)
            bb_pos.append(None)
        else:
            window = closes[i - period + 1 : i + 1]
            std = (sum((x - mid[i]) ** 2 for x in window) / period) ** 0.5
            u = mid[i] + mult * std
            l = mid[i] - mult * std
            upper.append(u)
            lower.append(l)
            bw = u - l
            bb_pos.append((closes[i] - l) / bw if bw > 0 else 0.5)
    return upper, mid, lower, bb_pos


def atr(candles, period=14):
    """ATR。"""
    if len(candles) < 2:
        return [None] * len(candles)
    trs = []
    for i in range(1, len(candles)):
        h, l, pc = candles[i]["h"], candles[i]["l"], candles[i - 1]["c"]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    result = [None] * period
    if len(trs) >= period:
        avg = sum(trs[:period]) / period
        result.append(avg)
        for i in range(period, len(trs)):
            avg = (avg * (period - 1) + trs[i]) / period
            result.append(avg)
    return result


def support_resistance(candles, lookback=48):
    """识别局部高低点作为支撑/阻力。"""
    swings_l, swings_h = [], []
    for i in range(1, len(candles) - 1):
        if i > len(candles) - lookback:
            if candles[i]["l"] <= candles[i - 1]["l"] and candles[i]["l"] <= candles[i + 1]["l"]:
                swings_l.append(candles[i]["l"])
            if candles[i]["h"] >= candles[i - 1]["h"] and candles[i]["h"] >= candles[i + 1]["h"]:
                swings_h.append(candles[i]["h"])
    sup = sum(swings_l[-3:]) / len(swings_l[-3:]) if swings_l else candles[-1]["l"]
    res = sum(swings_h[-3:]) / len(swings_h[-3:]) if swings_h else candles[-1]["h"]
    return round(sup, 2), round(res, 2)


# ═══════════════════════════════════════
#  市场状态判断
# ═══════════════════════════════════════

def detect_market_state(ema9_15m, ema21_15m, ema20_1h, ema60_1h, slope_1h, atr_vals, closes_15m):
    """趋势市 / 震荡市 / 高波动扰动市。"""
    # 高波动：ATR 显著放大
    if atr_vals and atr_vals[-1] is not None and len(atr_vals) >= 24:
        recent = [v for v in atr_vals[-24:] if v is not None]
        if recent:
            avg_atr = sum(recent) / len(recent)
            if avg_atr > 0 and atr_vals[-1] > avg_atr * 2:
                return "高波动扰动市"

    # 趋势市需要 15min 和 1H 同时确认
    if ema20_1h is not None and ema60_1h is not None and ema9_15m is not None and ema21_15m is not None:
        price = closes_15m[-1]
        gap_1h = abs(ema20_1h - ema60_1h) / ema60_1h
        gap_15m = abs(ema9_15m - ema21_15m) / price
        direction_aligned = (
            (ema20_1h > ema60_1h and slope_1h > 0) or
            (ema20_1h < ema60_1h and slope_1h < 0)
        )
        # 1H 均线发散 + 方向一致 + 斜率足够 + 15min 也有发散（非横盘）
        if gap_1h > 0.003 and abs(slope_1h) > 0.5 and direction_aligned and gap_15m > 0.001:
            return "趋势市"

    return "震荡市"


# ═══════════════════════════════════════
#  主流程
# ═══════════════════════════════════════

def main():
    # ── 1. 获取数据 ──
    candles_15m = fetch_candles(INST_ID, "15m", 96)
    candles_1h  = fetch_candles(INST_ID, "1H",  48)
    candles_4h  = fetch_candles(INST_ID, "4H",  30)
    ticker      = fetch_ticker(INST_ID)

    # ── 2. 计算指标 ──
    closes_15m = [c["c"] for c in candles_15m]
    closes_1h  = [c["c"] for c in candles_1h]
    closes_4h  = [c["c"] for c in candles_4h]

    # 15min
    ema9   = ema(closes_15m, 9)
    ema21  = ema(closes_15m, 21)
    rsi_15 = rsi(closes_15m, 14)
    m_line, m_sig, m_hist = macd(closes_15m)
    bb_u, bb_m, bb_l, bb_pos = bollinger(closes_15m)
    atr_15 = atr(candles_15m, 14)
    sup, res = support_resistance(candles_15m)

    # 1H
    ema20_1h = ema(closes_1h, 20)
    ema60_1h = sma(closes_1h, 48)   # SMA48 代理 EMA60
    rsi_1h   = rsi(closes_1h, 14)

    # 4H
    ema20_4h = ema(closes_4h, 20)

    # 最近3根15min K线（用于结构判断）
    last3 = candles_15m[-3:]

    # ── 3. 组装输出 ──
    def safe(v, d=2):
        """安全取值，None 返回 None，否则四舍五入。"""
        return round(v, d) if v is not None else None

    e9  = safe(ema9[-1])
    e21 = safe(ema21[-1])
    e20_1h = safe(ema20_1h[-1])
    e60_1h = safe(ema60_1h[-1])
    e20_4h = safe(ema20_4h[-1])

    # EMA 斜率
    def slope(arr, n=3):
        valid = [v for v in arr[-n:] if v is not None]
        if len(valid) < 2:
            return 0
        return round(valid[-1] - valid[0], 2)

    sl_15_e9  = slope(ema9)
    sl_15_e21 = slope(ema21)
    sl_1h     = slope(ema20_1h)
    sl_4h     = slope(ema20_4h)

    # 市场状态
    market_state = detect_market_state(e9, e21, e20_1h, e60_1h, sl_1h, atr_15, closes_15m)

    result = {
        "price": ticker["last"],
        "bid": ticker["bid"],
        "ask": ticker["ask"],
        "vol24h": round(ticker["vol24"], 0),
        "marketState": market_state,
        "indicators": {
            "15m": {
                "ema9": e9,
                "ema21": e21,
                "ema9_slope": sl_15_e9,
                "ema21_slope": sl_15_e21,
                "rsi": safe(rsi_15[-1], 1),
                "macd": safe(m_line[-1]),
                "macd_signal": safe(m_sig[-1]),
                "macd_hist": safe(m_hist[-1]),
                "atr": safe(atr_15[-1]),
                "bb_upper": safe(bb_u[-1]),
                "bb_middle": safe(bb_m[-1]),
                "bb_lower": safe(bb_l[-1]),
                "bb_pos": safe(bb_pos[-1], 3),
                "vol_last": candles_15m[-1]["v"],
                "vol_ma20": safe(sum(c["v"] for c in candles_15m[-20:]) / 20, 0),
            },
            "1h": {
                "ema20": e20_1h,
                "ema60": e60_1h,
                "ema20_slope": sl_1h,
                "rsi": safe(rsi_1h[-1], 1),
            },
            "4h": {
                "ema20": e20_4h,
                "ema20_slope": sl_4h,
            },
        },
        "structure": {
            "support": sup,
            "resistance": res,
        },
        "last3": [
            {"o": c["o"], "h": c["h"], "l": c["l"], "c": c["c"], "v": round(c["v"], 0)}
            for c in last3
        ],
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
