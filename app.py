import os
import csv
import logging
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, request, jsonify

APP_VERSION = "1.7-chatty-genius-stable"

# 将来のpandas仕様変更による冗長な警告を抑制（根本修正も実施済み）
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

# -------------------- ログ --------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)

# ---------------- ヘルパー：安全にfloat化 ----------------
def to_float(x):
    """単一要素Series / numpy scalar / Python number を float に安全変換"""
    try:
        if hasattr(x, "iloc"):  # 単一要素Series想定
            return float(x.iloc[0])
        return float(x)
    except Exception:
        try:
            return float(np.asarray(x).item())
        except Exception:
            return float(np.asarray(x, dtype="float64"))

def as_scalar(x):
    """
    x が Series/ndarray/np scalar でも、とにかく「最後の1要素の float 」を返す。
    NaNなら float('nan') を返す。
    """
    try:
        arr = np.asarray(x, dtype="float64").reshape(-1)
        return float(arr[-1]) if arr.size else float("nan")
    except Exception:
        return float("nan")

# ---------------- ユニバース（CSV優先 / 日経225フォールバック） ----------------
NIKKEI_225 = [
    "1332.T","1333.T","1605.T","1721.T","1801.T","1802.T","1803.T","1808.T","1812.T",
    "1925.T","1928.T","1963.T","2002.T","2269.T","2282.T","2413.T","2432.T","2501.T",
    "2502.T","2503.T","2531.T","2768.T","2801.T","2802.T","2871.T","2914.T","3086.T",
    "3099.T","3101.T","3103.T","3105.T","3289.T","3382.T","3401.T","3402.T","3405.T",
    "3407.T","3436.T","3606.T","3659.T","3861.T","3863.T","4061.T","4063.T","4151.T",
    "4183.T","4188.T","4208.T","4272.T","4324.T","4452.T","4502.T","4503.T","4506.T",
    "4507.T","4519.T","4523.T","4543.T","4568.T","4578.T","4689.T","4704.T","4751.T",
    "4755.T","4901.T","4902.T","4911.T","5020.T","5101.T","5108.T","5201.T","5202.T",
    "5214.T","5232.T","5233.T","5301.T","5332.T","5333.T","5401.T","5406.T","5411.T",
    "5541.T","5631.T","5703.T","5706.T","5707.T","5711.T","5713.T","5714.T","5801.T",
    "5802.T","5803.T","5901.T","6103.T","6113.T","6301.T","6302.T","6305.T","6326.T",
    "6361.T","6367.T","6471.T","6472.T","6473.T","6479.T","6501.T","6503.T","6504.T",
    "6506.T","6645.T","6674.T","6701.T","6702.T","6703.T","6723.T","6724.T","6752.T",
    "6753.T","6754.T","6758.T","6762.T","6770.T","6841.T","6857.T","6902.T","6952.T",
    "6954.T","6971.T","6976.T","6981.T","6988.T","6995.T","7003.T","7004.T","7011.T",
    "7012.T","7013.T","7180.T","7201.T","7202.T","7203.T","7205.T","7211.T","7261.T",
    "7267.T","7269.T","7270.T","7272.T","7731.T","7733.T","7735.T","7751.T","7752.T",
    "7762.T","7832.T","8031.T","8035.T","8053.T","8058.T","8267.T","8270.T","8303.T",
    "8304.T","8306.T","8308.T","8309.T","8316.T","8331.T","8354.T","8355.T","8410.T",
    "8411.T","8591.T","8593.T","8601.T","8604.T","8628.T","8630.T","8697.T","8725.T",
    "8729.T","8750.T","8766.T","8769.T","8795.T","8801.T","8802.T","8804.T","8830.T",
    "9001.T","9005.T","9007.T","9008.T","9009.T","9020.T","9021.T","9022.T","9064.T",
    "9065.T","9069.T","9101.T","9104.T","9107.T","9201.T","9202.T","9301.T","9303.T",
    "9412.T","9432.T","9433.T","9434.T","9501.T","9502.T","9503.T","9531.T","9532.T",
    "9602.T","9613.T","9681.T","9735.T","9766.T","9983.T","9984.T","9989.T"
]
CSV_PATH = os.environ.get("UNIVERSE_CSV", "universe.csv")

def load_universe():
    if os.path.exists(CSV_PATH):
        try:
            with open(CSV_PATH, newline="") as f:
                tickers = [row["ticker"].strip() for row in csv.DictReader(f) if row.get("ticker")]
            if tickers:
                log.info(f"Loaded universe from CSV ({len(tickers)} tickers)")
                return tickers
        except Exception as e:
            log.warning(f"Failed to load CSV universe: {e}")
    log.info(f"Using built-in Nikkei 225 universe ({len(NIKKEI_225)} tickers)")
    return NIKKEI_225

UNIVERSE = load_universe()

# ---------------- yfinance DL（列を単純化して欠損除去） ----------------
def fetch_ticker_df(ticker: str, period: str = "90d"):
    try:
        df = yf.download(
            ticker,
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False
        )
        if df is None or df.empty:
            return None
        df = df.rename(columns=str.title)
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        if df.empty:
            return None
        return df
    except Exception as e:
        log.warning(f"Download failed for {ticker}: {e}")
        return None

# -------------------- インジケーター --------------------
def rsi(series: pd.Series, n: int = 14):
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1/n, adjust=False).mean()
    roll_down = down.ewm(alpha=1/n, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-9)
    return 100.0 - (100.0 / (1.0 + rs))

def bollinger(series: pd.Series, n: int = 20, k: float = 2.0):
    ma = series.rolling(n).mean()
    std = series.rolling(n).std(ddof=0)
    upper = ma + k * std
    lower = ma - k * std
    return ma, upper, lower

# ---------------- 指標・スコア（可変lookback / RSI / BB） ----------------
def score_today_vs_past(
    today: pd.Series, df: pd.DataFrame,
    lookback=20, rsi_n=14, bb_n=20, bb_k=2.0, use_close_breakout=False
):
    # 十分なデータがない場合のフォールバック
    if len(df) < lookback + 1:
        return {
            "close": None, "turnover": 0, "rvol": 0.0, "breakout": False,
            "atr_pct": 0.0, "breakout_gap": 0.0, "level": None, "score": 0.0,
            "rsi": None, "bb_ma": None, "bb_upper": None
        }

    # 当日を除く直近N日
    pastN = df.iloc[-(lookback+1):-1]

    # スカラ化して安全に取り出し
    close = to_float(today["Close"])
    vol   = to_float(today["Volume"])

    # ブレイク基準（終値 or 高値）
    base_series = pastN["Close"] if use_close_breakout else pastN["High"]
    level = to_float(base_series.max())

    vol_mean = to_float(pastN["Volume"].mean())
    turnover = float(close * vol)
    rvol = float(vol / (vol_mean + 1e-9))
    breakout = bool(close > level)

    # ATR%・ブレイク距離
    atr = to_float((pastN["High"] - pastN["Low"]).mean())
    atr_pct = float(atr / close) if close else 0.0
    breakout_gap = float((close - level) / level) if level else 0.0

    # RSI（当日値を安全に）
    rsi_series = rsi(df["Close"], n=rsi_n)
    rsi_val = as_scalar(rsi_series.iloc[-1])

    # ボリンジャー（当日値を安全に）
    bb_ma, bb_upper, bb_lower = bollinger(df["Close"], n=bb_n, k=bb_k)
    bb_ma_last    = as_scalar(bb_ma.iloc[-1])
    bb_upper_last = as_scalar(bb_upper.iloc[-1])
    bb_ma_t   = None if (bb_ma_last != bb_ma_last) else bb_ma_last   # NaN判定
    bb_upper_t= None if (bb_upper_last != bb_upper_last) else bb_upper_last

    # 簡易スコア
    score = 0.5 * rvol + 100 * atr_pct + 50 * max(breakout_gap, 0.0)

    return {
        "close": float(close),
        "turnover": int(turnover),
        "rvol": float(rvol),
        "breakout": breakout,
        "atr_pct": float(atr_pct),
        "breakout_gap": float(breakout_gap),
        "level": float(level),
        "score": float(score),
        "rsi": rsi_val,
        "bb_ma": bb_ma_t,
        "bb_upper": bb_upper_t,
    }

# ---------------- スクリーニング（並列化 & 新ルール対応） ----------------
def screen_candidates(
    universe,
    min_turnover_jpy=3e8,
    max_workers=12,
    lookback=20,
    use_close_breakout=False,
    rsi_n=14, rsi_min=None, rsi_max=None,
    bb_n=20, bb_k=2.0, bb_mode="none"  # none / above_upper / near_upper
):
    results = []

    def bb_pass(m):
        if bb_mode == "none":
            return True
        # None/NaN を一括で弾く
        for key in ("close", "bb_ma", "bb_upper"):
            v = m.get(key, None)
            if v is None or not np.isfinite(v):
                return False
        c, ma, up = float(m["close"]), float(m["bb_ma"]), float(m["bb_upper"])
        if bb_mode == "above_upper":
            return c > up
        if bb_mode == "near_upper":
            # MA と Upper の中間より上ならOK（「上限近辺」定義）
            return c > (ma + (up - ma) * 0.5)
        return True

    def rsi_pass(m):
        if m["rsi"] is None or not np.isfinite(m["rsi"]):
            return False
        if rsi_min is not None and not (m["rsi"] >= float(rsi_min)):
            return False
        if rsi_max is not None and not (m["rsi"] <= float(rsi_max)):
            return False
        return True

    def one(t):
        try:
            df = fetch_ticker_df(t, period="90d")
            if df is None or len(df) < max(lookback + 1, 25):
                return None
            today = df.iloc[-1]
            m = score_today_vs_past(
                today, df,
                lookback=lookback,
                rsi_n=rsi_n,
                bb_n=bb_n, bb_k=bb_k,
                use_close_breakout=use_close_breakout
            )
            if not (m["breakout"] and (m["rvol"] > 2.0)):
                return None
            if not bool(m["turnover"] >= min_turnover_jpy):
                return None
            if not bb_pass(m):
                return None
            if (rsi_min is not None) or (rsi_max is not None):
                if not rsi_pass(m):
                    return None

            return {
                "ticker": t,
                "close": float(m["close"]),
                "turnover": int(m["turnover"]),
                "rvol": float(m["rvol"]),
                "atr_pct": float(m["atr_pct"]),
                "breakout_gap": float(m["breakout_gap"]),
                "score": float(m["score"]),
                "rsi": float(m["rsi"]) if m["rsi"] is not None else None,
                "bb_ma": float(m["bb_ma"]) if m["bb_ma"] is not None else None,
                "bb_upper": float(m["bb_upper"]) if m["bb_upper"] is not None else None,
                "reason_tags": [
                    "BREAKOUT_10D" if lookback == 10 else "BREAKOUT_20D",
                    "CLOSE_BREAK" if use_close_breakout else "HIGH_BREAK",
                    "RVOL_GT_2", "LIQ_OK"
                ]
            }
        except Exception as e:
            log.error(f"[ERROR] {t}: {e}")
            return None

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(one, t) for t in universe]
        for fut in as_completed(futures):
            r = fut.result()
            if r:
                results.append(r)

    results.sort(key=lambda x: x["score"], reverse=True)
    return results

# --------------------------- API ---------------------------
@app.route("/")
def health():
    return jsonify({
        "ok": True,
        "version": APP_VERSION,
        "ts": datetime.now(timezone(timedelta(hours=9))).isoformat(),
        "universe_size": len(UNIVERSE)
    })

@app.route("/screen", methods=["GET"])
def screen():
    top_n = int(request.args.get("top", 8))
    min_turnover = float(request.args.get("min_turnover", 3e8))
    workers = int(request.args.get("workers", 12))

    lookback = int(request.args.get("lookback", 20))  # 例: 10
    use_close_breakout = request.args.get("use_close_breakout", "false").lower() == "true"

    rsi_n = int(request.args.get("rsi_n", 14))
    rsi_min = request.args.get("rsi_min")
    rsi_max = request.args.get("rsi_max")
    rsi_min = float(rsi_min) if rsi_min is not None else None
    rsi_max = float(rsi_max) if rsi_max is not None else None

    bb_n = int(request.args.get("bb_n", 20))
    bb_k = float(request.args.get("bb_k", 2.0))
    bb_mode = request.args.get("bb_mode", "none")  # none / above_upper / near_upper

    res = screen_candidates(
        UNIVERSE,
        min_turnover_jpy=min_turnover,
        max_workers=workers,
        lookback=lookback,
        use_close_breakout=use_close_breakout,
        rsi_n=rsi_n, rsi_min=rsi_min, rsi_max=rsi_max,
        bb_n=bb_n, bb_k=bb_k, bb_mode=bb_mode
    )
    return jsonify({"count": len(res[:top_n]), "candidates": res[:top_n]})

@app.route("/explain", methods=["GET"])
def explain():
    ticker = request.args.get("ticker")
    if not ticker:
        return jsonify({"error": "ticker is required"}), 400

    # /explain でも同じパラメータで計算できるように
    lookback = int(request.args.get("lookback", 20))
    use_close_breakout = request.args.get("use_close_breakout", "false").lower() == "true"
    rsi_n = int(request.args.get("rsi_n", 14))
    bb_n = int(request.args.get("bb_n", 20))
    bb_k = float(request.args.get("bb_k", 2.0))

    df = fetch_ticker_df(ticker, period="90d")
    if df is None or len(df) < max(lookback + 1, 25):
        return jsonify({"error": "not enough data"}), 400

    today = df.iloc[-1]
    m = score_today_vs_past(
        today, df,
        lookback=lookback,
        rsi_n=rsi_n,
        bb_n=bb_n, bb_k=bb_k,
        use_close_breakout=use_close_breakout
    )

    info = {
        "ticker": ticker,
        "today_close": float(m["close"]) if m["close"] is not None else None,
        "break_level": float(m["level"]) if m["level"] is not None else None,
        "rvol": float(m["rvol"]),
        "atr_pct": float(m["atr_pct"]),
        "breakout_gap": float(m["breakout_gap"]),
        "turnover": int(m["turnover"]),
        "rsi": float(m["rsi"]) if m["rsi"] is not None else None,
        "bb_ma": float(m["bb_ma"]) if m["bb_ma"] is not None else None,
        "bb_upper": float(m["bb_upper"]) if m["bb_upper"] is not None else None,
        "reason_tags": [
            ("BREAKOUT_10D" if lookback == 10 else "BREAKOUT_20D")
            if m["level"] is not None else "NO_LEVEL",
            "CLOSE_BREAK" if use_close_breakout else "HIGH_BREAK",
            "RVOL_GT_2" if m["rvol"] > 2.0 else "RVOL_NOT_ENOUGH",
            "LIQ_OK" if m["turnover"] >= 3e8 else "LIQ_LOW"
        ]
    }
    return jsonify(info)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
