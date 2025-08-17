# strategy_domestic_intraday.py — KOSPI/KOSDAQ intraday strategy
from __future__ import annotations
import json, time
from datetime import datetime, timedelta

from kis_common import (
    create_kis_context, get_domestic_last_price, call_with_retry,
    dbuy_limit, dsell_market, dsell_limit
)

# ===== 설정 (국내) =====
TICKER = "005930"   # KOSPI 6자리 (예: 삼성전자 005930)
QTY    = 10
ENTRY_PRICE = 72500       # 국내는 정수 호가 (원 단위)
STOP_PRICE  = 71800       # 스탑 라인 (원)
POLL_SEC    = 10
MAX_HOURS   = 6
SELL_METHOD = "market"    # "market" | "limit"
LIMIT_SLIP  = 0.003       # 지정가 사용 시, stop/last 보다 0.3% 낮춰 던지기

# --- KRX 호가단위 보정 ---
def krx_tick_size(price: int) -> int:
    p = int(price)
    if p < 1000: return 1
    if p < 5000: return 5
    if p < 10000: return 10
    if p < 50000: return 50
    if p < 100000: return 100
    if p < 500000: return 500
    return 1000

def align_krx_price(price: float | int, direction: str = "down") -> int:
    """direction: 'down' or 'up'"""
    p = int(round(float(price)))
    tick = krx_tick_size(p)
    if direction == "up":
        return ((p + tick - 1) // tick) * tick
    return (p // tick) * tick

def run():
    ctx = create_kis_context()
    token = ctx["token"]; appkey = ctx["appkey"]; appsecret = ctx["appsecret"]

    # 1) 시초가 매수(지정가, 호가단위 보정)
    try:
        px = align_krx_price(ENTRY_PRICE, "up")
        print(f"[INTRA-KRX] 매수 {TICKER}: {QTY} @ {px}")
        resp_buy = call_with_retry(dbuy_limit, token, appkey, appsecret, TICKER, int(QTY), int(px))
        print(json.dumps(resp_buy, ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"[INTRA-KRX] 매수 실패: {e}"); return

    half = max(1, int(QTY // 2)); sold_half = False
    stop = align_krx_price(STOP_PRICE, "down")
    print(f"[INTRA-KRX] 스탑: {stop} | method={SELL_METHOD}, slip={LIMIT_SLIP if SELL_METHOD=='limit' else 'N/A'}")

    start = datetime.now()
    while (datetime.now() - start) < timedelta(hours=MAX_HOURS):
        last = get_domestic_last_price(TICKER)
        if last is not None:
            last = int(round(float(last)))
            print(f"[INTRA-KRX] 현재가: {last}")

            # (A) 스탑 트리거 → 50% 매도
            if (not sold_half) and last <= stop:
                try:
                    resp_s1 = None
                    if SELL_METHOD.lower() == "market":
                        print(f"[INTRA-KRX] 50% 시장가 매도 시도: {half}")
                        try:
                            resp_s1 = call_with_retry(dsell_market, token, appkey, appsecret, TICKER, int(half))
                        except Exception as e:
                            print(f"[INTRA-KRX] 시장가 실패 → 지정가 폴백: {e}")
                    if (resp_s1 is None) or (isinstance(resp_s1, dict) and resp_s1.get("rt_cd") != "0"):
                        # 지정가 폴백: stop/last 중 낮은 값에서 slip 만큼 더 낮춰 보수적으로
                        base_px = min(stop, last)
                        px = align_krx_price(int(base_px * (1.0 - float(LIMIT_SLIP))), "down")
                        print(f"[INTRA-KRX] 50% 지정가 폴백: {half} @ {px}")
                        resp_s1 = call_with_retry(dsell_limit, token, appkey, appsecret, TICKER, int(half), int(px))
                    print(json.dumps(resp_s1, ensure_ascii=False, indent=2))
                    sold_half = True
                except Exception as e:
                    print(f"[INTRA-KRX] 50% 매도 실패: {e}")

            # (B) 시간봉 간이 조건: 매 분 59분, 가격이 여전히 stop 이하 → 잔여 50%
            if sold_half and datetime.now().minute == 59 and last <= stop:
                try:
                    remain = max(1, int(QTY - half))
                    if SELL_METHOD.lower() == "market":
                        print(f"[INTRA-KRX] 잔여 50% 시장가 매도: {remain}")
                        resp_s2 = call_with_retry(dsell_market, token, appkey, appsecret, TICKER, int(remain))
                    else:
                        base_px = min(stop, last)
                        px = align_krx_price(int(base_px * (1.0 - float(LIMIT_SLIP))), "down")
                        print(f"[INTRA-KRX] 잔여 50% 지정가: {remain} @ {px}")
                        resp_s2 = call_with_retry(dsell_limit, token, appkey, appsecret, TICKER, int(remain), int(px))
                    print(json.dumps(resp_s2, ensure_ascii=False, indent=2))
                    break
                except Exception as e:
                    print(f"[INTRA-KRX] 잔여 매도 실패: {e}")
        time.sleep(POLL_SEC)


# Add main guard at top-level
if __name__ == "__main__":
    run()