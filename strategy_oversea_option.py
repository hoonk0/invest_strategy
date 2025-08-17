# strategy_intraday.py
from __future__ import annotations
import json, time
from datetime import datetime, timedelta

from kis_common import (
    create_kis_context, get_overseas_last_price, call_with_retry,
    obuy_limit, osell_market, osell_limit
)

# ===== 설정 =====
TICKER = "PLTR"
EXCG   = "NYSE"   # 실제 상장 거래소 맞춰주세요 (PLTR=NYSE)
QTY    = 100
ENTRY_PRICE = 186.00
STOP_PRICE  = 185.40
POLL_SEC    = 10
MAX_HOURS   = 6
SELL_METHOD = "market"    # "market" | "limit"
LIMIT_SLIP  = 0.003       # 지정가일 때 stop/last 보다 0.3% 낮춰 던지기

def run():
    ctx = create_kis_context()
    token = ctx["token"]; appkey = ctx["appkey"]; appsecret = ctx["appsecret"]

    # 1) 시초가 매수(지정가)
    try:
        print(f"[INTRA] 매수 {TICKER}: {QTY} @ {ENTRY_PRICE}")
        resp_buy = call_with_retry(obuy_limit, token, appkey, appsecret, EXCG, TICKER, int(QTY), float(ENTRY_PRICE))
        print(json.dumps(resp_buy, ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"[INTRA] 매수 실패: {e}"); return

    half = max(1, int(QTY // 2)); sold_half = False
    stop = round(float(STOP_PRICE), 2)
    print(f"[INTRA] 스탑: {stop} | method={SELL_METHOD}, slip={LIMIT_SLIP if SELL_METHOD=='limit' else 'N/A'}")

    start = datetime.now()
    while (datetime.now() - start) < timedelta(hours=MAX_HOURS):
        last = get_overseas_last_price(TICKER, EXCG)
        if last is not None:
            last = round(float(last), 2)
            print(f"[INTRA] 현재가: {last}")

            # (A) 스탑 트리거 → 50% 매도
            if (not sold_half) and last <= stop:
                try:
                    resp_s1 = None
                    if SELL_METHOD.lower() == "market":
                        print(f"[INTRA] 50% 시장가 매도 시도: {half}")
                        try:
                            resp_s1 = call_with_retry(osell_market, token, appkey, appsecret, EXCG, TICKER, int(half))
                        except Exception as e:
                            print(f"[INTRA] 시장가 실패 → 지정가 폴백: {e}")
                    if (resp_s1 is None) or (isinstance(resp_s1, dict) and resp_s1.get("rt_cd") != "0"):
                        px = max(0.01, round(min(stop, last) * (1.0 - float(LIMIT_SLIP)), 2))
                        print(f"[INTRA] 50% 지정가 폴백: {half} @ {px}")
                        resp_s1 = call_with_retry(osell_limit, token, appkey, appsecret, EXCG, TICKER, int(half), float(px))
                    print(json.dumps(resp_s1, ensure_ascii=False, indent=2))
                    sold_half = True
                except Exception as e:
                    print(f"[INTRA] 50% 매도 실패: {e}")

            # (B) 시간봉 간이 조건: 매 분 59분, 가격이 여전히 stop 이하 → 잔여 50%
            if sold_half:
                if datetime.now().minute == 59 and last <= stop:
                    try:
                        remain = max(1, int(QTY - half))
                        if SELL_METHOD.lower() == "market":
                            print(f"[INTRA] 잔여 50% 시장가 매도: {remain}")
                            resp_s2 = call_with_retry(osell_market, token, appkey, appsecret, EXCG, TICKER, int(remain))
                        else:
                            px = max(0.01, round(min(stop, last) * (1.0 - float(LIMIT_SLIP)), 2))
                            print(f"[INTRA] 잔여 50% 지정가: {remain} @ {px}")
                            resp_s2 = call_with_retry(osell_limit, token, appkey, appsecret, EXCG, TICKER, int(remain), float(px))
                        print(json.dumps(resp_s2, ensure_ascii=False, indent=2))
                        break
                    except Exception as e:
                        print(f"[INTRA] 잔여 매도 실패: {e}")
        time.sleep(POLL_SEC)

# === main guard ===
if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print("\n[INTRA] Stopped by user")
    except Exception as e:
        print(f"[INTRA] FATAL: {e}")