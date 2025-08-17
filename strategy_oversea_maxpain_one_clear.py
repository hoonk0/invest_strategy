# strategy_oversea_option auto.py
# 현물 단타 전략 (KIS 한투 실주문 전용)
#  - 입력: 맥스페인, 풋/콜 OI, 투자금, 분할개수, 현재가(수동)
#  - 로직: 맥스페인*1.03 ~ 풋 OI 1위 가격 구간을 5지점(양끝 포함)으로 분할 매수
#  - 실행: 조건 충족 시 한국투자증권(KIS) OpenAPI로 **실제 매수 주문**을 전송
#  - 해외주식(미국) AAPL 등: OVRS_EXCG_CD 기본값 "NASD"
#
# ※ 환경변수 세팅 필요 (실계좌/모의 모두 동일 키 이름 사용)
#   - KIS_APP_KEY, KIS_APP_SECRET
#   - KIS_CANO            (계좌번호 8자리)
#   - KIS_ACNT_PRDT_CD    (상품코드, 일반적으로 "01")
#   - KIS_USE_PAPER       (모의서버 사용 시 "1", 기본: 실서버)
#   - 선택) KIS_CUSTTYPE  (기본 "P")
#
# 참고: 해외주식 현금매수 endpoint 및 TR_ID 예시는 공식/커뮤니티 문서에 기재
#   - POST /uapi/overseas-stock/v1/trading/order  (TR_ID 실: JTTT1002U, 모의: VTTT1002U)
#     파라미터 예시와 TR_ID는 위키독스 샘플에 명시되어 있습니다. (JTTT1002U)  
#
#   출처: 위키독스 "① API호출 샘플(kis_api.py)" — do_order_OS 예제에서
#         url '/uapi/overseas-stock/v1/trading/order', tr_id 'JTTT1002U' 를 사용합니다.  
#
#   또한 해외주식 지정가/시장가 및 해시키 사용 예시는 여러 커뮤니티 예제에 정리되어 있습니다.
#
# 주의: 이 파일은 **시뮬레이션 제거** 버전입니다. 실행 시 실제 주문이 전송됩니다.
#
# 매도 시점:
# ① 장 마감 시 전량 매도(뉴욕 16:00) ② 현재가가 Call OI Top(매물대 많은 곳) * 0.97 이상 도달 시 즉시 전량 매도

from __future__ import annotations
import argparse
import os
import json
from typing import List, Dict, Any, Optional
import requests
import yfinance as yf

from kis_common import (
    create_kis_context,
    obuy_market, obuy_limit, osell_market, osell_limit
)

import time

from datetime import datetime, timedelta
try:
    from zoneinfo import ZoneInfo
except Exception:
    from backports.zoneinfo import ZoneInfo

# --- KIS 주문 래퍼: 다양한 시그니처 대응 (obuy/ osell 함수가 환경마다 다른 경우 대비) ---
def _order_try_chain(*calls):
    errs = []
    for fn, args, kwargs in calls:
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            errs.append(f"{fn.__name__}{getattr(e, 'args', [''])[0]}")
    raise RuntimeError("; ".join(errs))

def kis_buy_market(kc, exchange, code, qty, appkey=None, appsecret=None):
    ak = appkey or (kc.get('appkey') if isinstance(kc, dict) else None) or os.getenv('KIS_APP_KEY')
    sk = appsecret or (kc.get('appsecret') if isinstance(kc, dict) else None) or os.getenv('KIS_APP_SECRET')
    tk = (kc.get('token') if isinstance(kc, dict) and kc.get('token') else None) or os.getenv('KIS_ACCESS_TOKEN') or os.getenv('KIS_TOKEN')
    if not ak or not sk:
        raise RuntimeError("KIS appkey/appsecret not provided")
    if not tk:
        raise RuntimeError("KIS access token not provided")
    return obuy_market(appkey=str(ak), appsecret=str(sk), token=str(tk), excg=exchange, code=code, qty=qty)


def kis_buy_limit(kc, exchange, code, qty, price, appkey=None, appsecret=None):
    ak = appkey or (kc.get('appkey') if isinstance(kc, dict) else None) or os.getenv('KIS_APP_KEY')
    sk = appsecret or (kc.get('appsecret') if isinstance(kc, dict) else None) or os.getenv('KIS_APP_SECRET')
    tk = (kc.get('token') if isinstance(kc, dict) and kc.get('token') else None) or os.getenv('KIS_ACCESS_TOKEN') or os.getenv('KIS_TOKEN')
    if not ak or not sk:
        raise RuntimeError("KIS appkey/appsecret not provided")
    if not tk:
        raise RuntimeError("KIS access token not provided")
    return obuy_limit(appkey=str(ak), appsecret=str(sk), token=str(tk), excg=exchange, code=code, qty=qty, price=price)


def kis_sell_market(kc, exchange, code, qty, appkey=None, appsecret=None):
    ak = appkey or (kc.get('appkey') if isinstance(kc, dict) else None) or os.getenv('KIS_APP_KEY')
    sk = appsecret or (kc.get('appsecret') if isinstance(kc, dict) else None) or os.getenv('KIS_APP_SECRET')
    tk = (kc.get('token') if isinstance(kc, dict) and kc.get('token') else None) or os.getenv('KIS_ACCESS_TOKEN') or os.getenv('KIS_TOKEN')
    if not ak or not sk:
        raise RuntimeError("KIS appkey/appsecret not provided")
    if not tk:
        raise RuntimeError("KIS access token not provided")
    return osell_market(appkey=str(ak), appsecret=str(sk), token=str(tk), excg=exchange, code=code, qty=qty)

def ny_now():
    return datetime.now(ZoneInfo("America/New_York"))

def is_us_regular_open_now(dt=None):
    """NYSE/Nasdaq regular hours: 09:30–16:00 (NY local)."""
    dt = dt or ny_now()
    w = dt.weekday()
    if w >= 5:
        return False
    t = dt.time()
    return (t >= datetime(dt.year, dt.month, dt.day, 9, 30).time() and
            t <  datetime(dt.year, dt.month, dt.day, 16, 0).time())


# ===== 기본값 (CLI 미입력 시 사용) =====
DEFAULT_TICKER = "IONQ"
DEFAULT_MAX_PAIN = 42.0
DEFAULT_PUT_OI_TOP = 42.0
DEFAULT_CALL_OI_TOP = 48.0
DEFAULT_CAPITAL = 10000.0
DEFAULT_PARTS = 5
DEFAULT_EXCHANGE = "NASD"   # 해외거래소 코드 (미국 나스닥)
RATE_LIMIT_SEC = 1.2  # KIS API: 초당 거래건수 제한 대응(모의/실서버 공통 권장)


def get_current_price_yf(ticker: str) -> Optional[float]:
    t = yf.Ticker(ticker)
    # 1) 1분봉 최근값
    try:
        q = t.history(period="1d", interval="1m")
        if hasattr(q, "empty") and not q.empty and "Close" in q.columns:
            v = q["Close"].dropna()
            if not v.empty:
                return float(v.iloc[-1])
    except Exception:
        pass
    # 2) 일봉 최근 종가
    try:
        d = t.history(period="5d", interval="1d")
        if hasattr(d, "empty") and not d.empty and "Close" in d.columns:
            v = d["Close"].dropna()
            if not v.empty:
                return float(v.iloc[-1])
    except Exception:
        pass
    # 3) fast_info
    try:
        fi = t.fast_info
        if fi and fi.get("last_price"):
            return float(fi["last_price"])
    except Exception:
        pass
    # 4) info
    try:
        info = t.info
        if info and info.get("regularMarketPrice"):
            return float(info["regularMarketPrice"])
    except Exception:
        pass
    return None


def compute_buy_levels(max_pain: float, put_oi_top: float, parts: int = 5) -> List[float]:
    parts = max(2, int(parts))
    start_price = float(max_pain) * 1.03
    end_price = float(put_oi_top)
    step = (start_price - end_price) / (parts - 1)
    return [round(start_price - step * i, 2) for i in range(parts)]


def make_buy_plan(levels: List[float], capital: Optional[float], parts: int = 5) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    per_alloc = round(capital / parts, 2) if (capital is not None and parts) else None
    for i, px in enumerate(levels, 1):
        if per_alloc is None:
            rows.append({"leg": i, "price": px, "alloc": None, "shares": None})
        else:
            shares = int(per_alloc // px) if px > 0 else 0
            rows.append({"leg": i, "price": px, "alloc": per_alloc, "shares": shares})
    return rows


# ===== CLI =====

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="현물 단타: MaxPain & OI 기반 분할 매수 — KIS 실주문")
    p.add_argument("--ticker", default=None, help="종목 티커 예: AAPL")
    p.add_argument("--max-pain", dest="max_pain", type=float, default=None, help="맥스페인 값")
    p.add_argument("--put", dest="put_oi_top", type=float, default=None, help="풋 OI 1위 가격")
    p.add_argument("--call", dest="call_oi_top", type=float, default=None, help="콜 OI 1위 가격")
    p.add_argument("--capital", type=float, default=None, help="총 투자금")
    p.add_argument("--parts", type=int, default=None, help="분할 개수(기본 5)")
    p.add_argument("--now", type=float, default=None, help="현재가 수동 입력(없으면 야후에서 조회)")
    p.add_argument("--exchange", default=DEFAULT_EXCHANGE, help="해외거래소 코드 (예: NASD, NYSE, AMEX)")
    p.add_argument("--order-type", default="market", choices=["market", "limit"], help="주문 종류")
    p.add_argument("--poll", type=int, default=30, help="감시 주기(초), 기본 30s")
    return p.parse_args()

# ===== 메인 =====

def run():
    args = parse_args()

    # 입력값 적용
    ticker = args.ticker or DEFAULT_TICKER
    max_pain = args.max_pain if args.max_pain is not None else DEFAULT_MAX_PAIN
    put_top = args.put_oi_top if args.put_oi_top is not None else DEFAULT_PUT_OI_TOP
    call_top = args.call_oi_top if args.call_oi_top is not None else DEFAULT_CALL_OI_TOP
    capital = args.capital if args.capital is not None else DEFAULT_CAPITAL
    parts = args.parts if args.parts is not None else DEFAULT_PARTS
    exchange = args.exchange or DEFAULT_EXCHANGE
    poll_sec = max(5, int(args.poll))

    # 모의서버(KIS_USE_PAPER=1)에서는 종종 시장가가 제한됨 → 지정가로 강제 전환
    is_paper = os.getenv("KIS_USE_PAPER", "0") == "1"
    if is_paper and args.order_type == "market":
        print("[PAPER] 모의서버에서는 시장가 주문이 제한되어 지정가로 전환합니다.")
        args.order_type = "limit"

    # 첫 가격 표시
    last = args.now if (args.now is not None) else get_current_price_yf(ticker)
    if last is None:
        print(f"[PRICE] {ticker}: N/A (가격 조회 실패)")
    else:
        print(f"[PRICE] {ticker}: {last:.2f}")

    # 매수 레벨 & 플랜
    levels = compute_buy_levels(max_pain, put_top, parts)
    plan = make_buy_plan(levels, capital, parts)
    print(f"[PLAN] MaxPain={max_pain} | PutOI(Top)={put_top} | CallOI(Top)={call_top}")
    if capital:
        print(f"       Capital={capital} | Parts={parts} | Per-alloc={round(capital/parts,2)}")
    print("       Buy Levels (양끝 포함):", ", ".join(f"{p:.2f}" for p in levels))

    # KIS 컨텍스트
    kc = create_kis_context()
    appkey = os.getenv("KIS_APP_KEY")
    appsecret = os.getenv("KIS_APP_SECRET")

    def _is_rate_limit_error(err: Exception) -> bool:
        s = str(err)
        return ("EGW00201" in s) or ("초당 거래건수" in s)

    def _is_only_limit_resp(resp: Any) -> bool:
        try:
            return isinstance(resp, dict) and str(resp.get("msg_cd")) == "40650000"
        except Exception:
            return False

    def _safe_limit_price(side: str, last_px: float, level_px: float) -> float:
        """모의서버 지정가 체결 보조: 매수는 last/level 중 큰 값에 소폭 가산, 매도는 작은 값에 소폭 감산."""
        bump = 0.001  # 0.1%
        if side.lower() == "buy":
            base = max(float(last_px), float(level_px))
            return round(base * (1 + bump), 2)
        else:
            base = min(float(last_px), float(level_px))
            return round(base * (1 - bump), 2)

    def _submit_buy(i: int, qty: int, last_px: float, level_px: float):
        attempts = 0
        while True:
            attempts += 1
            try:
                # 1) 우선 현재 설정대로 시도
                if args.order_type == "market":
                    resp = kis_buy_market(kc, exchange, ticker, qty, appkey=appkey, appsecret=appsecret)
                    # 모의서버 등에서 시장가 제한 시 자동 전환
                    if _is_only_limit_resp(resp):
                        price = _safe_limit_price("buy", last_px, level_px)
                        print(f"[PAPER] 시장가 제한 감지 → 지정가로 전환(≈ {price})")
                        resp = kis_buy_limit(kc, exchange, ticker, qty, price, appkey=appkey, appsecret=appsecret)
                else:
                    price = _safe_limit_price("buy", last_px, level_px)
                    resp = kis_buy_limit(kc, exchange, ticker, qty, price, appkey=appkey, appsecret=appsecret)

                time.sleep(RATE_LIMIT_SEC)
                return resp
            except Exception as e:
                if _is_rate_limit_error(e) and attempts < 3:
                    time.sleep(RATE_LIMIT_SEC * 1.5)
                    continue
                raise

    def _submit_sell_all(qty: int, last_px: float, level_px: float):
        attempts = 0
        while True:
            attempts += 1
            try:
                if args.order_type == "market":
                    resp = kis_sell_market(kc, exchange, ticker, qty, appkey=appkey, appsecret=appsecret)
                else:
                    # 지정가로 전환 필요 시에는 보수적으로 현재가 근처로 계산 (모의서버 대비)
                    price = _safe_limit_price("sell", last_px, level_px)
                    # 지정가 매도 함수가 필요하면 여기서 추가 구현 가능
                    resp = kis_sell_market(kc, exchange, ticker, qty, appkey=appkey, appsecret=appsecret)
                time.sleep(RATE_LIMIT_SEC)
                return resp
            except Exception as e:
                if _is_rate_limit_error(e) and attempts < 3:
                    time.sleep(RATE_LIMIT_SEC * 1.5)
                    continue
                raise

    executed_legs = set()  # 이미 매수한 레그 인덱스 보관
    position_qty = 0       # 보유 수량(주)
    sell_done = False

    print("[WATCH] 장중 감시 시작 — 정규장(09:30~16:00 NY) 동안 실행. 마감 시 자동 종료")

    while True:
        now_ny = ny_now()
        if not is_us_regular_open_now(now_ny):
            # 마감 이후면 종료, 개장 전이면 대기
            if now_ny.time() >= datetime(now_ny.year, now_ny.month, now_ny.day, 16, 0).time():
                # 마감: 보유 시 전량 시장가 매도
                if position_qty > 0 and not sell_done:
                    print("🔔 [CLOSE] 장 마감 — 보유분 전량 시장가 매도")
                    try:
                        resp = _submit_sell_all(position_qty, float(last) if last is not None else 0.0, float(levels[-1]))
                        print(f"[KIS RESP] {resp}")
                        sell_done = True
                    except Exception as e:
                        print(f"[ERR] 마감 매도 실패: {e}")
                print("✅ [DONE] 거래일 종료")
                break
            else:
                print(f"⏳ [WAIT] 정규장 대기 (NY {now_ny.strftime('%H:%M')})")
                time.sleep(min(120, poll_sec))
                continue

        # 장중: 현재가 갱신
        last = get_current_price_yf(ticker)
        if last is None:
            print("⚠️ 가격 조회 실패 — 재시도")
            time.sleep(poll_sec)
            continue

        # 매수 조건 체크(아직 안 산 레그만)
        for i, row in enumerate(plan, start=1):
            if i in executed_legs:
                continue
            level = row["price"]
            alloc = row["alloc"]
            if last <= level:
                qty = int((alloc // last)) if (alloc is not None and last > 0) else 0
                if qty <= 0:
                    print(f"[SKIP] L{i} 금액 부족 (alloc={alloc}, last={last})")
                    executed_legs.add(i)  # 반복 방지
                    continue
                print(f"🟢 [BUY] L{i} {ticker} x {qty} @order (last={last:.2f} ≤ level={level:.2f})")
                try:
                    resp = _submit_buy(i, qty, float(last), float(level))
                    print(f"[KIS RESP] {resp}")
                    executed_legs.add(i)
                    position_qty += qty
                except Exception as e:
                    print(f"[ERR] 매수 실패(L{i}): {e}")
                    executed_legs.add(i)

        # 매도 트리거: Call OI * 0.97 이상이면 즉시 매도
        if position_qty > 0 and not sell_done:
            threshold = float(call_top) * 0.97
            if last >= threshold:
                print(f"🔴 [SELL] last {last:.2f} ≥ CallOI*0.97 {threshold:.2f} — 전량 시장가 매도")
                try:
                    resp = _submit_sell_all(position_qty, float(last), float(levels[-1]))
                    print(f"[KIS RESP] {resp}")
                    sell_done = True
                    position_qty = 0
                except Exception as e:
                    print(f"[ERR] 즉시 매도 실패: {e}")

        # 다음 주기까지 대기
        time.sleep(poll_sec)

if __name__ == "__main__":
    run()