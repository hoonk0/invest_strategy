from __future__ import annotations
#!/usr/bin/env python3
"""
KRX 스크리너 (심플)
- 조건: (1) 거래대금 상위 N, (2) 시가총액 범위, (3) 5/20/60 정배열, (4) 최소 등락률(%) 필터
- 기능:
    a) --date 로 해당 일자 스크리닝 리스트 출력
    b) --symbol 로 특정 종목의 해당 일자 상태 출력
    c) 시가 매수→TP/SL/EOD 단순 가정 시뮬 + 요약/상세 출력
    d) (옵션) KIS 모의투자 지정가 매수 API 호출

사전 준비
  pip install pykrx pandas
"""

import argparse
from datetime import datetime, timedelta
from typing import Optional, Dict
import pandas as pd
import math

# --- KIS broker (mock/live via kis_common) ---
from kis_common import (
    create_kis_context, call_with_retry,
    dbuy_limit, get_domestic_last_price
)

# ===== 기본 파라미터 =====
DEFAULT_DATE: Optional[str] = "2025-08-14"           # YYYY-MM-DD, 미지정 시 오늘
DEFAULT_TOPN: int = 30                       # 거래대금 상위 N
DEFAULT_CAP_MIN: int = 300_000_000     # 3,000억원
DEFAULT_CAP_MAX: int = 10_000_000_000_000    # 10조원
DEFAULT_MIN_CHANGE_PCT: float = 5.0          # 기본 진입 최소 등락률(%)
DEFAULT_TP_PCT: float = 3.0                  # 기본 익절 퍼센트(양수, %)
DEFAULT_SL_PCT: float = 1.5                  # 기본 손절 퍼센트(양수, %)
DEFAULT_SEED_PER_STOCK: int = 1_000_000      # 시뮬 종목당 투입금액(원)

# ---- pykrx 로딩 ----
try:
    from pykrx import stock
except Exception:
    raise SystemExit("pykrx 가 설치되어 있지 않습니다. 먼저 'pip install pykrx' 를 실행하세요.")

# ===== 유틸 =====

def _normalize_yyyymmdd(s: str) -> str:
    s = s.strip().replace("-", "").replace("/", "")
    if len(s) != 8 or not s.isdigit():
        raise ValueError("날짜는 YYYY-MM-DD 또는 YYYYMMDD 형식이어야 합니다.")
    return s

def _resolve_date_arg(date_str: str | None) -> str:
    """--date 파라미터 없으면 오늘(YYYYMMDD), 입력 시 YYYY-MM-DD/YYYMMDD 허용"""
    if date_str and str(date_str).strip():
        return _normalize_yyyymmdd(str(date_str))
    return datetime.now().strftime("%Y%m%d")

# ---- KRX 호가단위 & 보정 ----
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

def _get_prev_business_day(date_yyyymmdd: str) -> Optional[str]:
    dt = datetime.strptime(date_yyyymmdd, "%Y%m%d")
    for i in range(1, 11):
        d = (dt - timedelta(days=i)).strftime("%Y%m%d")
        try:
            tmp = stock.get_market_ohlcv_by_ticker(d)
            if tmp is not None and not tmp.empty:
                return d
        except Exception:
            pass
    return None

def _get_ohlcv(date: str) -> pd.DataFrame:
    kospi = stock.get_market_ohlcv_by_ticker(date, market="KOSPI")
    kosdaq = stock.get_market_ohlcv_by_ticker(date, market="KOSDAQ")
    df = pd.concat([kospi, kosdaq])
    df.index.name = "티커"
    return df

def _get_cap(date: str) -> pd.DataFrame:
    kospi = stock.get_market_cap_by_ticker(date, market="KOSPI")
    kosdaq = stock.get_market_cap_by_ticker(date, market="KOSDAQ")
    df = pd.concat([kospi, kosdaq])
    df.index.name = "티커"
    return df[["시가총액"]]

def _get_names() -> pd.DataFrame:
    kospi = stock.get_market_ticker_list(market="KOSPI")
    kosdaq = stock.get_market_ticker_list(market="KOSDAQ")
    rows = []
    for t in (kospi + kosdaq):
        try:
            rows.append({"티커": t, "종목명": stock.get_market_ticker_name(t)})
        except Exception:
            pass
    return pd.DataFrame(rows).set_index("티커")

# ===== 매수 실행 (KIS 모의투자/실계좌 공통 컨텍스트 사용) =====
def buy_stock_kis(symbol: str, qty: int, limit_price: int | None = None) -> dict:
    """
    한국투자증권(KIS) 국내주식 지정가 매수.
    - symbol: 6자리 코드 (예: '005930')
    - qty: 수량(정수)
    - limit_price: 지정가(원). None이면 현재가 조회 후 호가단위 상향 보정.
    반환: KIS 응답(dict)
    """
    sym = str(symbol).zfill(6)
    ctx = create_kis_context()
    token = ctx["token"]; appkey = ctx["appkey"]; appsecret = ctx["appsecret"]

    px = limit_price
    if px is None:
        last = get_domestic_last_price(sym)
        if last is None:
            raise RuntimeError("현재가 조회 실패(get_domestic_last_price 반환 None)")
        px = int(round(float(last)))
    px = align_krx_price(px, "up")

    print(f"[KIS BUY] {sym} x{int(qty)} @ {px} (limit)")
    resp = call_with_retry(dbuy_limit, token, appkey, appsecret, sym, int(qty), int(px))
    return resp

def fetch_krx(date: str) -> pd.DataFrame:
    """symbol, name, close, open, trading_value, market_cap, change_pct 반환"""
    ohlcv = _get_ohlcv(date)
    cap = _get_cap(date)
    names = _get_names()

    # 충돌 방지
    for col in ["시가총액", "종목명", "등락률"]:
        if col in ohlcv.columns:
            ohlcv = ohlcv.drop(columns=[col])
    if "종목명" in cap.columns:
        cap = cap.drop(columns=["종목명"])

    df = ohlcv.join(cap, how="left").join(names, how="left")
    df = df.rename(columns={
        "종가": "close",
        "시가": "open",
        "거래대금": "trading_value",
        "시가총액": "market_cap",
        "종목명": "name",
    })

    # 수치형 변환
    for c in ["close", "open", "trading_value", "market_cap"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # change_pct 계산: ((close - open) / open) * 100
    if "등락률" in ohlcv.columns:
        df["change_pct"] = pd.to_numeric(ohlcv["등락률"], errors="coerce")
    else:
        df["change_pct"] = ((df["close"] - df["open"]) / df["open"]) * 100

    df = df.reset_index().rename(columns={"티커": "symbol"})
    cols = ["symbol", "name", "close", "open", "trading_value", "market_cap", "change_pct"]
    return df[[c for c in cols if c in df.columns]]

def _compute_ma_alignment_for_symbol(symbol: str, date: str) -> Dict[str, object]:
    """5/20/60 정배열 여부와 이유"""
    try:
        dt = datetime.strptime(date, "%Y%m%d")
        start_dt = dt - timedelta(days=150)
        df = stock.get_market_ohlcv_by_date(start_dt.strftime("%Y%m%d"), date, symbol)
        if df.empty or len(df) < 60:
            return {"is_bullish": False, "reason": "데이터 부족"}
        close = df["종가"]
        ma5  = close.rolling(5).mean().iloc[-1]
        ma20 = close.rolling(20).mean().iloc[-1]
        ma60 = close.rolling(60).mean().iloc[-1]
        if pd.isna(ma5) or pd.isna(ma20) or pd.isna(ma60):
            return {"is_bullish": False, "reason": "이동평균 계산 불가"}
        ok = (ma5 > ma20) and (ma20 > ma60)
        return {"is_bullish": bool(ok), "reason": f"MA5={ma5:.2f}, MA20={ma20:.2f}, MA60={ma60:.2f}"}
    except Exception as e:
        return {"is_bullish": False, "reason": f"오류: {e}"}

# ===== 스크리닝 / 상태조회 =====

def screen_three_conditions(
    date_yyyymmdd: str,
    topn: int,
    cap_min: int,
    cap_max: int,
    min_change_pct: Optional[float] = None
) -> pd.DataFrame:
    """네 조건(거래대금 상위, 시총 범위, 최소 등락률, 정배열)을 모두 만족하는 리스트 반환"""
    df = fetch_krx(date_yyyymmdd)
    if df is None or df.empty:
        prev = _get_prev_business_day(date_yyyymmdd)
        if prev:
            print(f"[INFO] 당일 데이터 없음 → 이전 영업일({prev})로 대체")
            df = fetch_krx(prev)
            date_yyyymmdd = prev
        else:
            return pd.DataFrame()

    # 1) 거래대금 상위 N
    df_top = df.sort_values("trading_value", ascending=False).head(max(1, int(topn))).copy()

    # 2) 시가총액 범위
    df_cap = df_top[(df_top["market_cap"] >= cap_min) & (df_top["market_cap"] <= cap_max)].copy()

    # 3) 최소 등락률(항상 적용: 값 미지정이면 DEFAULT 사용)
    threshold = float(min_change_pct) if (min_change_pct is not None) else DEFAULT_MIN_CHANGE_PCT
    df_cap = df_cap[df_cap["change_pct"] >= threshold].copy()

    # 4) 정배열
    ma_info = {sym: _compute_ma_alignment_for_symbol(sym, date_yyyymmdd) for sym in df_cap["symbol"]}
    df_cap["MA_align"]  = df_cap["symbol"].map(lambda s: "O" if ma_info.get(s, {}).get("is_bullish") else "X")

    out = df_cap[df_cap["MA_align"] == "O"].copy()
    cols = [c for c in [
        "symbol","name","trading_value","close","open","change_pct","MA_align"
    ] if c in out.columns]
    return out.sort_values("trading_value", ascending=False)[cols]

def symbol_status(
    date_yyyymmdd: str,
    symbol: str,
    topn: int,
    cap_min: int,
    cap_max: int,
    min_change_pct: Optional[float] = None
) -> pd.DataFrame:
    """특정 종목에 대한 조건 충족 여부/근거를 표로 반환"""
    df = fetch_krx(date_yyyymmdd)
    if df is None or df.empty:
        prev = _get_prev_business_day(date_yyyymmdd)
        if prev:
            print(f"[INFO] 당일 데이터 없음 → 이전 영업일({prev})로 대체")
            df = fetch_krx(prev)
            date_yyyymmdd = prev
        else:
            return pd.DataFrame()

    # 거래대금 순위
    df = df.dropna(subset=["trading_value"]).copy()
    df["tv_rank"] = df["trading_value"].rank(ascending=False, method="min")

    row = df[df["symbol"] == str(symbol).zfill(6)].copy()
    if row.empty:
        return pd.DataFrame([{"symbol": str(symbol).zfill(6), "status": "해당 일자 데이터 없음"}])

    tv_rank = int(row.iloc[0]["tv_rank"]) if pd.notna(row.iloc[0]["tv_rank"]) else None
    in_topn = (tv_rank is not None) and (tv_rank <= topn)

    cap = float(row.iloc[0]["market_cap"]) if pd.notna(row.iloc[0]["market_cap"]) else None
    cap_ok = (cap is not None) and (cap_min <= cap <= cap_max)

    ma = _compute_ma_alignment_for_symbol(str(symbol).zfill(6), date_yyyymmdd)
    ma_ok = bool(ma.get("is_bullish"))

    change_pct = float(row.iloc[0].get("change_pct", 0) or 0)
    threshold = float(min_change_pct) if (min_change_pct is not None) else DEFAULT_MIN_CHANGE_PCT
    change_ok = change_pct >= threshold

    result = {
        "symbol": str(symbol).zfill(6),
        "name": row.iloc[0].get("name"),
        "trading_value": float(row.iloc[0].get("trading_value", 0) or 0),
        "tv_rank": tv_rank,
        "in_topn": bool(in_topn),
        "market_cap": cap,
        "cap_ok": bool(cap_ok),
        "change_pct": change_pct,
        "change_ok": change_ok,
        "MA_align": "O" if ma_ok else "X",
        "ALL_OK": bool(in_topn and cap_ok and ma_ok and change_ok),
    }
    return pd.DataFrame([result])

# ===== 시뮬(시가 매수→TP/SL/EOD) =====

def simulate_day(df_screened: pd.DataFrame, seed_per_stock: int, tp_pct: float, sl_pct: float) -> dict:
    """
    매우 단순한 데이 시뮬:
      - 진입가 = open
      - TP = open * (1 + tp_pct/100)  (tp_pct는 양수)
      - SL = open * (1 - sl_pct/100)  (sl_pct는 양수)
      - 종가가 TP 이상이면 TP에 청산, 종가가 SL 이하이면 SL에 청산, 아니면 종가에 청산
      - 수량 = floor(seed_per_stock / 진입가)
    반환:
      {
        "rows": [
            {
                symbol, name, entry, close, tp_px, sl_px, exit, exit_tag, qty, invested, pnl,
                buy_price, buy_time, sell_price, sell_time, pnl_pct
            } ...
        ],
        "total_invested": int,
        "total_pnl": int,
        "total_return_pct": float
      }
    """
    rows = []
    total_invested = 0
    total_pnl = 0

    if df_screened is None or df_screened.empty:
        return {
            "rows": [],
            "total_invested": 0,
            "total_pnl": 0,
            "total_return_pct": 0.0
        }

    for _, r in df_screened.iterrows():
        sym = str(r["symbol"]).zfill(6)
        name = r.get("name", "")
        open_px = float(r.get("open", 0) or 0)
        close_px = float(r.get("close", 0) or 0)
        if open_px <= 0 or math.isnan(open_px) or close_px <= 0 or math.isnan(close_px):
            continue

        tp_px = open_px * (1.0 + float(tp_pct)/100.0)
        sl_px = open_px * (1.0 - float(sl_pct)/100.0)
        qty = int(seed_per_stock // open_px)
        if qty <= 0:
            continue

        invested = int(qty * open_px)

        # 간단한 종료 로직(보수적 가정)
        if close_px >= tp_px:
            exit_px = tp_px
            exit_tag = "TP"
        elif close_px <= sl_px:
            exit_px = sl_px
            exit_tag = "SL"
        else:
            exit_px = close_px
            exit_tag = "EOD"

        pnl = int(round(qty * (exit_px - open_px)))
        pnl_pct = (exit_px - open_px) / open_px * 100.0

        rows.append({
            "symbol": sym,
            "name": name,
            "entry": int(round(open_px)),
            "close": int(round(close_px)),
            "tp_px": int(round(tp_px)),
            "sl_px": int(round(sl_px)),
            "exit": int(round(exit_px)),
            "exit_tag": exit_tag,
            "qty": qty,
            "invested": invested,
            "pnl": pnl,
            "buy_price": int(round(open_px)),
            "buy_time": "09:00",  # 가정: 시가 체결
            "sell_price": int(round(exit_px)),
            "sell_time": exit_tag,
            "pnl_pct": float(round(pnl_pct, 4))
        })
        total_invested += invested
        total_pnl += pnl

    total_return_pct = (total_pnl / total_invested * 100.0) if total_invested > 0 else 0.0
    return {
        "rows": rows,
        "total_invested": int(total_invested),
        "total_pnl": int(total_pnl),
        "total_return_pct": float(total_return_pct)
    }

# ===== 인터랙티브 셋업 =====

def interactive_setup(args) -> tuple[str, float, float, float, int]:
    """
    실행 직후 간단 설정 프롬프트.
    - 날짜: YYYY-MM-DD/YYYMMDD (엔터=오늘 또는 기존값)
    - 최소 등락률: % (엔터=기본 또는 기존값)
    - 익절 퍼센트: % (엔터=기본 또는 기존값)
    - 손절 퍼센트: % (엔터=기본 또는 기존값)
    - 종목당 시드(원): (엔터=기본 또는 기존값)
    """
    today_str = datetime.now().strftime("%Y%m%d")
    current_date = _resolve_date_arg(args.date) if getattr(args, "date", None) else today_str
    current_min_pct = args.min_change_pct if getattr(args, "min_change_pct", None) is not None else DEFAULT_MIN_CHANGE_PCT
    current_tp_pct = args.tp_pct if getattr(args, "tp_pct", None) is not None else DEFAULT_TP_PCT
    current_sl_pct = args.sl_pct if getattr(args, "sl_pct", None) is not None else DEFAULT_SL_PCT
    current_seed   = args.seed_per_stock if getattr(args, "seed_per_stock", None) is not None else DEFAULT_SEED_PER_STOCK

    print("[SETUP] 실행 전 간단 설정 (엔터 = 기본값 유지)")
    print(f"  - 기본 날짜: {current_date[:4]}-{current_date[4:6]}-{current_date[6:8]}")
    print(f"  - 기본 진입 최소 등락률: {current_min_pct:.2f}%")
    print(f"  - 기본 익절 퍼센트: {current_tp_pct:.2f}%")
    print(f"  - 기본 손절 퍼센트: {current_sl_pct:.2f}%")
    print(f"  - 기본 종목당 투입금액: {current_seed:,}원")

    try: date_in = input("📅 조회 날짜 입력 (YYYY-MM-DD, 엔터시 기본 유지): ").strip()
    except Exception: date_in = ""
    try: pct_in  = input("📈 진입 최소 등락률(%) 입력 (엔터시 기본 유지): ").strip()
    except Exception: pct_in = ""
    try: tp_in   = input("🎯 익절 퍼센트(%) 입력 (엔터시 기본 유지): ").strip()
    except Exception: tp_in = ""
    try: sl_in   = input("🛑 손절 퍼센트(%) 입력 (엔터시 기본 유지): ").strip()
    except Exception: sl_in = ""
    try: seed_in = input("💰 종목당 투입금액 입력 (원, 엔터시 기본 유지): ").strip()
    except Exception: seed_in = ""

    if date_in:
        try:
            current_date = _normalize_yyyymmdd(date_in)
        except Exception as e:
            print(f"[SETUP] 날짜 형식 오류: {e} → 기본값 유지({current_date})")
    if pct_in:
        try:
            current_min_pct = float(pct_in)
        except Exception:
            print(f"[SETUP] 등락률 입력 오류 → 기본값 유지({current_min_pct}%)")
    if tp_in:
        try:
            current_tp_pct = float(tp_in)
        except Exception:
            print(f"[SETUP] 익절 입력 오류 → 기본값 유지({current_tp_pct}%)")
    if sl_in:
        try:
            current_sl_pct = float(sl_in)
        except Exception:
            print(f"[SETUP] 손절 입력 오류 → 기본값 유지({current_sl_pct}%)")
    if seed_in:
        try:
            current_seed = int(seed_in.replace(",", ""))
        except Exception:
            print(f"[SETUP] 시드 입력 오류 → 기본값 유지({current_seed:,}원)")

    print(f"[SETUP] 적용값 → 날짜={current_date[:4]}-{current_date[4:6]}-{current_date[6:8]}, "
          f"최소 등락률={current_min_pct:.2f}%, 익절={current_tp_pct:.2f}%, 손절={current_sl_pct:.2f}%, "
          f"종목당 시드={current_seed:,}원")
    return current_date, current_min_pct, current_tp_pct, current_sl_pct, current_seed

# ===== CLI =====

def parse_args():
    p = argparse.ArgumentParser(description="KRX 스크리너: 거래대금 상위 + 시총범위 + 정배열 + 최소 등락률")
    p.add_argument("--date", default=DEFAULT_DATE, help="YYYY-MM-DD (미지정 시 오늘)")
    p.add_argument("--topn", type=int, default=DEFAULT_TOPN, help="거래대금 상위 N")
    p.add_argument("--cap_min", type=int, default=DEFAULT_CAP_MIN, help="시총 하한(원)")
    p.add_argument("--cap_max", type=int, default=DEFAULT_CAP_MAX, help="시총 상한(원)")
    p.add_argument("--symbol", default=None, help="특정 종목 상태 확인 (6자리 코드)")
    p.add_argument("--min_change_pct", type=float, default=DEFAULT_MIN_CHANGE_PCT, help="최소 등락률(%) 필터")

    # --- KIS 매수 옵션 (선택) ---
    p.add_argument("--buy", action="store_true", help="한국투자증권 모의투자 계좌로 매수 실행(지정가)")
    p.add_argument("--buy_symbol", default=None, help="매수할 6자리 종목코드(미지정 시 --symbol 사용)")
    p.add_argument("--buy_qty", type=int, default=None, help="매수 수량(기본 1)")
    p.add_argument("--buy_price", type=int, default=None, help="지정가(원). 미지정 시 현재가 기반 자동 산출")

    # --- 시뮬레이션 옵션 ---
    p.add_argument("--simulate", action="store_true", help="단순 데이 시뮬: 시가 매수 → TP/SL/EOD")
    p.add_argument("--seed_per_stock", type=int, default=DEFAULT_SEED_PER_STOCK, help="시뮬 종목당 투입금액(원)")
    p.add_argument("--tp_pct", type=float, default=DEFAULT_TP_PCT, help="시뮬 익절 퍼센트(%)")
    p.add_argument("--sl_pct", type=float, default=DEFAULT_SL_PCT, help="시뮬 손절 퍼센트(%)")

    return p.parse_args()

# ===== 메인 =====

def main():
    args = parse_args()

    # --- 인터랙티브 프롬프트 제거: 상단 DEFAULT_* 만 사용 ---
    date_choice = _resolve_date_arg(args.date)
    min_pct_choice = args.min_change_pct if args.min_change_pct is not None else DEFAULT_MIN_CHANGE_PCT
    tp_choice = args.tp_pct if args.tp_pct is not None else DEFAULT_TP_PCT
    sl_choice = args.sl_pct if args.sl_pct is not None else DEFAULT_SL_PCT
    seed_choice = args.seed_per_stock if args.seed_per_stock is not None else DEFAULT_SEED_PER_STOCK

    # 적용
    args.date = date_choice
    args.min_change_pct = min_pct_choice
    args.tp_pct = tp_choice
    args.sl_pct = sl_choice
    args.seed_per_stock = seed_choice

    date = _resolve_date_arg(args.date)

    # --- 심볼 상태 모드 ---
    if args.symbol:
        df = symbol_status(date, args.symbol, args.topn, args.cap_min, args.cap_max, args.min_change_pct)
        print("[STATUS]", date, args.symbol)
        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            print(df)

        # 선택적 매수 실행 (--symbol과 함께 사용 권장)
        if args.buy:
            sym_to_buy = args.buy_symbol or args.symbol
            qty = args.buy_qty or 1
            try:
                resp = buy_stock_kis(sym_to_buy, int(qty), args.buy_price)
                print("[KIS BUY RESP]")
                print(resp if isinstance(resp, dict) else str(resp))
            except Exception as e:
                print(f"[KIS BUY ERR] {e}")
        return

    # --- 리스트(스크리닝) 모드 ---
    df = screen_three_conditions(date, args.topn, args.cap_min, args.cap_max, args.min_change_pct)
    print(f"[SCREEN] {date} | 조건: 거래대금 TOP{args.topn} ∧ 시총 {args.cap_min:,}~{args.cap_max:,} ∧ 정배열 ∧ 등락률 ≥ {args.min_change_pct}%")
    if df.empty:
        print("🚫 조건을 만족하는 종목이 없습니다.")
        return
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(df)

    # --- 요약 수익 리포트 (항상 출력: 시가 매수→TP/SL/EOD 가정) ---
    sim_quick = simulate_day(df, args.seed_per_stock, args.tp_pct, args.sl_pct)
    delta_quick = sim_quick['total_pnl']
    sign_quick = "+" if delta_quick >= 0 else ""
    print("\n[SUMMARY]")
    print(f"  - 총투입금액(가정): {sim_quick['total_invested']:,}원")
    print(f"  - 오늘하루 총자산 증가액(가정): {sign_quick}{delta_quick:,}원")
    print(f"  - 오늘하루 수익률(가정): {sim_quick['total_return_pct']:.2f}%")

    # --- 종목별 상세 테이블 (항상 출력) ---
    if sim_quick["rows"]:
        df_quick_rows = pd.DataFrame(sim_quick["rows"])
        cols_kr = {
            "symbol": "symbol",
            "name": "name",
            "buy_price": "매수가격",
            "buy_time": "매수가격(시간)",
            "sell_price": "매도가격",
            "sell_time": "매도가격(시간)",
            "pnl": "수익",
            "pnl_pct": "수익률(%)"
        }
        show_cols = ["symbol","name","buy_price","buy_time","sell_price","sell_time","pnl","pnl_pct"]
        df_show = df_quick_rows[show_cols].rename(columns=cols_kr)
        print("\n[SUMMARY] 종목별 상세")
        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            print(df_show)

    # --- 시뮬레이션 출력(옵션) ---
    if args.simulate:
        sim = simulate_day(df, args.seed_per_stock, args.tp_pct, args.sl_pct)
        if sim["rows"]:
            df_sim = pd.DataFrame(sim["rows"])
            print("\n[SIM TABLE] (시가 매수 → TP/SL/EOD)")
            with pd.option_context("display.max_rows", None, "display.max_columns", None):
                print(df_sim[["symbol","name","qty","entry","tp_px","sl_px","close","exit","exit_tag","invested","pnl"]])

        print("\n[SIM REPORT]")
        print(f"  - 총투입금액: {sim['total_invested']:,}원")
        delta = sim["total_pnl"]
        sign = "+" if delta >= 0 else ""
        print(f"  - 오늘하루 총자산 증가액: {sign}{delta:,}원")
        print(f"  - 오늘하루 수익률: {sim['total_return_pct']:.2f}%")

    # --- 리스트 모드에서 --buy 사용 시: 첫 번째 종목을 매수 (또는 --buy_symbol 우선) ---
    if args.buy:
        if args.buy_symbol:
            target_sym = args.buy_symbol
        elif not df.empty:
            target_sym = str(df.iloc[0]["symbol"]).zfill(6)
            print(f"[INFO] --buy 지정: 스크리닝 1순위 {target_sym} 매수 시도")
        else:
            print("[INFO] --buy 무시: 매수 대상이 없습니다.")
            return
        qty = args.buy_qty or 1
        try:
            resp = buy_stock_kis(target_sym, int(qty), args.buy_price)
            print("[KIS BUY RESP]")
            print(resp if isinstance(resp, dict) else str(resp))
        except Exception as e:
            print(f"[KIS BUY ERR] {e}")

if __name__ == "__main__":
    main()