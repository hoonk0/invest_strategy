from __future__ import annotations
#!/usr/bin/env python3
"""
KRX 스크리너 (심플)
- 조건: (1) 거래대금 상위 N, (2) 시가총액 범위, (3) 5/20/60 정배열
- 기능:
    a) --date 로 해당 일자 스크리닝 리스트 출력
    b) --symbol 로 특정 종목의 해당 일자 상태 출력

사전 준비
  pip install pykrx pandas
"""

from datetime import datetime, timedelta
import time
from typing import Optional, Dict
import pandas as pd

# KIS helper funcs (모의투자/실계좌 공통 래퍼)
try:
    from kis_common import (
        create_kis_context,
        get_domestic_last_price,
        call_with_retry,
        dbuy_limit
    )
    # 선택적: 모듈에 있으면 시장가 매수 사용
    try:
        from kis_common import dbuy_market  # optional
    except Exception:
        dbuy_market = None  # type: ignore
    # 선택적: 모듈에 있으면 매도 사용
    try:
        from kis_common import dsell_market, dsell_limit  # optional
    except Exception:
        dsell_market = None  # type: ignore
        dsell_limit = None   # type: ignore
except Exception as _e:
    create_kis_context = None  # type: ignore
    get_domestic_last_price = None  # type: ignore
    call_with_retry = None  # type: ignore
    dbuy_limit = None  # type: ignore
    dbuy_market = None  # type: ignore
    dsell_market = None  # type: ignore
    dsell_limit = None   # type: ignore

# ===== 기본 파라미터 =====
DEFAULT_DATE: Optional[str] = None   # YYYY-MM-DD, 미지정 시 오늘
DEFAULT_TOPN: int = 30               # 거래대금 상위 N
DEFAULT_CAP_MIN: int = 300_000_000_000      # 3,000억원
DEFAULT_CAP_MAX: int = 10_000_000_000_000   # 10조원
DEFAULT_MIN_CHANGE_PCT: float = 5.0         # 등락률(%) 하한 (필수)

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

# ===== 기본 설정 (여기만 고치고 ▶ 실행) =====
# 날짜: None 이면 오늘, "YYYY-MM-DD" 형식으로 특정일 지정 가능
AUTO_DATE: Optional[str] = None
# 거래대금 상위 개수
AUTO_TOPN: int = 30
# 시총 범위 (원)
AUTO_CAP_MIN: int = 300_000_000_000
AUTO_CAP_MAX: int = 10_000_000_000_000
# (필수) 등락률 하한(%) — 이 값 미만은 제외
AUTO_MIN_CHANGE_PCT: float = 5.0
# 매수: ▶ 버튼 누르면 자동 매수할지 여부
AUTO_BUY: bool = True
# 종목당 투입금액(원) — 수량 = seed // 단가
AUTO_ENTRY_SEED: int = 1_000_000
# 매수 방식: "market"(가능하면 시장가), "limit"(지정가; 단가를 1틱 상향해 즉시체결 유도)
AUTO_BUY_MODE: str = "market"
# 매수 단가 고정 (원). None이면 현재가(없으면 종가→시가)를 사용
AUTO_BUY_PRICE: Optional[int] = None

# ===== 매도 설정 =====
# (1) 이익 실현: 진입가 대비 +x% 이상이면 전량 매도 (기본 3.0%)
AUTO_ENABLE_SELL: bool = True
AUTO_TAKE_PROFIT_PCT: float = 3.0
# (2) 장 종료 강제 매도 시각 (HH:MM)
AUTO_EXIT_TIME: str = "15:20"
# 매도 방식: "market"(가능하면 시장가) | "limit"(지정가; 즉시체결 유도)
AUTO_SELL_MODE: str = "market"
# 지정가 매도 시, 현재가 대비 여유폭(슬립) — 예: 0.3% 낮춰 호가 하향
AUTO_SELL_LIMIT_SLIP: float = 0.003
# 실시간 감시 주기(초)
AUTO_SELL_POLL_SEC: int = 10

# ======================


# ---- pykrx 로딩 ----
try:
    from pykrx import stock
except Exception:
    raise SystemExit("pykrx 가 설치되어 있지 않습니다. 먼저 'pip install pykrx' 를 실행하세요.")

# ===== 유틸 =====

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

def screen_three_conditions(date_yyyymmdd: str, topn: int, cap_min: int, cap_max: int, min_change_pct: float) -> pd.DataFrame:
    """세 조건(거래대금 상위, 시총 범위, 정배열)을 모두 만족하는 리스트 반환"""
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

    # (필수) 등락률 필터링
    df_cap = df_cap[df_cap["change_pct"] >= float(min_change_pct)].copy()

    # 3) 정배열
    ma_info = {sym: _compute_ma_alignment_for_symbol(sym, date_yyyymmdd) for sym in df_cap["symbol"]}
    df_cap["MA_align"]  = df_cap["symbol"].map(lambda s: "O" if ma_info.get(s, {}).get("is_bullish") else "X")
    df_cap["MA_reason"] = df_cap["symbol"].map(lambda s: ma_info.get(s, {}).get("reason", ""))

    out = df_cap[df_cap["MA_align"] == "O"].copy()
    cols = [c for c in [
        "symbol","name","trading_value","market_cap","close","open","change_pct","MA_align","MA_reason"
    ] if c in out.columns]
    return out.sort_values("trading_value", ascending=False)[cols]

def symbol_status(date_yyyymmdd: str, symbol: str, topn: int, cap_min: int, cap_max: int, min_change_pct: float) -> pd.DataFrame:
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
    change_ok = (change_pct >= float(min_change_pct))

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
        "MA_reason": ma.get("reason", ""),
        "ALL_OK": bool(in_topn and cap_ok and ma_ok and change_ok),
    }
    return pd.DataFrame([result])

def _resolve_date_arg(date_arg: Optional[str]) -> str:
    if date_arg:
        return date_arg.replace("-", "")
    return datetime.now().strftime("%Y%m%d")

def main():
    # === 런타임 입력(재생 버튼 전용) ===
    # - 엔터만 치면 기본값 유지
    global AUTO_DATE, AUTO_ENTRY_SEED, AUTO_TAKE_PROFIT_PCT
    try:
        print("\n[SETUP] 실행 전 간단 설정 (엔터 = 기본값 유지)")
        print(f"  - 기본 날짜: {AUTO_DATE or '오늘'}")
        print(f"  - 기본 종목당 투입금액: {AUTO_ENTRY_SEED:,}원")
        print(f"  - 기본 익절 퍼센트: {AUTO_TAKE_PROFIT_PCT:.2f}%")

        _in_date = input("📅 매매 날짜 입력 (YYYY-MM-DD, 엔터시 오늘): ").strip()
        if _in_date:
            AUTO_DATE = _in_date

        _in_seed = input("💰 종목당 투입금액 입력 (원, 엔터시 기본 유지): ").strip().replace(",", "")
        if _in_seed:
            AUTO_ENTRY_SEED = int(float(_in_seed))

        _in_tp = input("📈 익절 퍼센트 입력 (기본 3, 엔터시 기본 유지): ").strip()
        if _in_tp:
            AUTO_TAKE_PROFIT_PCT = float(_in_tp)

        print(f"[SETUP] 적용값 → 날짜={AUTO_DATE or '오늘'}, 종목당 시드={AUTO_ENTRY_SEED:,}원, 익절={AUTO_TAKE_PROFIT_PCT:.2f}%\n")
    except Exception as _e:
        print(f"[SETUP] 입력 처리 중 오류(기본값 사용): {_e}")
    # 1) 날짜 결정
    date = _resolve_date_arg(AUTO_DATE)

    # 2) 스크리닝
    df = screen_three_conditions(date, AUTO_TOPN, AUTO_CAP_MIN, AUTO_CAP_MAX, AUTO_MIN_CHANGE_PCT)
    print(f"[SCREEN] {date} | 조건: 거래대금 TOP{AUTO_TOPN} ∧ 시총 {AUTO_CAP_MIN:,}~{AUTO_CAP_MAX:,} ∧ 정배열 ∧ 등락률 ≥ {AUTO_MIN_CHANGE_PCT}%")
    if df.empty:
        print("🚫 조건을 만족하는 종목이 없습니다.")
        return
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(df)

    # 3) 자동 매수 (설정으로 제어)
    if not AUTO_BUY:
        print("[AUTO-BUY] 비활성화(AUTO_BUY=False). 매수는 실행하지 않습니다.")
        return

    # kis_common 확인
    if create_kis_context is None or call_with_retry is None or dbuy_limit is None:
        print("❌ kis_common이 없거나 필수 함수가 없습니다. 자동매수를 진행할 수 없습니다.")
        return

    ctx = create_kis_context()
    token = ctx.get("token"); appkey = ctx.get("appkey"); appsecret = ctx.get("appsecret")
    if not token or not appkey or not appsecret:
        print("❌ KIS 인증정보가 없습니다. kis_common 설정을 확인하세요.")
        return

    print(f"[AUTO-BUY] 방식={AUTO_BUY_MODE} | 종목당 시드={AUTO_ENTRY_SEED:,}원 | buy_price={AUTO_BUY_PRICE if AUTO_BUY_PRICE else '현재가/종가/시가'}")
    placed: list[dict] = []

    for _, r in df.iterrows():
        sym = str(r["symbol"]).zfill(6)

        # 기준 단가: 설정값 > 실시간현재가 > 종가 > 시가
        unit_px = None
        if AUTO_BUY_PRICE:
            unit_px = int(AUTO_BUY_PRICE)
        else:
            last = None
            if get_domestic_last_price is not None:
                try:
                    last = get_domestic_last_price(sym)
                except Exception:
                    last = None
            if last is not None and float(last) > 0:
                unit_px = int(round(float(last)))
            else:
                cp = int(r.get("close") or 0)
                op = int(r.get("open") or 0)
                unit_px = cp if cp > 0 else op

        if not unit_px or unit_px <= 0:
            print(f"[AUTO-BUY] {sym}: 유효단가 없음 → 스킵")
            continue

        qty = max(1, int(int(AUTO_ENTRY_SEED) // int(unit_px)))
        if qty <= 0:
            print(f"[AUTO-BUY] {sym}: 시드 부족(entry_seed={AUTO_ENTRY_SEED:,}, 단가={unit_px}) → 스킵")
            continue

        entry_px = int(unit_px)

        # 시장가 우선 시도 (지원 시)
        if AUTO_BUY_MODE.lower() == "market" and 'dbuy_market' in globals() and dbuy_market is not None:
            print(f"[AUTO-BUY] MARKET {sym} x{qty}")
            try:
                resp = call_with_retry(dbuy_market, token, appkey, appsecret, sym, int(qty))
                placed.append({"symbol": sym, "qty": int(qty), "side": "buy", "mode": "market", "price": None, "entry_price": entry_px, "sold": False, "resp": resp})
                print(resp)
                continue
            except Exception as e:
                print(f"[AUTO-BUY] 시장가 실패 → 지정가 폴백: {e}")

        # 지정가(즉시체결 유도: 1틱 상향)
        px = align_krx_price(unit_px, "up")
        print(f"[AUTO-BUY] LIMIT {sym} x{qty} @ {px}")
        try:
            resp = call_with_retry(dbuy_limit, token, appkey, appsecret, sym, int(qty), int(px))
            placed.append({"symbol": sym, "qty": int(qty), "side": "buy", "mode": "limit", "price": int(px), "entry_price": entry_px, "sold": False, "resp": resp})
            print(resp)
        except Exception as e:
            print(f"[AUTO-BUY] 지정가 실패: {e}")

    if not placed:
        print("[AUTO-BUY] 체결 시도 내역 없음(전부 스킵)")
    else:
        print(f"[AUTO-BUY] 주문 건수: {len(placed)}건 완료")

    # 4) 자동 매도: 수익률 +AUTO_TAKE_PROFIT_PCT% 또는 장 종료시 전량 매도
    if AUTO_ENABLE_SELL and placed:
        if get_domestic_last_price is None:
            print("❌ 실시간 현재가 함수(get_domestic_last_price)가 없어 매도 관리가 불가합니다.")
        else:
            # 장 종료 시각 계산
            try:
                eh, em = map(int, AUTO_EXIT_TIME.split(":"))
            except Exception:
                print(f"[AUTO-SELL] 잘못된 종료시각 형식: {AUTO_EXIT_TIME} → 매도 관리 생략")
                eh, em = 15, 20
            from datetime import datetime
            exit_dt = datetime.now().replace(hour=eh, minute=em, second=0, microsecond=0)

            print(f"[AUTO-SELL] 시작 — TP={AUTO_TAKE_PROFIT_PCT:.2f}% | 종료시각={AUTO_EXIT_TIME} | mode={AUTO_SELL_MODE}")
            try:
                while datetime.now() < exit_dt:
                    all_sold = True
                    for pos in placed:
                        if pos.get("sold"):
                            continue
                        all_sold = False
                        sym = pos["symbol"]; qty = int(pos["qty"])
                        entry_px = int(pos.get("entry_price") or 0)
                        if entry_px <= 0 or qty <= 0:
                            pos["sold"] = True
                            continue
                        last = None
                        try:
                            last = get_domestic_last_price(sym)
                        except Exception:
                            last = None
                        if last is None or float(last) <= 0:
                            continue
                        last_i = int(round(float(last)))

                        # 이익 실현 조건
                        tp_px = int(round(entry_px * (1.0 + float(AUTO_TAKE_PROFIT_PCT) / 100.0)))
                        if last_i >= tp_px:
                            print(f"[TP] {sym}: last={last_i} ≥ target={tp_px} (entry={entry_px}) → 매도 시도")
                            try:
                                if AUTO_SELL_MODE.lower() == "market" and 'dsell_market' in globals() and dsell_market is not None:
                                    resp_s = call_with_retry(dsell_market, token, appkey, appsecret, sym, qty)
                                else:
                                    # 지정가: 현재가 기준 약간 낮춰 신속 체결 유도
                                    base_px = int(round(last_i * (1.0 - float(AUTO_SELL_LIMIT_SLIP))))
                                    px_out = align_krx_price(base_px, "down")
                                    print(f"[TP] LIMIT {sym} x{qty} @ {px_out}")
                                    if 'dsell_limit' in globals() and dsell_limit is not None:
                                        resp_s = call_with_retry(dsell_limit, token, appkey, appsecret, sym, qty, int(px_out))
                                    else:
                                        print("[AUTO-SELL] dsell_limit 미제공 → 매도 스킵")
                                        resp_s = None
                                print(resp_s)
                                pos["sold"] = True
                            except Exception as e:
                                print(f"[TP] 매도 실패: {e}")
                    if all_sold:
                        print("[AUTO-SELL] 전량 매도 완료")
                        break
                    time.sleep(int(AUTO_SELL_POLL_SEC))
            except KeyboardInterrupt:
                print("[AUTO-SELL] 사용자 중단")

            # 장 종료 강제 매도 (남아있으면)
            remains = [p for p in placed if not p.get("sold")]
            if remains:
                print("[AUTO-SELL] 장 종료 — 잔여 물량 강제 매도")
                for pos in remains:
                    sym = pos["symbol"]; qty = int(pos["qty"])
                    try:
                        if AUTO_SELL_MODE.lower() == "market" and 'dsell_market' in globals() and dsell_market is not None:
                            resp_e = call_with_retry(dsell_market, token, appkey, appsecret, sym, qty)
                        else:
                            # 지정가: 현재가가 있으면 약간 낮춰, 없으면 진입가 기준 하향
                            last = None
                            try:
                                last = get_domestic_last_price(sym)
                            except Exception:
                                last = None
                            base = int(round(float(last))) if (last is not None and float(last) > 0) else int(pos.get("entry_price") or 0)
                            px_out = align_krx_price(int(base * (1.0 - float(AUTO_SELL_LIMIT_SLIP))), "down") if base > 0 else None
                            if px_out is not None and 'dsell_limit' in globals() and dsell_limit is not None:
                                print(f"[EOD] LIMIT {sym} x{qty} @ {px_out}")
                                resp_e = call_with_retry(dsell_limit, token, appkey, appsecret, sym, qty, int(px_out))
                            elif 'dsell_market' in globals() and dsell_market is not None:
                                # 지정가 미제공 시 시장가 폴백
                                resp_e = call_with_retry(dsell_market, token, appkey, appsecret, sym, qty)
                            else:
                                print("[EOD] 매도 함수 미제공 → 스킵")
                                resp_e = None
                        print(resp_e)
                        pos["sold"] = True
                    except Exception as e:
                        print(f"[EOD] 매도 실패: {e}")
            else:
                print("[AUTO-SELL] 잔여 물량 없음")

if __name__ == "__main__":
    main()