# strategy_uvix.py
from __future__ import annotations
import math
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import yfinance as yf

from kis_common import (
    create_kis_context, get_position_qty_overseas, call_with_retry,
    obuy_limit, osell_limit, is_mock_env
)

# ===== 설정 =====
MODE = "live"  # "live" or "backtest"
FUNC1_TICKER   = "UVIX"
FUNC1_START    = "2025-01-01"
FUNC1_END      = None
FUNC1_CAPITAL  = 10_000.0

SPIKE_CHECK = True
SPIKE_BASIS = "close"       # "close" | "high"
SPIKE_THRESHOLD_PCT = 5.0
SPIKE_EXIT_AT = "open_next" # "open_next" | "close_same"

# LIVE
LIVE_TICKER = "UVIX"
LIVE_EXCG   = "AMEX"     # UVIX(ARCA)는 AMEX 코드 사용되는 경우가 일반적
LIVE_BUDGET = 1000.0
LIVE_BUY_UP_PCT = 0.003  # 전일 종가 대비 0.3% 위 지정가
LIVE_SELL_PCT   = 1.0    # 100% 청산

def _get_yf_daily(ticker: str, start: str, end: Optional[str]) -> pd.DataFrame:
    # 견고한 다운로드: 컬럼 묶임 방지 옵션 + 진행바 끔
    df = yf.download(
        ticker,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        group_by="column",
        progress=False,
    )

    # yfinance가 Series를 줄 때 대비
    if isinstance(df, pd.Series):
        df = df.to_frame("Close")

    # 1) 멀티인덱스/이상 컬럼 평탄화
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([str(x) for x in col if x is not None]).strip() for col in df.columns]

    # 2) 정상 컬럼 조합 찾기(Open/High/Low/Close)
    want = ["Open", "High", "Low", "Close"]

    def _find_field(name: str) -> Optional[str]:
        cols = list(df.columns)
        cands = [
            name,
            name.lower(),
            f"{name}_{ticker}",
            f"{ticker}_{name}",
            f"{name.lower()}_{ticker.lower()}",
            f"{ticker.lower()}_{name.lower()}",
            f"{name.upper()}_{ticker.upper()}",
            f"{ticker.upper()}_{name.upper()}",
        ]
        for c in cands:
            if c in cols:
                return c
        return None

    mapping = {k: _find_field(k) for k in want}

    # 3) 매핑 실패 시 폴백: Ticker().history()로 재시도
    if any(v is None for v in mapping.values()):
        tkr = yf.Ticker(ticker)
        df2 = tkr.history(start=start, end=end, interval="1d", auto_adjust=False)
        if isinstance(df2, pd.Series):
            df2 = df2.to_frame("Close")
        if isinstance(df2.columns, pd.MultiIndex):
            df2.columns = ["_".join([str(x) for x in col if x is not None]).strip() for col in df2.columns]
        df = df2
        mapping = {k: (k if k in df.columns else _find_field(k)) for k in want}

    # 여전히 못 찾으면 에러를 명확히
    if any(v is None for v in mapping.values()):
        raise ValueError(f"{ticker}에 'Open/High/Low/Close' 컬럼을 찾지 못했습니다: {list(df.columns)}")

    out = df[[mapping["Open"], mapping["High"], mapping["Low"], mapping["Close"]]].copy()
    out.columns = ["Open", "High", "Low", "Close"]

    # 숫자화 + 정렬 + 결측 제거
    for c in ["Open", "High", "Low", "Close"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["Open", "High", "Low", "Close"]).sort_index()

    return out

def _nth_trading_day_after(index: pd.DatetimeIndex, base_ts, n: int):
    pos = index.searchsorted(base_ts, side="right")
    target = pos + (n - 1)
    return index[target] if target < len(index) else None

def backtest():
    end_date = FUNC1_END or datetime.now().strftime("%Y-%m-%d")
    df = _get_yf_daily(FUNC1_TICKER, FUNC1_START, end_date)
    if getattr(df.index, "tz", None) is None:
        df.index = df.index.tz_localize("UTC").tz_convert("America/New_York")
    else:
        df.index = df.index.tz_convert("America/New_York")

    def _month_min_with_date(sub: pd.DataFrame):
        return pd.Series({"min_low": float(sub["Low"].min()), "min_date": sub["Low"].idxmin()})

    by_month = (df.assign(YM=df.index.to_period("M"))
                  .groupby("YM", group_keys=False)
                  .apply(_month_min_with_date)
                  .reset_index())

    print("\n[UVIX] 월별 최저 Low 및 발생일")
    for _, row in by_month.iterrows():
        ym = str(row["YM"]); v = row["min_low"]; d = row["min_date"]
        print(f"{ym}\t{v:.4f}\t@{d.strftime('%Y-%m-%d')}")

    # Always run the full backtest routine now; RUN_BACKTEST flag removed

    results = []; idx = df.index
    for _, r in by_month.iterrows():
        ym = r["YM"]; min_day = r["min_date"]
        buy_days = [ _nth_trading_day_after(idx, min_day, k) for k in range(4, 9) ]
        buy_days = [ts for ts in buy_days if ts is not None][:5]
        if not buy_days: continue

        capital = float(FUNC1_CAPITAL); cash = capital; qty = 0.0
        each = capital * 0.2; buys = []; spike_end = False

        for ts in buy_days:
            if SPIKE_CHECK and qty > 0:
                i = int(df.index.get_indexer([ts])[0])
                if i > 0:
                    prev_close = float(df["Close"].iloc[i-1])
                    today_close = float(df["Close"].iloc[i])
                    today_high  = float(df["High"].iloc[i])
                    basis = today_close if SPIKE_BASIS=="close" else today_high
                    thres = prev_close * (1 + SPIKE_THRESHOLD_PCT/100)
                    if basis >= thres:
                        if SPIKE_EXIT_AT == "open_next":
                            if i+1 < len(df.index): exit_dt, exit_px = df.index[i+1], float(df["Open"].iloc[i+1])
                            else: exit_dt, exit_px = df.index[i], today_close
                        else:
                            exit_dt, exit_px = df.index[i], today_close
                        equity = cash + qty * exit_px; roi = (equity/capital - 1)*100
                        buys_detail = ", ".join(f"{pd.Timestamp(b[0]).strftime('%Y-%m-%d')} {b[1]:.4f}" for b in buys) if buys else ""
                        results.append({
                            "month": str(ym), "signal_date": min_day.strftime("%Y-%m-%d"),
                            "num_buys": len(buys), "buys_detail": buys_detail,
                            "avg_price": (sum(b[3] for b in buys) / sum(b[2] for b in buys)) if buys else 0.0,
                            "invested": sum(b[3] for b in buys) if buys else 0.0,
                            "remaining_cash": cash,
                            "eval_d8_date": None, "close_d8": math.nan, "equity_d8": math.nan, "roi_d8_%": math.nan,
                            "eom_date": None, "close_eom": math.nan, "equity_eom": math.nan, "roi_eom_%": math.nan,
                            "spike_exit_date": exit_dt.strftime("%Y-%m-%d"), "spike_exit_px": exit_px,
                            "equity_spike": equity, "roi_spike_%": roi,
                        })
                        spike_end = True; break

            px = float(df.at[ts, "Close"])
            invest = min(each, cash)
            if invest <= 0 or px <= 0: continue
            add = invest / px
            qty += add; cash -= invest; buys.append((ts, px, add, invest))

        if spike_end or qty <= 0: continue

        total = sum(b[3] for b in buys); avg = total / qty if qty else 0.0
        d8 = buy_days[-1]; close_d8 = float(df.at[d8, "Close"]); eq_d8 = cash + qty * close_d8; roi_d8 = (eq_d8/capital-1)*100

        month_df = df.loc[df.index.to_period("M") == ym]
        if not month_df.empty:
            last_ts = month_df.index[-1]; close_eom = float(df.at[last_ts, "Close"])
            eq_eom = cash + qty * close_eom; roi_eom = (eq_eom/capital-1)*100
        else:
            last_ts = None; close_eom = None; eq_eom = None; roi_eom = None

        buys_detail = ", ".join(f"{pd.Timestamp(b[0]).strftime('%Y-%m-%d')} {b[1]:.4f}" for b in buys) if buys else ""
        results.append({
            "month": str(ym), "signal_date": min_day.strftime("%Y-%m-%d"),
            "num_buys": len(buys), "buys_detail": buys_detail,
            "avg_price": avg, "invested": total, "remaining_cash": cash,
            "eval_d8_date": d8.strftime("%Y-%m-%d"), "close_d8": close_d8, "equity_d8": eq_d8, "roi_d8_%": roi_d8,
            "eom_date": last_ts.strftime("%Y-%m-%d") if last_ts else None,
            "close_eom": close_eom, "equity_eom": eq_eom, "roi_eom_%": roi_eom,
            "spike_exit_date": None, "spike_exit_px": None, "equity_spike": None, "roi_spike_%": None,
        })

    res_df = pd.DataFrame(results)
    print("\n[UVIX] 신저점 D+4~D+8 분할매수 백테스트 (초기자본 ${:,.2f})".format(FUNC1_CAPITAL))
    for _, rr in res_df.iterrows():
        eom_eq = "nan" if pd.isna(rr.get("equity_eom")) else f"{rr['equity_eom']:.2f}"
        eom_roi = "nan" if pd.isna(rr.get("roi_eom_%")) else f"{rr['roi_eom_%']:.2f}%"
        spike_part = ""
        if rr.get("spike_exit_date"):
            spike_part = f" | SPIKE exit {rr['spike_exit_date']} px {rr['spike_exit_px']:.4f} equity ${rr['equity_spike']:.2f} ({rr['roi_spike_%']:.2f}%)"
        buys_part = f" ({rr['buys_detail']})" if rr.get("buys_detail") else ""
        print(f"{rr['month']} | signal {rr['signal_date']} | buys {int(rr['num_buys'])}{buys_part} | avg {rr['avg_price']:.4f} | "
              f"D+8 {rr['eval_d8_date']} equity ${rr['equity_d8']:.2f} ({rr['roi_d8_%']:.2f}%) | "
              f"EOM {rr['eom_date']} equity ${eom_eq} ({eom_roi}){spike_part}")

def is_us_regular_open_kst() -> bool:
    # KST 기준 (서머타임 22:30~05:00) 대략 체크
    from datetime import timezone
    kst = datetime.utcnow() + timedelta(hours=9)
    open_dt = (kst - timedelta(days=1)).replace(hour=22, minute=30, second=0, microsecond=0) if kst.hour < 5 else kst.replace(hour=22, minute=30, second=0, microsecond=0)
    close_dt = kst.replace(hour=5, minute=0, second=0, microsecond=0) if kst.hour < 5 else (kst + timedelta(days=1)).replace(hour=5, minute=0, second=0, microsecond=0)
    return open_dt <= kst < close_dt

def live_once():
    # 1) 미국 정규장 체크
    if not is_us_regular_open_kst():
        print("⛔ [LIVE] 현재는 미국 정규장이 아닙니다. (모의/실주문 제한)")
        print("    - 조건: KST 기준 22:30 ~ 05:00 사이에만 주문 시도")
        return

    # 2) KIS 컨텍스트 확보
    ctx = create_kis_context()
    token = ctx["token"]
    headers = ctx["headers"]

    # 3) 데이터 조회 (최근 120일)
    end_dt = datetime.utcnow().strftime("%Y-%m-%d")
    start_dt = (datetime.utcnow() - timedelta(days=120)).strftime("%Y-%m-%d")
    try:
        df = _get_yf_daily(LIVE_TICKER, start_dt, end_dt).tz_localize("UTC").tz_convert("America/New_York")
    except Exception as e:
        print(f"❌ [LIVE] 데이터 조회 실패: {e}")
        return

    # 월 데이터 및 이번 달 신저점
    month_df = df.loc[df.index.to_period("M") == df.index[-1].to_period("M")]
    if month_df.empty:
        print("⚠️ [LIVE] 이번 달 데이터가 없습니다. (거래일 부족)")
        return

    min_day = month_df["Low"].idxmin()
    print(f"[LIVE] 이번 달 신저점 발생일(min_day): {min_day.strftime('%Y-%m-%d')}")

    # D+4 ~ D+8 후보일 계산
    idx = df.index
    cands = [ _nth_trading_day_after(idx, min_day, k) for k in range(4,9) ]
    cands = [ts.normalize() for ts in cands if ts is not None]
    cands_str = ", ".join(ts.strftime('%Y-%m-%d') for ts in cands) if cands else "(없음)"

    # 오늘(뉴욕) 날짜
    today_ny = pd.Timestamp(datetime.now(timezone.utc)).tz_convert("America/New_York").normalize()
    print(f"[LIVE] D+4~D+8 후보일: {cands_str} | 오늘(NY): {today_ny.strftime('%Y-%m-%d')}")

    # === 오늘 상태 표시: MIN_DAY / D+1~D+8 / 일반일 ===
    try:
        # D+1 ~ D+8 전부 계산 (거래일 기준)
        raw_days = [ _nth_trading_day_after(idx, min_day, k) for k in range(1, 9) ]
        dmap = {}
        for k, ts in zip(range(1, 9), raw_days):
            if ts is not None:
                dmap[ts.normalize()] = k

        if today_ny == min_day.normalize():
            print("🟡 [LIVE] 오늘은 신저점 발생일(MIN_DAY)입니다. (※ 확정은 종가 이후)")
        elif today_ny in dmap:
            k = dmap[today_ny]
            print(f"🔶 [LIVE] 오늘은 신저점 D+{k} 입니다.")
        else:
            print("⚪ [LIVE] 오늘은 신저점 관련 일반일입니다.")
    except Exception as _e:
        print(f"⚠️ [LIVE] 오늘 상태 판별 중 오류: {_e}")

    # === 전체 D+일차 표시 (D+0부터 무한) ===
    try:
        base = min_day.normalize()
        # 인덱스를 날짜 단위로 정규화하여 비교
        idx_norm = pd.DatetimeIndex([ts.normalize() for ts in idx])
        pos_min_right = idx_norm.searchsorted(base, side="right")
        pos_today_right = idx_norm.searchsorted(today_ny, side="right")
        d_all = max(0, int(pos_today_right - pos_min_right))  # D+0, D+1, ...
        print(f"🧭 [LIVE] 오늘은 신저점 발생 후 D+{d_all}일차 입니다.")
    except Exception as _e:
        print(f"⚠️ [LIVE] D+일차 계산 오류: {_e}")

    # 보유 수량 진단 (예외 무시)
    try:
        pos_qty = get_position_qty_overseas(headers, LIVE_EXCG, LIVE_TICKER)
        print(f"[LIVE] 보유수량 {LIVE_TICKER}: {pos_qty}")
    except Exception as e:
        print(f"⚠️ [LIVE] 보유수량 조회 실패(무시): {e}")
        pos_qty = 0

    # (선택) 스파이크 청산 로직 (있는 그대로 유지)
    if len(df.index) >= 3:
        prev = df.index[-1].normalize(); prev1 = df.index[-2]; prev2 = df.index[-3]
        prev_close = float(df.loc[prev1, "Close"]); prev_prev = float(df.loc[prev2, "Close"]) 
        if prev_close >= prev_prev*(1+SPIKE_THRESHOLD_PCT/100) and pos_qty>0 and today_ny==prev:
            ref_px = float(df.loc[prev, "Open"]) if "Open" in df.columns else prev_close
            sell_qty = max(1, int(pos_qty * 1.0))
            px = round(ref_px * 0.99, 2)
            try:
                resp = call_with_retry(osell_limit, token, ctx["appkey"], ctx["appsecret"], LIVE_EXCG, LIVE_TICKER, sell_qty, px)
                print("🔴 청산(스파이크 다음날 시초가 근처):"); import json; print(json.dumps(resp, ensure_ascii=False, indent=2))
            except Exception as e:
                print(f"⚠️ [LIVE] 청산 실패: {e}")

    # 4) 날짜 조건: 오늘이 D+4~D+8인가?
    if today_ny not in cands:
        print("⏳ [LIVE] 오늘은 신저점 발생일의 D+4~D+8 구간이 아닙니다. 매수 대기.")
        return

    # 5) 예산/수량 계산
    ref_close = float(df.iloc[-1]["Close"])  # 전일 종가 기준
    buy_px = round(ref_close*(1+LIVE_BUY_UP_PCT), 2)
    qty = int(LIVE_BUDGET // buy_px)
    print(f"[LIVE] 매수 지정가 계산: ref_close={ref_close:.4f}, buy_px={buy_px:.4f}, budget={LIVE_BUDGET}, qty={qty}")

    if qty < 1:
        print(f"💸 [LIVE] 예산 부족으로 매수 불가 (예산: {LIVE_BUDGET}, 지정가: {buy_px}, 계산수량: {qty})")
        return

    # 6) 주문 요청
    try:
        resp = call_with_retry(obuy_limit, token, ctx["appkey"], ctx["appsecret"], LIVE_EXCG, LIVE_TICKER, qty, buy_px)
    except Exception as e:
        print(f"❌ [LIVE] 주문 중 예외 발생: {e}")
        return

    if resp is None or (isinstance(resp, dict) and resp.get("rt_cd") != "0"):
        print(f"⚠️ [LIVE] 주문 요청 실패. 거래소코드/티커/헤더를 확인하세요. 응답: {resp}")
        return

    print(f"✅ [LIVE] 매수 주문 성공! {LIVE_TICKER} {qty}주 @ {buy_px}")

# For compatibility: define clear entry points for backtest/live
def run_backtest():
    print("[*] UVIX 전략 실행 — mode=backtest")
    backtest()

def run_live():
    print("[*] UVIX 전략 실행 — mode=live")
    live_once()
# === main guard ===
if __name__ == "__main__":
    try:
        if MODE == "backtest":
            run_backtest()
        elif MODE == "live":
            run_live()
        else:
            raise ValueError(f"Unknown MODE: {MODE}")
    except KeyboardInterrupt:
        print("\n[UVIX] Stopped by user")
    except Exception as e:
        print(f"[UVIX] FATAL: {e}")