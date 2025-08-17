from __future__ import annotations
import concurrent.futures as cf
import math
import time
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import pandas as pd
import numpy as np
import yfinance as yf

from datetime import datetime, timezone
from math import log, sqrt, exp, pi

# 표준정규분포 PDF 함수 (scipy.stats.norm.pdf 대체)
def _norm_pdf(x: float) -> float:
    return (1.0 / sqrt(2.0 * pi)) * exp(-0.5 * x * x)

# (optional) 나스닥 전체 티커 수집용 — 없으면 위키피디아 NASDAQ-100으로 폴백
try:
    from yahoo_fin import stock_info as si
    HAVE_YFIN = True
except Exception:
    HAVE_YFIN = False

# ===== 설정 =====
# 정렬 기준: 'abs' → 절대차이(달러), 'pct' → 퍼센트 차이
SORT_BY: str = "pct"          # 'abs' | 'pct'
TOP_N: int = 30                # 상위 N개만 출력
MAX_WORKERS: int = 8           # 동시 요청 개수 (네트워크 상황에 맞춰 조절)
NEAREST_EXPIRY_INDEX: int = 0  # 0=가장 가까운 만기, 1=그 다음 만기...
REQUEST_DELAY_SEC: float = 0.05  # 티커별 딜레이(서버 부하 방지)

# 수집 실패 시 건너뛰기. True면 옵션 데이터 없으면 스킵
SKIP_IF_NO_OPTIONS: bool = True

# 특정 티커만 테스트하고 싶으면 목록 지정. None이면 S&P500 전체 사용
OVERRIDE_TICKERS: Optional[List[str]] = None

DIFF_THRESHOLD: float = 0.02  # 최대 허용 diff_pct

WALL_TOP_K: int = 3  # 콜/풋 OI가 많이 모여있는 상위 K개 행사가만 표시

RISK_FREE_RATE: float = 0.03  # 연 3% 가정 (필요시 조정)
CONTRACT_SIZE: int = 100      # 미국 옵션 1계약 = 100주


@dataclass
class MaxPainResult:
    symbol: str
    price: float
    max_pain: float
    diff_abs: float
    diff_pct: float
    expiry: str
    n_calls_oi: int
    n_puts_oi: int
    call_walls: str
    put_walls: str
    atm_iv: float
    gex: float


# ---------- 유틸 ----------

# ----- 옵션 그릭스/IV 유틸 -----

def _bs_d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if S <= 0 or K <= 0 or sigma <= 0 or T <= 0:
        return float('nan')
    return (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T))


def bs_gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes 감마. 콜/풋 동일."""
    try:
        d1 = _bs_d1(S, K, T, r, sigma)
        if np.isnan(d1):
            return float('nan')
        return _norm_pdf(d1) / (S * sigma * sqrt(T))
    except Exception:
        return float('nan')


def parse_expiry_days(expiry_str: str) -> float:
    """만기 문자열(YYYY-MM-DD)로부터 잔여기간(년)을 계산"""
    try:
        dt_exp = datetime.strptime(expiry_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)        
        days = max(0.0, (dt_exp - now).total_seconds() / 86400.0)
        return days / 365.0
    except Exception:
        return float('nan')


def _clean_symbol(sym: str) -> str:
    """Wikipedia 심볼 표기(예: BRK.B)를 yfinance 호환(BRK-B)으로 정리."""
    return sym.replace(".", "-").strip()


def get_sp500_tickers() -> List[str]:
    """Wikipedia에서 S&P500 종목 리스트를 가져온다."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    if not tables:
        raise RuntimeError("S&P500 테이블을 찾지 못했습니다.")
    df = tables[0]
    col = 'Symbol' if 'Symbol' in df.columns else df.columns[0]
    tickers = [ _clean_symbol(s) for s in df[col].astype(str).tolist() ]
    # 중복/결측 제거
    tickers = [t for t in tickers if t and t.lower() != 'nan']
    return tickers


def get_nasdaq_tickers() -> List[str]:
    """나스닥 상장 종목 티커 리스트. 
    - 1순위: yahoo_fin으로 전량 수집
    - 폴백: Wikipedia의 NASDAQ-100 구성 종목
    """
    # 1) yahoo_fin 사용 가능 시 (전량)
    if HAVE_YFIN:
        try:
            tickers = si.tickers_nasdaq()
            return [t.replace('.', '-').strip() for t in tickers if isinstance(t, str) and t.strip()]
        except Exception:
            pass
    
    # 2) 폴백: Wikipedia NASDAQ-100
    url = "https://en.wikipedia.org/wiki/NASDAQ-100"
    tables = pd.read_html(url)
    # 'Ticker' 또는 'Symbol' 컬럼이 있는 테이블을 탐색
    for tb in tables:
        cols = [str(c) for c in tb.columns]
        if 'Ticker' in cols or 'Symbol' in cols:
            col = 'Ticker' if 'Ticker' in cols else 'Symbol'
            tickers = [str(s) for s in tb[col].tolist()]
            return [s.replace('.', '-').strip() for s in tickers if s and s.lower() != 'nan']
    raise RuntimeError("NASDAQ 티커 테이블을 찾지 못했습니다.")


def safe_current_price(tk: yf.Ticker) -> Optional[float]:
    """현재가 조회(우선순위: fast_info → info → 최근 종가)."""
    try:
        fi = getattr(tk, 'fast_info', None)
        if fi and getattr(fi, 'last_price', None):
            return float(fi.last_price)
    except Exception:
        pass
    try:
        info = tk.info
        if info and 'currentPrice' in info and info['currentPrice']:
            return float(info['currentPrice'])
    except Exception:
        pass
    try:
        hist = tk.history(period="2d", interval="1d", auto_adjust=False)
        if not hist.empty and 'Close' in hist.columns:
            return float(hist['Close'].iloc[-1])
    except Exception:
        pass
    return None


def calc_max_pain(calls: pd.DataFrame, puts: pd.DataFrame) -> Optional[float]:
    """맥스페인 계산. strike별 콜/풋 OI 손익 합이 최소가 되는 strike 반환."""
    if calls is None or puts is None or calls.empty or puts.empty:
        return None

    # 필수 컬럼 방어
    for need in ("strike", "openInterest"):
        if need not in calls.columns or need not in puts.columns:
            return None

    # 유효 OI만 사용
    calls = calls.copy()
    puts = puts.copy()
    calls['openInterest'] = pd.to_numeric(calls['openInterest'], errors='coerce').fillna(0)
    puts['openInterest'] = pd.to_numeric(puts['openInterest'], errors='coerce').fillna(0)

    # 공통 strike 세트 구성(옵션 체인에 따라 다를 수 있음)
    strikes = sorted(set(calls['strike'].dropna().tolist()) | set(puts['strike'].dropna().tolist()))
    if not strikes:
        return None

    # strike→OI 매핑(dict)으로 빠르게 합산
    call_oi_map = calls.groupby('strike')['openInterest'].sum().to_dict()
    put_oi_map = puts.groupby('strike')['openInterest'].sum().to_dict()

    pains = []
    for k in strikes:
        # Call 보유자 손실: max(0, k - S) * OI 를 S(최종 가격)별 합 → 여기서는 후보 S가 각 strike
        # Put 보유자 손실:  max(0, S - k) * OI
        call_loss = 0.0
        put_loss = 0.0
        for s, oi in call_oi_map.items():
            call_loss += max(0.0, k - s) * float(oi)
        for s, oi in put_oi_map.items():
            put_loss += max(0.0, s - k) * float(oi)
        pains.append((k, call_loss + put_loss))

    if not pains:
        return None

    pains.sort(key=lambda x: x[1])
    return float(pains[0][0])


def fetch_max_pain_for_symbol(symbol: str, expiry_idx: int = 0) -> Optional[MaxPainResult]:
    """단일 심볼에 대해 최근 만기 옵션체인을 가져와 맥스페인을 계산."""
    try:
        tk = yf.Ticker(symbol)
        # 만기일 목록
        expiries = tk.options or []
        if not expiries:
            if SKIP_IF_NO_OPTIONS:
                return None
            else:
                price = safe_current_price(tk)
                if price is None:
                    return None
                return MaxPainResult(symbol, price, math.nan, math.nan, math.nan, "NA", 0, 0, "", "", float('nan'), float('nan'))

        expiry = expiries[min(expiry_idx, len(expiries)-1)]
        chain = tk.option_chain(expiry)
        calls, puts = chain.calls, chain.puts
        mp = calc_max_pain(calls, puts)
        if mp is None or math.isnan(mp):
            return None

        price = safe_current_price(tk)
        if price is None or mp == 0:
            return None

        diff_abs = abs(price - mp)
        diff_pct = diff_abs / abs(mp)
        n_calls = int(pd.to_numeric(calls.get('openInterest', pd.Series([])), errors='coerce').fillna(0).sum()) if 'openInterest' in calls else 0
        n_puts  = int(pd.to_numeric(puts.get('openInterest', pd.Series([])), errors='coerce').fillna(0).sum()) if 'openInterest' in puts else 0

        # 상위 K개 콜/풋 OI 행사가(가격)만 문자열로 추출 (수량은 표시하지 않음)
        def _top_strikes(df_opt: pd.DataFrame, k: int) -> str:
            if df_opt is None or df_opt.empty or 'strike' not in df_opt.columns or 'openInterest' not in df_opt.columns:
                return ""
            tmp = df_opt[['strike','openInterest']].copy()
            tmp['openInterest'] = pd.to_numeric(tmp['openInterest'], errors='coerce').fillna(0)
            grp = tmp.groupby('strike', as_index=False)['openInterest'].sum().sort_values('openInterest', ascending=False).head(k)
            # 예쁘게: 정수에 가까우면 정수로, 아니면 소수 한두 자리
            def _fmt(x: float) -> str:
                try:
                    if abs(x - round(x)) < 1e-6:
                        return str(int(round(x)))
                    return f"{x:.2f}"
                except Exception:
                    return str(x)
            return ", ".join(_fmt(float(s)) for s in grp['strike'].tolist())

        call_walls = _top_strikes(calls, WALL_TOP_K)
        put_walls  = _top_strikes(puts,  WALL_TOP_K)

        # ----- ATM IV 계산 (현재가와 가장 가까운 strike의 IV 사용; calls 우선, 없으면 puts) -----
        atm_iv = float('nan')
        try:
            # calls
            if 'impliedVolatility' in calls.columns and 'strike' in calls.columns:
                calls_iv = calls[['strike', 'impliedVolatility']].dropna()
                if not calls_iv.empty:
                    k_atm = calls_iv.iloc[(calls_iv['strike'] - price).abs().argsort()].iloc[0]
                    atm_iv = float(k_atm['impliedVolatility'])
            # 보완: puts에서 시도
            if (np.isnan(atm_iv)) and ('impliedVolatility' in puts.columns) and ('strike' in puts.columns):
                puts_iv = puts[['strike', 'impliedVolatility']].dropna()
                if not puts_iv.empty:
                    k_atm = puts_iv.iloc[(puts_iv['strike'] - price).abs().argsort()].iloc[0]
                    atm_iv = float(k_atm['impliedVolatility'])
        except Exception:
            atm_iv = float('nan')

        # ----- GEX(감마 익스포저) 근사 계산 -----
        # 정의: 각 행사가의 감마 * OI * 계약크기 * S^2 를 합산(콜은 +, 풋은 -로 가중하는 방식; 설정에 따라 다를 수 있음)
        T_years = parse_expiry_days(expiry)
        gex_sum = 0.0
        try:
            if T_years > 0 and 'strike' in calls.columns and 'openInterest' in calls.columns and 'impliedVolatility' in calls.columns:
                tmp = calls[['strike','openInterest','impliedVolatility']].copy()
                tmp['openInterest'] = pd.to_numeric(tmp['openInterest'], errors='coerce').fillna(0.0)
                tmp['impliedVolatility'] = pd.to_numeric(tmp['impliedVolatility'], errors='coerce').fillna(np.nan)
                for _, row in tmp.iterrows():
                    K = float(row['strike'])
                    oi = float(row['openInterest'])
                    iv = float(row['impliedVolatility']) if not np.isnan(row['impliedVolatility']) else (atm_iv if not np.isnan(atm_iv) else np.nan)
                    if np.isnan(iv) or oi <= 0:
                        continue
                    gamma = bs_gamma(price, K, T_years, RISK_FREE_RATE, iv)
                    if np.isnan(gamma):
                        continue
                    gex_sum += gamma * oi * CONTRACT_SIZE * (price ** 2)
            if T_years > 0 and 'strike' in puts.columns and 'openInterest' in puts.columns and 'impliedVolatility' in puts.columns:
                tmp = puts[['strike','openInterest','impliedVolatility']].copy()
                tmp['openInterest'] = pd.to_numeric(tmp['openInterest'], errors='coerce').fillna(0.0)
                tmp['impliedVolatility'] = pd.to_numeric(tmp['impliedVolatility'], errors='coerce').fillna(np.nan)
                for _, row in tmp.iterrows():
                    K = float(row['strike'])
                    oi = float(row['openInterest'])
                    iv = float(row['impliedVolatility']) if not np.isnan(row['impliedVolatility']) else (atm_iv if not np.isnan(atm_iv) else np.nan)
                    if np.isnan(iv) or oi <= 0:
                        continue
                    gamma = bs_gamma(price, K, T_years, RISK_FREE_RATE, iv)
                    if np.isnan(gamma):
                        continue
                    # 풋은 음수 가중 (콜-풋 차이를 반영)
                    gex_sum -= gamma * oi * CONTRACT_SIZE * (price ** 2)
        except Exception:
            pass

        atm_iv_val = float(atm_iv) if (atm_iv is not None and not np.isnan(atm_iv)) else float('nan')
        gex_val = float(gex_sum)

        return MaxPainResult(symbol, float(price), float(mp), float(diff_abs), float(diff_pct), expiry, n_calls, n_puts, call_walls, put_walls, atm_iv_val, gex_val)
    except Exception:
        return None


# ---------- 메인 파이프라인 ----------

def scan_nasdaq_and_rank() -> pd.DataFrame:
    tickers = OVERRIDE_TICKERS if OVERRIDE_TICKERS else get_nasdaq_tickers()

    results: List[MaxPainResult] = []
    with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = []
        for i, t in enumerate(tickers):
            if REQUEST_DELAY_SEC > 0 and i:
                time.sleep(REQUEST_DELAY_SEC)
            futs.append(ex.submit(fetch_max_pain_for_symbol, t, NEAREST_EXPIRY_INDEX))
        for fu in cf.as_completed(futs):
            r = fu.result()
            if r is not None:
                results.append(r)

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame([r.__dict__ for r in results])
    # 필터링 diff_pct <= DIFF_THRESHOLD
    df = df[df['diff_pct'] <= DIFF_THRESHOLD].copy()

    if df.empty:
        return df

    # market_cap & volume 수집
    market_caps = []
    volumes = []
    for symbol in df['symbol']:
        try:
            tk = yf.Ticker(symbol)
            info = tk.info
            # market cap
            mc = info.get('marketCap', 0) if isinstance(info, dict) else 0
            if mc is None:
                mc = 0
            market_caps.append(mc)
            # volume (우선순위: fast_info → info → history)
            vol = None
            try:
                fi = getattr(tk, 'fast_info', None)
                if fi and getattr(fi, 'last_volume', None):
                    vol = int(fi.last_volume)
            except Exception:
                pass
            if vol is None:
                try:
                    if isinstance(info, dict) and info.get('volume') is not None:
                        vol = int(info.get('volume'))
                except Exception:
                    pass
            if vol is None:
                try:
                    h = tk.history(period="2d", interval="1d", auto_adjust=False)
                    if not h.empty and 'Volume' in h.columns:
                        vol = int(h['Volume'].iloc[-1])
                except Exception:
                    pass
            if vol is None:
                vol = 0
            volumes.append(vol)
        except Exception:
            market_caps.append(0)
            volumes.append(0)
        # 요청 부담 완화
        time.sleep(0.01)
    df['market_cap'] = market_caps
    df['volume'] = volumes
    df['atv_usd'] = df['price'] * df['volume']

    # 거래대금(atv_usd) 내림차순 정렬
    df = df.sort_values('atv_usd', ascending=False)

    return df.reset_index(drop=True)


def print_top_nasdaq(df: pd.DataFrame, top_n: int = TOP_N):
    if df.empty:
        print("⚠️ 결과가 비었습니다. 네트워크/요청 제한 또는 옵션 데이터 부재 가능.")
        return

    use_col = ['symbol', 'price', 'max_pain', 'diff_abs', 'diff_pct', 'expiry', 'market_cap', 'call_walls', 'put_walls', 'atm_iv', 'gex', 'atv_usd']
    df2 = df[use_col].head(top_n).copy()
    # 보기 좋게 반올림
    df2['price'] = df2['price'].round(4)
    df2['max_pain'] = df2['max_pain'].round(4)
    df2['diff_abs'] = df2['diff_abs'].round(4)
    df2['diff_pct'] = (df2['diff_pct'] * 100).round(3)
    df2['market_cap'] = df2['market_cap'].map('{:,.0f}'.format)
    df2['atv_usd'] = df2['atv_usd'].map('{:,.0f}'.format)
    df2['atm_iv'] = (df2['atm_iv'] * 100).round(2)  # %로 표시
    df2['gex'] = df2['gex'].round(0).map('{:,.0f}'.format)

    print(f"\n[NASDAQ] 맥스페인 근접도 상위 {top_n} 종목 (diff_pct ≤ {DIFF_THRESHOLD*100:.2f}%, 정렬기준: 거래대금 내림차순)")
    print(df2.to_string(index=False, justify='left'))


if __name__ == "__main__":
    try:
        # NASDAQ
        df_nasdaq = scan_nasdaq_and_rank()
        print_top_nasdaq(df_nasdaq, TOP_N)

    except KeyboardInterrupt:
        print("\n⏹️ 중단됨 (사용자)")
    except Exception as e:
        print(f"❌ 치명적 오류: {e}")