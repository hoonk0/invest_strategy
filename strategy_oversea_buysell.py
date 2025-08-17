#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path

# -*- coding: utf-8 -*-
"""
미국주식 프리장/본장 자동매수 스크립트 (KIS 실계좌)
- kis_devlp.yaml 설정을 읽어 OAuth 토큰 발급
- 프리장(04:00~09:30 ET), 본장(09:30~16:00 ET) 시간 체크
- 조건 충족 시 해외주식(미국) 시장가 매수 주문 실행

안전장치
- 기본은 DRY-RUN(모의실행)으로 동작합니다. 실제 주문은 --live 플래그를 넣어야 합니다.
- TR_ID 값은 증권사 문서에 따라 다를 수 있습니다. 아래 상수의 기본값은 None이며,
  KIS 문서의 실계좌 해외주식 ‘시장가 매수’ TR_ID로 교체하십시오.

사용 예시
  $ python strategy_oversea_maxpain_one_real_test.py --symbol AAPL --qty 2 --when pre --live
  $ python strategy_oversea_maxpain_one_real_test.py --symbol NVDA --qty 1 --when regular

주의
- 키/시크릿 등 민감정보는 kis_devlp.yaml 에서만 읽습니다. 코드에 절대 하드코딩하지 마세요.
- 미국 시장 휴장일은 별도 반영하지 않았습니다(기본 영업일/시간 기반). 필요시 보강하세요.
"""

# ===== 기본값 (CLI 미입력 시 사용) =====
DEFAULT_TICKER = "MSTR"
DEFAULT_MAX_PAIN = 390.0
DEFAULT_PUT_OI_TOP = 370.0
DEFAULT_CALL_OI_TOP = 390.0
DEFAULT_CAPITAL = 10000.0
DEFAULT_PARTS = 5
DEFAULT_EXCHANGE = "NASD"   # 해외거래소 코드 (미국 나스닥)
RATE_LIMIT_SEC = 1.2  # KIS API: 초당 거래건수 제한 대응(모의/실서버 공통 권장)

# --- 전략 요약(현재 파일) ---
# 입력값(필수/선택):
#   - symbol(티커), qty 또는 budget, lastprice(지정가), when(프리장/본장/즉시), exchange(거래소 코드)
#   - yaml에서 TR_ID/계좌/키를 읽음. 프리장/애프터는 ORD_SVR_DVSN_CD=1, 본장은 0 자동 적용
# 매수 트리거:
#   - 프로그램 실행 시점에 지정가(OVRS_ORD_UNPR=lastprice)로 매수 요청 전송
#   - 실패 시 거래소코드(NAS↔NASD) / 심볼(AAPL↔AAPL.O) 교차 재시도 + 해시키 적용
# 매도 트리거:
#   - **현재 파일에는 매도 로직 미구현**(원하면 장마감/목표가 도달 시 전량매도 로직 추가 가능)
# 참고(옵션/맥스페인 전략과의 관계):
#   - 아래 DEFAULT_MAX_PAIN/PUT/CALL/PARTS 값은 옵션 전략용 참고값으로만 정의됨.
#   - 이 파일은 실매수 1회 전송에 집중. 옵션 분할매수 루프는 포함하지 않음(요청 시 이식 가능).

# ==============================
# ✅ 사용자 설정 구역 (위에서부터 수정)
# ==============================
# 매수할 종목(티커)
USER_SYMBOL: str = DEFAULT_TICKER

# 거래소 코드 (KIS 기준)
#  - NAS=나스닥, NYS=뉴욕, ASE=아멕스  (필요 시 문서 확인)
USER_OVRS_EXCG_CD: str = DEFAULT_EXCHANGE

# 언제 매수할지: "pre"(프리장), "regular"(본장), "now"(즉시)
USER_WHEN: str = "now"

# 실제 주문 여부: False=DRY-RUN(모의), True=실주문
USER_LIVE: bool = True

# 수량 방식 선택
#  - True  -> 예산(달러)으로 계산해서 수량 산정
#  - False -> 고정 수량으로 주문
USER_USE_CASH_BUDGET: bool = False

# 예산(USD). 예: 1000달러 예산으로 수량 계산
USER_CASH_BUDGET_USD: float = DEFAULT_CAPITAL

# 예상 체결가(달러) 힌트 — 가격 API를 쓰지 않을 때 직접 지정
#  - None 이면 가격 API 시도(아래 QUOTE_TR_ID 가 필요)
#  - 숫자 지정 시 해당 값을 기준으로 수량 계산
USER_LAST_PRICE_HINT: float | None = 220.0  # 기본값(AAPL 예시). 다른 심볼이면 수정 권장
TOKEN_CACHE_PATH = Path(__file__).with_suffix('.token.json')
TOKEN_SAFETY_MARGIN_SEC = 60
# -----------------------------
# 액세스 토큰 캐시 유틸
# -----------------------------

def load_cached_token() -> Optional[dict]:
    try:
        if TOKEN_CACHE_PATH.exists():
            return json.loads(TOKEN_CACHE_PATH.read_text(encoding='utf-8'))
    except Exception:
        pass
    return None


def save_cached_token(token: str, expires_in: int) -> None:
    try:
        exp_ts = int(time.time()) + max(0, int(expires_in) - TOKEN_SAFETY_MARGIN_SEC)
        payload = {"access_token": token, "exp": exp_ts}
        TOKEN_CACHE_PATH.write_text(json.dumps(payload), encoding='utf-8')
    except Exception:
        pass

# 고정 수량 주문 시 사용
USER_FIXED_QTY: int = 1

# 슬리피지 여유율(%) — 예산 기반 수량 계산 시 살짝 보수적으로 계산
USER_SLIPPAGE_PCT: float = 0.2

# 최소/최대 수량 가드
USER_MIN_SHARES: int = 1
USER_MAX_SHARES: int = 9999

# (선택) 주문 전 출력만 확인하고 싶으면 True
USER_PRINT_PLAN_ONLY: bool = False
# -----------------------------
# 실매매 유틸: 레이트리밋/안전가/재시도
# -----------------------------

def _is_rate_limit_error_text(s: str) -> bool:
    s = str(s or "")
    return ("EGW00201" in s) or ("초당 거래건수" in s) or ("rate" in s.lower() and "limit" in s.lower())

def _safe_limit_price_buy(last_px: float, order_px: float) -> float:
    """매수 지정가를 소폭 상향(0.1%)하여 체결 보조"""
    try:
        base = max(float(last_px or 0), float(order_px or 0))
        return round(base * 1.001, 2)
    except Exception:
        return float(order_px)

def submit_buy_with_retry(cfg: "KisConfig", token: str, symbol: str, qty: int, live: bool,
                          ord_svr_dvsn_cd: str, order_price: float,
                          last_px_hint: float | None = None, max_attempts: int = 3):
    """
    place_overseas_market_buy를 감싸 레이트리밋 및 일시 오류에 대한 재시도 수행.
    필요 시 안전가(소폭 상향) 적용.
    """
    attempts = 0
    price_to_use = float(order_price)
    if last_px_hint is not None:
        price_to_use = _safe_limit_price_buy(last_px_hint, price_to_use)

    while True:
        attempts += 1
        try:
            resp = place_overseas_market_buy(cfg, token, symbol, qty, live, ord_svr_dvsn_cd, price_to_use)
            time.sleep(RATE_LIMIT_SEC)
            return resp
        except Exception as e:
            msg = str(e)
            if _is_rate_limit_error_text(msg) and attempts < max_attempts:
                time.sleep(RATE_LIMIT_SEC * 1.5)
                continue
            # 기타 오류는 즉시 전달
            raise
# ==============================

import argparse
import dataclasses
import datetime as dt
import json
import ssl
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any


import pytz
import yaml
import requests


# -----------------------------
# 설정 상수 (필수: TR_ID 확인)
# -----------------------------
# ⚠️ KIS 해외주식 "시장가 매수" 실계좌 TR_ID 를 문서에서 확인해 입력하세요.
# 예시는 비워둡니다. (잘못된 TR_ID 사용시 주문 실패)
ORDER_TR_ID_BUY_MARKET_REAL: Optional[str] = None  # 예: "TTTT1002U" (문서 확인 필요)

# 토큰 발급 엔드포인트 (실계좌)
TOKEN_URL = "https://openapi.koreainvestment.com:9443/oauth2/tokenP"
# 해외주식 주문 엔드포인트 (실계좌)
ORDER_URL = "https://openapi.koreainvestment.com:9443/uapi/overseas-stock/v1/trading/order"
HASHKEY_URL = "https://openapi.koreainvestment.com:9443/uapi/hashkey"

# 해외시세(가격) 조회 엔드포인트 & TR_ID (선택)
# 가격으로 수량을 계산하려면 QUOTE_TR_ID_OVERSEAS_PRICE 값을 KIS 문서에 따라 채우세요.
# (모르면 USER_LAST_PRICE_HINT 에 가격을 직접 넣어도 됩니다.)
QUOTE_URL = "https://openapi.koreainvestment.com:9443/uapi/overseas-price/v1/quotations/price"
QUOTE_TR_ID_OVERSEAS_PRICE: Optional[str] = None  # 예: "HHDFS76200200" 등 문서 확인 필요

# 미국 동부시간(뉴욕)
TZ_ET = pytz.timezone("US/Eastern")

# -----------------------------
# 데이터 구조
# -----------------------------
@dataclasses.dataclass
class KisConfig:
    env: str
    appkey: str
    appsecret: str
    account_no_8: str
    account_prod_2: str
    user_agent: str
    order_tr_id_buy_market_real: Optional[str] = None
    order_tr_id_us_extended: Optional[str] = None

    @staticmethod
    def from_yaml(path: Path) -> "KisConfig":
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        # 파일 키 매핑
        env = data.get("env", "prod")
        appkey = data.get("my_app") if env == "prod" else data.get("paper_app")
        appsecret = data.get("my_sec") if env == "prod" else data.get("paper_sec")
        account_no_8 = data.get("my_acct_stock") if env == "prod" else data.get("my_paper_stock")
        prod_2 = data.get("my_prod", "01")
        user_agent = data.get("my_agent", "Mozilla/5.0")
        order_tr_id = data.get("order_tr_id_buy_market_real")
        order_tr_id_ext = data.get("order_tr_id_us_extended")
        if not all([appkey, appsecret, account_no_8, prod_2]):
            raise ValueError("kis_devlp.yaml에서 필수 키가 누락되었습니다.")
        return KisConfig(env, appkey, appsecret, account_no_8, prod_2, user_agent, order_tr_id, order_tr_id_ext)

# -----------------------------
# 유틸: 세션/시간
# -----------------------------

def now_et() -> dt.datetime:
    return dt.datetime.now(tz=TZ_ET)


def is_weekday(d: dt.datetime) -> bool:
    return d.weekday() < 5  # 0=Mon ... 4=Fri


def in_premarket(d: dt.datetime) -> bool:
    """프리장 04:00~09:30 ET"""
    start = d.replace(hour=4, minute=0, second=0, microsecond=0)
    end = d.replace(hour=9, minute=30, second=0, microsecond=0)
    return start <= d < end


def in_regular(d: dt.datetime) -> bool:
    """본장 09:30~16:00 ET"""
    start = d.replace(hour=9, minute=30, second=0, microsecond=0)
    end = d.replace(hour=16, minute=0, second=0, microsecond=0)
    return start <= d < end

# -----------------------------
# 수량 계산/시세 유틸
# -----------------------------

def compute_qty_from_budget(price: float, budget_usd: float, slippage_pct: float,
                            min_shares: int, max_shares: int) -> int:
    if price <= 0:
        raise ValueError("가격이 0 이하입니다.")
    # 슬리피지(여유율) 반영
    eff_price = price * (1.0 + slippage_pct / 100.0)
    qty = int(budget_usd // eff_price)
    qty = max(min_shares, min(qty, max_shares))
    return qty


def try_fetch_last_price(cfg: "KisConfig", access_token: str, symbol: str, exchg_cd: str) -> float | None:
    """KIS 해외시세 API로 종가/현재가를 받아오는 예시. TR_ID가 없으면 None 반환."""
    if not QUOTE_TR_ID_OVERSEAS_PRICE:
        return None
    headers = {
        "content-type": "application/json",
        "authorization": f"Bearer {access_token}",
        "appkey": cfg.appkey,
        "appsecret": cfg.appsecret,
        "tr_id": QUOTE_TR_ID_OVERSEAS_PRICE,
        "User-Agent": cfg.user_agent,
    }
    params = {
        "AUTH": "",            # 문서에 따라 필요/불필요
        "EXCD": exchg_cd,       # 거래소 코드 (예: NAS)
        "SYMB": symbol,         # 심볼/코드
    }
    try:
        res = requests.get(QUOTE_URL, headers=headers, params=params, timeout=10)
        if res.status_code != 200:
            return None
        data = res.json()
        # 문서 포맷에 맞는 필드에서 가격을 추출하세요. 아래는 예시 키 이름입니다.
        # 우선순위: 현재가/체결가 -> 전일종가
        for key in ("last", "trade_price", "now_prc", "clos_prc"):
            val = data.get(key) or data.get("output", {}).get(key)
            if val is not None:
                try:
                    return float(val)
                except Exception:
                    pass
        return None
    except Exception:
        return None

# -----------------------------
# KIS API: 해시키 발급
# -----------------------------

def fetch_hashkey(cfg: "KisConfig", access_token: str, body: dict) -> str:
    headers = {
        "content-type": "application/json",
        "authorization": f"Bearer {access_token}",
        "appkey": cfg.appkey,
        "appsecret": cfg.appsecret,
        "User-Agent": cfg.user_agent,
    }
    res = requests.post(HASHKEY_URL, headers=headers, data=json.dumps(body), timeout=10)
    if res.status_code != 200:
        raise RuntimeError(f"해시키 발급 실패({res.status_code}): {res.text}")
    j = res.json()
    # 응답 키 이름은 문서에 따라 'HASH' 또는 'hash'일 수 있음
    return j.get("HASH") or j.get("hash")

# -----------------------------
# KIS API: 토큰/주문
# -----------------------------

def fetch_access_token(cfg: KisConfig) -> str:
    # 1) 캐시 사용
    cached = load_cached_token()
    now_ts = int(time.time())
    if cached and cached.get('access_token') and int(cached.get('exp', 0)) > now_ts:
        return cached['access_token']

    headers = {
        "content-type": "application/json",
        "User-Agent": cfg.user_agent,
    }
    body = {"grant_type": "client_credentials", "appkey": cfg.appkey, "appsecret": cfg.appsecret}

    # 2) 요청 + EGW00133 백오프 1회 재시도
    for attempt in range(2):
        res = requests.post(TOKEN_URL, headers=headers, data=json.dumps(body), timeout=15)
        if res.status_code == 200:
            j = res.json()
            access_token = j.get("access_token")
            expires_in = j.get("expires_in", 0)
            if not access_token:
                raise RuntimeError(f"토큰 응답 파싱 실패: {j}")
            save_cached_token(access_token, int(expires_in) if isinstance(expires_in, int) else 0)
            return access_token
        else:
            try:
                j = res.json()
            except Exception:
                j = {}
            if str(j.get("error_code")) == "EGW00133" and attempt == 0:
                time.sleep(65)
                continue
            raise RuntimeError(f"토큰 발급 실패({res.status_code}): {res.text}")


def place_overseas_market_buy(
    cfg: KisConfig,
    access_token: str,
    symbol: str,
    qty: int,
    live: bool,
    ord_svr_dvsn_cd: str,  # "0"=본장, "1"=확장세션(프리/애프터)
    order_price: float,
) -> Dict[str, Any]:
    """
    해외주식(미국) 시장가 매수 주문. 실계좌 기준.
    - ⚠️ ORDER_TR_ID_BUY_MARKET_REAL 값을 반드시 채워야 함
    - KIS 문서 기준으로 body/params가 다를 수 있으니 필요시 조정하세요.
    """
    if not live:
        return {"dry_run": True, "symbol": symbol, "qty": qty, "message": "DRY-RUN: 주문 미전송"}

    tr_id = None
    if ord_svr_dvsn_cd == "1":
        tr_id = cfg.order_tr_id_us_extended or cfg.order_tr_id_buy_market_real or ORDER_TR_ID_BUY_MARKET_REAL
    else:
        tr_id = cfg.order_tr_id_buy_market_real or ORDER_TR_ID_BUY_MARKET_REAL
    if not tr_id:
        raise RuntimeError(
            "실계좌 TR_ID가 없습니다. kis_devlp.yaml의 order_tr_id_buy_market_real(본장) 또는 order_tr_id_us_extended(확장세션)를 설정하세요."
        )

    headers = {
        "content-type": "application/json",
        "authorization": f"Bearer {access_token}",
        "appkey": cfg.appkey,
        "appsecret": cfg.appsecret,
        "tr_id": tr_id,
        "User-Agent": cfg.user_agent,
        "custtype": "P",  # P=개인, B=법인
    }

    excg_cd = USER_OVRS_EXCG_CD

    # ⚠️ 아래 요청 포맷은 증권사 문서에 따라 다를 수 있습니다. 대표적인 예시 형태입니다.
    # acct_no: 계좌번호(앞 8자리 + 뒤 2자리)
    acct_no = f"{cfg.account_no_8}{cfg.account_prod_2}"

    body = {
        "CANO": cfg.account_no_8,       # 계좌 앞 8자리
        "ACNT_PRDT_CD": cfg.account_prod_2,  # 계좌 뒤 2자리
        "OVRS_EXCG_CD": excg_cd,        # 예: NASD/NYSE/AMEX
        "PDNO": symbol,                 # 심볼(티커)
        "ORD_DVSN": "00",             # 00=지정가 (해외는 시장가 미지원, LOO/LOC는 32/34)
        "ORD_QTY": str(qty),
        "ORD_SVR_DVSN_CD": ord_svr_dvsn_cd,  # 0=본장, 1=확장세션
        "OVRS_ORD_UNPR": str(order_price),  # 지정가 가격 필수
        "SLL_BUY_DVSN_CD": "02",      # 02=매수, 01=매도
        "ORD_CNDT_CD": "0",          # 0=일반
    }

    # 재시도 후보 생성 (거래소/심볼 표기 교차 시도)
    excg_candidates = []
    if excg_cd not in ("NAS", "NASD", "NYS", "NYSE", "ASE", "AMEX"):
        excg_candidates.append(excg_cd)
    else:
        excg_candidates.append(excg_cd)
        if excg_cd == "NAS":
            excg_candidates.append("NASD")
        elif excg_cd == "NASD":
            excg_candidates.append("NAS")
        elif excg_cd == "NYS":
            excg_candidates.append("NYSE")
        elif excg_cd == "NYSE":
            excg_candidates.append("NYS")
        elif excg_cd == "ASE":
            excg_candidates.append("AMEX")
        elif excg_cd == "AMEX":
            excg_candidates.append("ASE")

    sym_candidates = [symbol]
    if "." not in symbol:
        sym_candidates.append(f"{symbol}.O")  # 나스닥 접미사 표기 시도
    else:
        base = symbol.split(".")[0]
        if base and base != symbol:
            sym_candidates.append(base)

    last_error = None
    for excg_try in excg_candidates:
        for sym_try in sym_candidates:
            body_try = dict(body)
            body_try["OVRS_EXCG_CD"] = excg_try
            body_try["PDNO"] = sym_try
            try:
                hk = fetch_hashkey(cfg, access_token, body_try)
                headers["hashkey"] = hk
                res = requests.post(ORDER_URL, headers=headers, data=json.dumps(body_try), timeout=15)
                if res.status_code != 200:
                    last_error = f"주문 실패({res.status_code}): {res.text}"
                    continue
                j = res.json()
                # 성공 판단: 일반적으로 rt_cd == "0" 또는 output이 존재
                if str(j.get("rt_cd")) == "0" or j.get("output"):
                    return j
                msg_cd = str(j.get("msg_cd"))
                msg1 = j.get("msg1", "")
                if msg_cd in ("APBK0656", "IGW00036"):
                    last_error = f"{msg_cd}: {msg1} (excg={excg_try}, sym={sym_try})"
                    continue
                # 다른 오류면 즉시 반환(상세 메시지 확인)
                return j
            except Exception as e:
                last_error = str(e)
                continue

    # 모든 조합 실패 시 에러
    raise RuntimeError(last_error or "해외주문 실패: 원인 불명")

# -----------------------------
# 메인 로직
# -----------------------------

def wait_until_session(when: str, poll_sec: int = 10) -> None:
    """원하는 세션(프리장/본장)까지 대기. 휴장일은 고려하지 않음."""
    assert when in ("pre", "regular"), "when 은 'pre' 또는 'regular' 만 허용"
    print(f"[WAIT] 세션 대기 시작 — 목표: {when}")
    while True:
        t = now_et()
        if not is_weekday(t):
            print("[WAIT] 주말입니다. 60분 대기...")
            time.sleep(3600)
            continue
        if when == "pre" and in_premarket(t):
            print(f"[WAIT] 프리장 시작 감지(ET {t.strftime('%Y-%m-%d %H:%M:%S')})")
            return
        if when == "regular" and in_regular(t):
            print(f"[WAIT] 본장 시작 감지(ET {t.strftime('%Y-%m-%d %H:%M:%S')})")
            return
        time.sleep(poll_sec)


def main():
    parser = argparse.ArgumentParser(description="KIS 미국주식 자동매수")
    parser.add_argument("--symbol", default=USER_SYMBOL, help="심볼 예: AAPL, NVDA, TSLA")
    parser.add_argument("--qty", type=int, default=None, help="매수 수량(미입력 시 예산 기반 계산)")
    parser.add_argument("--when", choices=["pre", "regular", "now"], default=USER_WHEN,
                        help="pre=프리장, regular=본장, now=즉시")
    parser.add_argument("--live", action="store_true" if not USER_LIVE else "store_false",
                        help=("실주문 전송(기본: DRY-RUN)" if not USER_LIVE else "DRY-RUN(기본: 실주문)"))
    parser.add_argument("--budget", type=float, default=USER_CASH_BUDGET_USD,
                        help="예산(USD). qty 미지정 시 예산으로 수량 계산")
    parser.add_argument("--use_budget", action="store_true", default=USER_USE_CASH_BUDGET,
                        help="예산 기반 모드 (기본값은 사용자 설정 구역 참조)")
    parser.add_argument("--lastprice", type=float, default=USER_LAST_PRICE_HINT,
                        help="예상 체결가(달러) 힌트. 가격 API 미사용 시 지정")
    parser.add_argument("--fixed_qty", type=int, default=USER_FIXED_QTY, help="고정 수량 모드에서 사용할 수량")
    parser.add_argument("--excg", default=USER_OVRS_EXCG_CD, help="거래소 코드(NAS/NYS/ASE)")
    parser.add_argument("--print_plan_only", action="store_true", default=USER_PRINT_PLAN_ONLY,
                        help="주문 전 계획만 출력하고 종료")
    parser.add_argument("--yaml", default="/Users/kyunghoon/Desktop/python/stock_program/invest_strategy/kis_devlp.yaml", help="설정 파일 경로")
    args = parser.parse_args()

    cfg = KisConfig.from_yaml(Path(args.yaml))

    # 인자/사용자설정 병합
    symbol = args.symbol
    exchg = args.excg
    when = args.when
    live = (args.live if (USER_LIVE is False) else True) if args.live else USER_LIVE

    use_budget = args.use_budget
    budget = float(args.budget)
    lastprice = args.lastprice
    fixed_qty = int(args.fixed_qty)

    # 런타임 거래소 코드를 전역 기본값으로 반영 (주문 함수에서 사용)
    globals()["USER_OVRS_EXCG_CD"] = exchg

    # 세션 대기
    if when in ("pre", "regular"):
        wait_until_session(when)

    # 세션 코드 계산: 0=본장, 1=확장세션(프리/애프터)
    if when == "regular":
        ord_svr = "0"
    elif when == "pre":
        ord_svr = "1"
    else:  # when == "now"
        t_now = now_et()
        ord_svr = "1" if (in_premarket(t_now) or (not in_regular(t_now))) else "0"

    # 토큰 발급
    print("[AUTH] 액세스 토큰 발급 시도...")
    token = fetch_access_token(cfg)
    print("[AUTH] 액세스 토큰 발급 완료")

    # 수량 계산
    qty = args.qty
    if qty is None:
        if use_budget:
            # 가격 힌트가 없으면 API 시도
            if lastprice is None:
                lastprice = try_fetch_last_price(cfg, token, symbol, exchg)
            if lastprice is None:
                raise RuntimeError("수량 계산을 위해 가격이 필요합니다. USER_LAST_PRICE_HINT 또는 --lastprice 를 지정하거나, QUOTE_TR_ID_OVERSEAS_PRICE 를 설정하세요.")
            qty = compute_qty_from_budget(lastprice, budget, USER_SLIPPAGE_PCT, USER_MIN_SHARES, USER_MAX_SHARES)
        else:
            qty = fixed_qty

    # 주문 가격 산정(해외 지정가 필요)
    order_price = lastprice
    if order_price is None:
        order_price = try_fetch_last_price(cfg, token, symbol, exchg)
    if order_price is None:
        raise RuntimeError("해외주문은 지정가(ORD_DVSN=00)로 가격이 필요합니다. --lastprice 로 가격을 지정하거나 시세 TR_ID를 설정하세요.")

    # 계획 출력
    print("[PLAN]",
          json.dumps({
              "symbol": symbol,
              "exchange": exchg,
              "when": when,
              "live": live,
              "use_budget": use_budget,
              "budget_usd": budget if use_budget else None,
              "last_price": order_price,
              "qty": qty,
          }, ensure_ascii=False, indent=2))

    if args.print_plan_only:
        print("[EXIT] 계획만 출력하도록 지정되어 종료합니다.")
        return

    # 주문 전송 (재시도/안전가 적용)
    print(f"[ORDER] {symbol} {qty}주 **지정가** 매수 시도 — live={live}")
    try:
        result = submit_buy_with_retry(
            cfg, token, symbol, qty, live, ord_svr, float(order_price),
            last_px_hint=float(order_price)  # 힌트가 있으면 소폭 상향
        )
        print("[RESULT]", json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"[ERROR] 주문 실패: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # SSL 검증 문제 방지(필요시). 기업망/프록시 환경이면 조정하세요.
    try:
        ssl._create_default_https_context = ssl._create_unverified_context  # noqa
    except Exception:
        pass
    main()
