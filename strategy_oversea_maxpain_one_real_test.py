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
import yaml
from pathlib import Path

# ===== 기본값 (CLI 미입력 시 사용) =====
DEFAULT_TICKER = "IONQ"
DEFAULT_MAX_PAIN = 42.00
DEFAULT_PUT_OI_TOP = 40.0
DEFAULT_CALL_OI_TOP = 45.0
DEFAULT_CAPITAL = 300.0
DEFAULT_PARTS = 5
DEFAULT_EXCHANGE = "NASD"   # 해외거래소 코드 (미국 나스닥)
RATE_LIMIT_SEC = 1.2  # KIS API: 초당 거래건수 제한 대응(모의/실서버 공통 권장)

# KIS 실계좌 설정 yaml 경로
DEFAULT_KIS_YAML = "/Users/kyunghoon/Desktop/python/stock_program/invest_strategy/kis_devlp.yaml"


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


def is_us_premarket_open_now(dt=None):
    """US premarket: 04:00–09:30 (NY local)."""
    dt = dt or ny_now()
    w = dt.weekday()
    if w >= 5:
        return False
    t = dt.time()
    return (t >= datetime(dt.year, dt.month, dt.day, 4, 0).time() and
            t <  datetime(dt.year, dt.month, dt.day, 9, 30).time())




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


# ===== KIS 실계좌 설정/토큰/해시키/주문 (REST) =====

TOKEN_URL = "https://openapi.koreainvestment.com:9443/oauth2/tokenP"
ORDER_URL = "https://openapi.koreainvestment.com:9443/uapi/overseas-stock/v1/trading/order"
HASHKEY_URL = "https://openapi.koreainvestment.com:9443/uapi/hashkey"

class KisConfig:
    def __init__(self, appkey, appsecret, cano8, prdt_cd2, user_agent,
                 tr_buy_limit=None, tr_buy_mkt=None, tr_sell_limit=None, tr_sell_mkt=None):
        self.appkey = appkey
        self.appsecret = appsecret
        self.cano8 = cano8
        self.prdt_cd2 = prdt_cd2
        self.user_agent = user_agent
        self.tr_buy_limit = tr_buy_limit
        self.tr_buy_mkt = tr_buy_mkt
        self.tr_sell_limit = tr_sell_limit
        self.tr_sell_mkt = tr_sell_mkt

    @staticmethod
    def from_yaml(path: str | Path):
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        env = data.get("env", "prod")
        appkey   = data.get("my_app") if env == "prod" else data.get("paper_app")
        appsec   = data.get("my_sec") if env == "prod" else data.get("paper_sec")
        cano8    = data.get("my_acct_stock") if env == "prod" else data.get("my_paper_stock")
        prdt_cd2 = data.get("my_prod", "01")
        ua       = data.get("my_agent", "Mozilla/5.0")
        # TR_ID 설정 (없으면 주문 시 에러 안내)
        tr_buy_limit  = data.get("order_tr_id_buy_limit_real")
        tr_buy_mkt    = data.get("order_tr_id_buy_market_real")
        tr_sell_limit = data.get("order_tr_id_sell_limit_real")
        tr_sell_mkt   = data.get("order_tr_id_sell_market_real")
        if not all([appkey, appsec, cano8, prdt_cd2]):
            raise RuntimeError("kis_devlp.yaml에서 appkey/appsecret/계좌/상품코드 값을 확인하세요.")
        return KisConfig(appkey, appsec, cano8, prdt_cd2, ua, tr_buy_limit, tr_buy_mkt, tr_sell_limit, tr_sell_mkt)

# 토큰 캐시 (1분 제한 대응)
_TOKEN_CACHE_PATH = Path(__file__).with_suffix(".token.json")
_TOKEN_SAFE_MARGIN = 60

def _load_token_cache():
    try:
        if _TOKEN_CACHE_PATH.exists():
            return json.loads(_TOKEN_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return None

def _save_token_cache(token: str, expires_in: int):
    try:
        exp = int(time.time()) + max(0, int(expires_in) - _TOKEN_SAFE_MARGIN)
        _TOKEN_CACHE_PATH.write_text(json.dumps({"access_token": token, "exp": exp}), encoding="utf-8")
    except Exception:
        pass

def kis_fetch_token(cfg: KisConfig) -> str:
    cache = _load_token_cache()
    now = int(time.time())
    if cache and cache.get("access_token") and int(cache.get("exp", 0)) > now:
        return cache["access_token"]
    headers = {"content-type": "application/json", "User-Agent": cfg.user_agent}
    body = {"grant_type": "client_credentials", "appkey": cfg.appkey, "appsecret": cfg.appsecret}
    for attempt in range(2):
        r = requests.post(TOKEN_URL, headers=headers, data=json.dumps(body), timeout=15)
        if r.status_code == 200:
            j = r.json()
            tok = j.get("access_token")
            exp = j.get("expires_in", 0)
            if not tok:
                raise RuntimeError(f"토큰 응답 파싱 실패: {j}")
            _save_token_cache(tok, int(exp) if isinstance(exp, int) else 0)
            return tok
        else:
            try:
                j = r.json()
            except Exception:
                j = {}
            if str(j.get("error_code")) == "EGW00133" and attempt == 0:
                time.sleep(65)
                continue
            raise RuntimeError(f"토큰 발급 실패({r.status_code}): {r.text}")

def kis_hashkey(cfg: KisConfig, access_token: str, body: dict) -> str:
    headers = {
        "content-type": "application/json",
        "authorization": f"Bearer {access_token}",
        "appkey": cfg.appkey,
        "appsecret": cfg.appsecret,
        "User-Agent": cfg.user_agent,
    }
    r = requests.post(HASHKEY_URL, headers=headers, data=json.dumps(body), timeout=10)
    if r.status_code != 200:
        raise RuntimeError(f"해시키 실패({r.status_code}): {r.text}")
    j = r.json()
    return j.get("HASH") or j.get("hash")

def _choose_tr_id(cfg: KisConfig, side: str, ord_dvsn: str) -> str:
    side = side.upper()  # BUY/SELL
    if side == "BUY":
        if ord_dvsn == "00":
            tr = cfg.tr_buy_limit
            need = "order_tr_id_buy_limit_real"
        else:
            tr = cfg.tr_buy_mkt
            need = "order_tr_id_buy_market_real"
    else:
        if ord_dvsn == "00":
            tr = cfg.tr_sell_limit
            need = "order_tr_id_sell_limit_real"
        else:
            tr = cfg.tr_sell_mkt
            need = "order_tr_id_sell_market_real"
    if not tr:
        raise RuntimeError(f"kis_devlp.yaml에 {need} 값을 추가하세요.")
    return tr

def kis_overseas_order(cfg: KisConfig, access_token: str, *,
                       side: str,  # "BUY" / "SELL"
                       order_type: str,  # "market" / "limit"
                       symbol: str, exchange: str,
                       qty: int, price: Optional[float],
                       ord_svr_dvsn_cd: str = "0") -> dict:
    """
    해외주식 주문 (실계좌).
    - side: BUY/SELL
    - order_type: market/limit
    - exchange: NAS/NASD/NYS/NYSE/ASE/AMEX 중 하나
    - price: limit일 때 필수, market일 때 0 처리
    """
    ord_dvsn = "01" if order_type == "market" else "00"
    sll_buy = "02" if side.upper() == "BUY" else "01"
    tr_id = _choose_tr_id(cfg, side.upper(), ord_dvsn)

    headers = {
        "content-type": "application/json",
        "authorization": f"Bearer {access_token}",
        "appkey": cfg.appkey,
        "appsecret": cfg.appsecret,
        "tr_id": tr_id,
        "custtype": "P",
        "User-Agent": cfg.user_agent,
    }

    body_base = {
        "CANO": cfg.cano8,
        "ACNT_PRDT_CD": cfg.prdt_cd2,
        "OVRS_EXCG_CD": exchange,
        "PDNO": symbol,
        "SLL_BUY_DVSN_CD": sll_buy,     # 01=매도, 02=매수
        "ORD_DVSN": ord_dvsn,           # 00=지정가, 01=시장가
        "ORD_QTY": str(int(qty)),
        # === 가격 필드: 해외 주문은 일부 TR에서 ORD_UNPR 사용 ===
        # 시장가(01)일 때 0, 지정가(00)일 때 가격 문자열
        "ORD_UNPR": "0" if ord_dvsn == "01" else str(price),
        # 호환 목적: 일부 환경에서 OVRS_ORD_UNPR 키를 요구
        "OVRS_ORD_UNPR": "0" if ord_dvsn == "01" else str(price),
        "ORD_SVR_DVSN_CD": ord_svr_dvsn_cd,  # 0=본장, 1=확장세션(프리/애프터)
        "ORD_CNDT_CD": "0",
        "KRW_YN": "Y",
    }

    # 거래소/심볼 교차 재시도 (확장):
    # - 거래소: [요청값, 쌍(pair), NYS/NYSE, NAS/NASD, AMEX/ASE] 순으로 유니크하게 시도
    # - 티커: base, base+권장접미사(.O/.N/.A) 순으로 시도 (요청 심볼 원형도 포함)
    valid_exchanges = ("NAS", "NASD", "NYS", "NYSE", "ASE", "AMEX")
    pair_map = {"NAS": "NASD", "NASD": "NAS", "NYS": "NYSE", "NYSE": "NYS", "ASE": "AMEX", "AMEX": "ASE"}

    # 1) 거래소 후보 구성
    excg_candidates: list[str] = []
    if exchange and exchange not in excg_candidates:
        excg_candidates.append(exchange)
    if exchange in pair_map and pair_map[exchange] not in excg_candidates:
        excg_candidates.append(pair_map[exchange])
    # 우선순위로 주요 미국 거래소 전체를 추가 (중복 제거)
    for ex in ("NYS", "NYSE", "NAS", "NASD", "AMEX", "ASE"):
        if ex not in excg_candidates:
            excg_candidates.append(ex)

    # 2) 심볼 기반 / 접미사 정책
    base_sym = symbol.split(".")[0].upper() if symbol else ""

    def preferred_suffixes(excg: str) -> list[str]:
        if excg in ("NAS", "NASD"):
            return [".O"]
        if excg in ("NYS", "NYSE"):
            return [".N"]
        if excg in ("AMEX", "ASE"):
            return [".A"]
        return []

    last_error = None
    for excg_try in excg_candidates:
        # 심볼 후보는 거래소에 맞는 접미사를 우선 시도, 그다음 base, 그다음 원본 입력
        cand_syms: list[str] = []
        for suf in preferred_suffixes(excg_try):
            cand_syms.append(base_sym + suf)
        cand_syms.append(base_sym)
        if symbol and symbol.upper() not in cand_syms:
            cand_syms.append(symbol.upper())
        # 중복 제거 (순서 유지)
        seen = set()
        cand_syms = [s for s in cand_syms if not (s in seen or seen.add(s))]

        for sym_try in cand_syms:
            body = dict(body_base)
            body["OVRS_EXCG_CD"] = excg_try
            body["PDNO"] = sym_try
            try:
                hk = kis_hashkey(cfg, access_token, body)
                headers["hashkey"] = hk
                # DEBUG (간단 요약): TR/구분/가격 키
                print(f"[DEBUG] TR={tr_id} SLL_BUY={sll_buy} ORD_DVSN={ord_dvsn} ORD_UNPR={body['ORD_UNPR']} EXCG={body['OVRS_EXCG_CD']} PDNO={body['PDNO']}")
                r = requests.post(ORDER_URL, headers=headers, data=json.dumps(body), timeout=15)
                if r.status_code != 200:
                    last_error = f"주문 실패({r.status_code}): {r.text}"
                    continue
                j = r.json()
                if str(j.get("rt_cd")) == "0" or j.get("output"):
                    return j
                msg_cd = str(j.get("msg_cd"))
                msg1 = j.get("msg1", "")
                if msg_cd in ("APBK0656", "IGW00036"):
                    last_error = f"{msg_cd}: {msg1} (excg={excg_try}, sym={sym_try})"
                    continue
                return j
            except Exception as e:
                last_error = str(e)
                continue
    raise RuntimeError(last_error or "해외주문 실패: 원인 불명")

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

    # 현재 세션에 맞는 주문서버구분코드(본장=0, 확장세션=1)
    def current_session_ord_svr(dt=None) -> Optional[str]:
        d = dt or ny_now()
        if is_us_premarket_open_now(d):
            return "1"  # 프리장=확장세션
        if is_us_regular_open_now(d):
            return "0"  # 본장
        return None

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

    # KIS 설정/토큰 (실계좌)
    cfg = KisConfig.from_yaml(DEFAULT_KIS_YAML)
    access_token = kis_fetch_token(cfg)

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

    def _submit_buy(i: int, qty: int, last_px: float, level_px: float, ord_svr: str, order_type: str):
        attempts = 0
        while True:
            attempts += 1
            try:
                if order_type == "market":
                    resp = kis_overseas_order(cfg, access_token,
                                              side="BUY", order_type="market",
                                              symbol=ticker, exchange=exchange,
                                              qty=qty, price=None, ord_svr_dvsn_cd=ord_svr)
                else:
                    price = _safe_limit_price("buy", last_px, level_px)
                    resp = kis_overseas_order(cfg, access_token,
                                              side="BUY", order_type="limit",
                                              symbol=ticker, exchange=exchange,
                                              qty=qty, price=price, ord_svr_dvsn_cd=ord_svr)

                time.sleep(RATE_LIMIT_SEC)
                return resp
            except Exception as e:
                if _is_rate_limit_error(e) and attempts < 3:
                    time.sleep(RATE_LIMIT_SEC * 1.5)
                    continue
                raise

    def _submit_sell_all(qty: int, last_px: float, level_px: float, ord_svr: str, order_type: str):
        attempts = 0
        while True:
            attempts += 1
            try:
                if order_type == "market":
                    resp = kis_overseas_order(
                        cfg, access_token,
                        side="SELL", order_type="market",
                        symbol=ticker, exchange=exchange,
                        qty=qty, price=None, ord_svr_dvsn_cd=ord_svr
                    )
                else:
                    price = _safe_limit_price("sell", last_px, level_px)
                    resp = kis_overseas_order(
                        cfg, access_token,
                        side="SELL", order_type="limit",
                        symbol=ticker, exchange=exchange,
                        qty=qty, price=price, ord_svr_dvsn_cd=ord_svr
                    )
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

    print("[WATCH] 프리/본장 감시 시작 — 프리장(04:00~09:30)·본장(09:30~16:00) 모두 주문 허용")

    while True:
        now_ny = ny_now()
        ord_svr = current_session_ord_svr(now_ny)
        if ord_svr is None:
            # 마감 이후면 종료, 그 외 시간에는 대기
            if now_ny.time() >= datetime(now_ny.year, now_ny.month, now_ny.day, 16, 0).time():
                if position_qty > 0 and not sell_done:
                    print("🔔 [CLOSE] 장 마감 — 보유분 전량 시장가 매도")
                    try:
                        ord_svr_close = "0"  # 마감 청산은 본장 기준으로 처리
                        print(f"[ORDER] SELL type=market ord_svr={ord_svr_close} (close)")
                        resp = _submit_sell_all(
                            position_qty,
                            float(last) if last is not None else 0.0,
                            float(levels[-1]),
                            ord_svr_close,
                            "market"
                        )
                        print(f"[KIS RESP] {resp}")
                        sell_done = True
                    except Exception as e:
                        print(f"[ERR] 마감 매도 실패: {e}")
                print("✅ [DONE] 거래일 종료")
                break
            else:
                print(f"⏳ [WAIT] 프리/본장 외 시간 대기 (NY {now_ny.strftime('%H:%M')})")
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
                    # 프리장(확장세션=1)에서는 시장가 불가 → 지정가로 강제
                    order_type_effective = "limit" if ord_svr == "1" else args.order_type
                    print(f"[ORDER] BUY type={order_type_effective} ord_svr={ord_svr} level={level} last={last}")
                    resp = _submit_buy(i, qty, float(last), float(level), ord_svr, order_type_effective)
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
                    resp = _submit_sell_all(position_qty, float(last) if last is not None else 0.0, float(levels[-1]), ord_svr="0", order_type="market")
                    print(f"[KIS RESP] {resp}")
                    sell_done = True
                    position_qty = 0
                except Exception as e:
                    print(f"[ERR] 즉시 매도 실패: {e}")

        # 다음 주기까지 대기
        time.sleep(poll_sec)
if __name__ == "__main__":
    run()