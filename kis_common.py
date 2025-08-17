# kis_common.py
from __future__ import annotations
import os, json, time
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional

import requests
import pandas as pd

# ===== 기본 설정 (모의 기본) =====
BASE_URL = os.getenv("KIS_BASE_URL", "https://openapivts.koreainvestment.com:29443")
TOKEN_PATH = "/oauth2/tokenP"              # 실전은 "/oauth2/token"
HASHKEY_PATH = "/uapi/hashkey"
CONTENT_TYPE = "application/json"

# ===== 계정 (환경변수 우선, 없으면 아래 값 사용) =====
APP_KEY    = os.getenv("KIS_APP_KEY",    "PSLB6jwalqlluwOglRaeQfDWSuERe2YIpE9N")
APP_SECRET = os.getenv("KIS_APP_SECRET", "q/YXQ7du/sTdhQ3TobYiBhJqn1iODBhU5JmlW8mYUUronVgW3ZsJAkOAGQx3xat8bNnmyp1vofc5q3VY+blotMwgl1yTZZC/P25OjWOlfwJNaLNEK9U0JuWQMbEnLmfAFxaa2xdcaXxAn+AtgYNcxEhdbZ4u8nn6MtclPrlob96ZVjOlGTk=")
CANO       = os.getenv("KIS_CANO",       "50146178")
PRDT_CD    = os.getenv("KIS_ACNT_PRDT_CD", "01")   # 보통 01

def is_mock_env() -> bool:
    return "openapivts" in BASE_URL

def mask(s: str, head: int = 4, tail: int = 3) -> str:
    if not s: return s
    return s if len(s) <= head + tail else (s[:head] + "*" * (len(s) - head - tail) + s[-tail:])

# ===== 토큰 발급/캐시 =====
def _token_cache_path(appkey: str) -> str:
    env = "vts" if is_mock_env() else "real"
    ak = (appkey or "")[:6]
    return os.path.join(os.path.expanduser("~"), f".kis_token_{env}_{ak}.json")

def _load_token_cache(path: str):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        return None
    return None

def _save_token_cache(path: str, token: str, expire_at_ts: float):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"access_token": token, "expire_at_ts": float(expire_at_ts)}, f)

def get_access_token(appkey: str, appsecret: str) -> Tuple[str, int]:
    url = f"{BASE_URL}{TOKEN_PATH}"
    headers = {"content-type": CONTENT_TYPE}
    body = {"grant_type": "client_credentials", "appkey": appkey, "appsecret": appsecret}
    res = requests.post(url, headers=headers, data=json.dumps(body), timeout=20)
    if res.status_code != 200:
        raise RuntimeError(f"토큰 발급 실패: HTTP {res.status_code} -> {res.text}")
    data = res.json()
    token = data.get("access_token"); expires_in = int(data.get("expires_in", 0))
    if not token:
        raise RuntimeError(f"토큰 응답에 access_token 없음: {data}")
    return token, expires_in

def get_access_token_cached(appkey: str, appsecret: str, safety_margin_sec: int = 60):
    path = _token_cache_path(appkey); now_ts = time.time()
    cached = _load_token_cache(path)
    if cached:
        token = cached.get("access_token"); exp = float(cached.get("expire_at_ts", 0))
        if token and (exp - now_ts) > safety_margin_sec:
            return token, int(exp - now_ts), True
    token, expires_in = get_access_token(appkey, appsecret)
    if not expires_in or expires_in <= 0: expires_in = 23*3600
    _save_token_cache(path, token, now_ts + expires_in)
    return token, expires_in, False

def build_common_headers(token: str, appkey: str, appsecret: str) -> Dict[str, str]:
    return {"content-type": CONTENT_TYPE, "authorization": f"Bearer {token}", "appkey": appkey, "appsecret": appsecret}

def get_hashkey(body: dict, appkey: str, appsecret: str) -> str:
    url = f"{BASE_URL}{HASHKEY_PATH}"
    headers = {"content-type": CONTENT_TYPE, "appkey": appkey, "appsecret": appsecret}
    res = requests.post(url, headers=headers, data=json.dumps(body), timeout=10)
    if res.status_code != 200:
        raise RuntimeError(f"hashkey 생성 실패: HTTP {res.status_code} -> {res.text}")
    data = res.json()
    return data.get("HASH") or data.get("hash") or data.get("HASHKEY") or data.get("hashkey")

# ===== 해외 주문/잔고 =====
def _overseas_tr_id(side: str) -> str:
    base = "JTTT1002U" if side.upper()=="BUY" else "JTTT1006U"
    return ("V"+base[1:]) if is_mock_env() else base

def _order_overseas_headers(token: str, appkey: str, appsecret: str, tr_id: str, hashkey: str) -> Dict[str, str]:
    return {"content-type": CONTENT_TYPE, "authorization": f"Bearer {token}",
            "appkey": appkey, "appsecret": appsecret, "tr_id": tr_id, "custtype": "P", "hashkey": hashkey}

def _order_overseas_body(cano: str, prdt_cd: str, excg: str, code: str, qty: int, price: float|int, ord_dvsn: str) -> dict:
    return {"CANO": str(cano), "ACNT_PRDT_CD": str(prdt_cd), "OVRS_EXCG_CD": str(excg),
            "PDNO": str(code), "ORD_DVSN": str(ord_dvsn), "ORD_QTY": str(int(qty)),
            "OVRS_ORD_UNPR": str(price if ord_dvsn=="00" else 0), "ORD_SVR_DVSN_CD": "0"}

def order_overseas(token: str, appkey: str, appsecret: str, cano: str, prdt_cd: str,
                   excg: str, code: str, qty: int, price: float|int, side: str, ord_dvsn: str="00") -> dict:
    url = f"{BASE_URL}/uapi/overseas-stock/v1/trading/order"
    tr_id = _overseas_tr_id(side)
    body = _order_overseas_body(cano, prdt_cd, excg, code, qty, price, ord_dvsn)
    hashkey = get_hashkey(body, appkey, appsecret); headers = _order_overseas_headers(token, appkey, appsecret, tr_id, hashkey)
    res = requests.post(url, headers=headers, data=json.dumps(body), timeout=20)
    if res.status_code != 200:
        raise RuntimeError(f"해외 주문 실패: HTTP {res.status_code} -> {res.text}")
    return res.json()

def obuy_limit(token, appkey, appsecret, excg, code, qty, price):
    return order_overseas(token, appkey, appsecret, CANO, PRDT_CD, excg, code, qty, price, side="BUY", ord_dvsn="00")

def obuy_market(token, appkey, appsecret, excg, code, qty):
    return order_overseas(token, appkey, appsecret, CANO, PRDT_CD, excg, code, qty, 0, side="BUY", ord_dvsn="01")

def osell_limit(token, appkey, appsecret, excg, code, qty, price):
    return order_overseas(token, appkey, appsecret, CANO, PRDT_CD, excg, code, qty, price, side="SELL", ord_dvsn="00")

def osell_market(token, appkey, appsecret, excg, code, qty):
    return order_overseas(token, appkey, appsecret, CANO, PRDT_CD, excg, code, qty, 0, side="SELL", ord_dvsn="01")

def overseas_inquire_balance(headers: Dict[str, str], cano: str, prdt_cd: str, excg: str) -> Dict:
    url = f"{BASE_URL}/uapi/overseas-stock/v1/trading/inquire-balance"
    tr_id = "VTTT3012R" if is_mock_env() else "JTTT3012R"
    h = dict(headers); h["tr_id"] = tr_id
    params = {"CANO": cano, "ACNT_PRDT_CD": prdt_cd, "OVRS_EXCG_CD": excg, "TR_CRCY_CD":"USD",
              "CTX_AREA_FK200":"", "CTX_AREA_NK200":""}
    res = requests.get(url, headers=h, params=params, timeout=20)
    if res.status_code != 200:
        raise RuntimeError(f"해외 잔고조회 실패: HTTP {res.status_code} -> {res.text}")
    return res.json()

def get_position_qty_overseas(headers: Dict[str, str], excg: str, code: str) -> int:
    data = overseas_inquire_balance(headers, CANO, PRDT_CD, excg); total = 0
    for row in data.get("output1", []):
        sym = str(row.get("ovrs_pdno") or row.get("pdno") or "").upper()
        if sym == code.upper():
            qty = row.get("ovrs_cblc_qty") or row.get("hldg_qty") or "0"
            try: total += int(float(str(qty)))
            except Exception: pass
    return total

# ===== 시세 (KIS → 실패 시 yfinance 1분봉) =====
def get_overseas_last_price(code: str, exchg: str) -> Optional[float]:
    # KIS 시세 (일부 계정/모의에서 제한 가능)
    try:
        token, _, _ = get_access_token_cached(APP_KEY, APP_SECRET)
        url = f"{BASE_URL}/uapi/overseas-price/v1/quotations/price"
        headers = {"content-type": CONTENT_TYPE, "authorization": f"Bearer {token}",
                   "appkey": APP_KEY, "appsecret": APP_SECRET}
        params = {"EXCD": exchg, "SYMB": code}
        res = requests.get(url, headers=headers, params=params, timeout=5)
        if res.status_code == 200:
            data = res.json(); out = data.get("output") or {}
            for k in ("last","ovrs_prpr","stck_prpr","close","P"):
                v = out.get(k)
                if v is not None: return float(v)
            if data.get("last") is not None: return float(data["last"])
    except Exception:
        pass
    # Yahoo 1분봉
    try:
        import yfinance as yf
        hist = yf.Ticker(code).history(period="1d", interval="1m", auto_adjust=False)
        if isinstance(hist, pd.DataFrame) and not hist.empty and "Close" in hist.columns:
            return float(hist["Close"].dropna().iloc[-1])
    except Exception:
        pass
    return None

# ===== 재시도 =====
def call_with_retry(func, *args, retries=3, base_sleep=0.8, **kwargs):
    last_err = None
    for i in range(retries):
        try:
            return func(*args, **kwargs)
        except RuntimeError as e:
            msg = str(e); last_err = e
            if "EGW00201" in msg or "초당 거래건수" in msg:
                time.sleep(base_sleep * (2**i)); continue
            raise
    raise last_err if last_err else RuntimeError("주문 재시도 실패")

# ===== 컨텍스트 =====
def create_kis_context() -> Dict[str, str]:
    token, expires_in, from_cache = get_access_token_cached(APP_KEY, APP_SECRET)
    expire_at = datetime.now() + timedelta(seconds=expires_in or 23*3600)
    print("✅ 토큰 사용 (캐시)" if from_cache else "✅ 토큰 발급 성공")
    print(f"- access_token(head): {token[:12]}… (총 {len(token)} chars)")
    print(f"- expires_in: {expires_in or 'N/A'} sec (~ {expire_at.strftime('%Y-%m-%d %H:%M:%S')})")
    headers = build_common_headers(token, APP_KEY, APP_SECRET)
    return {"token": token, "headers": headers,
            "appkey": APP_KEY, "appsecret": APP_SECRET,
            "cano": CANO, "prdt": PRDT_CD}

# ===============================
# 국내 주식 주문/시세 유틸 (KOSPI/KOSDAQ)
# ===============================
def _order_cash_headers(token: str, appkey: str, appsecret: str, tr_id: str, hashkey: str) -> Dict[str, str]:
    return {
        "content-type": CONTENT_TYPE,
        "authorization": f"Bearer {token}",
        "appkey": appkey,
        "appsecret": appsecret,
        "tr_id": tr_id,
        "custtype": "P",
        "hashkey": hashkey,
    }

def _order_cash_body(code: str, qty: int, price: int | float, ord_dvsn: str) -> dict:
    # 국내 주식 주문 본문
    return {
        "CANO": str(CANO),
        "ACNT_PRDT_CD": str(PRDT_CD),
        "PDNO": str(code),            # 6자리 종목코드 (예: 005930)
        "ORD_DVSN": str(ord_dvsn),    # 00: 지정가, 01: 시장가
        "ORD_QTY": str(int(qty)),
        "ORD_UNPR": str(int(price) if ord_dvsn == "00" else 0),  # 시장가는 0
    }

def order_cash(token: str, appkey: str, appsecret: str,
               code: str, qty: int, price: int | float, side: str, ord_dvsn: str = "00") -> dict:
    """
    국내 주식 현금주문 (모의/실전 자동 대응)
    side: "BUY" / "SELL"
    ord_dvsn: "00"(지정가) / "01"(시장가)
    """
    path = "/uapi/domestic-stock/v1/trading/order-cash"
    tr_real = "TTTC0802U" if side.upper() == "BUY" else "TTTC0801U"
    tr_id = ("V" + tr_real[1:]) if is_mock_env() else tr_real

    body = _order_cash_body(code, qty, price, ord_dvsn)
    hashkey = get_hashkey(body, appkey, appsecret)
    headers = _order_cash_headers(token, appkey, appsecret, tr_id, hashkey)

    url = f"{BASE_URL}{path}"
    res = requests.post(url, headers=headers, data=json.dumps(body), timeout=20)
    if res.status_code != 200:
        raise RuntimeError(f"국내 주문 실패: HTTP {res.status_code} -> {res.text}")
    return res.json()

def dbuy_limit(token: str, appkey: str, appsecret: str, code: str, qty: int, price: int | float) -> dict:
    return order_cash(token, appkey, appsecret, code, qty, price, side="BUY", ord_dvsn="00")

def dbuy_market(token: str, appkey: str, appsecret: str, code: str, qty: int) -> dict:
    return order_cash(token, appkey, appsecret, code, qty, price=0, side="BUY", ord_dvsn="01")

def dsell_limit(token: str, appkey: str, appsecret: str, code: str, qty: int, price: int | float) -> dict:
    return order_cash(token, appkey, appsecret, code, qty, price, side="SELL", ord_dvsn="00")

def dsell_market(token: str, appkey: str, appsecret: str, code: str, qty: int) -> dict:
    return order_cash(token, appkey, appsecret, code, qty, price=0, side="SELL", ord_dvsn="01")

def get_domestic_last_price(code: str) -> Optional[float]:
    """
    국내 현재가 조회 (KIS → 실패 시 Yahoo 대체)
    KIS: /uapi/domestic-stock/v1/quotations/inquire-price (tr_id=FHKST01010100)
    """
    # 1) KIS 시도
    try:
        token, _, _ = get_access_token_cached(APP_KEY, APP_SECRET)
        url = f"{BASE_URL}/uapi/domestic-stock/v1/quotations/inquire-price"
        headers = {
            "content-type": CONTENT_TYPE,
            "authorization": f"Bearer {token}",
            "appkey": APP_KEY,
            "appsecret": APP_SECRET,
            "tr_id": "FHKST01010100",
        }
        params = {
            "FID_COND_MRKT_DIV_CODE": "J",   # J: 주식
            "FID_INPUT_ISCD": str(code),     # 6자리 코드
        }
        res = requests.get(url, headers=headers, params=params, timeout=5)
        if res.status_code == 200:
            data = res.json()
            out = data.get("output") or {}
            # 대표 필드: stck_prpr(현재가)
            for k in ("stck_prpr", "prpr", "last"):
                v = out.get(k)
                if v is not None:
                    return float(v)
    except Exception:
        pass

    # 2) Yahoo 대체 (.KS → .KQ 폴백)
    try:
        import yfinance as yf
        for suffix in (".KS", ".KQ"):
            hist = yf.Ticker(f"{code}{suffix}").history(period="1d", interval="1m", auto_adjust=False)
            if isinstance(hist, pd.DataFrame) and not hist.empty and "Close" in hist.columns:
                return float(hist["Close"].dropna().iloc[-1])
    except Exception:
        pass
    return None