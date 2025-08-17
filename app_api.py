# app_api.py
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import pandas as pd

from search_domestic_book_clear import (
    screen_three_conditions, symbol_status, _resolve_date_arg,
    DEFAULT_TOPN, DEFAULT_CAP_MIN, DEFAULT_CAP_MAX, DEFAULT_MIN_CHANGE_PCT
)

app = FastAPI(title="KRX Screener API")

# Flutter 웹/외부 도메인 접근 대비(CORS). 모바일 네이티브면 보통 불필요하지만 켜두면 편함.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # 배포 후엔 본인 도메인으로 제한 권장
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True}


@app.get("/")
def read_root():
    return {
        "message": "KRX Screener API is running 🚀",
        "docs": "/docs",
        "endpoints": ["/health", "/screen", "/status"]
    }

@app.get("/screen")
def api_screen(
    date: Optional[str] = Query(None, description="YYYY-MM-DD 또는 YYYYMMDD (미지정시 기본값)"),
    topn: int = Query(DEFAULT_TOPN, ge=1),
    cap_min: int = Query(DEFAULT_CAP_MIN, ge=0),
    cap_max: int = Query(DEFAULT_CAP_MAX, ge=0),
    min_change_pct: float = Query(DEFAULT_MIN_CHANGE_PCT)
):
    date_resolved = _resolve_date_arg(date or None)
    df = screen_three_conditions(date_resolved, topn, cap_min, cap_max, min_change_pct)
    rows = [] if df is None or df.empty else df.to_dict(orient="records")
    return {"date": date_resolved, "count": len(rows), "rows": rows}

@app.get("/status")
def api_status(
    symbol: str = Query(..., min_length=1, description="6자리 종목코드"),
    date: Optional[str] = Query(None),
    topn: int = Query(DEFAULT_TOPN, ge=1),
    cap_min: int = Query(DEFAULT_CAP_MIN, ge=0),
    cap_max: int = Query(DEFAULT_CAP_MAX, ge=0),
    min_change_pct: float = Query(DEFAULT_MIN_CHANGE_PCT)
):
    date_resolved = _resolve_date_arg(date or None)
    df = symbol_status(date_resolved, symbol, topn, cap_min, cap_max, min_change_pct)
    rows = [] if df is None or df.empty else df.to_dict(orient="records")
    return {"date": date_resolved, "symbol": symbol, "rows": rows}