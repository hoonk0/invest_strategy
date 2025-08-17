# -*- coding: utf-8 -*-
from __future__ import annotations

"""
test.py — 실행/입력/기간루프가 '무조건 보이는' 최소 작동 버전
- 콘솔에 아무것도 안 찍히는 문제를 막기 위해, 모든 단계에서 배너/로그를 출력합니다.
- 기간(start~end) 지정 시 간단 백테스트 루프를 돌며 날짜만 출력(스크리닝/시뮬 자리 표시자).
- 단일 날짜 모드일 때도 배너/요약을 출력합니다.
- 추후 여기에 fetch_krx/screen/simulate 함수들을 붙이면 바로 확장 가능합니다.
"""

import sys
import argparse
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple

print(">>> test.py LOADED (module import stage)", flush=True)
import sys as _sys_dbg
_sys_dbg.stderr.write(">>> [stderr] test.py import OK\n")
_sys_dbg.stderr.flush()

# ===== 날짜 유틸 =====
def _normalize_yyyymmdd(s: str) -> str:
    s = str(s).strip().replace("-", "").replace("/", "")
    if len(s) != 8 or not s.isdigit():
        raise ValueError("날짜는 YYYY-MM-DD 또는 YYYYMMDD 형식이어야 합니다.")
    return s

def _resolve_date_arg(date_str: Optional[str]) -> str:
    """--date 파라미터 없으면 오늘(YYYYMMDD), 입력 시 YYYY-MM-DD/YYYMMDD 허용"""
    if date_str and str(date_str).strip():
        return _normalize_yyyymmdd(str(date_str))
    return datetime.now().strftime("%Y%m%d")

# ===== 기본 파라미터 =====
DEFAULT_DATE: Optional[str] = None           # YYYY-MM-DD, 미지정 시 오늘
DEFAULT_TOPN: int = 30                       # 자리표시자(현재 미사용)
DEFAULT_CAP_MIN: int = 300_000_000           # 자리표시자(현재 미사용)
DEFAULT_CAP_MAX: int = 10_000_000_000_000    # 자리표시자(현재 미사용)
DEFAULT_MIN_CHANGE_PCT: float = 5.0          # 자리표시자(현재 미사용)
DEFAULT_TP_PCT: float = 3.0                  # 자리표시자(현재 미사용)
DEFAULT_SL_PCT: float = 1.5                  # 자리표시자(현재 미사용)
DEFAULT_SEED_PER_STOCK: int = 1_000_000      # 자리표시자(현재 미사용)

# 기간 기본값 — 둘 다 지정되면 기간 백테스트 모드
DEFAULT_START_DATE: Optional[str] = "2024-01-01"
DEFAULT_END_DATE: Optional[str] = "2024-12-31"

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="테스트 스켈레톤 (항상 출력)")
    p.add_argument("--date", default=DEFAULT_DATE, help="단일 조회 날짜 (YYYY-MM-DD/YYYMMDD). 미지정 시 오늘")

    # (자리표시자) 조건/시뮬 관련 인자 — 현재는 값만 보여줌
    p.add_argument("--topn", type=int, default=DEFAULT_TOPN)
    p.add_argument("--cap_min", type=int, default=DEFAULT_CAP_MIN)
    p.add_argument("--cap_max", type=int, default=DEFAULT_CAP_MAX)
    p.add_argument("--min_change_pct", type=float, default=DEFAULT_MIN_CHANGE_PCT)
    p.add_argument("--seed_per_stock", type=int, default=DEFAULT_SEED_PER_STOCK)
    p.add_argument("--tp_pct", type=float, default=DEFAULT_TP_PCT)
    p.add_argument("--sl_pct", type=float, default=DEFAULT_SL_PCT)

    # 기간(백테스트) 옵션
    p.add_argument("--start_date", default=DEFAULT_START_DATE, help="조회 시작일 (YYYY-MM-DD/YYYMMDD)")
    p.add_argument("--end_date", default=DEFAULT_END_DATE, help="조회 종료일 (YYYY-MM-DD/YYYMMDD)")
    p.add_argument("--export_csv", default=None, help="(옵션) 기간 실행 시 일자별 결과 CSV 저장 경로")
    p.add_argument("--backtest", action="store_true", help="기간 강제 백테스트 모드 (start/end 없이도 오늘 하루로 실행)")

    # 인터랙션 스킵 옵션 (입력 불가 환경 대비)
    p.add_argument("--no_input", action="store_true", help="프롬프트 생략(비대화식 실행)")
    return p.parse_args()

def interactive_setup(args: argparse.Namespace) -> Tuple[str, float, float, float, int, Optional[str], Optional[str]]:
    """
    실행 직후 간단 설정 프롬프트(항상 배너 출력). 입력이 불가능하면 기본값/인자값을 사용.
    반환: (date_yyyymmdd, min_change_pct, tp_pct, sl_pct, seed_per_stock, start_date_yyyymmdd, end_date_yyyymmdd)
    """
    print("=== 인터랙티브 셋업 시작 ===", flush=True)
    today_str = datetime.now().strftime("%Y%m%d")
    current_date = _resolve_date_arg(args.date) if getattr(args, "date", None) else today_str
    current_min_pct = args.min_change_pct if getattr(args, "min_change_pct", None) is not None else DEFAULT_MIN_CHANGE_PCT
    current_tp_pct = args.tp_pct if getattr(args, "tp_pct", None) is not None else DEFAULT_TP_PCT
    current_sl_pct = args.sl_pct if getattr(args, "sl_pct", None) is not None else DEFAULT_SL_PCT
    current_seed   = args.seed_per_stock if getattr(args, "seed_per_stock", None) is not None else DEFAULT_SEED_PER_STOCK

    print(f"[SETUP] 기본값 → 날짜={current_date[:4]}-{current_date[4:6]}-{current_date[6:8]}, 최소등락률={current_min_pct}%, TP={current_tp_pct}%, SL={current_sl_pct}%, 시드={current_seed:,}원", flush=True)

    # 입력 생략 모드면 바로 반환
    if getattr(args, "no_input", False):
        print("[SETUP] --no_input 지정 → 프롬프트 스킵", flush=True)
        start_date_ymd = _normalize_yyyymmdd(args.start_date) if args.start_date else None
        end_date_ymd   = _normalize_yyyymmdd(args.end_date) if args.end_date else None
        return current_date, current_min_pct, current_tp_pct, current_sl_pct, current_seed, start_date_ymd, end_date_ymd

    # 프롬프트 (입력이 막혀있으면 EOFError → except로 빠짐)
    try:
        period_in = input("📅 조회기간 (YYYY-MM-DD~YYYY-MM-DD, 공백시 단일 날짜): ").strip()
        date_in   = input("📅 단일 조회 날짜 (YYYY-MM-DD, 기간 미사용 시) [엔터=유지]: ").strip()
        pct_in    = input("📈 최소 등락률(%) [엔터=유지]: ").strip()
        tp_in     = input("🎯 익절 퍼센트(%) [엔터=유지]: ").strip()
        sl_in     = input("🛑 손절 퍼센트(%) [엔터=유지]: ").strip()
        seed_in   = input("💰 종목당 투입금액(원) [엔터=유지]: ").strip()
    except Exception:
        print("[SETUP] 입력 채널이 막혀 있습니다 → 기본값/인자값 유지", flush=True)
        period_in = date_in = pct_in = tp_in = sl_in = seed_in = ""

    start_date_ymd = None
    end_date_ymd = None
    if period_in:
        try:
            if "~" in period_in:
                s, e = [x.strip() for x in period_in.split("~", 1)]
            else:
                raise ValueError("형식은 YYYY-MM-DD~YYYY-MM-DD 입니다.")
            start_date_ymd = _normalize_yyyymmdd(s)
            end_date_ymd   = _normalize_yyyymmdd(e)
        except Exception as e:
            print(f"[SETUP] 조회기간 형식 오류: {e} → 기간 입력 무시", flush=True)
            start_date_ymd = None
            end_date_ymd = None

    if date_in:
        try:
            current_date = _normalize_yyyymmdd(date_in)
        except Exception as e:
            print(f"[SETUP] 날짜 입력 오류: {e} → 기존값 유지", flush=True)

    if pct_in:
        try: current_min_pct = float(pct_in)
        except Exception: print("[SETUP] 최소 등락률 입력 오류 → 기존값 유지", flush=True)
    if tp_in:
        try: current_tp_pct = float(tp_in)
        except Exception: print("[SETUP] TP 입력 오류 → 기존값 유지", flush=True)
    if sl_in:
        try: current_sl_pct = float(sl_in)
        except Exception: print("[SETUP] SL 입력 오류 → 기존값 유지", flush=True)
    if seed_in:
        try: current_seed = int(seed_in.replace(",", ""))
        except Exception: print("[SETUP] 시드 입력 오류 → 기존값 유지", flush=True)

    # 요약 배너
    if start_date_ymd and end_date_ymd:
        print(f"[SETUP] 적용값 → 기간={start_date_ymd}~{end_date_ymd}, min%={current_min_pct}, TP={current_tp_pct}, SL={current_sl_pct}, seed={current_seed:,}원", flush=True)
    else:
        print(f"[SETUP] 적용값 → 날짜={current_date}, min%={current_min_pct}, TP={current_tp_pct}, SL={current_sl_pct}, seed={current_seed:,}원", flush=True)

    return current_date, current_min_pct, current_tp_pct, current_sl_pct, current_seed, start_date_ymd, end_date_ymd

# ===== 기간 루프(자리표시자) =====
def backtest_range(
    start_yyyymmdd: str,
    end_yyyymmdd: str,
    *,
    topn: int,
    cap_min: int,
    cap_max: int,
    min_change_pct: float,
    seed_per_stock: int,
    tp_pct: float,
    sl_pct: float,
    export_csv: Optional[str] = None
) -> Dict[str, float | int]:
    """
    자리표시자 버전: 날짜만 하루씩 증가시키며 콘솔에 출력.
    추후 여기에 실제 스크리닝/시뮬 로직을 끼워넣으면 됩니다.
    """
    try:
        s = datetime.strptime(_normalize_yyyymmdd(start_yyyymmdd), "%Y%m%d").date()
        e = datetime.strptime(_normalize_yyyymmdd(end_yyyymmdd), "%Y%m%d").date()
    except Exception as ex:
        print(f"[PERIOD] 날짜 파싱 실패: {ex}", flush=True)
        return {"total_pnl": 0, "total_invested": 0, "win_days": 0, "loss_days": 0, "max_drawdown_pct": 0.0}

    if s > e:
        s, e = e, s

    total_pnl = 0
    total_invested = 0
    win_days = loss_days = 0
    peak_equity = 0
    equity = 0
    cur = s
    print("=== 기간 루프 시작 ===", flush=True)
    while cur <= e:
        ymd = cur.strftime("%Y%m%d")
        print(f"[DAY] {ymd} 실행(자리표시자) — 조건: TOP{topn}, cap[{cap_min:,}~{cap_max:,}], min%={min_change_pct}, seed={seed_per_stock:,}, TP={tp_pct}, SL={sl_pct}", flush=True)

        # TODO: 여기서 screen_three_conditions(...) → simulate_day(...) 호출
        # 자리표시자 결과(0원 수익)로 누적
        day_pnl = 0
        invested = 0

        equity += day_pnl
        peak_equity = max(peak_equity, equity)
        dd = 0 if peak_equity == 0 else (peak_equity - equity) / peak_equity * 100.0

        total_pnl += day_pnl
        total_invested += invested
        if day_pnl >= 0: win_days += 1
        else: loss_days += 1

        cur += timedelta(days=1)
    max_dd_pct = 0.0 if peak_equity == 0 else (peak_equity - equity) / peak_equity * 100.0
    print("=== 기간 루프 종료 ===", flush=True)
    return {
        "total_pnl": int(total_pnl),
        "total_invested": int(total_invested),
        "win_days": win_days,
        "loss_days": loss_days,
        "max_drawdown_pct": float(max_dd_pct)
    }

def main() -> None:
    print("=== ENTER main() ===", flush=True)
    print("=== 스크립트 진입 ===", flush=True)
    args = parse_args()
    print(f"[ARGS] {vars(args)}", flush=True)

    try:
        date_choice, min_pct_choice, tp_choice, sl_choice, seed_choice, start_choice, end_choice = interactive_setup(args)
    except Exception as ex:
        print(f"[SETUP] 인터랙티브 오류: {ex} → 인자/기본값 사용", flush=True)
        date_choice = _resolve_date_arg(args.date) if getattr(args, "date", None) else datetime.now().strftime("%Y%m%d")
        min_pct_choice = args.min_change_pct if getattr(args, "min_change_pct", None) is not None else DEFAULT_MIN_CHANGE_PCT
        tp_choice = args.tp_pct if getattr(args, "tp_pct", None) is not None else DEFAULT_TP_PCT
        sl_choice = args.sl_pct if getattr(args, "sl_pct", None) is not None else DEFAULT_SL_PCT
        seed_choice = args.seed_per_stock if getattr(args, "seed_per_stock", None) is not None else DEFAULT_SEED_PER_STOCK
        start_choice = _normalize_yyyymmdd(args.start_date) if getattr(args, "start_date", None) else None
        end_choice   = _normalize_yyyymmdd(args.end_date) if getattr(args, "end_date", None) else None

    # 적용
    args.date = date_choice
    args.min_change_pct = min_pct_choice
    args.tp_pct = tp_choice
    args.sl_pct = sl_choice
    args.seed_per_stock = seed_choice
    args.start_date = start_choice or args.start_date
    args.end_date = end_choice or args.end_date

    date_banner = _resolve_date_arg(args.date) if args.date else datetime.now().strftime("%Y%m%d")
    print(f"=== {date_banner} 전략 시작 ===", flush=True)

    # 기간 모드
    if (args.start_date and args.end_date) or getattr(args, "backtest", False):
        if not (args.start_date and args.end_date):
            print("[PERIOD] --backtest 지정 → 시작/종료일 없음 → 오늘 하루만 실행", flush=True)
            bt_start = bt_end = date_banner
        else:
            try:
                bt_start = _normalize_yyyymmdd(args.start_date)
                bt_end   = _normalize_yyyymmdd(args.end_date)
            except Exception as e:
                print(f"[PERIOD] 날짜 형식 오류: {e}", flush=True)
                return
        print(f"[PERIOD] 실행 준비: {bt_start}~{bt_end}", flush=True)
        out = backtest_range(
            bt_start, bt_end,
            topn=args.topn,
            cap_min=args.cap_min,
            cap_max=args.cap_max,
            min_change_pct=args.min_change_pct,
            seed_per_stock=args.seed_per_stock,
            tp_pct=args.tp_pct,
            sl_pct=args.sl_pct,
            export_csv=args.export_csv
        )
        sign = "+" if out["total_pnl"] >= 0 else ""
        print("\n[PERIOD SUMMARY]", flush=True)
        print(f"  - 기간 손익 합계: {sign}{out['total_pnl']:,}원", flush=True)
        ti = out["total_invested"]
        rr = (out["total_pnl"]/ti*100.0) if ti > 0 else 0.0
        print(f"  - 기간 총투입금(가정): {ti:,}원 | 수익률(투입대비): {rr:.2f}%", flush=True)
        print(f"  - 승일수/패일수: {out['win_days']}/{out['loss_days']}", flush=True)
        print(f"  - 최대 낙폭(Max DD): {out['max_drawdown_pct']:.2f}%", flush=True)
        if args.export_csv:
            print(f"  - 일별 결과 CSV 저장: {args.export_csv}", flush=True)
        print("=== 종료 ===", flush=True)
        return

    # 단일 날짜 모드
    print("[SINGLE] 기간 모드가 아니므로 단일 날짜 모드로 종료합니다.", flush=True)
    print("=== 종료 ===", flush=True)

if __name__ == "__main__":
    print(">>> __main__ guard reached — starting program", flush=True)
    try:
        main()
    except Exception as e:
        # 어떤 예외가 나더라도 '조용히 죽지 않도록' 마지막 로그를 남깁니다.
        print(f"[FATAL] 예외 발생: {e}", file=sys.stderr, flush=True)
        import traceback as _tb
        _tb.print_exc()
        raise