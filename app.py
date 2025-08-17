# app.py
from __future__ import annotations
import sys
import subprocess
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

import kis_common as kc

MENU = """
[전략 선택]
1) 국장
2) 해외 옵션 자동
3) 해외 옵션 연습
4) 유빅스
5) 국장 전략1 테스트(책기반)
선택: """

def main():
    print(f"ℹ️ BASE_URL={kc.BASE_URL}")
    print(f"- APP_KEY : {kc.mask(kc.APP_KEY)}")
    print(f"- APP_SECRET: {kc.mask(kc.APP_SECRET)}")
    print(f"- CANO/PRDT: {kc.CANO}/{kc.PRDT_CD}")

    choice = input(MENU).strip()
    if choice == "1":
        subprocess.run([sys.executable, os.path.join(BASE_DIR, "strategy_domestic_intraday.py")])
    elif choice == "2":
        subprocess.run([sys.executable, os.path.join(BASE_DIR, "strategy_oversea_option_auto.py")])
    elif choice == "3":
        subprocess.run([sys.executable, os.path.join(BASE_DIR, "strategy_oversea_option_exercise.py")])
    elif choice == "4":
        subprocess.run([sys.executable, os.path.join(BASE_DIR, "strategy_oversea_uvix.py")])
    elif choice == "5":
        subprocess.run([sys.executable, os.path.join(BASE_DIR, "strategy_domestic_book_test.py")])
    else:
        print("유효한 번호를 선택하세요.")

if __name__ == "__main__":
    main()