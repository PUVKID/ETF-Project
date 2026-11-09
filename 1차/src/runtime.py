"""로컬 비공개 상태와 단일 작업 잠금을 관리합니다."""

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sys
import traceback

from .core import now_iso, read_json, save_json


ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "work" / "비공개"
STATE = PRIVATE / "운영상태.json"
CONFIG = PRIVATE / "개인설정.json"
STOP = PRIVATE / "중지요청"
LOCK = PRIVATE / "작업잠금"
SYMBOLS = {"QQQM": "Invesco NASDAQ 100 ETF", "SCHG": "Schwab US Large-Cap Growth ETF",
           "SPYM": "State Street SPDR Portfolio S&P 500 ETF"}


def load_config():
    config = read_json(CONFIG)
    if config is None:
        config = {"monthly_contribution": 0, "currency": "KRW",
                  "symbols": list(SYMBOLS), "train_steps": 20, "batch_size": 2,
                  "context_length": 256, "gpu_memory_fraction": 0.25,
                  "refresh_seconds": 21600, "fee_bps": 10, "fx_spread_bps": 50}
        save_json(CONFIG, config)
    return config


def initial_state():
    return {"schema_version": 1, "updated_at": now_iso(), "holdings": [],
            "portfolio": {"total_value": 1000000, "currency": "KRW", "as_of": None,
                          "cost_basis": 1000000, "pnl": 0, "monthly_contribution": 0},
            "paper": {"status": "waiting_fx", "initial_capital_krw": 1000000,
                      "current_balance_krw": 1000000, "cash_krw": 1000000, "cash_usd": None,
                      "positions": [], "pnl_krw": 0, "pnl_pct": 0, "fx": None,
                      "fees_krw": 0, "exchange_cost_krw": 0, "message": "환율 확보 전 환산·체결 대기"},
            "history": [], "data": {"status": "idle", "source": "Yahoo Finance / yfinance",
                "rows": 0, "new_rows": 0, "symbols": [], "as_of": None,
                "adjustment": "배당·분할 조정 OHLC로 연구; 보유 평가는 비조정 마지막 종가"},
            "training": {"status": "idle", "step": 0, "total_steps": 0, "loss": None,
                "round": 0, "new_rows": 0, "checkpoint": None, "device": None},
            "validation": {"status": "pending", "metrics": []},
            "simulation": {"strategies": [], "assumptions": []},
            "job": {"status": "idle", "action": None, "message": "실행 대기", "continuous": False,
                    "next_run_at": None}, "logs": [], "forecasts": [], "forward": {"status": "pending", "n": 0},
            "notes": ["실제 주문 기능은 없습니다. 토스 Open API 계좌 인증·주문은 연결하지 않았습니다.",
                "QQQM은 2020년 설정으로 세 ETF의 공통 실측 기간이 10년 미만입니다.",
                "SPYM의 과거 티커는 SPLG이며 과거 추종 지수 변경 이력이 있습니다.",
                "무료 일봉은 지연 시세입니다. 당일 미완결 봉은 학습에서 제외합니다.",
                "개인 실제 보유와 별개로 원금 100만원의 가상계좌만 운용합니다. 추가입금은 0원입니다.",
                "조정 가격 연구에는 배당 재투자 가정이 있으며 환전·세금·실제 호가 차이는 포함하지 않습니다.",
                "Chronos-2 사전학습의 개별 자산·시점 경계가 확인되지 않아 엄격한 역사적 미관측 시험을 보장하지 않습니다.",
                "같은 고정 학습·검증 구간의 반복 학습은 새 독립 성과가 아닙니다. 검증 구간 선택 편향이 누적될 수 있습니다.",
                "최초 시험 종료일 이후 새 일봉이 2개 이상 쌓이면 별도 시계열로 이어 학습합니다. 그 구간은 독립 시험 성과로 사용하지 않습니다.",
                "PC가 켜져 있고 로컬 프로세스가 실행 중일 때만 반복합니다. 전원 종료 중에는 동작하지 않습니다.",
                "10년 이상은 투자 계획 기간입니다. 단기 모델 출력은 10년 수익 예측이 아닙니다."]}


def load_state():
    return read_json(STATE, initial_state())


def update_state(**sections):
    state = load_state()
    for key, value in sections.items():
        if isinstance(value, dict) and isinstance(state.get(key), dict):
            state[key].update(value)
        else:
            state[key] = value
    state["updated_at"] = now_iso()
    save_json(STATE, state)
    return state


def log(message, level="info"):
    state = load_state()
    entries = state.get("logs", [])
    entries.append({"time": now_iso(), "level": level, "message": str(message)})
    update_state(logs=entries[-120:])


def record_failure(error, argv=None):
    PRIVATE.mkdir(parents=True, exist_ok=True)
    entry = {"한국시각": datetime.now(timezone(timedelta(hours=9))).isoformat(),
             "argv": argv or sys.argv, "cwd": os.getcwd(), "exit_code": 1,
             "error": str(error), "traceback": traceback.format_exc(),
             "당시변경": "운영상태·로그는 갱신 가능; 소스는 이 함수에서 수정하지 않음"}
    with (PRIVATE / "실패이력.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(entry, ensure_ascii=False) + "\n")


@contextmanager
def worker_lock():
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with LOCK.open("a+b") as stream:
        stream.seek(0)
        # Windows의 배타 잠금 byte는 읽기 자체도 금지하므로 크기만 확인합니다.
        if os.fstat(stream.fileno()).st_size == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise RuntimeError("다른 수집·학습 작업이 실행 중입니다") from error
        else:
            import fcntl
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("다른 수집·학습 작업이 실행 중입니다") from error
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def is_busy():
    try:
        with worker_lock():
            return False
    except RuntimeError:
        return True
