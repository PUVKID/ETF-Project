"""단일 작업과 중지 가능한 지속 학습 루프의 명령 진입점입니다."""

import argparse
from datetime import datetime, timedelta, timezone
import time
import subprocess
import sys

from src.core import now_iso
from src.market import load_market, refresh_market
from src.modeling import reconcile_forward, train_round
from src.paper import reconcile_paper
from src.runtime import ENV_ROOT, PRIVATE, ROOT, STOP, initial_state, load_config, log, record_failure, update_state, worker_lock
from src.intraday import market_monitor


def wait_cooperatively(seconds):
    until = time.monotonic() + seconds
    while time.monotonic() < until:
        if STOP.exists():
            return False
        time.sleep(min(1, max(0, until - time.monotonic())))
    return not STOP.exists()


def fatal_cuda(error):
    message = str(error).lower()
    return isinstance(error, RuntimeError) and "cuda error:" in message and any(
        marker in message for marker in ("unspecified launch failure", "device-side assert",
                                        "illegal memory access", "the launch timed out", "illegal instruction"))


def supervise():
    """치명 CUDA 문맥을 폐기하고 마지막 저장 가중치로 새 작업을 시작합니다."""
    with worker_lock(PRIVATE / "운용감시잠금"):
        failures = 0
        while not STOP.exists():
            executable = ENV_ROOT / ".venv" / "Scripts" / "python.exe"
            command = [str(executable) if executable.is_file() else sys.executable,
                       "-B", "-X", "utf8", str(ROOT / "run_pipeline.py"), "start", "--worker"]
            started = time.monotonic()
            result = subprocess.run(command, cwd=ROOT, shell=False,
                                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            if STOP.exists():
                return 0
            if result.returncode != 75:
                return result.returncode
            failures = 1 if time.monotonic() - started >= 300 else failures + 1
            delay = min(300, 30 * 2 ** min(failures - 1, 4))
            message = f"치명 CUDA 오류로 프로세스를 종료했습니다. {delay}초 뒤 마지막 저장 가중치에서 재실행합니다."
            log(message, "error")
            update_state(job={"status": "backoff", "continuous": True, "message": message,
                              "next_run_at": (datetime.now(timezone.utc) + timedelta(seconds=delay)).isoformat()})
            if not wait_cooperatively(delay):
                return 0
        return 0


def run(action):
    with worker_lock():
        config = load_config()
        continuous = action == "start"
        monitor_enabled = continuous and bool(config.get("autotrading", {}).get("enabled", False))
        last_refresh = 0.0
        errors = 0
        update_state(notes=initial_state()["notes"], job={"status": "running", "action": action, "continuous": continuous,
                          "message": "로컬 작업 실행 중", "next_run_at": None})
        with market_monitor(monitor_enabled):
            return run_loop(action, config, continuous, last_refresh, errors)


def run_loop(action, config, continuous, last_refresh, errors):
    while not STOP.exists():
        try:
            update_state(job={"status": "running", "message": "수집·학습 작업을 실행하고 있습니다", "next_run_at": None})
            market = load_market()
            if action == "refresh" or not market["series"] or set(market["series"]) != set(config.get("symbols", market["series"])) or (continuous and time.monotonic() - last_refresh >= config.get("refresh_seconds", 21600)):
                market = refresh_market()
                last_refresh = time.monotonic()
            reconcile_forward(market["series"])
            reconcile_paper(market["series"])
            if action != "refresh" and not STOP.exists():
                train_round()
            errors = 0
            if not continuous:
                break
            update_state(job={"status": "running", "message": "같은 Chronos-2 체크포인트에서 이어 학습합니다", "next_run_at": None})
        except InterruptedError as error:
            log(str(error))
            break
        except Exception as error:
            record_failure(error)
            errors += 1
            log(str(error), "error")
            update_state(training={"status": "error"}, data={"status": "error"} if not load_market()["series"] else {},
                         job={"status": "error", "message": str(error)})
            if continuous and fatal_cuda(error):
                # 손상된 문맥에서 현재 미저장 가중치를 커밋하지 않습니다.
                return 75
            if not continuous:
                raise
            delay = min(3600, 60 * 2 ** min(errors - 1, 6))
            update_state(job={"status": "backoff", "message": f"오류 후 {delay}초 대기 · 원문을 비공개 기록에 보존",
                              "next_run_at": (datetime.now(timezone.utc) + timedelta(seconds=delay)).isoformat()})
            if not wait_cooperatively(delay):
                break
    stopped = STOP.exists()
    update_state(job={"status": "stopped" if stopped else "idle", "message": "사용자 요청으로 중지했습니다" if stopped else "요청 작업을 마쳤습니다",
                      "continuous": False, "next_run_at": None},
                 training={"status": "stopped"} if stopped else {})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["refresh", "train", "start"])
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    raise SystemExit(supervise() if args.action == "start" and not args.worker else run(args.action))
