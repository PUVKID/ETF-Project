"""단일 작업과 중지 가능한 지속 학습 루프의 명령 진입점입니다."""

import argparse
from datetime import datetime, timedelta, timezone
import time

from src.core import now_iso
from src.market import load_market, refresh_market
from src.modeling import reconcile_forward, train_round
from src.paper import reconcile_paper
from src.runtime import STOP, load_config, log, record_failure, update_state, worker_lock


def wait_cooperatively(seconds):
    until = time.monotonic() + seconds
    while time.monotonic() < until:
        if STOP.exists():
            return False
        time.sleep(min(1, max(0, until - time.monotonic())))
    return not STOP.exists()


def run(action):
    with worker_lock():
        config = load_config()
        continuous = action == "start"
        last_refresh = 0.0
        errors = 0
        update_state(job={"status": "running", "action": action, "continuous": continuous,
                          "message": "로컬 작업 실행 중", "next_run_at": None})
        while not STOP.exists():
            try:
                market = load_market()
                if action == "refresh" or not market["series"] or (continuous and time.monotonic() - last_refresh >= config.get("refresh_seconds", 21600)):
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
    run(parser.parse_args().action)
