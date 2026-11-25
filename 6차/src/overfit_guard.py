"""새 관측별 제한 학습과 저장 전 검증으로 과적합 위험을 줄입니다."""

from datetime import date
import math
from pathlib import Path
import uuid

from .core import now_iso, read_json, save_json


MONITOR_SCOPE = "최초3종목 고정 검증의 반복 모니터링; 독립 평가·15종목 성과·수익 또는 과적합 부재 보장 아님"


def _load_ledger(path):
    path = Path(path)
    if not path.exists():
        return {"schema_version": 1, "consumed_dates": {}, "attempts": []}
    try:
        ledger = read_json(path)
        if (not isinstance(ledger, dict) or ledger.get("schema_version") != 1
                or not isinstance(ledger.get("consumed_dates"), dict)
                or not isinstance(ledger.get("attempts"), list) or not ledger["attempts"]):
            raise ValueError("필수 필드 또는 실행 이력이 없습니다")
        union, ids = {}, set()
        for attempt in ledger["attempts"]:
            if (not isinstance(attempt, dict) or not isinstance(attempt.get("attempt_id"), str)
                    or attempt["attempt_id"] in ids or attempt.get("status") not in
                    ("reserved", "accepted", "rejected", "failed", "interrupted")
                    or type(attempt.get("budget_steps")) is not int or not 1 <= attempt["budget_steps"] <= 25
                    or not isinstance(attempt.get("dates"), dict) or not attempt["dates"]):
                raise ValueError("실행 이력이 유효하지 않습니다")
            ids.add(attempt["attempt_id"])
            for symbol, days in attempt["dates"].items():
                _validate_dates(symbol, days)
                if union.setdefault(symbol, set()).intersection(days):
                    raise ValueError("이미 소비한 날짜가 중복 예약되었습니다")
                union[symbol].update(days)
        for symbol, days in ledger["consumed_dates"].items():
            _validate_dates(symbol, days)
        if {key: sorted(value) for key, value in union.items()} != ledger["consumed_dates"]:
            raise ValueError("예약 이력과 소비 날짜가 다릅니다")
        return ledger
    except (ValueError, TypeError, KeyError, AttributeError) as error:
        raise RuntimeError("과적합 방어 원장이 손상되어 추가 학습을 차단했습니다") from error


def _validate_dates(symbol, days):
    if not isinstance(symbol, str) or not symbol or not isinstance(days, list) or not days:
        raise ValueError("자산·날짜가 유효하지 않습니다")
    if any(not isinstance(day, str) or date.fromisoformat(day).isoformat() != day for day in days):
        raise ValueError("날짜 형식이 유효하지 않습니다")
    if days != sorted(set(days)):
        raise ValueError("날짜가 정렬되지 않았거나 중복되었습니다")


def reserve_training(path, corpus, requested_steps, batch_size):
    """fit 전에 날짜를 원자적으로 소비하며 실패·재시작 때 되돌리지 않습니다."""
    if type(requested_steps) is not int or not 1 <= requested_steps <= 1000 or type(batch_size) is not int or batch_size < 1:
        raise ValueError("학습 단계·배치 설정이 유효하지 않습니다")
    ledger = _load_ledger(path)
    available, windows = {}, 0
    for segment in corpus["segments"]:
        if not segment["eligible"]:
            continue
        days = sorted({row["date"] for row in segment["rows"]})
        _validate_dates(segment["symbol"], days)
        if len(days) < 2:
            raise ValueError("학습 가능 구간은 최소 두 관측이 필요합니다")
        available.setdefault(segment["symbol"], set()).update(days)
        windows += len(days) - 1
    new_dates = {symbol: sorted(days - set(ledger["consumed_dates"].get(symbol, [])))
                 for symbol, days in available.items()}
    new_dates = {symbol: days for symbol, days in new_dates.items() if days}
    count = sum(map(len, new_dates.values()))
    result = {"status": "waiting_new_data", "reason": "새 학습 가능 관측 날짜를 기다립니다. 수집·모의운용은 계속합니다.",
              "new_observations": count, "budget_steps": 0, "max_steps": 25,
              "monitor_scope": MONITOR_SCOPE, "wait_seconds": 300,
              "sampling_note": "전체 허용 자료를 재사용하되 무작위 표본 때문에 모든 행의 매회 학습을 보장하지 않음"}
    if not count or not windows:
        return result
    budget = min(requested_steps, 25, math.ceil(count / batch_size), max(1, math.ceil(windows / batch_size)))
    result.update(status="reserved", reason="새 날짜의 제한 학습을 예약했습니다. 검증 전 운영 가중치를 바꾸지 않습니다.",
                  budget_steps=budget, bootstrap=not ledger["attempts"], attempt_id=uuid.uuid4().hex)
    attempt = {**result, "reserved_at": now_iso(), "dates": new_dates}
    ledger["attempts"].append(attempt)
    ledger["consumed_dates"] = {symbol: sorted(set(ledger["consumed_dates"].get(symbol, [])) | available.get(symbol, set()))
                                for symbol in sorted(set(ledger["consumed_dates"]) | set(available))}
    save_json(path, ledger)
    return result


def finish_attempt(path, attempt_id, status, report=None):
    if status not in ("accepted", "rejected", "failed", "interrupted"):
        raise ValueError("학습 예약 종료 상태가 유효하지 않습니다")
    ledger = _load_ledger(path)
    matches = [entry for entry in ledger["attempts"] if entry["attempt_id"] == attempt_id]
    if len(matches) != 1 or matches[0]["status"] != "reserved":
        raise RuntimeError("과적합 방어 원장의 예약 상태가 일치하지 않습니다")
    matches[0].update(status=status, finished_at=now_iso(), report=report or {})
    save_json(path, ledger)


def validation_decision(before, after):
    """같은 고정 검증 MAE·MAPE 중 하나라도 악화하면 저장을 거부합니다."""
    result = {"accepted": False, "reason": "고정 검증의 유한한 동일 표본 지표를 확인하지 못했습니다.",
              "tolerance_fraction": 0.0, "monitor_scope": MONITOR_SCOPE}
    for metrics in (before, after):
        if (not isinstance(metrics, dict) or type(metrics.get("n")) is not int or metrics["n"] <= 0
                or any(type(metrics.get(key)) not in (int, float) or not math.isfinite(metrics[key])
                       or metrics[key] < 0 for key in ("mae", "mape"))):
            return result
    if before["n"] != after["n"]:
        return result
    worsened = [key for key in ("mae", "mape") if after[key] > before[key]]
    result.update(accepted=not worsened, reason="고정 검증 오차 악화: " + ", ".join(worsened) if worsened
                  else "같은 고정 검증에서 MAE·MAPE 악화 없음; 미래 성능 보장 아님")
    return result
