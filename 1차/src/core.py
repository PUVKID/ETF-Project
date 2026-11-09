"""외부 패키지 없이 검증 가능한 시세·분할·모의체결 계산입니다."""

from datetime import date, datetime, time, timezone
import json
import math
import os
from pathlib import Path
import tempfile


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    with path.open(encoding="utf-8-sig") as stream:
        return json.load(stream)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".저장-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def session_completed(day, current=None):
    # 미 동부 표준시 기준 장 종료 후 15분까지 기다려 서머타임에도 보수적으로 처리합니다.
    current = datetime.fromisoformat(current) if isinstance(current, str) else current
    current = current or datetime.now(timezone.utc)
    close = datetime.combine(date.fromisoformat(day), time(21, 15), tzinfo=timezone.utc)
    return current >= close


def validate_rows(rows):
    by_date = {}
    for original in rows:
        row = dict(original)
        date.fromisoformat(row["date"])
        for field in ("open", "close"):
            value = float(row[field])
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{row['date']} {field} 가격이 유효하지 않습니다")
            row[field] = value
        if row["date"] in by_date and by_date[row["date"]] != row:
            raise ValueError(f"{row['date']} 중복 날짜의 값이 다릅니다")
        by_date[row["date"]] = row
    return [by_date[key] for key in sorted(by_date)]


def make_split(dates):
    dates = sorted(set(dates))
    if len(dates) < 20:
        raise ValueError("시간순 분할에는 최소 20개 관측값이 필요합니다")
    first = int(len(dates) * 0.6)
    second = int(len(dates) * 0.8)
    return {"train_start": dates[0], "train_end": dates[first - 1],
            "validation_start": dates[first], "validation_end": dates[second - 1],
            "test_start": dates[second], "test_end": dates[-1], "rows": len(dates),
            "method": "고정 시간순 60/20/20; 동일 구간 반복은 새 독립 검증 아님"}


def common_dates(series):
    if not series:
        return []
    return sorted(set.intersection(*(set(row["date"] for row in rows) for rows in series.values())))


def forecast_metrics(actual, predicted, previous):
    if not actual or len(actual) != len(predicted) or len(actual) != len(previous):
        raise ValueError("예측과 실제값 길이가 일치해야 합니다")
    if any(not math.isfinite(float(value)) for values in (actual, predicted, previous) for value in values):
        raise ValueError("유한하지 않은 예측값입니다")
    if any(value <= 0 for value in actual):
        raise ValueError("양의 실제 가격이 필요합니다")
    count = len(actual)
    return {"mae": sum(abs(a - p) for a, p in zip(actual, predicted)) / count,
            "mape": sum(abs(a - p) / a for a, p in zip(actual, predicted)) / count,
            "direction_accuracy": sum((a > old) == (p > old) for a, p, old in zip(actual, predicted, previous)) / count,
            "n": count}


def backtest(series, fee_bps=10, rebalance=False, weights=None):
    """전일 종가로 결정하고 다음 관측일 조정 시가에만 모의 체결합니다."""
    if not 0 <= fee_bps <= 1000:
        raise ValueError("비용 범위를 확인해 주세요")
    series = {symbol: validate_rows(rows) for symbol, rows in series.items()}
    symbols = sorted(series)
    days = common_dates(series)
    if len(days) < 2:
        raise ValueError("공통 거래일이 부족합니다")
    weights = weights or {symbol: 1 / len(symbols) for symbol in symbols}
    if set(weights) != set(symbols) or any(not math.isfinite(value) or value < 0 for value in weights.values()):
        raise ValueError("배분이 유효하지 않습니다")
    if abs(sum(weights.values()) - 1) > 1e-8:
        raise ValueError("배분 합계는 1이어야 합니다")
    lookup = {symbol: {row["date"]: row for row in rows} for symbol, rows in series.items()}
    units = {symbol: 0.0 for symbol in symbols}
    cash = 100.0
    costs = turnover = 0.0
    history = [{"date": days[0], "value": 100.0}]
    trades = []
    rate = fee_bps / 10000
    for index, day in enumerate(days[1:], 1):
        previous = days[index - 1]
        # 월말 신호는 다음 거래일 시가에 체결되며 당일 종가를 참조하지 않습니다.
        should_trade = index == 1 or (rebalance and previous[:7] != day[:7])
        if should_trade:
            old_values = {symbol: units[symbol] * lookup[symbol][day]["open"] for symbol in symbols}
            wealth = cash + sum(old_values.values())
            low, high = 0.0, wealth
            # 비용을 제외한 자산과 거래대금의 방정식을 풀어 음수 현금을 방지합니다.
            for _ in range(70):
                after_fee = (low + high) / 2
                charge = rate * sum(abs(after_fee * weights[symbol] - old_values[symbol]) for symbol in symbols)
                if after_fee + charge > wealth:
                    high = after_fee
                else:
                    low = after_fee
            after_fee = (low + high) / 2
            notional = sum(abs(after_fee * weights[symbol] - old_values[symbol]) for symbol in symbols)
            charge = wealth - after_fee
            costs += charge
            turnover += notional / 100
            units = {symbol: after_fee * weights[symbol] / lookup[symbol][day]["open"] for symbol in symbols}
            cash = 0.0
            trades.append({"signal_date": previous, "execution_date": day, "cost": charge, "notional": notional})
        value = cash + sum(units[symbol] * lookup[symbol][day]["close"] for symbol in symbols)
        history.append({"date": day, "value": value})
    peak = 100.0
    drawdown = 0.0
    for row in history:
        peak = max(peak, row["value"])
        drawdown = min(drawdown, row["value"] / peak - 1)
    years = (date.fromisoformat(days[-1]) - date.fromisoformat(days[0])).days / 365.25
    end = history[-1]["value"]
    return {"history": history, "trades": trades, "total_return": end / 100 - 1,
            "cagr": (end / 100) ** (1 / years) - 1 if years > 0 else 0,
            "max_drawdown": drawdown, "turnover": turnover, "costs": costs,
            "start": days[0], "end": days[-1], "years": years, "fee_bps": fee_bps}
