"""무료 일봉의 출처·완결 여부와 조정 기준을 보존합니다."""

from datetime import datetime, timedelta, timezone
import hashlib
import json
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from .core import backtest, common_dates, now_iso, read_json, save_json, session_completed, validate_rows
from .runtime import PRIVATE, load_config, log, record_failure, update_state


MARKET = PRIVATE / "시세.json"


def load_market():
    return read_json(MARKET, {"series": {}, "metadata": {}})


def fetch_chart(symbol):
    parameters = urlencode({"period1": 946684800, "period2": int(datetime.now(timezone.utc).timestamp()),
                            "interval": "1d", "events": "div,splits"})
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{quote(symbol, safe='')}?{parameters}"
    with urlopen(Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=30) as response:
        payload = json.loads(response.read())
    if payload.get("chart", {}).get("error"):
        raise RuntimeError(f"{symbol}: {payload['chart']['error']}")
    result = payload.get("chart", {}).get("result")
    if not result:
        raise RuntimeError(f"{symbol}: 무료 시세 응답이 비어 있습니다")
    return result[0], url


def refresh_market():
    config = load_config()
    old = load_market()
    series = {}
    metadata = {}
    update_state(data={"status": "running"}, job={"message": "무료 과거 일봉을 확인하고 있습니다"})
    end = (datetime.now(timezone.utc).date() + timedelta(days=1)).isoformat()
    for symbol in config["symbols"]:
        chart, source_url = fetch_chart(symbol)
        quotes = chart["indicators"]["quote"][0]
        adjusted_prices = chart["indicators"]["adjclose"][0]["adjclose"]
        events = chart.get("events", {})
        dividends = {datetime.fromtimestamp(int(stamp), timezone.utc).date().isoformat(): value["amount"]
                     for stamp, value in events.get("dividends", {}).items()}
        splits = {datetime.fromtimestamp(int(stamp), timezone.utc).date().isoformat(): value["numerator"] / value["denominator"]
                  for stamp, value in events.get("splits", {}).items()}
        rows = []
        for index, timestamp in enumerate(chart["timestamp"]):
            day = datetime.fromtimestamp(timestamp, timezone.utc).date().isoformat()
            if not session_completed(day):
                continue
            if any(value is None for value in (quotes["close"][index], quotes["open"][index], adjusted_prices[index])):
                continue
            raw_close = float(quotes["close"][index])
            adjusted = float(adjusted_prices[index])
            factor = adjusted / raw_close
            rows.append({"date": day, "open": float(quotes["open"][index]) * factor,
                         "close": adjusted, "raw_close": raw_close, "raw_open": float(quotes["open"][index]),
                         "dividend": float(dividends.get(day, 0)), "split": float(splits.get(day, 0))})
        rows = validate_rows(rows)
        if len(rows) < 100:
            raise RuntimeError(f"{symbol}: 완결 일봉 {len(rows)}개로 학습 자료가 부족합니다")
        series[symbol] = rows
        encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
        metadata[symbol] = {"symbol": symbol, "rows": len(rows), "start": rows[0]["date"],
                            "end": rows[-1]["date"], "sha256": hashlib.sha256(encoded).hexdigest(),
                            "source": "Yahoo Finance chart 일봉 · 표준 HTTPS 직접 수집", "url": source_url, "fetched_at": now_iso(),
                            "adjustment": "Adj Close / Close 비율을 Open에 적용; 배당·분할 조정",
                            "alias_note": "과거 SPLG 티커; 과거 지수 변경 이력" if symbol == "SPYM" else None}
    # 전 종목 성공 후 하나의 스냅샷으로 저장해 일부 종목만 바뀌는 상태를 막습니다.
    payload = {"series": series, "metadata": metadata, "fetched_at": now_iso()}
    save_json(MARKET, payload)
    old_dates = {symbol: {row["date"] for row in rows} for symbol, rows in old["series"].items()}
    new_rows = sum(len({row["date"] for row in rows} - old_dates.get(symbol, set())) for symbol, rows in series.items())
    dates = common_dates(series)
    update_state(data={"status": "ready", "as_of": min(rows[-1]["date"] for rows in series.values()),
                       "start": dates[0], "end": dates[-1], "rows": sum(len(rows) for rows in series.values()),
                       "common_rows": len(dates), "new_rows": new_rows, "symbols": list(metadata.values()),
                       "fetched_at": payload["fetched_at"], "source": "Yahoo Finance chart / 무료 일봉"},
                 training={"new_rows": new_rows})
    try:
        refresh_fx()
    except Exception as error:
        record_failure(error)
        log(f"무료 환율 수집 실패 · 환산 상태를 확인해 주세요: {error}", "error")
        update_state(fx_status="error")
    rebuild_simulation(payload)
    log(f"완결 일봉 확보: 공통 {len(dates)}일, 신규 종목별 합계 {new_rows}행")
    return payload


def refresh_fx():
    from .paper import FX
    chart, source_url = fetch_chart("KRW=X")
    quotes = chart["indicators"]["quote"][0]
    rates = [{"date": datetime.fromtimestamp(stamp, timezone.utc).date().isoformat(), "rate": float(quotes["close"][index])}
             for index, stamp in enumerate(chart["timestamp"])
             if quotes["close"][index] is not None and session_completed(datetime.fromtimestamp(stamp, timezone.utc).date().isoformat())]
    rates = [row for row in rates if row["rate"] > 0]
    if not rates:
        raise RuntimeError("USD/KRW 실제 무료 환율 자료가 비어 있습니다")
    value = {"rate": rates[-1]["rate"], "as_of": rates[-1]["date"], "fetched_at": now_iso(),
             "source": "Yahoo Finance KRW=X 일봉 · KRW per USD", "url": source_url, "rates": rates,
             "note": "토스증권 실제 적용 환율 아님; 시장 일봉과 설정 스프레드 가정"}
    save_json(FX, value)
    update_state(fx_status="ready")
    return value


def rebuild_simulation(market=None):
    market = market or load_market()
    # 과거 단순 정책 비교는 최초 세 종목 범위를 유지하며 확장 후보의 모델 성과와 혼합하지 않습니다.
    series = {symbol: market["series"][symbol] for symbol in ("QQQM", "SCHG", "SPYM") if symbol in market["series"]}
    if not series:
        return
    from .paper import FX
    fx = read_json(FX)
    if not fx:
        update_state(history=[], simulation={"strategies": [], "assumptions": ["환율 확보 전 원화 비교를 대기합니다"]})
        return
    days = common_dates(series)
    trimmed = {symbol: [row for row in rows if row["date"] in set(days)] for symbol, rows in series.items()}
    hold = historical_policy(trimmed, fx)
    monthly = historical_policy(trimmed, fx, monthly=True)
    benchmark = historical_policy({"SPYM": trimmed["SPYM"]}, fx)
    history = [{"date": row["date"], "portfolio": row["value_krw"] / 10000,
                "rebalance": monthly["history"][index]["value_krw"] / 10000,
                "benchmark": benchmark["history"][index]["value_krw"] / 10000}
               for index, row in enumerate(hold["history"])]
    strategies = []
    for name, key, result in (("동일가중 단순 보유", "portfolio", hold), ("월간 리밸런싱", "rebalance", monthly), ("SPYM 단순 보유", "benchmark", benchmark)):
        values = [row[key] for row in history]
        peak, drawdown = 100.0, 0.0
        for value in values:
            peak = max(peak, value)
            drawdown = min(drawdown, value / peak - 1)
        years = (datetime.fromisoformat(days[-1]) - datetime.fromisoformat(days[0])).days / 365.25
        strategies.append({"name": name, "start": days[0], "end": days[-1], "years": years,
            "total_return": values[-1] / 100 - 1, "cagr": (values[-1] / 100) ** (1 / years) - 1,
            "max_drawdown": drawdown, "costs": result["fees_krw"] + result["exchange_cost_krw"],
            "turnover": sum(trade["amount_usd"] for trade in result["trades"]) / result["initial_value_usd"],
            "currency": "KRW"})
    update_state(history=history, simulation={"strategies": strategies,
        "assumptions": ["가상 초기자본 100만원, 원화 평가 · 일별 실제 USD/KRW 환율 반영 · 원금 100 기준 지수",
                        "2026-11-10 확인한 토스 일반 공시 매매·SEC 요율을 과거 전체에 가정 적용한 비교",
                        "야간 환전 스프레드50bp · 10달러 이하 매매수수료 면제 · 센트 절사/SEC 반올림 반영",
                        "다음 거래일 배당·분할 조정 시가 체결, 조정 단위 6자리 수량 가정 · 실제 과거 수수료 재현 아님",
                        "배당 재투자 조정 가격의 비교 · 세금·호가차이 미포함",
                        "추가 납입 0, 공통 기간에서 같은 시작일·종료일 사용",
                        "역사적 단순 정책 비교이며 AI 예측 기반 거래 수익이 아님"]})


def historical_policy(series, fx, monthly=False):
    from .paper import advance_account, create_account, fx_for_day, session_open
    days = common_dates(series)
    initial_fx = fx_for_day(fx, days[0])
    if not initial_fx:
        raise RuntimeError("과거 비교 시작일의 원/달러 환율이 없습니다")
    account = create_account(initial_fx, days[0], spread_bps=50)
    account["history"] = [{"date": days[0], "value_krw": account["current_balance_krw"], "fx_rate": initial_fx["rate"]}]
    adjusted = {symbol: [{**row, "raw_open": row["open"], "raw_close": row["close"], "dividend": 0, "split": 0}
                          for row in rows] for symbol, rows in series.items()}
    weights = {symbol: 1 / len(series) for symbol in series}
    signals = []
    for index, day in enumerate(days[:-1]):
        if index == 0 or (monthly and day[:7] != days[index + 1][:7]):
            # 정책이 정해진 과거 반사실 비교이며 실제 전진 신호 원장과 분리합니다.
            generated = session_open(day) + timedelta(hours=7)
            signals.append({"id": day, "as_of": day, "created_at": generated.isoformat(), "weights": weights})
    advance_account(account, adjusted, fx, signals)
    return account
