"""무료 정규장 완결 1분 관측을 학습과 별개로 수집합니다."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import json
import math
import threading
import traceback
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from .autotrading import timestamp
from .core import now_iso, read_json, save_json
from .runtime import PRIVATE, STOP, load_config, log, record_failure


QUOTES = PRIVATE / "단기시세.json"


def fetch_intraday(symbol):
    query = urlencode({"range": "1d", "interval": "1m", "includePrePost": "false"})
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{quote(symbol, safe='')}?{query}"
    with urlopen(Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=12) as response:
        payload = json.load(response)
    result = payload.get("chart", {}).get("result")
    if payload.get("chart", {}).get("error") or not result:
        raise RuntimeError(f"{symbol}: 무료 1분봉을 확보하지 못했습니다: {payload.get('chart', {}).get('error')}")
    return result[0], url


def parse_quote(symbol, chart, fetched_at, source_url):
    meta = chart.get("meta", {})
    if meta.get("symbol") != symbol or meta.get("currency") != "USD" or meta.get("instrumentType") != "ETF" or meta.get("dataGranularity") != "1m":
        raise ValueError("요청 ETF·달러 통화·1분 간격 메타데이터가 일치하지 않습니다")
    periods = meta.get("tradingPeriods", [])
    if isinstance(periods, dict):
        periods = periods.get("regular", [])
    def flatten(value):
        if isinstance(value, dict) and "start" in value and "end" in value:
            return [value]
        return [p for part in value for p in flatten(part)] if isinstance(value, list) else []
    regular = flatten(periods)
    if not regular:
        fallback = meta.get("currentTradingPeriod", {}).get("regular")
        regular = [fallback] if fallback and "start" in fallback and "end" in fallback else []
    if not regular:
        raise ValueError("공급자의 정규장 거래시간을 확인할 수 없습니다")
    fetched = timestamp(fetched_at)
    indicators = chart.get("indicators", {}).get("quote", [])
    if not indicators:
        raise ValueError("무료 1분봉 가격이 없습니다")
    candidates = []
    for index, stamp in enumerate(chart.get("timestamp", [])):
        values = indicators[0]
        price = values.get("close", [])[index]
        volume = values.get("volume", [])[index]
        if price is None or volume is None or volume <= 0 or not math.isfinite(float(price)) or price <= 0:
            continue
        start = datetime.fromtimestamp(stamp, timezone.utc)
        end = start + timedelta(minutes=1)
        session = next((p for p in regular if p["start"] <= stamp and stamp + 60 <= p["end"]), None)
        if session is None or end > fetched:
            continue
        candidates.append({"symbol": symbol, "price": float(price), "bar_start": start.isoformat(), "bar_end": end.isoformat(),
            "fetched_at": fetched_at, "session_open": datetime.fromtimestamp(session["start"], timezone.utc).isoformat(),
            "session_close": datetime.fromtimestamp(session["end"], timezone.utc).isoformat(), "regular_session": True,
            "interval_seconds": 60, "source": "Yahoo Finance chart 무료 1분봉", "url": source_url,
            "provider_delay_seconds": meta.get("exchangeDataDelayedBy"),
            "observed_age_seconds": (fetched - end).total_seconds(), "volume": volume,
            "note": "무료 지연 정보; 실제 토스 체결 호가나 모든 거래소 최우선 호가가 아닙니다"})
    if not candidates:
        raise ValueError("공급자 정규장 범위에 완결된 거래 관측이 없습니다")
    return max(candidates, key=lambda row: timestamp(row["bar_end"]))


def collect_quotes(symbols, fetched_at=None, fetcher=None):
    fetcher = fetcher or fetch_intraday
    quotes, errors, details = {}, {}, {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(fetcher, symbol): symbol for symbol in symbols}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                chart, url = future.result()
                observed = fetched_at or now_iso()
                quotes[symbol] = parse_quote(symbol, chart, observed, url)
            except Exception as error:
                errors[symbol] = str(error)
                details[symbol] = traceback.format_exc()
    return {"quotes": quotes, "errors": errors, "error_details": details, "fetched_at": fetched_at or now_iso(),
            "status": "ready" if not errors else "partial" if quotes else "unavailable"}


def monitor_loop(finished):
    errors = 0
    while not finished.is_set() and not STOP.exists():
        try:
            config = load_config()
            def stoppable_fetch(symbol):
                if finished.is_set() or STOP.exists():
                    raise InterruptedError("시세 수집 중 중지 요청")
                return fetch_intraday(symbol)
            snapshot = collect_quotes(config["symbols"], fetcher=stoppable_fetch)
            if finished.is_set() or STOP.exists():
                return
            # 실패 ETF는 개별 대기하며 이전 캐시를 새 시세로 재표시하지 않습니다.
            for symbol, message in snapshot["errors"].items():
                record_failure(RuntimeError(f"무료 단기시세 {symbol}: {message}\n{snapshot['error_details'][symbol]}"))
            save_json(QUOTES, snapshot)
            from .market import load_market
            from .paper import reconcile_paper
            reconcile_paper(load_market()["series"])
            errors = errors + 1 if not snapshot["quotes"] else 0
            delay = min(900, config["autotrading"]["quote_poll_seconds"] * 2 ** min(errors, 4))
        except Exception as error:
            record_failure(error)
            log(f"단기 시세·모의주문 갱신 실패: {error}", "error")
            errors += 1
            delay = min(900, 60 * 2 ** min(errors, 4))
        # 이 대기는 시세 호출 간격이며 모델 학습은 별도 주 스레드에서 계속 실행됩니다.
        if finished.wait(delay):
            return


@contextmanager
def market_monitor(enabled):
    finished = threading.Event()
    thread = threading.Thread(target=monitor_loop, args=(finished,), daemon=True) if enabled else None
    if thread:
        thread.start()
    try:
        yield
    finally:
        finished.set()
        if thread:
            thread.join(20)
