"""주문·새 정규장 관측·원가 원장을 분리한 자동 모의매매입니다."""

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import math
import threading

from .core import now_iso, read_json, save_json
from .fees import POLICY, affordable_quantity, convert_won, floor_quantity, trade_fees
from .runtime import PRIVATE, load_config, update_state
from .universe import DEFAULT_POLICY, SYMBOLS, UNIVERSE_HASH, universe


ACCOUNT_MUTEX = threading.RLock()
POLICY_NAME = "Chronos-2 전망과 비용을 비교한 ETF·현금 배분"


def timestamp(value):
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("시세와 주문 시각에는 시간대가 필요합니다")
    return parsed.astimezone(timezone.utc)


def select_allocation(forecasts, equity_usd, policy=None):
    policy = {**DEFAULT_POLICY, **(policy or {})}
    scores, reasons = {}, {}
    budget = max(1, float(equity_usd) * (1 - policy["cash_reserve"]) / policy["max_positions"])
    # 예상 왕복 일반 수수료와 최소 SEC를 보수적으로 비교합니다. 실제 비용은 체결 때 따로 계산합니다.
    cost = 0.002 + 0.01 / budget
    for row in forecasts:
        symbol = row["symbol"]
        predicted, previous = float(row["adjusted_price"]), float(row["last_adjusted_price"])
        if symbol not in SYMBOLS or not all(math.isfinite(v) and v > 0 for v in (predicted, previous)):
            raise ValueError("후보 범위와 유한한 양의 모델 전망을 확인해 주세요")
        change = predicted / previous - 1
        edge = change - cost
        scores[symbol] = max(0.0, edge) if edge >= policy["minimum_edge"] else 0.0
        reasons[symbol] = f"모델 1거래일 변화 전망 {change:.3%}, 예상 왕복비용 {cost:.3%}, 비용 차감 여유 {edge:.3%}"
    selected = sorted((s for s in scores if scores[s] > 0), key=lambda s: (-scores[s], s))[:policy["max_positions"]]
    total = sum(scores[s] for s in selected)
    weights = {s: min(policy["max_weight"], (1 - policy["cash_reserve"]) * scores[s] / total)
               if s in selected else 0.0 for s in scores}
    for symbol in reasons:
        reasons[symbol] += " · 배분 대상" if weights[symbol] else " · 비용 기준 미달 또는 상위 후보 외, 현금/기존 보유 축소"
    return {"weights": weights, "cash_weight": max(0.0, 1 - sum(weights.values())), "reasons": reasons,
            "name": POLICY_NAME, "description": "최대 3개 ETF, 종목당 최대45%, 현금 최소10%; 전망 부족 시 전액 현금",
            "max_positions": policy["max_positions"], "policy_version": 3, "universe_hash": UNIVERSE_HASH}


def migrate_account(account):
    # 기존 현금·체결·환전 원본은 보존하고 없는 표시 필드만 보강합니다.
    account.setdefault("orders", [])
    account.setdefault("trades", [])
    account.setdefault("cash_movements", [])
    account.setdefault("processed_signal_ids", [])
    account.setdefault("corporate_actions_through", {})
    account.setdefault("realized_pnl_krw", 0.0)
    account.setdefault("fees_usd", 0.0)
    account.setdefault("fees_krw", 0.0)
    account["schema_version"] = 3
    for position in account.get("positions", []):
        position.setdefault("name", SYMBOLS.get(position["symbol"], position["symbol"]))
        for key in ("average_cost_usd", "cost_basis_usd", "cost_basis_krw", "unrealized_pnl_krw", "unrealized_pnl_pct"):
            position.setdefault(key, None)
    for index, trade in enumerate(account["trades"]):
        trade.setdefault("id", "legacy-" + hashlib.sha256(f"{index}:{trade}".encode()).hexdigest()[:20])
        trade.setdefault("name", SYMBOLS.get(trade["symbol"], trade["symbol"]))
        trade.setdefault("executed_at", trade.get("execution_open_utc"))
        trade.setdefault("reason", None)
        trade.setdefault("realized_pnl_krw", None)
        trade.setdefault("assumption", None)
    receipt = account.get("fx_receipt")
    if receipt and not any(row.get("id") == "initial-fx" for row in account["cash_movements"]):
        account["cash_movements"].append({"id": "initial-fx", "type": "환전", "created_at": receipt.get("created_at"),
            "amount_krw": receipt.get("initial_krw", account.get("initial_capital_krw")),
            "amount_usd": receipt.get("received_usd", account.get("initial_value_usd")),
            "fee_krw": receipt.get("cost_krw"), "rate": receipt.get("rate", account.get("initial_fx")),
            "receipt": deepcopy(receipt)})
    return account


def fund_account(account, fx, spread_bps):
    migrate_account(account)
    if account.get("cash_usd") is not None or not fx:
        return account
    receipt = convert_won(account["cash_krw"], fx["rate"], spread_bps)
    account.update(cash_krw=0, cash_usd=receipt["received_usd"], initial_fx=fx["rate"],
                   initial_value_usd=receipt["received_usd"], exchange_cost_krw=receipt["cost_krw"],
                   fx_receipt={**receipt, "source": fx["source"], "as_of": fx["as_of"], "created_at": now_iso()})
    migrate_account(account)
    return account


def queue_orders(account, signal, prices):
    migrate_account(account)
    if signal.get("policy_version") != 3 or signal["id"] in account["processed_signal_ids"]:
        return account
    weights = signal["weights"]
    if any(s not in SYMBOLS or not math.isfinite(float(w)) or w < 0 for s, w in weights.items()) or sum(weights.values()) > 1 + 1e-9:
        raise ValueError("후보 밖 종목·공매도·현금 초과 배분은 허용하지 않습니다")
    for order in account["orders"]:
        if order["status"] == "pending":
            order.update(status="cancelled", cancelled_at=signal["created_at"], cancel_reason="새 3차 정책 신호로 미체결 주문을 대체했습니다")
    positions = {p["symbol"]: p for p in account["positions"]}
    wealth = account.get("cash_usd")
    if wealth is None:
        return account
    for symbol, position in positions.items():
        price = prices.get(symbol, position.get("price_usd"))
        if price is None or not math.isfinite(float(price)) or price <= 0:
            raise ValueError("기존 보유의 유효한 기준 가격이 없어 신규 배분을 대기합니다")
        wealth += position["quantity"] * price
    for symbol in sorted(set(weights) | set(positions)):
        price = prices.get(symbol)
        if price is None or not math.isfinite(float(price)) or price <= 0:
            continue
        existing = positions.get(symbol, {}).get("quantity", 0)
        target = wealth * weights.get(symbol, 0) / price
        difference = target - existing
        quantity = floor_quantity(abs(difference))
        if quantity <= 0:
            continue
        side = "매수" if difference > 0 else "매도"
        account["orders"].append({"id": f"{signal['id']}:{symbol}:{side}", "signal_id": signal["id"],
            "symbol": symbol, "name": SYMBOLS.get(symbol, symbol), "side": side, "quantity": quantity,
            "target_weight": weights.get(symbol, 0), "status": "pending", "created_at": signal["created_at"],
            "policy_version": 3, "universe_hash": signal.get("universe_hash"),
            "reason": signal.get("reasons", {}).get(symbol, "선택 대상에서 제외되어 보유를 줄입니다"),
            "waiting_reason": "주문 생성 뒤 시작된 새 정규장 완결 1분 관측을 기다립니다"})
    account["processed_signal_ids"].append(signal["id"])
    account["policy"] = {key: signal.get(key) for key in ("name", "description", "max_positions", "cash_weight", "policy_version", "universe_hash")}
    account["policy"].update(as_of=signal["created_at"], waiting_reason=None)
    return account


def quote_waiting_reason(quote, created_at, current, max_age=1200):
    if quote is None:
        return "해당 ETF의 무료 단기 시세를 확보하지 못했습니다"
    try:
        now, created = timestamp(current), timestamp(created_at)
        start, end, fetched = (timestamp(quote[k]) for k in ("bar_start", "bar_end", "fetched_at"))
        opening, closing = timestamp(quote["session_open"]), timestamp(quote["session_close"])
        if not math.isfinite(float(quote["price"])) or quote["price"] <= 0:
            return "유효한 양의 관측 가격이 없습니다"
        if quote.get("interval_seconds") not in (60, 300) or (end - start).total_seconds() != quote["interval_seconds"]:
            return "완결 1~5분봉이 아닙니다"
        if not quote.get("regular_session") or not opening <= start < end <= closing:
            return "정규장 관측이 아닙니다"
        if not opening <= now < closing:
            return "현재 미국 정규장이 열려 있지 않습니다"
        if start <= created:
            return "주문 생성 뒤 시작된 새 관측이 아직 없습니다"
        if end > now or fetched > now or fetched < end:
            return "미래 시각 또는 아직 완결되지 않은 관측입니다"
        if (now - end).total_seconds() > max_age or (now - fetched).total_seconds() > max_age:
            return "무료 관측이 허용 지연 20분을 넘었습니다"
    except (ValueError, TypeError, KeyError):
        return "시세 시간대·정규장·가격 메타데이터가 불완전합니다"
    return None


def fill_orders(account, quotes, fx, current=None, max_age=1200):
    migrate_account(account)
    current = current or now_iso()
    positions = {p["symbol"]: p for p in account["positions"] if p["quantity"] > 0}
    completed = {trade["id"] for trade in account["trades"]}
    for order in sorted(account["orders"], key=lambda row: row.get("side") != "매도"):
        if order["status"] != "pending":
            continue
        if order["id"] in completed:
            order["status"] = "filled"
            continue
        symbol = order["symbol"]
        quote = quotes.get(symbol)
        waiting = quote_waiting_reason(quote, order["created_at"], current, max_age)
        if not fx or account.get("cash_usd") is None:
            waiting = "실제 시장 환율·달러 현금 확보 전 체결 대기"
        elif not math.isfinite(float(fx["rate"])) or fx["rate"] <= 0 or fx.get("as_of", "9999") > current[:10]:
            waiting = "유효한 현재 또는 이전 관측 환율이 없습니다"
        if waiting:
            order["waiting_reason"] = waiting
            continue
        price, rate = float(quote["price"]), float(fx["rate"])
        existing = positions.get(symbol, {}).get("quantity", 0)
        side = "sell" if order["side"] == "매도" else "buy"
        quantity = floor_quantity(min(existing, order["quantity"])) if side == "sell" else affordable_quantity(account["cash_usd"], price, order["quantity"])
        if quantity <= 0:
            order.update(status="cancelled", cancelled_at=current, cancel_reason="매도 가능 보유 또는 비용 포함 가용 달러가 부족합니다")
            continue
        amount = quantity * price
        fees = trade_fees(amount, side)
        fee = fees["total_usd"]
        if side == "sell" and amount <= fee:
            order["waiting_reason"] = "매도 대금이 공시 최소 비용보다 작아 체결을 대기합니다"
            continue
        position = positions.get(symbol, {"symbol": symbol, "name": SYMBOLS.get(symbol, symbol), "quantity": 0,
                                           "cost_basis_usd": 0.0, "cost_basis_krw": 0.0})
        old_cost, old_krw = position.get("cost_basis_usd"), position.get("cost_basis_krw")
        realized = None
        if side == "buy":
            new_cost = old_cost + amount + fee if old_cost is not None else None
            new_krw = old_krw + (amount + fee) * rate if old_krw is not None else None
            new_quantity = existing + quantity
        else:
            fraction = quantity / existing
            released = old_krw * fraction if old_krw is not None else None
            realized = (amount - fee) * rate - released if released is not None else None
            new_cost = old_cost * (1 - fraction) if old_cost is not None else None
            new_krw = old_krw * (1 - fraction) if old_krw is not None else None
            new_quantity = max(0, existing - quantity)
        account["cash_usd"] += amount - fee if side == "sell" else -amount - fee
        if account["cash_usd"] < -1e-8:
            raise ArithmeticError("비용을 포함한 모의 주문이 가용 현금을 초과했습니다")
        account["cash_usd"] = max(0.0, account["cash_usd"])
        if new_quantity > 1e-9:
            position.update(quantity=new_quantity, price_usd=price, cost_basis_usd=new_cost, cost_basis_krw=new_krw,
                            average_cost_usd=new_cost / new_quantity if new_cost is not None else None)
            positions[symbol] = position
        else:
            positions.pop(symbol, None)
        trade = {"id": order["id"], "order_id": order["id"], "symbol": symbol, "name": SYMBOLS.get(symbol, symbol),
                 "side": order["side"], "executed_at": quote["bar_end"], "recorded_at": current,
                 "quantity": quantity, "price_usd": price, "amount_usd": amount, "amount_krw": amount * rate,
                 "fee_usd": fee, "fee_krw": fee * rate, **fees, "fx_rate": rate, "fx_as_of": fx["as_of"],
                 "reason": order["reason"], "realized_pnl_krw": realized, "signal_created_at": order["created_at"],
                 "quote": deepcopy(quote), "execution_date": quote["bar_end"][:10],
                 "assumption": "무료 지연 정규장 완결 1분 종가 가상체결; 실제 토스 호가·주문·체결 아님, 소수점 6자리 가정"}
        account["trades"].append(trade)
        account["fees_usd"] += fee
        account["fees_krw"] += fee * rate
        if realized is not None:
            account["realized_pnl_krw"] += realized
        order.update(status="filled", executed_at=quote["bar_end"], filled_quantity=quantity, waiting_reason=None)
    account["positions"] = list(positions.values())
    return account


def apply_corporate_actions(account, series):
    migrate_account(account)
    positions = {p["symbol"]: p for p in account["positions"]}
    through = account["corporate_actions_through"]
    for symbol, rows in series.items():
        previous = through.get(symbol, account.get("last_processed", account.get("as_of", "9999")))
        position = positions.get(symbol)
        for row in sorted(rows, key=lambda r: r["date"]):
            if row["date"] <= previous or position is None:
                continue
            split = row.get("split", 0)
            if split > 0:
                position["quantity"] *= split
                if position.get("cost_basis_usd") is not None:
                    position["average_cost_usd"] = position["cost_basis_usd"] / position["quantity"]
            if account.get("cash_usd") is not None:
                account["cash_usd"] += position["quantity"] * row.get("dividend", 0)
        if rows:
            through[symbol] = max(previous, max(row["date"] for row in rows))
    return account


def account_view(account):
    public = migrate_account(deepcopy(account))
    public["trade_count"] = len(public["trades"])
    public["trades"] = list(reversed(public["trades"][-100:]))
    public["orders"] = list(reversed(public["orders"][-100:]))
    public["universe"] = universe()
    public.setdefault("policy", {"name": POLICY_NAME, "description": "동일 latest 모델에서 전망한 최대3개 ETF와 현금", "max_positions": 3,
                                  "cash_weight": None, "as_of": None, "waiting_reason": "새 3차 모델 전망 생성 대기"})
    if public.get("fx"):
        public["fx"] = {k: v for k, v in public["fx"].items() if k != "rates"}
    for key in ("processed_signal_ids", "corporate_actions_through", "last_processed", "last_signal_id"):
        public.pop(key, None)
    return public


def trade_page(account, offset=0, limit=50, side="all"):
    if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 100 or side not in ("all", "buy", "sell"):
        raise ValueError("offset은 0이상, limit은 1~100, side는 all/buy/sell이어야 합니다")
    normalized = migrate_account(deepcopy(account))
    wanted = {"buy": "매수", "sell": "매도"}.get(side)
    trades = [row for row in reversed(normalized["trades"]) if wanted is None or row["side"] == wanted]
    return {"total": len(trades), "offset": offset, "limit": limit, "trades": trades[offset:offset + limit], "as_of": now_iso()}


def reconcile(series, quotes=None):
    from .paper import ACCOUNT, FX, create_account, mark_account
    config = load_config()
    with ACCOUNT_MUTEX:
        fx = read_json(FX)
        account = read_json(ACCOUNT)
        if account is None:
            latest = min((rows[-1]["date"] for rows in series.values() if rows), default=now_iso()[:10])
            account = create_account(fx, latest)
        from .fees import fx_spread_bps
        fund_account(account, fx, fx_spread_bps())
        migrate_account(account)
        apply_corporate_actions(account, series)
        prices = {s: rows[-1].get("raw_close", rows[-1]["close"]) for s, rows in series.items() if rows}
        signals = read_json(PRIVATE / "모의신호.json", [])
        eligible = [row for row in signals if row.get("policy_version") == 3 and row.get("universe_hash") == UNIVERSE_HASH]
        if eligible:
            queue_orders(account, max(eligible, key=lambda r: timestamp(r["created_at"])), prices)
        snapshot = read_json(PRIVATE / "단기시세.json", {})
        quotes = quotes if quotes is not None else snapshot.get("quotes", {})
        fill_orders(account, quotes, fx, max_age=config["autotrading"]["quote_max_age_seconds"])
        marks = {s: rows[-1] for s, rows in series.items() if rows}
        for symbol, quote in quotes.items():
            try:
                daily_date = marks.get(symbol, {}).get("date", "0000")
                if timestamp(quote["bar_end"]) <= timestamp(now_iso()) and timestamp(quote["fetched_at"]) <= timestamp(now_iso()) and quote["price"] > 0 and quote["bar_end"][:10] >= daily_date:
                    marks[symbol] = {"raw_close": quote["price"], "date": quote["bar_end"][:10],
                                     "price_as_of": quote["bar_end"], "price_source": quote["source"]}
            except (KeyError, ValueError, TypeError):
                continue
        mark_account(account, marks, fx)
        for position in account["positions"]:
            cost = position.get("cost_basis_krw")
            position["unrealized_pnl_krw"] = position["value_krw"] - cost if cost is not None else None
            position["unrealized_pnl_pct"] = position["value_krw"] / cost - 1 if cost and cost > 0 else None
            mark = marks.get(position["symbol"], {})
            position["price_as_of"] = mark.get("price_as_of", mark.get("date"))
            position["price_source"] = mark.get("price_source", "Yahoo Finance 완결 일봉 종가")
        pending = [row for row in account["orders"] if row["status"] == "pending"]
        account["status"] = "waiting_quote" if pending else "running" if account["trades"] else "ready"
        account["message"] = pending[0].get("waiting_reason", "새 관측 대기") if pending else "현재 배분 유지 · 다음 모델 전망을 기다립니다"
        account["as_of"] = now_iso()
        account["quote_status"] = snapshot.get("status", "pending")
        account["fee_policy"] = deepcopy(POLICY)
        if config.get("fee_confirmation", {}).get("commission_rate") == 0.001:
            account["fee_policy"]["account_confirmation"] = "사용자가 기본0.1% 적용 확인 · 실제 계좌 API 미연결"
            account["fee_policy"]["limitations"] = [v for v in POLICY["limitations"] if "이벤트" not in v]
        account["assumptions"] = ["무료 지연 1분봉, 주문 이후 시작된 완결 정규장 관측만 체결; 최대 지연20분",
            "시장 USD/KRW는 실제 토스 은행 환율 아님; 공시 기본0.1%·소액면제·SEC·환전 원장 유지",
            "지정15개 비레버리지 ETF 후보이며 전체 ETF·토스 실제 지원 종목 검증 아님",
            "소수점6자리·전액 현금 증거금 가정, 세금·실제 호가차이·부분체결·현금결제기간 미반영",
            "배당은 지급일 대신 완결 일봉 관측일 현금 반영; 원가 미상은 추정하지 않음"]
        save_json(ACCOUNT, account)
        update_state(paper=account_view(account), holdings=[], portfolio={"total_value": account["current_balance_krw"],
            "currency": "KRW", "as_of": account["as_of"], "pnl": account["pnl_krw"]})
        return account
