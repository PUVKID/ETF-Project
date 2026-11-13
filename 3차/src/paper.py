"""원화 모의계좌 공통 계산과 과거 일봉 비교입니다. 실제 운용 체결은 새 완결 1분봉 경로로 위임합니다."""

from datetime import datetime, time, timezone
import math
from zoneinfo import ZoneInfo

from .core import common_dates, now_iso, read_json, save_json
from .runtime import PRIVATE, load_config, update_state
from .fees import POLICY, affordable_quantity, convert_won, floor_quantity, fx_spread_bps, trade_fees


ACCOUNT = PRIVATE / "모의계좌.json"
FX = PRIVATE / "환율.json"


def validate_weights(weights):
    if not weights or any(not math.isfinite(float(value)) or value < 0 for value in weights.values()):
        raise ValueError("공매도나 유효하지 않은 배분은 허용하지 않습니다")
    if abs(sum(weights.values()) - 1) > 1e-8:
        raise ValueError("레버리지 없이 배분 합계가 1이어야 합니다")


def session_open(day):
    # 미국 현지 09:30의 서머타임을 반영하며 고정 UTC 시각을 쓰지 않습니다.
    return datetime.combine(datetime.fromisoformat(day).date(), time(9, 30), ZoneInfo("America/New_York")).astimezone(timezone.utc)


def create_account(fx, as_of, spread_bps=None):
    spread_bps = fx_spread_bps() if spread_bps is None else spread_bps
    if not 0 <= spread_bps <= 1000:
        raise ValueError("환전 스프레드 가정 범위를 확인해 주세요")
    account = {"status": "waiting_fx", "initial_capital_krw": 1000000,
               "current_balance_krw": 1000000, "cash_krw": 1000000, "cash_usd": None,
               "cash_usd_value_krw": None, "positions": [], "pnl_krw": 0, "pnl_pct": 0,
               "fees_krw": 0.0, "fees_usd": 0.0, "exchange_cost_krw": 0.0,
               "initial_fx": None, "initial_value_usd": None, "fx": fx,
               "fx_receipt": None, "trades": [], "last_processed": as_of,
               "last_signal_id": None, "as_of": as_of, "created_at": now_iso(),
               "message": "실제 시장 환율 확보 전 환산·체결 대기", "history": []}
    if fx:
        rate = float(fx["rate"])
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError("유효한 원/달러 환율이 필요합니다")
        receipt = convert_won(1000000, rate, spread_bps)
        cost = receipt["cost_krw"]
        dollars = receipt["received_usd"]
        account.update(status="ready", cash_krw=0, cash_usd=dollars,
                       cash_usd_value_krw=dollars * rate, current_balance_krw=dollars * rate,
                       pnl_krw=-cost, pnl_pct=-cost / 1000000, initial_fx=rate,
                       initial_value_usd=dollars, exchange_cost_krw=cost,
                       message="가상 환전 완료 · 신호 생성 뒤의 새 거래일 시가를 기다립니다",
                       fx_receipt={**receipt,
                                   "source": fx["source"], "as_of": fx["as_of"], "created_at": now_iso()})
    return account


def mark_account(account, latest_rows, fx):
    if not fx or account["cash_usd"] is None:
        return account
    rate = float(fx["rate"])
    total_usd = account["cash_usd"]
    for position in account["positions"]:
        row = latest_rows.get(position["symbol"])
        price = row["raw_close"] if row else position["price_usd"]
        position.update(price_usd=price, value_usd=position["quantity"] * price,
                        price_krw=price * rate, value_krw=position["quantity"] * price * rate)
        total_usd += position["value_usd"]
    value = account["cash_krw"] + total_usd * rate
    account.update(current_balance_krw=value, pnl_krw=value - account["initial_capital_krw"],
                   pnl_pct=value / account["initial_capital_krw"] - 1, cash_usd_value_krw=account["cash_usd"] * rate,
                   fx=fx, valuation_at=now_iso())
    return account


def advance_account(account, series, fx, signals):
    if not fx or account["cash_usd"] is None:
        return account
    dates = common_dates(series)
    lookup = {symbol: {row["date"]: row for row in rows} for symbol, rows in series.items()}
    positions = {position["symbol"]: dict(position) for position in account["positions"]}
    for day in dates:
        if day <= account["last_processed"]:
            continue
        opening = session_open(day)
        day_fx = fx_for_day(fx, day)
        if not day_fx:
            break
        # 실제 생성시각보다 앞선 시가에 소급 체결하지 않습니다.
        available = [signal for signal in signals if signal["as_of"] < day
                     and datetime.fromisoformat(signal["created_at"]) < opening]
        eligible = max(available, key=lambda signal: signal["created_at"]) if available else None
        for symbol, position in positions.items():
            row = lookup[symbol][day]
            split = row.get("split", 0)
            if split > 0:
                position["quantity"] *= split
            account["cash_usd"] += position["quantity"] * row.get("dividend", 0)
        if eligible and eligible.get("id", eligible["created_at"]) != account["last_signal_id"]:
            weights = eligible["weights"]
            validate_weights(weights)
            if set(weights) != set(series):
                raise ValueError("신호와 시세의 종목이 일치하지 않습니다")
            open_prices = {symbol: lookup[symbol][day].get("raw_open", lookup[symbol][day]["open"]) for symbol in series}
            old_values = {symbol: positions.get(symbol, {}).get("quantity", 0) * open_prices[symbol] for symbol in series}
            wealth = account["cash_usd"] + sum(old_values.values())
            # 매도 후 가용 달러 현금 안에서 매수하며 수수료·소수점 수량을 함께 계산합니다.
            for side in ("sell", "buy"):
                for symbol in sorted(series):
                    price = open_prices[symbol]
                    existing = positions.get(symbol, {}).get("quantity", 0)
                    difference = wealth * weights[symbol] / price - existing
                    if (side == "sell" and difference >= 0) or (side == "buy" and difference <= 0):
                        continue
                    quantity = floor_quantity(min(existing, -difference)) if side == "sell" else affordable_quantity(account["cash_usd"], price, difference)
                    if quantity <= 0:
                        continue
                    amount = quantity * price
                    fee_parts = trade_fees(amount, side)
                    fee = fee_parts["total_usd"]
                    if side == "sell" and amount <= fee:
                        continue
                    account["cash_usd"] += amount - fee if side == "sell" else -amount - fee
                    if account["cash_usd"] < -1e-8:
                        raise ArithmeticError("모의 주문이 가용 달러 현금을 초과했습니다")
                    account["cash_usd"] = max(0.0, account["cash_usd"])
                    new_quantity = existing - quantity if side == "sell" else existing + quantity
                    positions[symbol] = {"symbol": symbol, "quantity": new_quantity, "price_usd": price}
                    account["trades"].append({"symbol": symbol, "side": "매수" if side == "buy" else "매도",
                        "quantity": quantity, "price_usd": price, "amount_usd": amount,
                        "amount_krw": amount * day_fx["rate"], "fee_usd": fee, "fee_krw": fee * day_fx["rate"],
                        **fee_parts, "fx_rate": day_fx["rate"], "signal_date": eligible["as_of"],
                        "signal_created_at": eligible["created_at"], "execution_date": day,
                        "execution_open_utc": opening.isoformat(),
                        "assumption": "실제 일봉 시가의 가상체결 · 일반 공시 요율 적용 · 실제 토스 체결 아님"})
                    account["fees_usd"] += fee
                    account["fees_krw"] += fee * day_fx["rate"]
            account["last_signal_id"] = eligible.get("id", eligible["created_at"])
        account["positions"] = list(positions.values())
        account["last_processed"] = day
        account["as_of"] = day
        mark_account(account, {symbol: lookup[symbol][day] for symbol in series}, day_fx)
        account["history"].append({"date": day, "value_krw": account["current_balance_krw"], "fx_rate": day_fx["rate"]})
    if dates:
        mark_account(account, {symbol: lookup[symbol][dates[-1]] for symbol in series}, fx)
    account["status"] = "running" if account["trades"] else "ready"
    account["message"] = "모델 신호의 다음 거래일 모의체결 기록" if account["trades"] else "새 신호 생성 이후의 거래일 시가를 기다립니다"
    return account


def reconcile_paper(series):
    # 실제 운용은 주문 이후 완결된 단기 관측 경로를 사용합니다. advance_account는 과거 정책 비교용입니다.
    from .autotrading import reconcile
    return reconcile(series)


def fx_for_day(fx, day):
    rates = [row for row in fx.get("rates", []) if row["date"] <= day]
    if rates:
        row = max(rates, key=lambda value: value["date"])
        return {"rate": row["rate"], "as_of": row["date"], "source": fx["source"],
                "fetched_at": fx.get("fetched_at")}
    if fx.get("as_of", "9999") <= day:
        return {key: value for key, value in fx.items() if key != "rates"}
    return None


