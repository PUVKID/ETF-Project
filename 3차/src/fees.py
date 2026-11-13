"""2026-11-10 확인한 토스증권 일반 공시 요율의 모의 계산입니다.

실제 계좌 우대·체결 호가·은행 매매기준율을 인증해 확인한 것은 아닙니다.
"""

from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP
import math
from zoneinfo import ZoneInfo


POLICY = {"checked_at": "2026-11-10", "broker": "토스증권 일반 공시 기준",
          "commission_rate": 0.001, "commission_exempt_notional_usd": 10,
          "sec_sell_rate": 0.0000206, "sec_minimum_usd": 0.01,
          "fx_regular_spread_bps": 5, "fx_other_spread_bps": 50,
          "taf_usd": 0, "taf_note": "2026년 4분기 TAF 휴지 및 토스 고객 전가 근거 없음. 임의 부과하지 않음",
          "policy_review_after": "2026-12-31",
          "sources": ["https://home.tossinvest.com/ko/open-api",
                      "https://corp.tossinvest.com/ko/business?tab=commission",
                      "https://corp.tossinvest.com/ko/faq?page=1&searchTitleKeyword=%EC%88%98%EC%88%98%EB%A3%8C&tab=101",
                      "https://www.sec.gov/files/rules/sro/finra/2026/34-106409.pdf"],
          "limitations": ["계좌별 수수료 이벤트·우대 미확인", "무료 시장 환율은 토스 실제 은행 호가가 아님",
                          "한국 외환영업일 미확인 시 낮 시간도 50bp 보수 가정", "세금은 포함하지 않음",
                          "2027년 요율·TAF 고객 적용은 재확인 필요"]}


def fx_spread_bps(timestamp=None, business_day=None):
    moment = datetime.fromisoformat(timestamp) if isinstance(timestamp, str) else timestamp
    moment = (moment or datetime.now(timezone.utc)).astimezone(ZoneInfo("Asia/Seoul"))
    minutes = moment.hour * 60 + moment.minute
    if business_day is True and moment.weekday() < 5 and 9 * 60 <= minutes < 15 * 60 + 30:
        return 5
    return 50


def convert_won(won, market_rate, spread_bps):
    if any(not math.isfinite(float(value)) for value in (won, market_rate, spread_bps)) or won < 0 or market_rate <= 0 or not 0 <= spread_bps <= 1000:
        raise ValueError("환전 금액·시장 환율·스프레드가 유효하지 않습니다")
    offered = market_rate * (1 + spread_bps / 10000)
    dollars = won / offered
    return {"initial_krw": won, "rate": market_rate, "applied_rate": offered, "spread_bps": spread_bps,
            "received_usd": dollars, "cost_krw": won - dollars * market_rate,
            "basis": "토스 공시 스프레드 + 무료 시장 환율을 적용한 가정 · 토스 실제 은행 호가 아님"}


def trade_fees(notional, side):
    if not math.isfinite(float(notional)) or notional < 0 or side not in ("buy", "sell"):
        raise ValueError("매매 금액과 방향이 유효하지 않습니다")
    amount = Decimal(str(notional))
    # 공시의 매매 수수료는 센트 미만 절사, SEC는 센트 단위 반올림입니다.
    commission = Decimal(0) if amount <= Decimal("10") else amount * Decimal("0.001")
    commission = commission.quantize(Decimal("0.01"), rounding=ROUND_DOWN)
    sec = max(Decimal("0.01"), amount * Decimal("0.0000206")) if side == "sell" and amount > 0 else Decimal(0)
    sec = sec.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return {"commission_usd": float(commission), "sec_usd": float(sec), "taf_usd": 0.0,
            "total_usd": float(commission + sec)}


def floor_quantity(quantity):
    return float(Decimal(str(max(0.0, quantity))).quantize(Decimal("0.000001"), rounding=ROUND_DOWN))


def affordable_quantity(cash_usd, price_usd, max_quantity):
    if not all(math.isfinite(float(value)) and value >= 0 for value in (cash_usd, price_usd, max_quantity)) or price_usd == 0:
        raise ValueError("주문 가능 현금·가격·수량이 유효하지 않습니다")
    low = 0
    high = int(Decimal(str(max_quantity)) * 1000000)
    while low < high:
        middle = (low + high + 1) // 2
        quantity = middle / 1000000
        amount = quantity * price_usd
        if amount + trade_fees(amount, "buy")["total_usd"] <= cash_usd:
            low = middle
        else:
            high = middle - 1
    return low / 1000000
