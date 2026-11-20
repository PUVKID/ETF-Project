"""공식 운용사 자료로 범위를 정한 모의운용 후보와 정책 기본값입니다."""

import hashlib
import json


SYMBOLS = {
    "QQQM": "Invesco NASDAQ 100 ETF", "SCHG": "Schwab US Large-Cap Growth ETF",
    "SPYM": "State Street SPDR Portfolio S&P 500 ETF", "VTI": "Vanguard Total Stock Market ETF",
    "VEA": "Vanguard FTSE Developed Markets ETF", "VWO": "Vanguard FTSE Emerging Markets ETF",
    "BND": "Vanguard Total Bond Market ETF", "IEF": "iShares 7-10 Year Treasury Bond ETF",
    "TLT": "iShares 20+ Year Treasury Bond ETF", "SGOV": "iShares 0-3 Month Treasury Bond ETF",
    "GLD": "SPDR Gold Shares", "XLK": "Technology Select Sector SPDR Fund",
    "XLF": "Financial Select Sector SPDR Fund", "XLV": "Health Care Select Sector SPDR Fund",
    "XLE": "Energy Select Sector SPDR Fund",
}
UNIVERSE_HASH = hashlib.sha256(json.dumps(sorted(SYMBOLS)).encode()).hexdigest()
DEFAULT_POLICY = {"enabled": True, "policy_version": 3, "max_positions": 3,
                  "cash_reserve": 0.1, "max_weight": 0.45, "minimum_edge": 0.0025,
                  "quote_poll_seconds": 60, "quote_max_age_seconds": 1200,
                  "universe_hash": UNIVERSE_HASH}


def universe():
    return [{"symbol": symbol, "name": name} for symbol, name in SYMBOLS.items()]
