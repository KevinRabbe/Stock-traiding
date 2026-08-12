from stock_trading.core import TradeDirection


TRANSACTION_CODE_MEANINGS: dict[str, str] = {
    "A": "grant_or_award",
    "C": "conversion",
    "D": "disposition_to_issuer",
    "E": "short_derivative_expiration",
    "F": "exercise_price_or_tax_withholding",
    "G": "gift",
    "H": "long_derivative_expiration",
    "I": "rule_16b3_discretionary_transaction",
    "J": "other",
    "L": "small_acquisition",
    "M": "derivative_exercise_or_conversion",
    "O": "out_of_money_derivative_exercise",
    "P": "open_market_or_private_purchase",
    "S": "open_market_or_private_sale",
    "U": "change_of_control_tender",
    "W": "will_or_descent",
    "X": "in_or_at_money_derivative_exercise",
    "Z": "voting_trust",
}


def classify_direction(transaction_code: str | None, acquired_disposed: str | None) -> TradeDirection:
    code = (transaction_code or "").upper()
    acquired_disposed = (acquired_disposed or "").upper()

    if code == "P":
        return TradeDirection.BUY
    if code == "S":
        return TradeDirection.SELL
    if acquired_disposed == "A":
        return TradeDirection.ACQUIRE
    if acquired_disposed == "D":
        return TradeDirection.DISPOSE
    return TradeDirection.UNKNOWN


def classify_intent(transaction_code: str | None, acquired_disposed: str | None) -> str:
    code = (transaction_code or "").upper()

    if code == "P":
        return "DISCRETIONARY_BUY"
    if code == "S":
        return "DISCRETIONARY_SELL"
    if code == "A":
        return "COMPENSATION"
    if code == "F":
        return "TAX_OR_EXERCISE_PAYMENT"
    if code == "G":
        return "GIFT"
    if code in {"C", "M", "O", "X"}:
        return "DERIVATIVE_EXERCISE_OR_CONVERSION"
    if code == "I":
        return "RULE_16B3_DISCRETIONARY"
    if code in TRANSACTION_CODE_MEANINGS:
        return TRANSACTION_CODE_MEANINGS[code].upper()

    direction = classify_direction(code, acquired_disposed)
    return direction.value.upper()
