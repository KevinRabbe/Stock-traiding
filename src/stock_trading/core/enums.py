from enum import StrEnum


class Source(StrEnum):
    SEC_EDGAR = "sec_edgar"
    SEC_QUARTERLY = "sec_quarterly"
    TIINGO = "tiingo"
    USASPENDING = "usaspending"
    LDA = "lda"
    EXCHANGERATE_HOST = "exchangerate_host"
    CONGRESS = "congress"


class EventType(StrEnum):
    MARKET_BAR = "market_bar"
    INSIDER_TRANSACTION = "insider_transaction"
    GOVERNMENT_CONTRACT = "government_contract"
    LOBBYING_ACTIVITY = "lobbying_activity"
    FX_RATE = "fx_rate"
    CONGRESS_TRANSACTION = "congress_transaction"
    CORPORATE_ACTION = "corporate_action"


class TradeDirection(StrEnum):
    BUY = "buy"
    SELL = "sell"
    ACQUIRE = "acquire"
    DISPOSE = "dispose"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"


class SemanticDirection(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    MIXED = "mixed"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"
