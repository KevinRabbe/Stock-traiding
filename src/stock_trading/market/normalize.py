import json
from datetime import date
from decimal import Decimal

from stock_trading.core import RawRecord, Source

from .models import MarketBar, SecurityMapping
from .tiingo import normalize_tiingo_ticker


class TiingoNormalizer:
    def parse_metadata(self, raw: RawRecord, *, company_id: str) -> SecurityMapping:
        self._require_tiingo_json(raw)
        payload = json.loads(self._text(raw))
        ticker = normalize_tiingo_ticker(str(payload["ticker"]))
        start_date = payload.get("startDate")
        if not start_date:
            raise ValueError("Tiingo metadata has no startDate")
        end_date = payload.get("endDate")
        return SecurityMapping(
            company_id=company_id,
            ticker=ticker,
            exchange_code=payload.get("exchangeCode") or None,
            valid_from=date.fromisoformat(str(start_date)[:10]),
            valid_to=date.fromisoformat(str(end_date)[:10]) if end_date else None,
        )

    def parse_prices(
        self,
        raw: RawRecord,
        *,
        company_id: str,
        ticker: str,
    ) -> tuple[MarketBar, ...]:
        self._require_tiingo_json(raw)
        normalized_ticker = normalize_tiingo_ticker(ticker)
        payload = json.loads(self._text(raw))
        if not isinstance(payload, list):
            raise ValueError("Tiingo prices response must be a JSON list")

        bars: list[MarketBar] = []
        for row in payload:
            bars.append(
                MarketBar(
                    company_id=company_id,
                    ticker=normalized_ticker,
                    date=date.fromisoformat(str(row["date"])[:10]),
                    open=self._decimal(row["open"]),
                    high=self._decimal(row["high"]),
                    low=self._decimal(row["low"]),
                    close=self._decimal(row["close"]),
                    volume=self._decimal(row["volume"]),
                    adj_open=self._decimal(row["adjOpen"]),
                    adj_high=self._decimal(row["adjHigh"]),
                    adj_low=self._decimal(row["adjLow"]),
                    adj_close=self._decimal(row["adjClose"]),
                    adj_volume=self._decimal(row["adjVolume"]),
                    dividend_cash=self._decimal(row.get("divCash", 0)),
                    split_factor=self._decimal(row.get("splitFactor", 1)),
                )
            )
        return tuple(bars)

    @staticmethod
    def _require_tiingo_json(raw: RawRecord) -> None:
        if raw.source is not Source.TIINGO:
            raise ValueError("Tiingo normalizer requires Source.TIINGO")
        if raw.content_type != "application/json":
            raise ValueError("Tiingo normalizer requires application/json")

    @staticmethod
    def _text(raw: RawRecord) -> str:
        if isinstance(raw.content, bytes):
            return raw.content.decode("utf-8")
        return raw.content

    @staticmethod
    def _decimal(value) -> Decimal:
        return Decimal(str(value))
