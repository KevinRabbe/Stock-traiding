import json
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, DecimalException

from pydantic import ValidationError

from stock_trading.core import RawRecord, Source

from .models import MarketBar, TiingoMetadata
from .tiingo import normalize_tiingo_ticker


@dataclass(frozen=True, slots=True)
class TiingoPriceRowIssue:
    row_index: int
    date: str | None
    reason: str


class TiingoNormalizer:
    def parse_metadata(self, raw: RawRecord) -> TiingoMetadata:
        self._require_tiingo_json(raw)
        payload = json.loads(self._text(raw))
        ticker = normalize_tiingo_ticker(str(payload["ticker"]))
        name = str(payload.get("name") or "").strip()
        start_date = payload.get("startDate")
        if not name:
            raise ValueError("Tiingo metadata has no company name")
        if not start_date:
            raise ValueError("Tiingo metadata has no startDate")
        end_date = payload.get("endDate")
        return TiingoMetadata(
            ticker=ticker,
            name=name,
            exchange_code=payload.get("exchangeCode") or None,
            start_date=date.fromisoformat(str(start_date)[:10]),
            end_date=date.fromisoformat(str(end_date)[:10]) if end_date else None,
        )

    def parse_prices(
        self,
        raw: RawRecord,
        *,
        security_id: str | None = None,
        ticker: str,
        company_id: str | None = None,
        invalid_rows: list[TiingoPriceRowIssue] | None = None,
    ) -> tuple[MarketBar, ...]:
        """Normalize a Tiingo series onto security identity.

        ``company_id`` is accepted temporarily as a compatibility alias for old
        benchmark/setup call sites. It is interpreted only as the security ID;
        no company attribution is written into ``MarketBar``.

        Normalization is strict by default. Backfill callers may pass
        ``invalid_rows`` to quarantine malformed provider rows while preserving
        the valid rows from the same response. The rejected rows are never
        repaired or inserted into the market store.
        """

        if security_id is not None and company_id is not None and security_id != company_id:
            raise ValueError("security_id and legacy company_id alias disagree")
        resolved_security_id = security_id or company_id
        if not resolved_security_id:
            raise ValueError("security_id is required")

        self._require_tiingo_json(raw)
        normalized_ticker = normalize_tiingo_ticker(ticker)
        payload = json.loads(self._text(raw))
        if not isinstance(payload, list):
            raise ValueError("Tiingo prices response must be a JSON list")

        bars: list[MarketBar] = []
        for row_index, row in enumerate(payload):
            try:
                if not isinstance(row, dict):
                    raise TypeError("Tiingo price row must be a JSON object")
                bars.append(
                    MarketBar(
                        security_id=resolved_security_id,
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
            except (ValidationError, KeyError, TypeError, ValueError, DecimalException) as exc:
                if invalid_rows is None:
                    raise
                invalid_rows.append(
                    TiingoPriceRowIssue(
                        row_index=row_index,
                        date=self._row_date_hint(row),
                        reason=self._row_failure_reason(exc),
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

    @staticmethod
    def _row_date_hint(row) -> str | None:
        if not isinstance(row, dict):
            return None
        value = row.get("date")
        if value is None:
            return None
        return str(value)[:10]

    @staticmethod
    def _row_failure_reason(exc: Exception) -> str:
        if isinstance(exc, ValidationError):
            errors = exc.errors()
            if errors:
                message = errors[0].get("msg")
                if message:
                    return str(message)
        message = str(exc).strip()
        return message or type(exc).__name__
