from datetime import date
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MarketBar(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    company_id: str = Field(min_length=1)
    ticker: str = Field(min_length=1)
    date: date
    open: Decimal = Field(gt=0)
    high: Decimal = Field(gt=0)
    low: Decimal = Field(gt=0)
    close: Decimal = Field(gt=0)
    volume: Decimal = Field(ge=0)
    adj_open: Decimal = Field(gt=0)
    adj_high: Decimal = Field(gt=0)
    adj_low: Decimal = Field(gt=0)
    adj_close: Decimal = Field(gt=0)
    adj_volume: Decimal = Field(ge=0)
    dividend_cash: Decimal = Field(default=Decimal("0"), ge=0)
    split_factor: Decimal = Field(default=Decimal("1"), gt=0)

    @model_validator(mode="after")
    def validate_ranges(self) -> "MarketBar":
        if self.high < max(self.open, self.close) or self.low > min(self.open, self.close):
            raise ValueError("raw OHLC range is inconsistent")
        if self.adj_high < max(self.adj_open, self.adj_close) or self.adj_low > min(
            self.adj_open, self.adj_close
        ):
            raise ValueError("adjusted OHLC range is inconsistent")
        return self


class TiingoMetadata(BaseModel):
    """Descriptive market-source metadata before canonical entity resolution."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ticker: str = Field(min_length=1)
    name: str = Field(min_length=1)
    exchange_code: str | None = None
    start_date: date
    end_date: date | None = None


class SecurityMapping(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    company_id: str = Field(min_length=1)
    ticker: str = Field(min_length=1)
    exchange_code: str | None = None
    valid_from: date
    valid_to: date | None = None

    @model_validator(mode="after")
    def validate_interval(self) -> "SecurityMapping":
        if self.valid_to is not None and self.valid_to < self.valid_from:
            raise ValueError("valid_to must be >= valid_from")
        return self

    def contains(self, day: date) -> bool:
        return self.valid_from <= day and (self.valid_to is None or day <= self.valid_to)
