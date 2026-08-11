from datetime import datetime

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator

from .enums import Source
from .ids import content_sha256, raw_artifact_id
from .time import as_utc


class RawRecord(BaseModel):
    """Immutable raw source response captured before normalization."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: Source
    source_record_id: str = Field(min_length=1)
    fetched_at: AwareDatetime
    content_type: str = Field(min_length=1)
    content: bytes | str
    sha256: str = Field(min_length=64, max_length=64)

    @field_validator("fetched_at", mode="before")
    @classmethod
    def normalize_fetched_at(cls, value: datetime) -> datetime:
        return as_utc(value)

    @field_validator("sha256")
    @classmethod
    def normalize_hash(cls, value: str) -> str:
        return value.lower()

    @model_validator(mode="after")
    def verify_content_hash(self) -> "RawRecord":
        expected = content_sha256(self.content)
        if self.sha256 != expected:
            raise ValueError("sha256 does not match raw content")
        return self

    @property
    def artifact_id(self) -> str:
        return raw_artifact_id(self.sha256)
