from pydantic import BaseModel, ConfigDict, Field, field_validator

from stock_trading.core import SemanticAnnotation, SemanticDirection


ALLOWED_TOPICS = frozenset(
    {
        "TECH.AI",
        "TECH.SEMICONDUCTORS",
        "TECH.CLOUD",
        "TECH.CYBERSECURITY",
        "TRADE.EXPORT_CONTROLS",
        "TRADE.TARIFFS",
        "DEFENSE.AIRCRAFT",
        "DEFENSE.MISSILES",
        "DEFENSE.NAVAL",
        "DEFENSE.SPACE",
        "DEFENSE.CYBERSECURITY",
        "ENERGY.OIL_GAS",
        "ENERGY.NUCLEAR",
        "ENERGY.RENEWABLE",
        "HEALTH.PHARMA",
        "HEALTH.MEDICAL_DEVICES",
        "FINANCE.BANKING",
        "FINANCE.SECURITIES",
        "REGULATION.ANTITRUST",
        "REGULATION.DATA_PRIVACY",
        "REGULATION.ENVIRONMENT",
        "INFRASTRUCTURE.TRANSPORT",
        "INFRASTRUCTURE.TELECOM",
        "GOVERNMENT.PROCUREMENT",
        "OTHER",
    }
)


class SemanticResult(BaseModel):
    """Strict model-produced semantic fields before provenance is attached."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    topics: tuple[str, ...] = ()
    direction: SemanticDirection = SemanticDirection.UNKNOWN
    novelty: float | None = Field(default=None, ge=0.0, le=1.0)
    importance: float | None = Field(default=None, ge=0.0, le=1.0)
    company_relevance: float | None = Field(default=None, ge=0.0, le=1.0)
    policy_relevance: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("topics")
    @classmethod
    def validate_topics(cls, topics: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(dict.fromkeys(topic.strip().upper() for topic in topics if topic.strip()))
        unknown = sorted(set(normalized) - ALLOWED_TOPICS)
        if unknown:
            raise ValueError(f"unknown semantic topics: {', '.join(unknown)}")
        return normalized

    def to_annotation(
        self,
        *,
        model: str,
        extractor_version: str,
        schema_version: str = "semantic-v1",
    ) -> SemanticAnnotation:
        return SemanticAnnotation(
            topics=self.topics,
            direction=self.direction,
            novelty=self.novelty,
            importance=self.importance,
            company_relevance=self.company_relevance,
            policy_relevance=self.policy_relevance,
            confidence=self.confidence,
            model=model,
            extractor_version=extractor_version,
            schema_version=schema_version,
        )
