import json
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np

from .dataset import FeatureSchema, TrainingRow


@dataclass(frozen=True, slots=True)
class OpportunityPrediction:
    expected_alpha_20d: float
    expected_downside_20d: float
    probability_positive_alpha: float
    opportunity_score: float


@dataclass(frozen=True, slots=True)
class LightGbmTrainingConfig:
    num_boost_round: int = 500
    early_stopping_rounds: int = 50
    learning_rate: float = 0.03
    num_leaves: int = 31
    min_data_in_leaf: int = 20
    feature_fraction: float = 0.9
    bagging_fraction: float = 0.9
    bagging_freq: int = 1
    downside_penalty: float = 0.5
    balance_companies: bool = True
    seed: int = 42


class LightGbmModelBundle:
    def __init__(
        self,
        *,
        feature_schema: FeatureSchema,
        alpha_model: lgb.Booster,
        downside_model: lgb.Booster,
        probability_model: lgb.Booster,
        downside_penalty: float,
        positive_alpha_threshold: float,
    ) -> None:
        self.feature_schema = feature_schema
        self.alpha_model = alpha_model
        self.downside_model = downside_model
        self.probability_model = probability_model
        self.downside_penalty = downside_penalty
        self.positive_alpha_threshold = positive_alpha_threshold

    def predict(self, features: dict[str, float | None]) -> OpportunityPrediction:
        matrix = np.asarray([self.feature_schema.vector(features)], dtype=np.float32)
        alpha = float(self.alpha_model.predict(matrix)[0])
        downside = max(0.0, float(self.downside_model.predict(matrix)[0]))
        probability = float(self.probability_model.predict(matrix)[0])
        probability = min(1.0, max(0.0, probability))
        score = alpha * probability - self.downside_penalty * downside
        return OpportunityPrediction(
            expected_alpha_20d=alpha,
            expected_downside_20d=downside,
            probability_positive_alpha=probability,
            opportunity_score=score,
        )

    def feature_importance(self, *, importance_type: str = "gain") -> dict[str, float]:
        values = self.alpha_model.feature_importance(importance_type=importance_type)
        return {
            name: float(value)
            for name, value in zip(self.feature_schema.names, values, strict=True)
        }

    def save(self, directory: str | Path) -> Path:
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        self.alpha_model.save_model(str(root / "alpha.txt"))
        self.downside_model.save_model(str(root / "downside.txt"))
        self.probability_model.save_model(str(root / "probability.txt"))
        metadata = {
            "feature_names": list(self.feature_schema.names),
            "downside_penalty": self.downside_penalty,
            "positive_alpha_threshold": self.positive_alpha_threshold,
        }
        (root / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return root

    @classmethod
    def load(cls, directory: str | Path) -> "LightGbmModelBundle":
        root = Path(directory)
        metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        return cls(
            feature_schema=FeatureSchema(tuple(metadata["feature_names"])),
            alpha_model=lgb.Booster(model_file=str(root / "alpha.txt")),
            downside_model=lgb.Booster(model_file=str(root / "downside.txt")),
            probability_model=lgb.Booster(model_file=str(root / "probability.txt")),
            downside_penalty=float(metadata["downside_penalty"]),
            positive_alpha_threshold=float(metadata["positive_alpha_threshold"]),
        )


class LightGbmTrainer:
    def __init__(self, config: LightGbmTrainingConfig | None = None) -> None:
        self.config = config or LightGbmTrainingConfig()

    def train(
        self,
        train_rows: tuple[TrainingRow, ...] | list[TrainingRow],
        validation_rows: tuple[TrainingRow, ...] | list[TrainingRow],
        *,
        positive_alpha_threshold: float = 0.02,
    ) -> LightGbmModelBundle:
        train_rows = tuple(train_rows)
        validation_rows = tuple(validation_rows)
        if not train_rows:
            raise ValueError("train_rows must not be empty")
        if not validation_rows:
            raise ValueError("validation_rows must not be empty")

        schema = FeatureSchema.from_rows(train_rows)
        train_x = schema.matrix(train_rows)
        validation_x = schema.matrix(validation_rows)
        train_weight = (
            _company_balanced_weights(train_rows)
            if self.config.balance_companies
            else None
        )
        validation_weight = (
            _company_balanced_weights(validation_rows)
            if self.config.balance_companies
            else None
        )

        alpha = self._train_one(
            train_x,
            np.asarray([row.alpha_20d for row in train_rows], dtype=np.float64),
            validation_x,
            np.asarray([row.alpha_20d for row in validation_rows], dtype=np.float64),
            train_weight=train_weight,
            validation_weight=validation_weight,
            feature_names=schema.names,
            objective="regression_l2",
            metric="l2",
        )
        downside = self._train_one(
            train_x,
            np.asarray([row.downside_20d for row in train_rows], dtype=np.float64),
            validation_x,
            np.asarray([row.downside_20d for row in validation_rows], dtype=np.float64),
            train_weight=train_weight,
            validation_weight=validation_weight,
            feature_names=schema.names,
            objective="regression_l1",
            metric="l1",
        )
        probability = self._train_one(
            train_x,
            np.asarray([row.positive_alpha_20d for row in train_rows], dtype=np.float64),
            validation_x,
            np.asarray([row.positive_alpha_20d for row in validation_rows], dtype=np.float64),
            train_weight=train_weight,
            validation_weight=validation_weight,
            feature_names=schema.names,
            objective="binary",
            metric="binary_logloss",
        )

        return LightGbmModelBundle(
            feature_schema=schema,
            alpha_model=alpha,
            downside_model=downside,
            probability_model=probability,
            downside_penalty=self.config.downside_penalty,
            positive_alpha_threshold=positive_alpha_threshold,
        )

    def _train_one(
        self,
        train_x: np.ndarray,
        train_y: np.ndarray,
        validation_x: np.ndarray,
        validation_y: np.ndarray,
        *,
        train_weight: np.ndarray | None,
        validation_weight: np.ndarray | None,
        feature_names: tuple[str, ...],
        objective: str,
        metric: str,
    ) -> lgb.Booster:
        config = self.config
        parameters = {
            "objective": objective,
            "metric": metric,
            "learning_rate": config.learning_rate,
            "num_leaves": config.num_leaves,
            "min_data_in_leaf": config.min_data_in_leaf,
            "feature_fraction": config.feature_fraction,
            "bagging_fraction": config.bagging_fraction,
            "bagging_freq": config.bagging_freq,
            "seed": config.seed,
            "feature_fraction_seed": config.seed,
            "bagging_seed": config.seed,
            "data_random_seed": config.seed,
            "deterministic": True,
            "force_col_wise": True,
            "verbosity": -1,
        }
        train_set = lgb.Dataset(
            train_x,
            label=train_y,
            weight=train_weight,
            feature_name=list(feature_names),
            free_raw_data=False,
        )
        validation_set = lgb.Dataset(
            validation_x,
            label=validation_y,
            weight=validation_weight,
            reference=train_set,
            free_raw_data=False,
        )
        return lgb.train(
            parameters,
            train_set,
            num_boost_round=config.num_boost_round,
            valid_sets=[validation_set],
            valid_names=["validation"],
            callbacks=[
                lgb.early_stopping(
                    config.early_stopping_rounds,
                    first_metric_only=True,
                    verbose=False,
                ),
                lgb.log_evaluation(period=0),
            ],
        )


def _company_balanced_weights(rows: tuple[TrainingRow, ...]) -> np.ndarray:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.company_id] = counts.get(row.company_id, 0) + 1
    company_count = len(counts)
    if company_count == 0:
        raise ValueError("cannot balance an empty company set")

    # Mean weight stays at 1.0 while every company contributes equal total
    # training mass inside the current train/validation split.
    row_count = len(rows)
    return np.asarray(
        [
            row_count / (company_count * counts[row.company_id])
            for row in rows
        ],
        dtype=np.float64,
    )
