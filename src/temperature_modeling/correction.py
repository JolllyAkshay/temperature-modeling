"""
Post-processing correction models for GraphCast temperature forecasts.

Trains a statistical model to predict GraphCast's systematic error from
``FeatureVector`` objects and applies the correction at inference time.

Supported model types
---------------------
``mean_bias``
    Per-lead-time mean bias removal.  No sklearn required.  Fast baseline.

``linear``
    Ordinary least-squares regression (``LinearRegression``).

``ridge``
    Ridge (L2-regularised) regression.  More robust than plain OLS when
    features are collinear.

``random_forest``
    ``RandomForestRegressor`` — captures non-linear interactions between
    lead time, season, and forecast anomaly.

``xgboost``
    ``XGBRegressor`` — gradient-boosted trees; typically the strongest
    performer on tabular weather data with modest sample sizes.

All models are trained on the ``error_c`` target (forecast − observed) and
predict the expected error; correcting a forecast means subtracting the
predicted error.

Typical usage
-------------
>>> from temperature_modeling.correction import train_and_evaluate
>>> model, evaluation = train_and_evaluate(vectors, model_type="xgboost")
>>> print(f"Skill score: {evaluation.window_skill_score:+.1%}")
>>> corrected = model.correct(new_vectors)
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .features import FEATURE_FIELDS, FeatureVector

# Lead-time window of primary interest.
_WINDOW = range(10, 16)  # days 10–15 inclusive


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class CorrectionResult:
    """Corrected prediction for a single FeatureVector."""
    lead_days: int
    raw_forecast_c: float
    corrected_forecast_c: float
    observed_c: float          # NaN-ish sentinel (0.0) if unknown at inference
    raw_error_c: float
    corrected_error_c: float


@dataclass
class ModelEvaluation:
    """
    Hold-out test-set evaluation of a correction model.

    ``window_*`` metrics are restricted to lead days 10–15 (the target window).
    ``per_lead_*`` dicts cover all lead times present in the test set.
    """
    model_type: str
    n_train: int
    n_test: int
    window_raw_rmse: float
    window_corrected_rmse: float
    window_skill_score: float          # 1 − corrected_rmse / raw_rmse
    per_lead_raw_rmse: Dict[int, float] = field(default_factory=dict)
    per_lead_corrected_rmse: Dict[int, float] = field(default_factory=dict)


# ── Public model wrapper ──────────────────────────────────────────────────────

class CorrectionModel:
    """
    Wraps a fitted sklearn-compatible estimator.

    Obtain instances via :func:`train_correction_model` or
    :func:`train_and_evaluate`.
    """

    def __init__(self, estimator, model_type: str) -> None:
        self._est = estimator
        self.model_type = model_type

    def predict_errors(self, vectors: List[FeatureVector]) -> List[float]:
        """Return predicted error_c for each vector."""
        X = _to_matrix(vectors)
        return list(self._est.predict(X))

    def correct(self, vectors: List[FeatureVector]) -> List[float]:
        """Return bias-corrected forecast_temp_c for each vector."""
        predicted_errors = self.predict_errors(vectors)
        return [
            v.forecast_temp_c - pred_err
            for v, pred_err in zip(vectors, predicted_errors)
        ]


# ── Public training functions ─────────────────────────────────────────────────

def train_correction_model(
    vectors: List[FeatureVector],
    model_type: str = "xgboost",
) -> "CorrectionModel":
    """
    Fit a correction model on *all* supplied vectors.

    Parameters
    ----------
    vectors:
        Output of :func:`~temperature_modeling.features.extract_features`.
    model_type:
        One of ``"mean_bias"``, ``"linear"``, ``"ridge"``,
        ``"random_forest"``, ``"xgboost"``.

    Returns
    -------
    CorrectionModel
    """
    estimator = _build_estimator(model_type)
    X = _to_matrix(vectors)
    y = [v.error_c for v in vectors]
    estimator.fit(X, y)
    return CorrectionModel(estimator, model_type)


def train_and_evaluate(
    vectors: List[FeatureVector],
    model_type: str = "xgboost",
    test_fraction: float = 0.2,
) -> Tuple["CorrectionModel", ModelEvaluation]:
    """
    Chronological train/test split, fit, and evaluate.

    The last ``test_fraction`` of *vectors* (by position, preserving temporal
    order) forms the test set.  The remainder is used for training.

    Parameters
    ----------
    vectors:
        Chronologically sorted feature vectors.
    model_type:
        Model type string (see :func:`train_correction_model`).
    test_fraction:
        Fraction of vectors held out for evaluation (default 0.2).

    Returns
    -------
    tuple[CorrectionModel, ModelEvaluation]

    Raises
    ------
    ValueError
        If there are fewer than 10 vectors or the test set is empty.
    """
    if len(vectors) < 10:
        raise ValueError(
            f"Need at least 10 vectors to train and evaluate; got {len(vectors)}"
        )
    split = max(1, int(len(vectors) * (1 - test_fraction)))
    train_vecs = vectors[:split]
    test_vecs = vectors[split:]
    if not test_vecs:
        raise ValueError("test_fraction too small; test set is empty")

    model = train_correction_model(train_vecs, model_type)

    # Evaluate on hold-out set.
    pred_errors = model.predict_errors(test_vecs)
    results = [
        CorrectionResult(
            lead_days=int(v.lead_days),
            raw_forecast_c=v.forecast_temp_c,
            corrected_forecast_c=v.forecast_temp_c - pred_err,
            observed_c=v.forecast_temp_c - v.error_c,  # recover observed
            raw_error_c=v.error_c,
            corrected_error_c=v.error_c - pred_err,
        )
        for v, pred_err in zip(test_vecs, pred_errors)
    ]

    evaluation = _compute_evaluation(model_type, len(train_vecs), results)
    return model, evaluation


# ── Internal helpers ──────────────────────────────────────────────────────────

def _to_matrix(vectors: List[FeatureVector]):
    """Convert FeatureVectors to a 2-D list (or numpy array if available)."""
    rows = [
        [getattr(v, f) for f in FEATURE_FIELDS]
        for v in vectors
    ]
    try:
        import numpy as np
        return np.array(rows, dtype=float)
    except ImportError:
        return rows  # _MeanBiasEstimator works on plain lists


def _build_estimator(model_type: str):
    """Return an unfitted sklearn-compatible estimator."""
    if model_type == "mean_bias":
        return _MeanBiasEstimator()
    if model_type == "linear":
        from sklearn.linear_model import LinearRegression
        return LinearRegression()
    if model_type == "ridge":
        from sklearn.linear_model import Ridge
        return Ridge(alpha=1.0)
    if model_type == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=5,
            random_state=42,
        )
    if model_type == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
        )
    raise ValueError(
        f"Unknown model_type {model_type!r}. "
        "Choose from: mean_bias, linear, ridge, random_forest, xgboost"
    )


def _rmse(errors: List[float]) -> float:
    if not errors:
        return float("nan")
    return math.sqrt(sum(e ** 2 for e in errors) / len(errors))


def _compute_evaluation(
    model_type: str,
    n_train: int,
    results: List[CorrectionResult],
) -> ModelEvaluation:
    """Compute RMSE and skill metrics from CorrectionResult objects."""
    # Per-lead aggregation.
    lead_raw: Dict[int, List[float]] = {}
    lead_corr: Dict[int, List[float]] = {}
    for r in results:
        lead_raw.setdefault(r.lead_days, []).append(r.raw_error_c)
        lead_corr.setdefault(r.lead_days, []).append(r.corrected_error_c)

    per_lead_raw_rmse = {ld: _rmse(errs) for ld, errs in lead_raw.items()}
    per_lead_corrected_rmse = {ld: _rmse(errs) for ld, errs in lead_corr.items()}

    # Window (10–15 day) aggregation.
    window_raw = [r.raw_error_c for r in results if r.lead_days in _WINDOW]
    window_corr = [r.corrected_error_c for r in results if r.lead_days in _WINDOW]

    raw_rmse = _rmse(window_raw)
    corr_rmse = _rmse(window_corr)
    skill = (1.0 - corr_rmse / raw_rmse) if raw_rmse else float("nan")

    return ModelEvaluation(
        model_type=model_type,
        n_train=n_train,
        n_test=len(results),
        window_raw_rmse=round(raw_rmse, 3),
        window_corrected_rmse=round(corr_rmse, 3),
        window_skill_score=round(skill, 4),
        per_lead_raw_rmse={ld: round(v, 3) for ld, v in per_lead_raw_rmse.items()},
        per_lead_corrected_rmse={ld: round(v, 3) for ld, v in per_lead_corrected_rmse.items()},
    )


# ── No-dependency mean-bias estimator ────────────────────────────────────────

class _MeanBiasEstimator:
    """
    Per-lead-time mean bias estimator.

    Sklearn-compatible interface (fit / predict) but requires no external
    packages.  Predicts the training mean error for each lead day; for lead
    days unseen in training the grand mean is used.
    """

    def __init__(self) -> None:
        self._lead_bias: Dict[int, float] = {}
        self._grand_bias: float = 0.0

    def fit(self, X, y) -> "_MeanBiasEstimator":
        # X column 0 is lead_days (see FEATURE_FIELDS).
        by_lead: Dict[int, List[float]] = {}
        for row, err in zip(X, y):
            ld = int(row[0])
            by_lead.setdefault(ld, []).append(err)
        self._lead_bias = {ld: sum(errs) / len(errs) for ld, errs in by_lead.items()}
        all_errors = list(y) if not hasattr(y, "__iter__") else list(y)
        self._grand_bias = sum(all_errors) / len(all_errors) if all_errors else 0.0
        return self

    def predict(self, X) -> List[float]:
        return [
            self._lead_bias.get(int(row[0]), self._grand_bias)
            for row in X
        ]
