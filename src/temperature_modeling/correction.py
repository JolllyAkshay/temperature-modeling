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

from .features import FEATURE_FIELDS, FEATURE_FIELDS_SAT, FeatureVector

# Lead-time window of primary interest.
_WINDOW = range(10, 16)  # days 10–15 inclusive

# Lead-time bands for per-band model training.
# Each band gets its own XGBoost model so hyper-parameters can specialise
# for short-lead (high signal) vs long-lead (high noise) regimes.
# Finer 3-day bands give each model a more homogeneous error structure.
LEAD_BANDS = [(1, 3), (4, 6), (7, 9), (10, 12), (13, 15)]


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

    def __init__(self, estimator, model_type: str, feature_fields: List[str] = None) -> None:
        self._est = estimator
        self.model_type = model_type
        self._feature_fields = feature_fields or FEATURE_FIELDS

    def predict_errors(self, vectors: List[FeatureVector]) -> List[float]:
        """Return predicted error_c for each vector."""
        X = _to_matrix(vectors, self._feature_fields)
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
    feature_fields: List[str] = None,
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
    feature_fields:
        Ordered list of FeatureVector field names to use as predictors.
        Defaults to ``FEATURE_FIELDS``; pass ``FEATURE_FIELDS_SAT`` to
        include satellite surface-state features.

    Returns
    -------
    CorrectionModel
    """
    fields = feature_fields or FEATURE_FIELDS
    estimator = _build_estimator(model_type)
    X = _to_matrix(vectors, fields)
    y = [v.error_c for v in vectors]
    estimator.fit(X, y)
    return CorrectionModel(estimator, model_type, fields)


def train_and_evaluate(
    vectors: List[FeatureVector],
    model_type: str = "xgboost",
    test_fraction: float = 0.2,
    feature_fields: List[str] = None,
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

    model = train_correction_model(train_vecs, model_type, feature_fields)

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


def train_and_evaluate_banded(
    vectors: List[FeatureVector],
    model_type: str = "xgboost",
    test_fraction: float = 0.2,
    feature_fields: List[str] = None,
    lead_bands: List[Tuple[int, int]] = None,
) -> Tuple[None, "ModelEvaluation"]:
    """
    Train one model per lead-time band and return a combined evaluation.

    Each band ``(low, high)`` gets its own independently fitted model so
    that XGBoost can specialise: short-lead models learn high-signal,
    low-variance corrections; long-lead models focus on regime/teleconnection
    patterns where the marginal predictors (z500, MJO) matter most.

    The chronological 80/20 split is applied independently within each band
    so test dates are consistent across bands.

    Parameters
    ----------
    vectors:
        Chronologically sorted feature vectors (all lead times mixed).
    model_type, test_fraction, feature_fields:
        Same as :func:`train_and_evaluate`.
    lead_bands:
        List of ``(low, high)`` inclusive lead-day ranges.
        Defaults to ``[(1, 5), (6, 10), (11, 15)]``.

    Returns
    -------
    tuple[None, ModelEvaluation]
        The first element is ``None`` because inference requires routing
        each sample to the correct band model; use :func:`train_and_evaluate`
        when you need a single deployable model.
    """
    if lead_bands is None:
        lead_bands = LEAD_BANDS

    all_test_results: List[CorrectionResult] = []
    total_train = 0

    for band_idx, (low, high) in enumerate(lead_bands):
        band = [v for v in vectors if low <= int(v.lead_days) <= high]
        if len(band) < 10:
            continue
        split = max(1, int(len(band) * (1 - test_fraction)))
        train_vecs, test_vecs = band[:split], band[split:]
        if not test_vecs:
            continue

        # Regularisation tightens progressively as lead increases.
        _BAND_OVERRIDES = [
            {"max_depth": 5, "reg_lambda": 0.5, "min_child_weight": 2},  # [1-3]
            {"max_depth": 5, "reg_lambda": 1.0, "min_child_weight": 3},  # [4-6]
            {"max_depth": 4, "reg_lambda": 2.0, "min_child_weight": 5},  # [7-9]
            {"max_depth": 3, "reg_lambda": 3.0, "min_child_weight": 6},  # [10-12]
            {"max_depth": 3, "reg_lambda": 4.0, "min_child_weight": 8},  # [13-15]
        ]
        band_model = _build_estimator_with_overrides(
            model_type, _BAND_OVERRIDES[band_idx] if band_idx < len(_BAND_OVERRIDES) else {}
        )
        fields = feature_fields or FEATURE_FIELDS
        X_train = _to_matrix(train_vecs, fields)
        y_train = [v.error_c for v in train_vecs]
        band_model.fit(X_train, y_train)
        band_model = CorrectionModel(band_model, model_type, fields)
        pred_errors = band_model.predict_errors(test_vecs)
        total_train += len(train_vecs)

        for v, pred_err in zip(test_vecs, pred_errors):
            all_test_results.append(
                CorrectionResult(
                    lead_days=int(v.lead_days),
                    raw_forecast_c=v.forecast_temp_c,
                    corrected_forecast_c=v.forecast_temp_c - pred_err,
                    observed_c=v.forecast_temp_c - v.error_c,
                    raw_error_c=v.error_c,
                    corrected_error_c=v.error_c - pred_err,
                )
            )

    if not all_test_results:
        raise ValueError("No band had enough samples to evaluate.")

    evaluation = _compute_evaluation(
        model_type + "_banded", total_train, all_test_results
    )
    return None, evaluation


# ── Internal helpers ──────────────────────────────────────────────────────────

def _to_matrix(vectors: List[FeatureVector], feature_fields: List[str] = None):
    """Convert FeatureVectors to a 2-D list (or numpy array if available)."""
    fields = feature_fields or FEATURE_FIELDS
    rows = [
        [getattr(v, f) for f in fields]
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
            reg_lambda=2.0,   # L2 regularisation — reduces overfitting on small bands
            min_child_weight=5,
            random_state=42,
            verbosity=0,
        )
    raise ValueError(
        f"Unknown model_type {model_type!r}. "
        "Choose from: mean_bias, linear, ridge, random_forest, xgboost"
    )


def _build_estimator_with_overrides(model_type: str, overrides: dict):
    """Build an estimator and apply hyperparameter overrides (XGBoost only)."""
    if model_type == "xgboost" and overrides:
        from xgboost import XGBRegressor
        base_params = dict(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            min_child_weight=5,
            random_state=42,
            verbosity=0,
        )
        base_params.update(overrides)
        return XGBRegressor(**base_params)
    return _build_estimator(model_type)


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
