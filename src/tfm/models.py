"""M3 - Model factory: the five classifiers compared in the study.

  - naive: always predicts the training majority class; the floor to beat
  - logreg: logistic regression, scaled inside a Pipeline (linear baseline)
  - random_forest: bagging ensemble (parallel trees)
  - xgboost: boosting ensemble (sequential trees)
  - svm: RBF-kernel SVC, scaled inside a Pipeline

Adding or removing a model only means editing `build_model_specs`.
Thesis: §4.4 (Listings 4.8 to 4.10; the grids are in Table 4.3).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from . import config


@dataclass
class ModelSpec:
    """A classifier plus its hyper-parameter grid for GridSearchCV."""

    name: str
    estimator: BaseEstimator
    param_grid: dict = field(default_factory=dict)  # empty grid = no tuning


def build_model_specs(random_state: int = config.RANDOM_STATE) -> dict[str, ModelSpec]:
    """Return the model registry {name: ModelSpec}."""
    specs: dict[str, ModelSpec] = {}

    specs["naive"] = ModelSpec(
        name="naive",
        estimator=DummyClassifier(strategy="most_frequent"),
    )

    # the scaler lives inside the pipeline, so it is fitted on training data
    # only, never before the split
    specs["logreg"] = ModelSpec(
        name="logreg",
        estimator=Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=random_state)),
            ]
        ),
        param_grid={"clf__C": [0.01, 0.1, 1.0, 10.0]},
    )

    specs["random_forest"] = ModelSpec(
        name="random_forest",
        estimator=RandomForestClassifier(
            random_state=random_state, n_jobs=config.N_JOBS_ESTIMATOR
        ),
        param_grid={
            "n_estimators": [200, 400],
            "max_depth": [5, 10, None],
            "min_samples_leaf": [1, 5],
        },
    )

    specs["xgboost"] = ModelSpec(
        name="xgboost",
        estimator=XGBClassifier(
            eval_metric="logloss",
            n_jobs=config.N_JOBS_ESTIMATOR,
            random_state=random_state,
        ),
        param_grid={
            "n_estimators": [200, 400],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.05, 0.1],
        },
    )

    specs["svm"] = ModelSpec(
        name="svm",
        estimator=Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf", probability=True, random_state=random_state)),
            ]
        ),
        param_grid={"clf__C": [0.1, 1.0, 10.0], "clf__gamma": ["scale", 0.01]},
    )

    return specs


def select_model_specs(names: list[str] | None = None) -> dict[str, ModelSpec]:
    """Registry restricted to `names`, or the full registry when None.

    Used by both the main run and the walk-forward, so --models means the
    same in both.
    """
    specs = build_model_specs()
    if names:
        return {name: spec for name, spec in specs.items() if name in names}
    return specs


CORE3 = ["logreg", "random_forest", "xgboost"]  # initial 3-model design
