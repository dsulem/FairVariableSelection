import pandas as pd
import numpy as np
from tqdm import tqdm
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn import base, metrics, model_selection, pipeline, preprocessing
from sklearn import linear_model, naive_bayes, tree
from fairfs.dataset_loader import get_all_datasets, get_simulated_data
from fairfs.unfairness_metrics import CombinedMetric, UnfairnessMetric, calc_unfairness, UNFAIRNESS_METRICS
from generate_data import simulate_logreg, simulate_linreg
import matplotlib.pyplot as plt




"""Fair feature selection for logistic regression.

Forward sequential feature selection picks the subset of features in X that minimises
    log loss + unfairness_weight * unfairness
estimated with 4-fold CV on the training set. Logistic regression is then fit on the selected
features and evaluated on the test set. The sensitive attribute S is only used to measure
unfairness; it is not a model input.

Usage:
    from fairfs import run_all
    results, predictions = run_all(X_train, S_train, y_train, X_test, S_test, y_test)
"""

UNFAIRNESS_WEIGHTS = [0, 1, 2, 3, 4]
SEED = 42  # inner CV folds and logistic regression


class FairLogLossScorer():
    def __init__(self, S, unfairness_metric, unfairness_weight):
        """Scorer with the scikit-learn signature scorer(estimator, X, y), returning
        -(log loss + weight * unfairness), since feature selection maximises its score.
        The selector only passes (estimator, X, y) for each CV fold, so S is stored here and
        each fold's rows are looked up through the index of y.
        """
        self.S = S
        self.unfairness_metric = unfairness_metric
        self.unfairness_weight = unfairness_weight

    def __call__(self, estimator, X, y):
        assert isinstance(y, pd.Series), 'pd.Series required for index matching'
        y_prob = estimator.predict_proba(X)
        y_pred = estimator.predict(X)
        unfairness = calc_unfairness(y, y_pred, self.S.loc[y.index],
                                                        self.unfairness_metric)
        loss = metrics.log_loss(y, y_prob, labels=estimator.classes_)
        return -(loss + self.unfairness_weight * unfairness)


def _as_series(values, name):
    return pd.Series(np.asarray(values).ravel(), name=name)  # fresh 0..n-1 index


def run_experiment(X_train, S_train, y_train, X_test, S_test, y_test, unfairness_metric,
                   unfairness_weight, feature_names=None):
    """Select features and fit on the training set, evaluate on the test set.

    Args:
        X_train, X_test: features, shape (n, d); arrays or DataFrames
        S_train, S_test: binary sensitive attribute, shape (n,)
        y_train, y_test: binary labels, shape (n,)
        unfairness_metric (str): one of unfairness_metrics.UNFAIRNESS_METRICS
        unfairness_weight (float): weight of the unfairness penalty, >= 0
        feature_names (list): optional; taken from the columns if X_train is a DataFrame

    Returns:
        dict of test-set results, DataFrame with one row of predictions per test sample
    """
    if feature_names is None:
        feature_names = (list(X_train.columns) if isinstance(X_train, pd.DataFrame)
                         else [f'x{j}' for j in range(np.shape(X_train)[1])])
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    # Series with a 0..n-1 index: the scorer uses the index of y to find each fold's S values
    S_train, y_train = _as_series(S_train, 'S'), _as_series(y_train, 'y')
    S_test, y_test = _as_series(S_test, 'S'), _as_series(y_test, 'y')
    assert len(X_train) == len(S_train) == len(y_train), 'training inputs differ in length'
    assert len(X_test) == len(S_test) == len(y_test), 'test inputs differ in length'

    clf = linear_model.LogisticRegression(random_state=SEED)
    sfs = SequentialFeatureSelector(
        clf, 'best', verbose=0, cv=model_selection.KFold(4, shuffle=True, random_state=SEED),
        scoring=FairLogLossScorer(S_train, unfairness_metric, unfairness_weight), n_jobs=2)
    pipe = pipeline.Pipeline([
        ('standardize', preprocessing.StandardScaler()),
        ('feature_selection', sfs),
        ('model', clf),
    ])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)
    selected = [int(j) for j in pipe.named_steps['feature_selection'].k_feature_idx_]
    results = {
        'unfairness_metric': unfairness_metric,
        'unfairness_weight': unfairness_weight,
        'unfairness': calc_unfairness(y_test, y_pred, S_test,
                                                         unfairness_metric),
        'log_loss': metrics.log_loss(y_test, y_prob, labels=pipe.classes_),
        'auc': metrics.roc_auc_score(y_test, y_prob[:, 1]),
        'accuracy': metrics.accuracy_score(y_test, y_pred),
        'n_selected': len(selected),
        'selected_features': ';'.join(str(feature_names[j]) for j in selected),
    }
    predictions = pd.DataFrame({
        'unfairness_metric': unfairness_metric,
        'unfairness_weight': unfairness_weight,
        'sample_index': np.arange(len(y_test)),  # row position in the test set
        'S': S_test.values,
        'y_true': y_test.values,
        'y_pred': y_pred,
        'y_prob': y_prob[:, 1],  # P(y = pipe.classes_[1])
    })
    return results, predictions


def run_all(X_train, S_train, y_train, X_test, S_test, y_test,
            unfairness_metric_names=UNFAIRNESS_METRICS,
            unfairness_weights=UNFAIRNESS_WEIGHTS, feature_names=None):
    """Run run_experiment for every unfairness metric and weight.

    Returns:
        results DataFrame (one row per metric x weight), predictions DataFrame (one row per
        test sample per metric x weight)
    """
    all_results, all_predictions = [], []
    for unfairness_metric in unfairness_metric_names:
        for unfairness_weight in unfairness_weights:
            res, preds = run_experiment(X_train, S_train, y_train, X_test, S_test, y_test,
                                        unfairness_metric, unfairness_weight, feature_names)
            print(f"{unfairness_metric} | weight {unfairness_weight} | "
                  f"unfairness={res['unfairness']:.3f}  log_loss={res['log_loss']:.3f}  "
                  f"auc={res['auc']:.3f} | features: {res['selected_features']}")
            all_results.append(res)
            all_predictions.append(preds)
    return pd.DataFrame(all_results), pd.concat(all_predictions, ignore_index=True)