import logging
from typing import Any, Tuple, TypedDict

import numpy as np
import pandas as pd
import xgboost as xgb
from pandas.core.frame import DataFrame
from sklearn.metrics import (auc, f1_score, precision_recall_curve,
                             precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold


CV_RESULTS_FILE = 'kf_results.csv'
RANDOM_SEED = 42
NUM_NEIGHBOURS = 10
FIXED_PRECISION = .95
GRID_SIZE = 5
FEATURES = [
    'lat_rad_x', 'lon_rad_x',
    'lat_rad_y', 'lon_rad_y',
    'location_distance',
    'phone_distance',
    'name_levenshtein_norm',
    'name_lcs_norm',
]
CLASSIFIER_PARAMS = {
    'n_estimators': [100],
    'learning_rate': [.1],
    'max_depth': range(1, 9, 1),
    'subsample': [.5, .7, .9, 1],
    'gamma': [0, .05, .25],
    'colsample_bytree': [1, .5, .75],
    'reg_alpha': [0, .1, .5, 1],
    'reg_lambda': [0, .1, .5, 1],
    'scale_pos_weight': [1, 10, 25, 50],
}

logger = logging.getLogger(__name__)


class ClassifierParams(TypedDict):
    '''
    Datatype of classifier parameters.
    '''
    param: str
    value: Any


def generate_grid(
    params: ClassifierParams,
    shape: int = 5
):
    '''
    Generates grid of classifier parameters of length "shape"

    Parameters:
        params : ClassifierParams
            Parameters to choose from.
        shape : int, default 5
            Number of distinct params to generate. Not Less than 1.
    Returns:
        dict_of_params : Dict[int, ClassifierParams]
    '''
    dict_of_params = {}
    i = 0
    while len(dict_of_params) < shape - 1:
        param = {}
        for p in params:
            param[p] = np.random.choice(params[p])
        if param not in dict_of_params.values():
            dict_of_params[i] = param
            i += 1
    # add default parameters
    # TODO: make it const
    dict_of_params[-1] = {
        'colsample_bylevel': 1,
        'colsample_bynode': 1,
        'colsample_bytree': 1,
        'gamma': 0,
        'learning_rate': 0.1,
        'max_depth': 3,
        'n_estimators': 100,
        'reg_alpha': 0,
        'reg_lambda': 1,
        'scale_pos_weight': 1,
    }
    return dict_of_params


def pair_metrics(
    y_true,
    pred_proba,
    fixed_precision
) -> Tuple[float, float, float]:
    '''
    Calculates
        - precision-recall AUC
        - recall@precision=fixed_precision
        - threshold (for given precision)

    Parameters:
        y_true : arrayLike
            True values.
        pred_proba : arrayLike
            Predicted probabilities of class 1.
        fixed_precision : float
            Value of precision at which to calculate recall.
    Returns:
        precision-recall AUC : float
        recall at fixed precision : float
        threshold : float
    '''
    precision, recall, thresholds = precision_recall_curve(
        y_true, pred_proba)
    pr_rec_auc = auc(recall, precision)
    recall_at_pr = 0
    threshold = 0
    for i in range(len(precision)):
        if precision[i] > fixed_precision:
            recall_at_pr = recall[i]
            threshold = thresholds[i]
            break
    return pr_rec_auc, recall_at_pr, threshold


def metrics_at_threshold(
        y_true,
        pred_proba,
        threshold) -> Tuple[float, float, float]:
    '''

    Calculates precision, recall and f1-Score at given threshold.

    Parameters:
        y_true : arrayLike
            True values.
        pred_proba : arrayLike
            Predicted probabilities of class 1.
        threshold : float
            Threshold for class prediction.
    Returns:
        precision : float
        recall : float
        f1-score : float
    '''
    return (
        precision_score(
            y_true,
            np.where(pred_proba > threshold, 1, 0)
        ),
        recall_score(
            y_true,
            np.where(pred_proba > threshold, 1, 0)
        ),
        f1_score(
            y_true,
            np.where(pred_proba > threshold, 1, 0)
        )
    )


def run_kfold(
    data: DataFrame,
    fixed_precision: float = FIXED_PRECISION,
    random_seed: int = RANDOM_SEED,
    grid_size: int = GRID_SIZE
) -> Tuple[xgb.XGBClassifier, float]:
    '''
    Runs kfold cross validation to select best parameters of a model.
    Selection based on recall@fixed_precision.

    Parameters:
        data : DataFrame
            Pairs dataset with FEATURES and target.
        fixed_precision : float, default FIXED_PRECISION
            Level of precision at which recall of a model is calculated
            for model selection.
        random_seed : int, default RANDOM_SEED
            Seed for random number generator.
    Returns:
        xgb.XGBClassifier
            Classifier with best parameters fitted on whole dataset.
        float
            Average theshold for classifier to get fixed precision.
    '''
    kf_results = pd.DataFrame()

    grid_of_params = generate_grid(CLASSIFIER_PARAMS, grid_size)
    logger.debug('%s parameters for CV', len(grid_of_params))

    kf = StratifiedKFold(
        n_splits=5,
        random_state=random_seed,
        shuffle=True,
    )

    num_fold: int = 0
    for train_index, test_index in kf.split(
        data[FEATURES],
        data.target,
    ):
        num_fold += 1
        logger.debug('Running fold #%s', num_fold)

        X_train = data[data.index.isin(train_index)][FEATURES]
        X_test = data[data.index.isin(test_index)][FEATURES]
        y_train = data[data.index.isin(train_index)].target
        y_test = data[data.index.isin(test_index)].target

        for i, _ in sorted(grid_of_params.items(), key=lambda x: x[0]):
            param_index = str(i)
            param = grid_of_params[i]
            logger.debug('Checking parameters #%s', param_index)

            clf = xgb.XGBClassifier(
                seed=random_seed,
                n_jobs=-1,
                **param,
            )

            clf.fit(X_train, y_train)
            # test metrics
            test_pr_rec_auc, test_rec_at_fixed_pr, threshold = \
                pair_metrics(y_test, clf.predict_proba(
                    X_test)[:, 1], fixed_precision)

            # train metrics
            train_pr_rec_auc, train_rec_at_fixed_pr, _ = \
                pair_metrics(y_test, clf.predict_proba(
                    X_test)[:, 1], fixed_precision)

            kf_results = kf_results.append(
                [[
                    param_index, param, threshold,
                    train_pr_rec_auc, train_rec_at_fixed_pr,
                    test_pr_rec_auc, test_rec_at_fixed_pr,
                ]]
            )
    kf_results.columns = [
        'param_index', 'param', 'threshold',
        'train_pr_rec_auc', 'train_rec_at_fixed_pr',
        'test_pr_rec_auc', 'test_rec_at_fixed_pr',
    ]
    kf_results.to_csv(CV_RESULTS_FILE)
    logger.debug('CV results written to %s', CV_RESULTS_FILE)
    # select param with best average test_rec_at_fixed_pr
    best_model = (
        kf_results
        .groupby('param_index')
        .test_rec_at_fixed_pr
        .mean()
        .sort_values(ascending=False)
        .index[0]
    )
    avg_threshold = kf_results[
        kf_results.param_index == best_model
    ].threshold.mean()

    best_param = kf_results[
        kf_results.param_index == best_model
    ].param.values[0]
    X_train = data[FEATURES]
    y_train = data.target

    clf = xgb.XGBClassifier(
        seed=random_seed,
        n_jobs=-1,
        **best_param,
    )
    clf.fit(X_train, y_train)
    # clf.dump()

    return clf, avg_threshold
