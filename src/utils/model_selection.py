from typing import Any, Optional, TypedDict

import numpy as np
import pandas as pd
import xgboost as xgb
from pandas.core.frame import DataFrame
from sklearn.metrics import auc, precision_recall_curve
from sklearn.model_selection import StratifiedKFold

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


class ClassifierParams(TypedDict):
    param: str
    value: Any


def generate_grid(
    params: ClassifierParams,
    shape: int = 5
):
    '''
    generates grid of params of length "shape"
    shape >= 1
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
    return dict_of_params


def pair_metrics(
    y_true,
    pred_proba,
    fixed_precision
):
    '''
    returns precision-recall AUC,
    recall@precision=fixed_precision and threshold
    '''
    precision, recall, thresholds = precision_recall_curve(
        y_true, pred_proba[:, 1])
    pr_rec_auc = auc(recall, precision)
    recall_at_pr = 0
    threshold = 0
    for i in range(len(precision)):
        if precision[i] > fixed_precision:
            recall_at_pr = recall[i]
            threshold = thresholds[i]
            break
    return pr_rec_auc, recall_at_pr, threshold


def run_kfold(
    data: DataFrame,
    fixed_precision: Optional[float] = FIXED_PRECISION,
    random_seed: Optional[int] = RANDOM_SEED,
    grid_size: Optional[int] = GRID_SIZE
) -> xgb.XGBClassifier:
    '''
    runs kfold cross validation to select
    best parameters of a model.
    selection on recall@fixed_precision

    arguments:
        data - pairwise dataset with FEATURES
        and target
        fixed_precision - level of precision at wich
        recall of a model is evaluated
        random_seed - random seed
    '''
    kf_results = pd.DataFrame()

    grid_of_params = generate_grid(CLASSIFIER_PARAMS, grid_size)

    kf = StratifiedKFold(
        n_splits=5,
        random_state=random_seed,
        shuffle=True,
    )
    for train_index, test_index in kf.split(
        data[FEATURES],
        data.target,
    ):
        X_train = data[data.index.isin(train_index)][FEATURES]
        X_test = data[data.index.isin(test_index)][FEATURES]
        y_train = data[data.index.isin(train_index)].target
        y_test = data[data.index.isin(test_index)].target

        for i, _ in sorted(grid_of_params.items(), key=lambda x: x[0]):
            param_index = str(i)
            param = grid_of_params[i]

            clf = xgb.XGBClassifier(
                seed=random_seed,
                n_jobs=-1,
                **param,
            )

            clf.fit(X_train, y_train)
            # test metrics
            test_pr_rec_auc, test_rec_at_fixed_pr, threshold = \
                pair_metrics(y_test, clf.predict_proba(
                    X_test), fixed_precision)

            # train metrics
            train_pr_rec_auc, train_rec_at_fixed_pr, _ = \
                pair_metrics(y_test, clf.predict_proba(
                    X_test), fixed_precision)

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
    kf_results.to_csv('kf_results.csv')
    # select param with best average test_rec_at_fixed_pr
    best_model = (
        kf_results
        .groupby('param_index')
        .test_rec_at_fixed_pr
        .mean()
        .sort_values(ascending=False)
        .index[0]
    )
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

    return clf
