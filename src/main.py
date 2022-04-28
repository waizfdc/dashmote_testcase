#!/usr/bin/env python3
'''A script for model training for a problem of outlet matchung.

usage: run with a command line arguments for configuration
of parameters.
'''
import logging
import pathlib
import sys

import pandas as pd
import yaml

from utils.data_manipulation import generate_pairs, test_split
from utils.feature_engineering import feature_preprocessing
from utils.model_selection import metrics_at_threshold, pair_metrics, run_kfold
from utils.pairwise_features import add_pairwise_features

APP_NAME = 'DASHMODE_TestCase'

# config
CONFIG_PATH = './config.yaml'

logger = logging.getLogger(APP_NAME)


def main():
    config = yaml.safe_load(open(CONFIG_PATH, 'r'))
    setup_logging(pathlib.Path(config['log_path']))
    logger.debug('configuration: %s', repr(config))

    # creating train and test data
    data = pd.read_parquet(pathlib.Path(config['data_path']))
    logger.info('Data loaded. shape is %s', data.shape)

    train, test = test_split(data, random_seed=config['random_seed'])
    logger.info(
        'Data splited. train shape is %s, test shape is %s',
        train.shape,
        test.shape
    )
    logger.debug('Test columns: %s', ' '.join(test.columns))

    # model selection
    feature_preprocessing(train, inplace=True)
    logger.debug('Train feature preprocessing finished. '
                 + 'Train shape is %s', train.shape
                 )
    # creating pairs dataset for pairwise classification
    train_pairs = generate_pairs(train)
    logger.debug('Generated train pairs dataset with shape '
                 + '%s', train_pairs.shape
                 )

    add_pairwise_features(train_pairs, inplace=True)
    logger.debug('Added pairwise features. '
                 + 'train_pairs shape is %s',
                 train_pairs.shape
                 )

    # select (kfold cross-validation) and train model
    logger.info('Cross Validation Started.')
    best_model, threshold = run_kfold(
        train_pairs,
        pathlib.Path(config['model_path']),
        pathlib.Path(config['cv_log_path']),
        fixed_precision=config['fixed_precision'],
        random_seed=config['random_seed'],
    )
    logger.info('Best model selected')
    logger.info('Average threshold on train for '
                + 'precision=%s is %s',
                config['fixed_precision'],
                threshold
                )

    # model evaluation
    feature_preprocessing(test, inplace=True, drop_old=False)
    logger.debug('Test feature preprocessing finished. '
                 + 'Test shape is %s',
                 test.shape
                 )
    logger.debug('Test columns: %s', ' '.join(test.columns))

    test_pairs = generate_pairs(train, test, keep_columns=['test_type_x'])
    logger.debug('Generated test pairs dataset with shape '
                 + '%s', test_pairs.shape
                 )
    logger.debug('Test pairs columns: %s', ' '.join(test_pairs.columns))

    add_pairwise_features(test_pairs, inplace=True)
    logger.debug('Added pairwise features. '
                 + 'test_pairs shape is %s',
                 test_pairs.shape
                 )
    logger.debug('Test pairs columns: %s', ' '.join(test_pairs.columns))

    test_pairs['score'] = best_model.predict_proba(
        test_pairs[best_model._Booster.feature_names]
    )[:, 1]
    logger.debug('Predicted scores for test pairs.')

    test_auc, test_recall, test_threshold = (
        pair_metrics(
            test_pairs.target,
            test_pairs.score,
            config['fixed_precision']
        )
    )
    logger.info('Metrics on test pairs are: '
                + 'precision-recall AUC = %s, '
                + 'recall at fixed precision = %s, '
                + 'threshold at fixed precision on test = %s',
                test_auc, test_recall, test_threshold
                )

    test_precision, test_recall, test_f1 = (
        metrics_at_threshold(test_pairs.target, test_pairs.score, threshold)
    )
    logger.info('Metrics on mixed test pairs with train threshold: '
                + 'precision = %s, '
                + 'recall = %s, '
                + 'f1_score = %s',
                test_precision, test_recall, test_f1
                )

    # test on test data consisting only new POI (not seen in train)
    filter = test_pairs.test_type_x == 'poi'
    test_precision, test_recall, test_f1 = (
        metrics_at_threshold(
            test_pairs[filter].target,
            test_pairs[filter].score,
            threshold
        )
    )
    logger.info('Metrics on new POI test pairs with train threshold: '
                + 'precision = %s, '
                + 'recall = %s, '
                + 'f1_score = %s',
                test_precision, test_recall, test_f1
                )


def setup_logging(log_path: pathlib.Path):
    """Setup logging via YAML if it is provided"""

    file_formatter = logging.Formatter(
        fmt='%(asctime)s\t%(levelname)s'
        + '\t%(name)s\t%(module)s\t%(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    file_handler = logging.FileHandler(
        filename=log_path,
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)

    stream_formatter = logging.Formatter(
        fmt='%(message)s',
    )
    stream_handler = logging.StreamHandler(
        stream=sys.stderr,
    )
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(stream_formatter)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    logging.getLogger('numexpr.utils').setLevel(logging.WARNING)


if __name__ == "__main__":
    main()
