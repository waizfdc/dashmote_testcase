import pathlib

import pandas as pd

from utils.data_manipulation import generate_pairs, test_split
from utils.feature_engineering import feature_preprocessing
from utils.model_selection import pair_metrics, run_kfold
from utils.pairwise_features import add_pairwise_features

# Parameters
RANDOM_SEED = 42
NUM_NEIGHBOURS = 10
FIXED_PRECISION = .95

# Log config
LOG_PATH = './logs'

# Data config
DATA_DIR = '.data/raw'
DATA_FILENAME = 'cs1_us_outlets.parquet.gzip'
DATA_PATH = pathlib.Path(f'{DATA_DIR}/{DATA_FILENAME}')


def main():
    # creating train and test data
    data = pd.read_parquet(DATA_PATH)
    train, test = test_split(data, random_seed=RANDOM_SEED)

    # model selection
    feature_preprocessing(train, inplace=True)
    # creating pairs dataset for pairwise classification
    train_pairs = generate_pairs(train)
    add_pairwise_features(train_pairs, inplace=True)

    # select (kfold cross-validation) and train model
    best_model = run_kfold(
        train_pairs,
        fixed_precision=FIXED_PRECISION,
        random_seed=RANDOM_SEED,
    )

    # model evaluation
    test = feature_preprocessing(test)

    test_pairs = generate_pairs(train, test)
    add_pairwise_features(test_pairs, inplace=True)

    test_pred = best_model.predict_proba(test_pairs)

    print(pair_metrics(test_pairs.target, test_pred))


if __name__ == "__main__":
    main()
