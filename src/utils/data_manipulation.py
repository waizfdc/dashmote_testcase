from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

# default parameters
RANDOM_SEED = 42
TEST_SIZE_POI = .15
TEST_SIZE_RECORD = .2
NUM_NEIGHBOURS = 10
RECORD_COLUMN = 'id_record'
POI_COLUMN = 'id_dashmote'
STRAT_COLUMN = 'persistent_cluster'
TEST_TYPE_COLUMN = 'test_type'
LOCATION_COLUMNS = [
    'lat_rad',
    'lon_rad',
]
EARTH_RADIUS = 6378100


def add_index_column(
    data: DataFrame,
    record_column: str = RECORD_COLUMN,
    inplace: bool = False
) -> Optional[DataFrame]:
    '''
    Adds a new index-like column to a dataframe.

    Parameters:
        data : DataFrame
            A dataframe in which a new column should be created.
        record_column : str, default True
            A name for a new column.
        inplace : bool, default False
            Modify the DataFrame in place (do not create a new object).

    Returns:
        DataFrame or None.
    '''
    if not inplace:
        data = data.copy()
    data[record_column] = range(len(data))
    if inplace:
        return
    return data


def test_split(
    data: DataFrame,
    random_seed: int = RANDOM_SEED,
    test_size_poi: float = TEST_SIZE_POI,
    test_size_record: float = TEST_SIZE_RECORD,
    record_column: str = RECORD_COLUMN,
    poi_column: str = POI_COLUMN,
    strat_column: str = STRAT_COLUMN,
) -> Tuple[DataFrame, DataFrame]:
    '''
    Splits data in two datasets.

    Test contains of POI data and record data.
    POI data contains POI ids exclusive for test dataset.
    record data contains POI ids which may be in train dataset.
    Test contains extra column 'test_type'.

    Parameters:
        data : DataFrame
            A dataframe to split.
        random_seed : int, default RANDOM_SEED
            Seed for random number generator.
        test_size_poi : float, default TEST_SIZE_POI
            Fraction of POI ids to form a test set.
        test_size_record : float, default TEST_SIZE_RECORD
            Fraction of records to form a test set where POI may be in train.
        record_column : str, default RECORD_COLUMN
            Column name for record id.
        strat_column : str, default STRAT_COLUMN
            Column name for stratification.

    Returns:
        DataFrame, DataFrame
            Returns train and test datasets.
    '''
    if record_column not in data.columns:
        add_index_column(
            data,
            record_column=record_column,
            inplace=True)

    # create unique POI dataset
    unique_poi = data\
        .groupby(poi_column)\
        .first()\
        .reset_index(drop=False, inplace=False)
    _, test_poi = train_test_split(
        unique_poi,
        test_size=test_size_poi,
        stratify=unique_poi[strat_column],
        random_state=random_seed,
    )
    test_poi = set(test_poi[poi_column].to_list())

    # create test dataset with selected POI
    test_poi_df = data[data[poi_column].isin(test_poi)].copy()
    test_poi_df[TEST_TYPE_COLUMN] = 'poi'

    # create test dataset with subsample of records from
    # remaining POI
    remaining_df = data[~data[poi_column].isin(test_poi)]
    _, test_record_df = train_test_split(
        remaining_df,
        test_size=test_size_record,
        stratify=remaining_df[strat_column],
        random_state=random_seed,
    )
    test_record_df = test_record_df.copy()
    test_record_df[TEST_TYPE_COLUMN] = 'record'

    # concat 2 types of test
    test_df = pd.concat([test_poi_df, test_record_df])

    # train is remaining records
    train_df = data[~data[record_column].isin(test_df[record_column])].copy()

    return train_df, test_df


def generate_pairs(
    data: DataFrame,
    test: Optional[DataFrame] = None,
    cluster_col: str = STRAT_COLUMN,
    n_neighbours: int = NUM_NEIGHBOURS,
    keep_columns: Optional[List[str]] = None,
) -> DataFrame:
    '''
    Generates a dataset of pairs of records with target.

    For each record generates `n_neighbours` pairs from nearest neighbours
    from the same cluster.
    Also calculates location distance between pairs and target variable.
    If test dataframe is provided, pairs are generated for
    test records from [data + test] dataset.

    Parameters:
        data : DataFrame
            Dataset from which pairs are generated.
        test : DataFrame | None
            Dataset of test records for which pairs should be generated.
        cluster_col : str, default STRAT_COLUMN
            Name of the cluster column.
        n_neighbours : int, default NUM_NEIGHBOURS
            Number of Neighbours to generate pairs.
        keep_columns: List[str] | None
            List of columns to keep in pairs dataset from test.

    Returns:
        DataFrame
            Returns dataset with features of both records, target
            and distance columns.
    '''
    data = data.copy()
    columns = data.columns

    # append test to find all neighbours
    # and generate pairs only for test rows
    if test is not None:
        test = test.copy()
        test['to_sample'] = True
        data['to_sample'] = False
        data = data.append(test, ignore_index=True)
    else:
        data['to_sample'] = True

    # generate pairs for each cluster in data
    pair_tdfs = []
    for cluster, tdf in data.groupby(cluster_col):
        # print(cluster)
        tdf = tdf.copy()
        add_index_column(tdf, record_column='index', inplace=True)

        # find n neighbours and respected distances
        knn = KNeighborsRegressor(
            n_neighbors=min(len(tdf), n_neighbours + 1),
            metric='haversine',
            n_jobs=-1,
        )
        knn.fit(tdf[LOCATION_COLUMNS], tdf.index)
        distances, neighbours = knn.kneighbors(
            tdf[LOCATION_COLUMNS],
            return_distance=True
        )
        # convert distances to meters
        distances *= EARTH_RADIUS
        # delete self refences
        distances = distances[:, 1:]
        neighbours = neighbours[:, 1:]
        # crete more convinient form for distance storing
        distances_pivot = pd.DataFrame()
        for i in range(len(neighbours)):
            for j in range(len(neighbours[i])):
                distances_pivot = distances_pivot.append(
                    [[i, neighbours[i, j], distances[i, j]]]
                )
        distances_pivot.columns = [
            'index',
            'index_n',
            'location_distance',
        ]

        # create pairs of nearest neighbours. would be more efficient
        # using Spark for large datasets.
        # first join neighbours ids, then join neighbours features
        pair_tdf = (
            tdf[tdf.to_sample]
            .merge(
                distances_pivot,
                on='index',
            )
            .merge(
                tdf,
                left_on='index_n',
                right_on='index',
            )
            .query('index_x != index_y')  # should always be True
        )

        # add target column
        pair_tdf['target'] = np.where(
            pair_tdf[POI_COLUMN+'_x'] == pair_tdf[POI_COLUMN+'_y'], 1, 0)

        ret_columns = [col+'_x' for col in columns] \
            + [col+'_y' for col in columns] \
            + ['target', 'location_distance']
        if keep_columns:
            ret_columns = keep_columns + ret_columns
        pair_tdfs.append(pair_tdf[ret_columns])
    return pd.concat(pair_tdfs).reset_index(drop=True)
