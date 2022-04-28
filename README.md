# dashmote_testcase
Code for Dashmote testcase for Data Scientist.

## Problem 
Matching outlets from different providers.

## Algorithm
Pairwise classification using XGBClassifier.

Pairs are generated from the same `persistent_cluster`.

Pairs contain only `num_neighbours` nearest neighbours.

## Usage
set 'config.yaml'

run ```python src/main.py```

### Model usage for test
- generate pairs for test data (using Train dataset)
- score this pairs
- sample some for assesors evaluation to estimate threshold for given precision level

## Results
### Metrics for Classifier
precision-recall AUC = 0.96

recall@precision==0.95 = 0.91

### Metrics with train threshold
precision = 0.90

recall = 0.96

f1-score = 0.93

### Metrics on test with new outlets and train threshold
precision = 0.87

recall = 0.95

f1-score = 0.91

TODO: calculate non-pairwise metrics like mean jacard index

## folder structure
```
DASHMOTE_TESTCASE
│   README.md
│   requirments.txt
|   config.yaml  
│
└───data
│   └───raw
│       │   cs1_us_outlets.parquet.gzip
│   
└───models
|   │   ...
|
└───src
│   │   main.py
│   │
│   └───utils
│       │   data_manipulation.py
│       │   feature_engineering.py
│       │   model_selection.py
│       │   pairwise_features.py
|
└───logs
    │   testcase.log
    │   cv.log
```

