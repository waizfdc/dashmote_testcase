import re
from math import radians
from typing import Optional

import regex
from pandas.core.frame import DataFrame

RECORD_COLUMN = 'id_record'
POI_COLUMN = 'id_dashmote'
STRAT_COLUMN = 'persistent_cluster'
INPUT_COLUMNS = [
    RECORD_COLUMN,
    POI_COLUMN,
    STRAT_COLUMN,
    'name',
    'telephone',
    'lat',
    'lon',
]
FINAL_COLUMNS = [
    RECORD_COLUMN,
    POI_COLUMN,
    STRAT_COLUMN,
    'stand_name',
    'stand_phone',
    'lat_rad',
    'lon_rad',
]


def stand_phone(s: str) -> str:
    if not s:
        return ''
    stand_phone = re.sub(r'[^0-9]', '', s)
    if stand_phone[0] != '1':
        stand_phone = '1'+stand_phone
    return stand_phone[:11]


def remove_text_in_parentheses(text: str) -> str:
    if not text:
        return ''
    return re.sub(r'[\(\[].*?[\)\]]', '', text)


def remove_non_alphanum(text: str) -> str:
    if not text:
        return ''
    # return regex.sub('[^\p{Latin}\p{posix_punct}\d\s]', ' ', text)
    return regex.sub(r'[^\p{Latin}\d\s]', ' ', text)


def remove_double_spaces(text: str) -> str:
    if not text:
        return
    return re.sub(' +', ' ', text)


def strip(text: str) -> str:
    if not text:
        return ''
    return text.strip()


def lower(text: str) -> str:
    if not text:
        return ''
    return text.lower()


def function_chaining(start, *funcs):
    res = start
    for func in funcs:
        res = func(res)
    return res


def stand_name(text: str) -> str:
    return function_chaining(text,
                             lower,
                             remove_text_in_parentheses,
                             remove_non_alphanum,
                             remove_double_spaces,
                             strip,
                             )


def feature_preprocessing(
    data: DataFrame,
    inplace: Optional[bool] = True
) -> Optional[DataFrame]:
    # check if data has needed columns
    dif_cols = (set(INPUT_COLUMNS) - set(data.columns))
    if len(dif_cols) != 0:
        raise KeyError(f"data should contain input columns {dif_cols}")
    if not inplace:
        data = data.copy()
    data['stand_name'] = data.name.apply(stand_name)
    data['stand_phone'] = data.telephone.apply(stand_phone)
    data['lat_rad'] = data.lat.apply(radians)
    data['lon_rad'] = data.lon.apply(radians)

    drop_cols = [col for col in data.columns if col not in FINAL_COLUMNS]
    data.drop(columns=drop_cols, inplace=True)
    data = data[FINAL_COLUMNS]
    if inplace:
        return
    return data
