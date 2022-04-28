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
    '''
    Convers string with US phone number to standart form.

    Standart form means:
        - contains only numeric characters
        - starts with '1'
        - length is 11 characters
    If given None returns empty string.

    Parameters:
        s : str | None
            Phone number string.
    Returns:
        str
    '''
    if not s:
        return ''
    stand_phone = re.sub(r'[^0-9]', '', s)
    if stand_phone[0] != '1':
        stand_phone = '1'+stand_phone
    return stand_phone[:11]


def remove_text_in_parentheses(text: str) -> str:
    '''Removes text in parenthesis including brackets etc.'''
    if not text:
        return ''
    return re.sub(r'[\(\[].*?[\)\]]', '', text)


def remove_non_alphanum(text: str) -> str:
    '''Changes all non Latin or numeric characters to whitespaces.'''
    if not text:
        return ''
    # return regex.sub('[^\p{Latin}\p{posix_punct}\d\s]', ' ', text)
    return regex.sub(r'[^\p{Latin}\d\s]', ' ', text)


def remove_double_spaces(text: str) -> str:
    '''Reduces sequential spaces to 1 space.'''
    if not text:
        return
    return re.sub(' +', ' ', text)


def strip(text: str) -> str:
    '''String strip() wrapper.'''
    if not text:
        return ''
    return text.strip()


def lower(text: str) -> str:
    '''String lower() wrapper.'''
    if not text:
        return ''
    return text.lower()


def function_chaining(start, *funcs):
    '''Helper function to organize function chaining.'''
    res = start
    for func in funcs:
        res = func(res)
    return res


def stand_name(text: str) -> str:
    '''
    Converts name of outlet to one format.

    Removes text in brackets, non LAtin or numeric
    characters, double whitespaces. Returns in lowercase.

    Parameters:
        text : str
            Name of an outlet
    Returns:
        str
    '''
    return function_chaining(text,
                             lower,
                             remove_text_in_parentheses,
                             remove_non_alphanum,
                             remove_double_spaces,
                             strip,
                             )


def feature_preprocessing(
    data: DataFrame,
    inplace: bool = False,
    drop_old: bool = True,
) -> Optional[DataFrame]:
    '''
    Adds new columns with parsed features.

    stand_name - standardized name of an outlet.
    stand_phone - standardized phone number of an outlet.
    lat_rad - latitude in radians.
    lon_rad - longitude in radians.

    Parameters:
        data : DataFrame
            Dataset with initial features.
        inplace : bool, default False
            Modify the DataFrame in place (do not create a new object).
        drop_old : bool, default True
    Returns:
        DataFrame or None
    '''
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
    if drop_old:
        data.drop(columns=drop_cols, inplace=True)
        data = data[FINAL_COLUMNS]
    if inplace:
        return
    return data
