from typing import Optional

from pandas.core.frame import DataFrame
# from Levenshtein import distance as text_distance


def longest_common_substring_norm(text1: str, text2: str) -> float:
    '''
    Returns longest common substring of two strings devided
    by their average length.
    '''
    if not text1:
        text1 = ''
    if not text2:
        text2 = ''
    len1, len2 = len(text1), len(text2)
    # check for empty strings
    if len1 * len2 == 0:
        return 0
    dp = [[0 for _ in range(len2+1)] for _ in range(len1+1)]

    for i in range(len1-1, -1, -1):
        for j in range(len2-1, -1, -1):
            if text1[i] == text2[j]:
                dp[i][j] = 1 + dp[i+1][j+1]
            else:
                dp[i][j] = max(dp[i+1][j], dp[i][j+1])

    return dp[0][0] / (len1 + len2) * 2


def levenshtein_distance_norm(text1: str, text2: str) -> float:
    '''
    Returns levenstein distance of two strings devided
    by their average length + 1 (to avoid devision by zero)
    '''
    if not text1:
        text1 = ''
    if not text2:
        text2 = ''
    # return text_distance(text1, text2) / ((len(text1) + len(text2))/2+1)
    return (len(text1) + len(text2))/2


def add_pairwise_features(
    data: DataFrame,
    inplace: bool = True
) -> Optional[DataFrame]:
    '''
    Adds pairwise features to data.

    - phone_distance: normalized Levenshtein distance
    of phone numbers
    - name_levenshtein_norm: normalized Levenshtein distance
    of names
    - name_lcs_norm: normalized Longest Common Substring
    of names

    Parameters:
        data : DataFrame
            Pairs dataset with initial features.
        inplace : bool, default False
            Modify the DataFrame in place (do not create a new object).
    Returns:
        DataFrame or None
    '''
    if not inplace:
        data = data.copy()
    data['phone_distance'] = data.apply(
        lambda row: levenshtein_distance_norm(
            row.stand_phone_x,
            row.stand_phone_y
        ),
        axis=1
    )
    data['name_levenshtein_norm'] = data.apply(
        lambda row: levenshtein_distance_norm(
            row.stand_name_x,
            row.stand_name_y
        ),
        axis=1
    )
    data['name_lcs_norm'] = data.apply(
        lambda row: longest_common_substring_norm(
            row.stand_name_x,
            row.stand_name_y
        ),
        axis=1
    )
    if inplace:
        return
    return data
