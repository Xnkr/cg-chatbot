import re
import string
import pandas as pd
from .contractions import CONTRACTION_MAP
from static.constants import *


def expand_contractions(text: str, contraction_mapping=CONTRACTION_MAP):
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                      flags=re.IGNORECASE | re.DOTALL)

    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match) \
            if contraction_mapping.get(match) \
            else contraction_mapping.get(match.lower())
        expanded_contraction = first_char + expanded_contraction[1:]
        return expanded_contraction

    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text


def clean_text(text):
    text = text.lower()
    text = expand_contractions(text)
    return text.translate(str.maketrans("", "", string.punctuation.replace('-', '') + '“”')).strip()


def clean_data(df, keep_duplicates='last'):
    df.drop_duplicates(subset=[QUESTION], keep=keep_duplicates, inplace=True)
    if df.isnull().values.any():
        print('Found null values. Dropping null values')
        df.dropna(inplace=True)
    df[CLEAN_QUESTION] = df[QUESTION].apply(clean_text)
    df[CLEAN_ANSWER] = df[ANSWER].apply(clean_text)
    return df


def get_dataframe(file, type=JSON):
    print('Reading file')
    if type == CSV:
        return pd.read_csv(file)
    if type == JSON:
        return pd.read_json(file)
