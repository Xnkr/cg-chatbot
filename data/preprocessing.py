import re
import string
import pandas as pd
from .contractions import CONTRACTION_MAP


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


def context_specific_cleaning(text):
    text = text.lower().replace('next ’18', 'next\'18').replace('’', '\'')
    text = expand_contractions(text)
    return text.translate(str.maketrans("", "", string.punctuation.replace('-', '') + '“”')).strip()


def clean_data(df, keep_duplicates='last'):
    df.drop_duplicates(subset=['question'], keep=keep_duplicates, inplace=True)
    if df.isnull().values.any():
        print('Found null values. Dropping null values')
        df.dropna(inplace=True)
    df['clean_question'] = df['question'].apply(context_specific_cleaning)
    df['clean_answer'] = df['answer'].apply(context_specific_cleaning)
    return df


def get_dataframe(file, type='csv'):
    print('Reading file')
    if type == 'csv':
        return pd.read_csv(file)
    if type == 'json':
        return pd.read_json(file)
