import pandas as pd
import pathlib
import itertools
import string
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from stop_words import get_stop_words


def read_data(filename: str):

    df = pd.read_csv(filename)
    return df


def get_responses(df: pd.DataFrame):
    """ Extract responses and return a pd.DataFrame
    with columns: authorId, questionId, formattedValue

    Args:
        df: dataframe from read_data

    Returns:
        pd.DataFrame with responses
    """
    responses = []
    for i, x in df.iterrows():
        df_tmp = (pd.DataFrame(x.responses).
                  filter(['questionId', 'formattedValue']).
                  assign(authorId=x.authorId))
        responses.append(df_tmp)

    return pd.concat(responses, ignore_index=True)

def extract_responses_by_id(responses: list, key: str='138'):
    """ Extract a specific question

    Args:
        responses: list (example df.iloc[0].responses)
        key: questionId (example '142')

    Returns:
        responses as a string
    """

    response = [x['formattedValue'] for x in responses
                if x['questionId'] == key]
    if len(response):
        return response[0]
    else:
        return None

def find_most_common_words(df: pd.DataFrame, featurename: str):
    x = df[featurename]
    answers = x.formattedValue.values.tolist()
    answers = ' '.join(answers)
    answers = answers.lower()

    stop_words = set(stopwords.words('french') +
                     list(string.punctuation) +
                     get_stop_words('fr'))
    word_tokens = word_tokenize(answers, language='french')

    words = [x for x in word_tokens if x not in stop_words]

    cnt = Counter(words)
    cnt.most_common(20)

