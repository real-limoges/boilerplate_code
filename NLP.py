import numpy as np
import pandas as pd

from multiprocessing import cpu_count, Pool

import gensim
import spacy

from keras.callbacks import EarlyStopping
from keras.layers import Dense, Embedding, LSTM, TimeDistributed
from keras.models import Sequential
from keras.optimizers import Nadam
from keras.utils impor np_utils

import keras_metrics as km

from sklearn.model_selection import train_test_split

gensim.models.word2vec.FAST_VERSION = 1


def convert_to_nlp(s):
    '''
    Converts a unicode string to a spaCy document
    
    @param s - unicode string
    @return spaCy document
    '''
    return nlp(s)


def clean_nlp(s):
    '''
    Removes URLs, puncutation, spaCy tokens that aren't alpha 

    @param spaCy document - document to be cleaned
    @return list of unicode tokens
    '''
    token_lst = [token.lemma_ for token in s if token.is_alpha and (not token.is_punct) and
                                             (not token.is_space) and (not token.like_url)]
    return token_lst


def transform_messages(data):
    '''
    Transforms the document to remove unwanted information
    
    @param data: Pandas Series that contains the raw text data
    @return Pandas Series with cleaned data
    '''
    data = data.str.lower()
    data = data.str.replace(r'\W', ' ')
    data = data.apply(convert_to_nlp)

    

if __name__ == '__main__':
    FILENAME = 'filename.parquet'

    df = pd.read_parquet(FILENAME)
    df['token_lst'] = transform_messages(df['document'])


