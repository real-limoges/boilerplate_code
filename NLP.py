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

    
def compile_lstm(embeddings, shape, settings):
    '''
    Compiles a 2 Cell LSTM model with shape['nr_class'] classes. Simple architecture
    that uses the same number of hidden units for each LSTM cell. Embeddings are learned
    separaately and passed as a numpy 2D array

    @param embeddings: Numpy 2D array that maps a token at position i to a
                       vector of weights
    @param shape: Dictionary of the shapes of each of the layers (e.g. 
    @param settings: Dictionary of the settings for the LSTM (e.g. batch size)
    '''
    model = Sequential()
    #Add embedding layer to translate indexes of tokens to vectors.
    model.add(
        Embedding(
                embeddings.shape[0],
                embeddings.shape[1],
                input_length=shape['max_length'],
                trainable=False,
                weights=[embeddings],
                mask_zero=True
        )
    )
    #Reshapes the data to be sequences of vectors rather than just vectors
    model.add(
        TimeDistributed(
            Dense(shape['nr_hidden1'], use_bias=False)
        )
    )
    #First LSTM cell that returns sequences to be used in second LSTM cell
    model.add(
        LSTM(shape['nr_hidden1'],
             recurrent_dropout=settings['recurrent_dropout'],
             dropout=settings['dropout'],
             return_sequences=True
        )
    )
    #Accepts sequences from previous LSTM cell
    model.add(
        LSTM(shape['nr_hidden1'],
             recurrent_dropout=settings['recurrent_dropout'],
             dropout=settings['dropout']
        )
    )
    #Add layer that maps LSTM output to shape['nr_class'] classes
    model.add(Dense(shape['nr_class'], activation='softmax', use_bias=False)) 



if __name__ == '__main__':
    FILENAME = 'filename.parquet'

    df = pd.read_parquet(FILENAME)
    df['token_lst'] = transform_messages(df['document'])


