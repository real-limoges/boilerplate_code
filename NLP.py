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

    
def train_gensim_word2vec(data, v_size=100, epcohs=20, min_count=1):
    '''
    Trains a Gensim Word2Vec model on a given Pandas series. Returns the model
    and saves it to a file called 'trained_word2vec.model'

    @param data - Pandas Series of text data
    @param v_size - Size of vectors produced by the word2vec model. E.g. 'the' may
                    be expressed as a vector of v_size floats
    @param epochs - How many iterations over the data that the model is trained on
    @param min_count - Minimum number of times a token appears to have embeddings trained

    @returns - trained Gensim word2vec model
    '''
    train_docs = []
    
    for doc in data:
        s_split = doc.split()
        train_docs.append(s_split)

    g_model = gensim.models.Word2Vec(train_docs, size=v_size, min_count=min_count, window=10, 
                                     workers=12)
    g_model.train(train_docs, total_examples=len(train_docs), epochs=epochs)
    g_model.save('trained_word2vec.model')
    return g_model


def generate_gensim_features(docs, max_length, model):
    '''
    Creates a 3D numpy array initialized to 0 and token labels added at the (document, word number)
    position.

    @param docs - list of lists of tokens
    @param max_length - parameter that controls how long a sequence of tokens is permitted. tokens
                        in a doc beyond this are ignored.
    @param model - Gensim word2vec model

    @return - 3D numpy array.
    '''
    docs = list(docs)
    Xs = np.zeros((len(docs), max_length), dtype='int32')

    for i, doc in enumerate(docs):
        j = 0
        for token in doc:
            try:
                Xs[i, j] = model.wv.vocab[token].index
            except KeyError:
                Xs[i, j] = 0
            j+=1
            if j >= max_length:
                break
    return Xs


def generate_gensim_embeddings(model)
    '''
    Removes the lookup between word and vector for the Gensim model to create a Keras
    embedding layer.

    @param model - Gensim Word2Vec model
    
    @return - Numpy 2D array
    '''
    embedding_matrix = np.zeros((len(model.wv.vocab), v_size))
    for i in xrange(len(model.wv.vocab)):
        embedding_vector = model.wv[model.wv.index2word[i]]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def compile_lstm(embeddings, shape, settings):
    '''
    Compiles a 2 Cell LSTM model with shape['nr_class'] classes. Simple architecture
    that uses the same number of hidden units for each LSTM cell. Embeddings are learned
    separaately and passed as a numpy 2D array

    @param embeddings: Numpy 2D array that maps a token at position i to a
                       vector of weights
    @param shape: Dictionary of the shapes of each of the layers (e.g. 
    @param settings: Dictionary of the settings for the LSTM (e.g. batch size)
    
    @returns - compiled LSTM model
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

    lstm_shape = {'nr_hidden1': 64,
                  'max_length': 25,
                  'v_size': 128,
                  'nr_class': 2}

    lstm_settings = {'dropout': 0.25,
                     'recurrent_dropout': 0.25}

    df = pd.read_parquet(FILENAME)
    df['token_lst'] = transform_messages(df['document'])

    word2vec = train_gensim_word2vec(df['token_str'], v_size=lstm_shape['v_size'])


