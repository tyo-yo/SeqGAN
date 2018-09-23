import numpy as np
import keras
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Lambda, Activation, Dropout
from keras.layers import Dense, Embedding, LSTM
from keras.layers.wrappers import TimeDistributed

def GeneratorPretraining(V, E, H):
    '''
    Model for Generator pretraining. This model's weights should be shared with
        Generator.
    # Arguments:
        V: int, Vocabrary size
        E: int, Embedding size
        H: int, LSTM hidden size
    # Returns:
        generator_pretraining: keras Model
            input: word ids, shape = (B, T)
            output: word probability, shape = (B, T, V)
    '''
    # in comment, B means batch size, T means lengths of time steps.
    input = Input(shape=(None,), dtype='int32', name='Input') # (B, T)
    out = Embedding(V, E, mask_zero=True, name='Embedding')(input) # (B, T, E)
    out = LSTM(H, return_sequences=True, name='LSTM')(out)  # (B, T, H)
    out = TimeDistributed(Dense(V, activation='softmax'), name='Dense')(out)    # (B, T, V)
    generator_pretraining = Model(input, out)
    return generator_pretraining
