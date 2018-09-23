import numpy as np
import keras
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Lambda, Activation, Dropout, Concatenate
from keras.layers import Dense, Embedding, LSTM, Conv1D, GlobalMaxPooling1D
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
    out = TimeDistributed(Dense(V, activation='softmax', name='Dense'), name='TimeDense')(out)    # (B, T, V)
    generator_pretraining = Model(input, out)
    return generator_pretraining

def Disciriminator(V, E, filter_sizes, num_filters, dropout):
    '''
    Disciriminator model.
    # Arguments:
        V: int, Vocabrary size
        E: int, Embedding size
        filter_sizes: list of int, list of each Conv1D filter sizes
        num_filters: list of int, list of each Conv1D num of filters
        dropout: float
    # Returns:
        discriminator: keras model
            input: word ids, shape = (B, T)
            output: probability of true data or not, shape = (B,)
    '''
    input = Input(shape=(None,), dtype='int32', name='Input')   # (B, T)
    out = Embedding(V, E, name='Embedding')(input)  # (B, T, E)
    out = VariousConv1D(out, filter_sizes, num_filters)
    out = Highway(out, num_layers=1)
    out = Dropout(dropout, name='Dropout')(out)
    out = Dense(1, activation='sigmoid', name='FC')(out)

    discriminator = Model(input, out)
    return discriminator

def VariousConv1D(x, filter_sizes, num_filters, name_prefix=''):
    '''
    Layer wrapper function for various filter sizes Conv1Ds
    # Arguments:
        x: tensor, shape = (B, T, E)
        filter_sizes: list of int, list of each Conv1D filter sizes
        num_filters: list of int, list of each Conv1D num of filters
        name_prefix: str, layer name prefix
    # Returns:
        out: tensor, shape = (B, sum(num_filters))
    '''
    conv_outputs = []
    for filter_size, n_filter in zip(filter_sizes, num_filters):
        conv_name = '{}VariousConv1D/Conv1D/filter_size_{}'.format(name_prefix, filter_size)
        pooling_name = '{}VariousConv1D/MaxPooling/filter_size_{}'.format(name_prefix, filter_size)
        conv_out = Conv1D(n_filter, filter_size, name=conv_name)(x)   # (B, time_steps, n_filter)
        conv_out = GlobalMaxPooling1D(name=pooling_name)(conv_out) # (B, n_filter)
        conv_outputs.append(conv_out)
    concatenate_name = '{}VariousConv1D/Concatenate'.format(name_prefix)
    out = Concatenate(name=concatenate_name)(conv_outputs)
    return out

def Highway(x, num_layers=1, activation='relu', name_prefix=''):
    '''
    Layer wrapper function for Highway network
    # Arguments:
        x: tensor, shape = (B, input_size)
    # Optional Arguments:
        num_layers: int, dafault is 1, the number of Highway network layers
        activation: keras activation, default is 'relu'
        name_prefix: str, default is '', layer name prefix
    # Returns:
        out: tensor, shape = (B, input_size)
    '''
    input_size = K.int_shape(x)[1]
    for i in range(num_layers):
        gate_ratio_name = '{}Highway/Gate_ratio_{}'.format(name_prefix, i)
        fc_name = '{}Highway/FC_{}'.format(name_prefix, i)
        gate_name = '{}Highway/Gate_{}'.format(name_prefix, i)

        gate_ratio = Dense(input_size, activation='sigmoid', name=gate_ratio_name)(x)
        fc = Dense(input_size, activation=activation, name=fc_name)(x)
        x = Lambda(lambda args: args[0] * args[2] + args[1] * (1 - args[2]), name=gate_name)([fc, x, gate_ratio])
    return x
