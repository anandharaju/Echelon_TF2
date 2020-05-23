from keras.models import Model, Sequential
from keras.layers import Dense, Embedding, Conv1D, multiply, GlobalMaxPool1D, Input, Activation, Flatten
from keras.layers import concatenate
from keras.layers import TimeDistributed, LSTM
import tensorflow as tf


def model(max_len, win_size, vocab_size=256):
    ###################################################################################################################
    #                                       Branch 1 - branch for byte sequence
    ###################################################################################################################
    inp = Input((max_len,))
    emb = Embedding(vocab_size, 8)(inp)

    conv1 = Conv1D(kernel_size=(win_size), filters=128, strides=(win_size), padding='same')(emb)
    conv2 = Conv1D(kernel_size=(win_size), filters=128, strides=(win_size), padding='same')(emb)
    a = Activation('sigmoid', name='sigmoid')(conv2)

    mul = multiply([conv1, a])
    a = Activation('relu', name='relu')(mul)
    p = GlobalMaxPool1D()(a)
    # p = GlobalMaxPool1D()(mul)
    d = Dense(128)(p)  # 64
    cnn = Dense(64, activation='sigmoid')(d)

    ###################################################################################################################
    #                                                  Merge - CNN & LSTM
    ###################################################################################################################
    # define LSTM model
    td = TimeDistributed(cnn)
    lstm = LSTM(64, return_sequences=True)(a)
    out = Dense(1, activation='sigmoid')(lstm)
    fused_model = Model(inp, out)

    return fused_model
