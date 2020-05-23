from keras.models import Model, Sequential
from keras.layers import Dense, Embedding, Conv1D, multiply, GlobalMaxPool1D, Input, Activation, Flatten
from keras.layers import concatenate
import tensorflow as tf


def model(max_len, win_size, vocab_size=256):
    ###################################################################################################################
    #                                       Branch 1 - branch for byte sequence
    ###################################################################################################################
    inp1 = Input((max_len,), name='ip1')
    emb = Embedding(vocab_size, 8)(inp1)

    conv1 = Conv1D(kernel_size=(win_size), filters=128, strides=(win_size), padding='same')(emb)
    conv2 = Conv1D(kernel_size=(win_size), filters=128, strides=(win_size), padding='same')(emb)
    a = Activation('sigmoid', name='sigmoid')(conv2)

    mul = multiply([conv1, a])
    a = Activation('relu', name='relu')(mul)
    p = GlobalMaxPool1D()(a)
    # p = GlobalMaxPool1D()(mul)
    d = Dense(128, activation='relu')(p)  # 64
    branch_1 = Dense(48, activation='relu')(d)

    ###################################################################################################################
    #                                    Branch 2 - branch for extracted features
    ###################################################################################################################
    inp2 = Input((52, ), name='ip2')
    d104 = Dense(104, activation='relu')(inp2)
    d52 = Dense(52, activation='relu')(d104)
    branch_2 = Dense(32, activation='relu')(d52)
    # features = Flatten()(features)

    ###################################################################################################################
    #                                                  Branch Merge
    ###################################################################################################################
    combined = concatenate([branch_1, branch_2])
    d80 = Dense(80, activation='relu')(combined)
    d40 = Dense(40, activation='relu')(d80)
    d16 = Dense(16, activation='relu')(d40)
    out = Dense(1, activation='sigmoid')(d16)
    fused_model = Model([inp1, inp2], out)

    return fused_model
