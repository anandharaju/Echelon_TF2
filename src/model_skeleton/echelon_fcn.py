from keras.models import Model
from keras.layers import Dense, Embedding, Conv1D, multiply, GlobalMaxPool1D, Input, Activation, concatenate, Flatten, Average, Maximum, merge


def model(max_len, win_size, vocab_size=256):
    inp = Input((None,))
    emb = Embedding(vocab_size, 8)(inp)

    conv1 = Conv1D(kernel_size=(win_size), filters=cnst.NUM_FILTERS, strides=(win_size), padding='same')(emb)
    conv2 = Conv1D(kernel_size=(win_size), filters=cnst.NUM_FILTERS, strides=(win_size), padding='same')(emb)
    a = Activation('sigmoid', name='sigmoid')(conv2)
    
    mul = multiply([conv1, a])
    a = Activation('relu', name='relu')(mul)
    p = GlobalMaxPool1D()(a)

    d = Dense(cnst.NUM_FILTERS)(p)  #64
    out = Dense(1, activation='sigmoid')(d)

    return Model(inp, out)
