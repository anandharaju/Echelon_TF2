from keras.models import Model
from keras.layers import Dense, Embedding, Conv1D, multiply, GlobalMaxPool1D, Input, Activation, Concatenate
from config import settings as cnst


def model(max_len, win_size, vocab_size=256):
    inp = Input((max_len,))
    emb = Embedding(vocab_size, 8)(inp)

    conv1 = Conv1D(kernel_size=(win_size), filters=cnst.NUM_FILTERS, strides=(win_size), padding='same')(emb)
    conv2 = Conv1D(kernel_size=(win_size), filters=cnst.NUM_FILTERS, strides=(win_size), padding='same')(emb)
    a = Activation('sigmoid', name='sigmoid')(conv2)
    
    mul = multiply([conv1, a])
    a = Activation('relu', name='relu')(mul)
    p = GlobalMaxPool1D()(a)
    # p = GlobalMaxPool1D()(mul)
    d = Dense(cnst.NUM_FILTERS)(p)  #64
    out = Dense(1, activation='sigmoid')(d)

    return Model(inp, out)


def model1(max_len, win_size, vocab_size=256):
    inp = Input((max_len,))
    emb = Embedding(vocab_size, 8)(inp)

    nbyte_win = int(win_size/32)
    kilo_win = int(win_size*2)
    win_size = int(win_size/2)

    conv1n = Conv1D(kernel_size=(nbyte_win), filters=cnst.NUM_FILTERS, strides=(nbyte_win), padding='same')(emb)
    conv2n = Conv1D(kernel_size=(nbyte_win), filters=cnst.NUM_FILTERS, strides=(nbyte_win), padding='same')(emb)
    an = Activation('sigmoid', name='sigmoid1')(conv2n)

    conv1 = Conv1D(kernel_size=(win_size), filters=cnst.NUM_FILTERS, strides=(win_size), padding='same')(emb)
    conv2 = Conv1D(kernel_size=(win_size), filters=cnst.NUM_FILTERS, strides=(win_size), padding='same')(emb)
    a = Activation('sigmoid', name='sigmoid2')(conv2)

    conv1k = Conv1D(kernel_size=(kilo_win), filters=cnst.NUM_FILTERS, strides=(kilo_win), padding='same')(emb)
    conv2k = Conv1D(kernel_size=(kilo_win), filters=cnst.NUM_FILTERS, strides=(kilo_win), padding='same')(emb)
    ak = Activation('sigmoid', name='sigmoid3')(conv2k)

    muln = multiply([conv1n, an])
    an = Activation('relu', name='relu1')(muln)
    pn = GlobalMaxPool1D()(an)

    mul = multiply([conv1, a])
    a = Activation('relu', name='relu2')(mul)
    p = GlobalMaxPool1D()(a)

    mulk = multiply([conv1k, ak])
    ak = Activation('relu', name='relu3')(mulk)
    pk = GlobalMaxPool1D()(ak)

    merged = Concatenate(axis=1)([pn, p, pk])

    # p = GlobalMaxPool1D()(mul)
    d = Dense(cnst.NUM_FILTERS*3)(merged)  # 64
    out = Dense(1, activation='sigmoid')(d)

    return Model(inp, out)