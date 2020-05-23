from keras.models import Model
from keras import optimizers
from keras.layers import Dense, Embedding, Conv1D, multiply, GlobalMaxPool1D, Input, Activation
from keras.layers import concatenate


def model(max_len, win_size, vocab_size=256):
    inp1 = Input((max_len,), name='ip1')
    emb1 = Embedding(vocab_size, 8)(inp1)

    inp2 = Input((max_len,), name='ip2')
    emb2 = Embedding(vocab_size, 8)(inp2)

    inp3 = Input((max_len,), name='ip3')
    emb3 = Embedding(vocab_size, 8)(inp3)

    emb = concatenate([emb1, emb2, emb3])

    conv1 = Conv1D(kernel_size=(win_size), filters=128, strides=(win_size), padding='same')(emb)
    conv2 = Conv1D(kernel_size=(win_size), filters=128, strides=(win_size), padding='same')(emb)
    a = Activation('sigmoid', name='sigmoid')(conv2)
    
    mul = multiply([conv1, a])
    a = Activation('relu', name='relu')(mul)
    p = GlobalMaxPool1D()(a)
    #p = GlobalMaxPool1D()(mul)
    d = Dense(128)(p)  #64
    out = Dense(1, activation='sigmoid')(d)

    return Model([inp1, inp2, inp3], out)
