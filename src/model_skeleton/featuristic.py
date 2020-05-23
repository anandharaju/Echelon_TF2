from keras.models import Model, Sequential
from keras import optimizers
from keras.layers import Dense, Embedding, Conv1D, multiply, GlobalMaxPool1D, Input, Activation
# from sklearn.model_selection import GridSearchCV


'''
def model(num_of_features):
    model = Sequential()
    model.add(Dense(num_of_features * 2, input_dim=num_of_features, activation='relu'))
    model.add(Dense(num_of_features, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model
'''


def model(num_of_features):
    inp = Input((num_of_features,))
    d = Dense(num_of_features * 2, activation='relu')(inp)
    d = Dense(num_of_features, activation='relu')(d)
    d = Dense(32, activation='relu')(d)
    d = Dense(16, activation='relu')(d)
    out = Dense(1, activation='sigmoid')(d)
    return Model(inp, out)
