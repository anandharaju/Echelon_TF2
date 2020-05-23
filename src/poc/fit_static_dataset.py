from keras.models import Model, Sequential
from keras import optimizers
from keras.layers import Dense, Embedding, Conv1D, multiply, GlobalMaxPool1D, Input, Activation
# from sklearn.model_selection import GridSearchCV
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
import random
from sklearn import metrics
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import numpy as np
from sklearn.utils import class_weight


#########################################################################
target_confidence = 50
num_of_features = 52  # X.shape[1]
#########################################################################
plt.rcParams.update({'font.size': 14.0})
dataframe = pd.read_csv("D:\\08_Dataset\\Huawei_DS\\data\\features.csv")
Y = dataframe.loc[:, "target"]
dataframe = dataframe.drop('target', axis=1)
dataframe = dataframe.drop(['VersionInformationSize'], axis=1)
X = dataframe.iloc[:, 0:52].values.astype(float)

min = X.min(0)
max = X.max(0)
X = (X - min) / (max - min)


def featuristic():
    model = Sequential()
    model.add(Dense(num_of_features, input_dim=num_of_features, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    return model


model = featuristic()

X_test = None
Y_test = None
pred = None
X = pd.DataFrame(X)
Y = pd.DataFrame(Y)

list_auc = []
list_bacc = []
list_acc = []
list_tpr = []
list_fpr = []
list_fnr = []

for i in range(0, 10):
    # random.seed(1)
    indexes = random.sample(range(0, len(Y)), int(len(Y) * 0.2))
    X_test = X.iloc[indexes].values
    Y_test = Y.iloc[indexes].values.ravel()
    X_train = X.drop(axis=0, index=indexes).values
    Y_train = Y.drop(axis=0, index=indexes).values.ravel()

    # SMOTE = For Oversampling
    # sm = SMOTE(random_state=1, ratio=1.0)  # sampling_strategy
    # X_train, Y_train = sm.fit_sample(X_train, Y_train)
    # X_test, Y_test = sm.fit_sample(X_test, Y_test)

    # Class Imbalance Tackling - Setting class weights
    class_weights = class_weight.compute_class_weight('balanced', np.unique(Y_train), Y_train)
    model.fit(X_train, Y_train, batch_size=100, epochs=1, class_weight=class_weights)

    # model.fit(X_train, Y_train, batch_size=1, epochs=1)
    pred = model.predict(X_test)
    print(metrics.confusion_matrix(Y_test, pred > 0.5))
    print("B.Acc", metrics.balanced_accuracy_score(Y_test, pred > 0.5))

    fpr, tpr, thds = metrics.roc_curve(Y_test, pred)
    auc = metrics.roc_auc_score(Y_test, pred)
    print("ROC AUC Score     : {:0.2f}".format(auc))  # , fpr, tpr)
    plt.plot(fpr, tpr, label="auc=" + str(auc))
    plt.plot([0, 1], [0, 1], linestyle='--')

    acc = metrics.accuracy_score(Y_test, pred > (target_confidence / 100)) * 100
    bacc = metrics.balanced_accuracy_score(Y_test, pred > (target_confidence / 100)) * 100
    cm = metrics.confusion_matrix(Y_test, pred > (target_confidence / 100))
    print('\t\t[Confidence: ' + str(target_confidence) + '%]')
    tn = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tp = cm[1][1]

    TPR = (tp / (tp + fn)) * 100
    FPR = (fp / (fp + tn)) * 100
    FNR = (fn / (fn + tp)) * 100

    list_auc.append(auc)
    list_acc.append(acc)
    list_bacc.append(bacc)
    list_tpr.append(TPR)
    list_fpr.append(FPR)
    list_fnr.append(FNR)

print("Average AUC  : {:0.2f}".format(np.mean(list_auc)))
print("Average Acc  : {:0.2f}".format(np.mean(list_acc)))
print("Average BAcc : {:0.2f}".format(np.mean(list_bacc)))
print("Average TPR  : {:0.2f}".format(np.mean(list_tpr)))
print("Average FPR  : {:0.2f}".format(np.mean(list_fpr)))
print("Average FNR  : {:0.2f}".format(np.mean(list_fnr)))

print("Class Weights:", class_weights)

plt.legend(loc=4)
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()


b = Y_test == 0
m = Y_test == 1

plt.plot(Y_test[b], range(0, len(Y_test[b])), 'bo', label='Target-Benign')
plt.plot(Y_test[m], range(0, len(Y_test[m])), 'ro', label='Target-Malware')

plt.plot(pred[b], range(0, len(pred[b])), 'b1', label='Predicted-Benign')
plt.plot(pred[m], range(0, len(pred[m])), 'r1', label='Predicted-Malware')

plt.xlabel("Probability Range")
plt.ylabel("Samples")
plt.xlim([-0.05, 1.05])
plt.xticks(np.linspace(0, 1, 21))
plt.legend()
plt.grid()
plt.show()




