import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import backend as K
import numpy as np
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from sklearn.utils import class_weight
from sklearn import metrics
import matplotlib.pyplot as plt
from keras.models import load_model, save_model
from keras.models import Model, Sequential
from keras import optimizers
from keras.layers import Dense, Embedding, Conv1D, multiply, GlobalMaxPool1D, Input, Activation, LeakyReLU
from scipy import interp
import time
from sklearn.metrics import plot_confusion_matrix
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import ModelCheckpoint
import seaborn as sns
import matplotlib as mpl
from keras.regularizers import l2
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.constraints import maxnorm
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_fscore_support
from sklearn.metrics import make_scorer
from scipy.stats import randint as sp_randint #sp_randint(0, 32)
from scipy.stats import expon # expon(scale=100, loc=5)
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
import keras
# from sklearn.model_selection import GridSearchCV

# LOAD DATA
data_path = "/content/drive/My Drive/dataset"
df = pd.read_csv(data_path+"/offline_unnormalized.csv")
ytier1 = df['target'].values
ydf = df['target']
df = df.drop(columns=['target'])


df = pd.DataFrame(StandardScaler().fit_transform(df), columns=df.columns)
#df = pd.DataFrame(MinMaxScaler().fit_transform(df), columns=df.columns)
xtier1 = df[:].values
xdf = df[:]

bdf = pd.read_csv(data_path+"/offline_benign_unnormalized.csv")
by = bdf['target'].values
bdf = bdf.drop(columns=['target'])

bdf = pd.DataFrame(StandardScaler().fit_transform(bdf), columns=bdf.columns)
#bdf = pd.DataFrame(MinMaxScaler().fit_transform(bdf), columns=bdf.columns)
bx = bdf[:].values

mdf = pd.read_csv(data_path+"/offline_malware_unnormalized.csv")
my = mdf['target'].values
mdf = mdf.drop(columns=['target'])

mdf = pd.DataFrame(StandardScaler().fit_transform(mdf), columns=mdf.columns)
#mdf = pd.DataFrame(MinMaxScaler().fit_transform(mdf), columns=mdf.columns)
mx = mdf[:].values

drop_keys = None
keys = df.keys()

mean_fpr = None
mean_tpr = None
val_len = 0
train_len = 0

list_auc1 = []
list_tpr1 = []
list_fpr1 = []
list_fnr1 = []
list_tpr_auc1 = []
list_fpr_auc1 = []
list_interp_tpr1 = []

list_aucr = []
list_tprr = []
list_fprr = []
list_fnrr = []
list_tpr_aucr = []
list_fpr_aucr = []
list_interp_tprr = []

list_train_time = []
list_pred_time = []
list_num_dropped_features = []
list_dropped_features = []

list_train_y = []
list_train_pred_val = []
list_test_y = []
list_test_pred = []
list_test_pred_val = []

mean_tier1_micro_precision = []
mean_tier1_micro_recall = []
mean_tier1_micro_f1_score = []

mean_tier1_macro_precision = []
mean_tier1_macro_recall = []
mean_tier1_macro_f1_score = []

mean_tier1_weighted_precision = []
mean_tier1_weighted_recall = []
mean_tier1_weighted_f1_score = []

mean_recon_micro_precision = []
mean_recon_micro_recall = []
mean_recon_micro_f1_score = []

mean_recon_macro_precision = []
mean_recon_macro_recall = []
mean_recon_macro_f1_score = []

mean_recon_weighted_precision = []
mean_recon_weighted_recall = []
mean_recon_weighted_f1_score = []

mean_tier1_class0_precision = []
mean_tier1_class0_recall = []
mean_tier1_class0_f1_score = []
mean_tier1_class0_support = []

mean_tier1_class1_precision = []
mean_tier1_class1_recall = []
mean_tier1_class1_f1_score = []
mean_tier1_class1_support = []

mean_recon_class0_precision = []
mean_recon_class0_recall = []
mean_recon_class0_f1_score = []
mean_recon_class0_support = []

mean_recon_class1_precision = []
mean_recon_class1_recall = []
mean_recon_class1_f1_score = []
mean_recon_class1_support = []

mean_tier1_precision_recall_f1score_support = []
mean_recon_precision_recall_f1score_support = []
count = 1
recon_cm = np.zeros((2,2))

def reset():
    global mean_fpr
    global mean_tpr
    global val_len
    global train_len

    global list_auc1
    global list_tpr1
    global list_fpr1
    global list_fnr1
    global list_tpr_auc1
    global list_fpr_auc1
    global list_interp_tpr1

    global list_aucr
    global list_tprr
    global list_fprr
    global list_fnrr
    global list_tpr_aucr
    global list_fpr_aucr
    global list_interp_tprr

    global list_train_time
    global list_pred_time
    global list_num_dropped_features
    global list_dropped_features

    global list_train_y
    global list_train_pred_val
    global list_test_y
    global list_test_pred
    global list_test_pred_val

    global mean_tier1_micro_precision
    global mean_tier1_micro_recall
    global mean_tier1_micro_f1_score

    global mean_tier1_macro_precision
    global mean_tier1_macro_recall
    global mean_tier1_macro_f1_score

    global mean_tier1_weighted_precision
    global mean_tier1_weighted_recall
    global mean_tier1_weighted_f1_score

    global mean_recon_micro_precision
    global mean_recon_micro_recall
    global mean_recon_micro_f1_score

    global mean_recon_macro_precision
    global mean_recon_macro_recall
    global mean_recon_macro_f1_score

    global mean_recon_weighted_precision
    global mean_recon_weighted_recall
    global mean_recon_weighted_f1_score

    global mean_tier1_class0_precision
    global mean_tier1_class0_recall
    global mean_tier1_class0_f1_score
    global mean_tier1_class0_support 

    global mean_tier1_class1_precision
    global mean_tier1_class1_recall 
    global mean_tier1_class1_f1_score
    global mean_tier1_class1_support 

    global mean_recon_class0_precision
    global mean_recon_class0_recall
    global mean_recon_class0_f1_score 
    global mean_recon_class0_support 

    global mean_recon_class1_precision
    global mean_recon_class1_recall 
    global mean_recon_class1_f1_score
    global mean_recon_class1_support 

    global mean_tier1_precision_recall_f1score_support
    global mean_recon_precision_recall_f1score_support

    global mean_tier1_precision_recall_f1score_support_perclass
    global mean_recon_precision_recall_f1score_support_perclass

    global recon_cm

    mean_fpr = 0
    mean_tpr = 0
    val_len = 0
    train_len = 0

    list_auc1 = []
    list_tpr1 = []
    list_fpr1 = []
    list_fnr1 = []
    list_tpr_auc1 = []
    list_fpr_auc1 = []
    list_interp_tpr1 = []

    list_aucr = []
    list_tprr = []
    list_fprr = []
    list_fnrr = []
    list_tpr_aucr = []
    list_fpr_aucr = []
    list_interp_tprr = []

    list_train_time = []
    list_pred_time = []
    list_num_dropped_features = []
    list_dropped_features = []

    list_train_y = []
    list_train_pred_val = []
    list_test_y = []
    list_test_pred = []
    list_test_pred_val = []

    mean_tier1_micro_precision = []
    mean_tier1_micro_recall = []
    mean_tier1_micro_f1_score = []

    mean_tier1_macro_precision = []
    mean_tier1_macro_recall = []
    mean_tier1_macro_f1_score = []

    mean_tier1_weighted_precision = []
    mean_tier1_weighted_recall = []
    mean_tier1_weighted_f1_score = []

    mean_recon_micro_precision = []
    mean_recon_micro_recall = []
    mean_recon_micro_f1_score = []

    mean_recon_macro_precision = []
    mean_recon_macro_recall = []
    mean_recon_macro_f1_score = []

    mean_recon_weighted_precision = []
    mean_recon_weighted_recall = []
    mean_recon_weighted_f1_score = []

    mean_tier1_class0_precision = []
    mean_tier1_class0_recall = []
    mean_tier1_class0_f1_score = []
    mean_tier1_class0_support = []

    mean_tier1_class1_precision = []
    mean_tier1_class1_recall = []
    mean_tier1_class1_f1_score = []
    mean_tier1_class1_support = []

    mean_recon_class0_precision = []
    mean_recon_class0_recall = []
    mean_recon_class0_f1_score = []
    mean_recon_class0_support = []

    mean_recon_class1_precision = []  
    mean_recon_class1_recall = []
    mean_recon_class1_f1_score = []
    mean_recon_class1_support = []

    mean_tier1_precision_recall_f1score_support = []
    mean_recon_precision_recall_f1score_support = []
    mean_tier1_precision_recall_f1score_support_perclass = []
    mean_recon_precision_recall_f1score_support_perclass = []
    count = 1
    recon_cm = np.zeros((2,2))
	
def select_decision_threshold(Y, pred, target_fpr):
    threshold = 0.0
    selected_threshold = 100
    while threshold <= 100.0:
        cm = metrics.confusion_matrix(Y, (pred >= (threshold / 100)).astype(int))
        tn = cm[0][0]
        fp = cm[0][1]
        fn = cm[1][0]
        tp = cm[1][1]

        TPR = (tp / (tp + fn)) * 100
        FPR = (fp / (fp + tn)) * 100
        if FPR <= target_fpr:
            selected_threshold = threshold
            print("Training Threshold :", threshold, "TPR: {:6.2f}".format(TPR), "\tFPR: {:6.2f}".format(FPR))
            break
        else:
            threshold += 0.1  
    return selected_threshold


def display_probability_chart(Y, pred, threshold, plot_name):
    b = Y == 0
    m = Y == 1

    dpi = 300
    figsize = (6, 2.7)
    plt.rcParams.update({'font.size': 6})
    plt.figure(num=None, figsize=figsize, dpi=dpi)

    ax = plt.subplot(111)
    ax.plot(pred[b], range(0, len(pred[b])), 'b<', markersize=5)
    ax.plot(pred[m], range(0, len(pred[m])), 'r>', markersize=5)

    ax.plot(Y[b], range(0, len(Y[b])), 'bo', label='Target-Benign', markersize=8)
    ax.plot(Y[m], range(0, len(Y[m])), 'ro', label='Target-Malware', markersize=8)

    if threshold is not None:
        ax.plot([threshold/100, threshold/100], [0, len(Y)], 'black', linestyle=':', label="Selected Threshold")

    ax.set_xticks(np.linspace(0, 1, 21))
    ax.set_xlabel("Probability Range", fontsize=6)
    ax.set_ylabel(plot_name[:9] + " Samples", fontsize=6)

    ax.tick_params(axis='both', which='major', labelsize=6)

    ax.set_xlim([-0.05, 1.05])
    ax.grid(linewidth=0.2)

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.10), ncol=3, fancybox=True, prop={'size': 5})
    plt.savefig(plot_name+".png", bbox_inches='tight')
    plt.close()
	
def plot_fpr_train_vs_test(list_train_y, list_train_pred_val, list_test_y, list_test_pred_val):
    plt.clf()
    dpi = 300
    figsize = (4, 4)
    plt.figure(num=None, figsize=figsize, dpi=dpi)

    list_train_fpr = []
    list_test_fpr = []

    for i in range(0, len(list_train_y)):
        fpr_list = []
        for confidence in np.arange(0, 101, 1):
            cm = metrics.confusion_matrix(list_train_y[i], list_train_pred_val[i] > (confidence/100))
            tn = cm[0][0]
            fp = cm[0][1]
            fn = cm[1][0]
            tp = cm[1][1]
            TPR = (tp / (tp + fn)) * 100
            FPR = (fp / (fp + tn)) * 100
            FNR = (fn / (fn + tp)) * 100
            fpr_list.append(FPR)
        list_train_fpr.append(fpr_list)

    for i in range(0, len(list_test_y)):
        fpr_list = []
        for confidence in np.arange(0, 101, 1):
            cm = metrics.confusion_matrix(list_test_y[i], list_test_pred_val[i] > (confidence/100))
            tn = cm[0][0]
            fp = cm[0][1]
            fn = cm[1][0]
            tp = cm[1][1]
            TPR = (tp / (tp + fn)) * 100
            FPR = (fp / (fp + tn)) * 100
            FNR = (fn / (fn + tp)) * 100
            fpr_list.append(FPR)
        list_test_fpr.append(fpr_list)

    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.title("FPR - Training vs Testing", fontdict = {'fontsize' : 6})
    plt.xlabel("Training FPR %", fontsize=6)
    plt.ylabel("Testing FPR %", fontsize=6)
    plt.plot([1, 1], [0, 1], 'black', linestyle=':', label="Target FPR")
    plt.plot([0, 1], [1, 1], 'black', linestyle=':')
    plt.xlim(0, 2)
    plt.ylim(0, 2)
    plt.plot(np.mean(list_train_fpr, axis=0)[1:], np.mean(list_test_fpr, axis=0)[1:], color='black', label="FPR - Train vs Test")
    plt.legend(loc=1, prop={'size': 5})
    plt.savefig("FPR_train_vs_test_"+str(time.time())+".png", bbox_inches='tight')
    plt.close()


def plot_auc_reconciled(Y, pred, prob, tier):
    plt.clf()
    # AUC PLOT
    dpi = 300
    figsize = (4, 2.7)
    plt.figure(num=None, figsize=figsize, dpi=dpi)
  
    acc = metrics.accuracy_score(Y, pred)
    bacc = metrics.balanced_accuracy_score(Y, pred)
    cm = metrics.confusion_matrix(Y, pred)

    tn = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tp = cm[1][1]

    TPR = (tp/(tp+fn))*100
    FPR = (fp/(fp+tn))*100
    FNR = (fn/(fn+tp))*100
    fpr_auc, tpr_auc, thds_auc = metrics.roc_curve(Y, prob, drop_intermediate=False)
    auc = metrics.roc_auc_score(Y, prob, max_fpr=cnst.OVERALL_TARGET_FPR/100)

    global mean_recon_micro_precision
    global mean_recon_micro_recall
    global mean_recon_micro_f1_score

    global mean_recon_macro_precision
    global mean_recon_macro_recall
    global mean_recon_macro_f1_score

    global mean_recon_weighted_precision
    global mean_recon_weighted_recall
    global mean_recon_weighted_f1_score

    global mean_recon_perclass_precision
    global mean_recon_perclass_recall
    global mean_recon_perclass_f1_score
    global mean_recon_perclass_support

    global mean_recon_class0_precision
    global mean_recon_class0_recall
    global mean_recon_class0_f1_score 
    global mean_recon_class0_support 

    global mean_recon_class1_precision
    global mean_recon_class1_recall 
    global mean_recon_class1_f1_score
    global mean_recon_class1_support 
        
    mean_recon_micro_precision.append(precision_recall_fscore_support(Y, pred, average='micro')[0])
    mean_recon_micro_recall.append(precision_recall_fscore_support(Y, pred, average='micro')[1])
    mean_recon_micro_f1_score.append(precision_recall_fscore_support(Y, pred, average='micro')[2])

    mean_recon_macro_precision.append(precision_recall_fscore_support(Y, pred, average='macro')[0])
    mean_recon_macro_recall.append(precision_recall_fscore_support(Y, pred, average='macro')[1])
    mean_recon_macro_f1_score.append(precision_recall_fscore_support(Y, pred, average='macro')[2])

    mean_recon_weighted_precision.append(precision_recall_fscore_support(Y, pred, average='weighted')[0])
    mean_recon_weighted_recall.append(precision_recall_fscore_support(Y, pred, average='weighted')[1])
    mean_recon_weighted_f1_score.append(precision_recall_fscore_support(Y, pred, average='weighted')[2])

    data = precision_recall_fscore_support(Y, pred, average=None, labels=[0,1])
    mean_recon_class0_precision.append(data[0][0])
    mean_recon_class0_recall.append(data[1][0])
    mean_recon_class0_f1_score.append(data[2][0])
    mean_recon_class0_support.append(data[3][0])

    mean_recon_class1_precision.append(data[0][1])
    mean_recon_class1_recall.append(data[1][1])
    mean_recon_class1_f1_score.append(data[2][1])
    mean_recon_class1_support.append(data[3][1])

    #mean_recon_precision += metrics.precision_score(Y, pred)
    #mean_recon_recall    += metrics.recall_score(Y, pred)
    #mean_recon_f1_score  += metrics.f1_score(Y, pred)

    print("%5s & %5s & %5s & %5s" % (str(tn), str(fp), str(fn), str(tp)), 
          "  [ FPR: {:5.2f} ]".format(FPR), 
          "[ TPR (Recall): {:5.2f} ]".format(metrics.recall_score(Y, pred)), 
          "[ Precision: {:5.2f} ]".format(metrics.precision_score(Y, pred)), 
          "[ F1-score: {:5.2f} ]".format(metrics.f1_score(Y, pred)))
          #, '[Acc: ' + str(acc)[:6] + "] [Balanced Acc: " + str(bacc)[:6] + ']')

    plt.plot([0.01, 0.01], [0, 1], 'black', linestyle=':', label="Target FPR")
    plt.plot([0, 0.022], [0.9, 0.9], 'black', linestyle='-.', label="Target TPR")

    plt.xlabel("FPR", fontsize=6)
    plt.ylabel("TPR", fontsize=6)
    plt.xticks(np.arange(0, 0.022, 0.002), fontsize=6)
    plt.yticks(fontsize=6)
    plt.xlim(0, 0.022)
    #plt.title(plot_title, fontdict = {'fontsize' : 6})
    plt.plot(fpr_auc, tpr_auc, label=r'Restricted ROC AUC = %0.5f' % (auc), lw=2, alpha=.8)
    plt.legend(loc=1, prop={'size': 4})
    plt.savefig("auc_"+tier+".png", bbox_inches='tight')
    plt.close()

    confusion_matrix_heatmap(cm, tier)
    global recon_cm
    cm = np.array(cm)
    recon_cm += cm
    return TPR, FPR, FNR, auc, fpr_auc, tpr_auc, thds_auc
	
def plot_auc(Y, prob, confidence, tier):
    pred = prob > (confidence / 100)

    plt.clf()
    # AUC PLOT
    dpi = 300
    figsize = (4, 2.7)
    plt.figure(num=None, figsize=figsize, dpi=dpi)
  
    acc = metrics.accuracy_score(Y, pred)
    bacc = metrics.balanced_accuracy_score(Y, pred)
    cm = metrics.confusion_matrix(Y, pred)

    tn = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tp = cm[1][1]
    TPR = (tp/(tp+fn))*100
    FPR = (fp/(fp+tn))*100
    FNR = (fn/(fn+tp))*100
    fpr_auc, tpr_auc, thds_auc = metrics.roc_curve(Y, prob, drop_intermediate=False)
    auc = metrics.roc_auc_score(Y, prob, max_fpr=cnst.OVERALL_TARGET_FPR/100)

    if "tier1" in tier:
        global mean_tier1_micro_precision
        global mean_tier1_micro_recall
        global mean_tier1_micro_f1_score

        global mean_tier1_macro_precision
        global mean_tier1_macro_recall
        global mean_tier1_macro_f1_score

        global mean_tier1_weighted_precision
        global mean_tier1_weighted_recall
        global mean_tier1_weighted_f1_score

        global mean_tier1_perclass_precision
        global mean_tier1_perclass_recall
        global mean_tier1_perclass_f1_score
        global mean_tier1_perclass_support

        global mean_tier1_class0_precision
        global mean_tier1_class0_recall
        global mean_tier1_class0_f1_score
        global mean_tier1_class0_support 

        global mean_tier1_class1_precision
        global mean_tier1_class1_recall 
        global mean_tier1_class1_f1_score
        global mean_tier1_class1_support 

        mean_tier1_micro_precision.append(precision_recall_fscore_support(Y, pred, average='micro')[0])
        mean_tier1_micro_recall.append(precision_recall_fscore_support(Y, pred, average='micro')[1])
        mean_tier1_micro_f1_score.append(precision_recall_fscore_support(Y, pred, average='micro')[2])

        mean_tier1_macro_precision.append(precision_recall_fscore_support(Y, pred, average='macro')[0])
        mean_tier1_macro_recall.append(precision_recall_fscore_support(Y, pred, average='macro')[1])
        mean_tier1_macro_f1_score.append(precision_recall_fscore_support(Y, pred, average='macro')[2])

        mean_tier1_weighted_precision.append(precision_recall_fscore_support(Y, pred, average='weighted')[0])
        mean_tier1_weighted_recall.append(precision_recall_fscore_support(Y, pred, average='weighted')[1])
        mean_tier1_weighted_f1_score.append(precision_recall_fscore_support(Y, pred, average='weighted')[2])

        data = precision_recall_fscore_support(Y, pred, average=None, labels=[0,1])
        mean_tier1_class0_precision.append(data[0][0])
        mean_tier1_class0_recall.append(data[1][0])
        mean_tier1_class0_f1_score.append(data[2][0])
        mean_tier1_class0_support.append(data[3][0])

        mean_tier1_class1_precision.append(data[0][1])
        mean_tier1_class1_recall.append(data[1][1])
        mean_tier1_class1_f1_score.append(data[2][1])
        mean_tier1_class1_support.append(data[3][1])

        #mean_tier1_precision += metrics.precision_score(Y, pred) 
        #mean_tier1_recall    += metrics.recall_score(Y, pred)
        #mean_tier1_f1_score  += metrics.f1_score(Y, pred)

        #print("mean_tier1_precision", mean_tier1_precision)

    print("%5s & %5s & %5s & %5s" % (str(tn), str(fp), str(fn), str(tp)), 
          "  [ FPR: {:5.2f} ]".format(FPR), 
          "[ TPR (Recall): {:5.2f} ]".format(metrics.recall_score(Y, pred)), 
          "[ Precision: {:5.2f} ]".format(metrics.precision_score(Y, pred)), 
          "[ F1-score: {:5.2f} ]".format(metrics.f1_score(Y, pred)), 
          '[Confidence: ' + str(confidence) + '%]') 
          #, '[Acc: ' + str(acc)[:6] + "] [Balanced Acc: " + str(bacc)[:6] + ']')

    plt.plot([0.01, 0.01], [0, 1], 'black', linestyle=':', label="Target FPR")
    plt.plot([0, 0.022], [0.9, 0.9], 'black', linestyle='-.', label="Target TPR")

    plt.xlabel("FPR", fontsize=6)
    plt.ylabel("TPR", fontsize=6)
    plt.xticks(np.arange(0, 0.022, 0.002), fontsize=6)
    plt.yticks(fontsize=6)
    plt.xlim(0, 0.022)
    #plt.title(plot_title, fontdict = {'fontsize' : 6})
    plt.plot(fpr_auc, tpr_auc, label=r'Restricted ROC AUC = %0.5f' % (auc), lw=2, alpha=.8)
    plt.legend(loc=1, prop={'size': 4})
    plt.savefig("auc_"+tier+".png", bbox_inches='tight')
    plt.close()
    return TPR, FPR, FNR, auc, fpr_auc, tpr_auc, thds_auc
	
def plot_cv_auc(mean_fpr, list_interp_tpr1, list_interp_tprr, list_auc1, list_aucr, recon_results, fpr1, fprr):
    dpi = 300
    figsize = (4, 2.7)
    plt.figure(num=None, figsize=figsize, dpi=dpi)

    plt.text(0.018, 0.52, "Tier - 1 FPR:{:6.2f}".format(np.mean(list_fpr1)) + " [#files: "+str(int(np.mean(list_fpr1)*recon_results.iloc[2]['support']/100))+"]", size=5, rotation=0, ha="center", va="center", bbox=dict(boxstyle="round", ec=(1, 0, 0), fc=(1, 0.6, 0.6)))
    plt.text(0.018, 0.45, "Overall FPR:{:6.2f}".format(np.mean(list_fprr)) + " [#files: "+str(int(np.mean(list_fprr)*recon_results.iloc[2]['support']/100))+"]", size=5, rotation=0, ha="center", va="center", bbox=dict(boxstyle="round", ec=(0, 0, 1), fc=(0.8, 1, 1)))

    table = plt.table(cellText=recon_results.values,
          rowLabels=recon_results.index,
          colLabels=recon_results.columns,
          colWidths = [0.15, 0.15, 0.15, 0.15, 0.15], 
          cellLoc = 'right', 
          rowLoc = 'center',
          #cellColours = ['grey'],
          edges = 'vertical',
          loc='right', zorder=1, bbox=[0.455,0.05,0.53,0.28])

    table.set_fontsize(6)
    table.scale(1.5, 1.5)

    plt.xlabel("FPR", fontsize=6)
    plt.ylabel("TPR", fontsize=6)
    plt.xticks(np.arange(0, 0.022, 0.002), fontsize=6)
    plt.yticks(fontsize=6)
    plt.xlim(0, 0.022)
    plt.title("5-FOLD CV Mean Results", fontdict = {'fontsize' : 6})

    mean_tpr1 = np.mean(list_interp_tpr1, axis=0)
    mean_tpr1[-1] = 1.0
    mean_full_auc1 = metrics.auc(mean_fpr, mean_tpr1)
    mean_auc1 = np.mean(list_auc1)
    std_auc1 = np.std(list_auc1)
    plt.plot(mean_fpr, mean_tpr1, color='r', linewidth=1, label=r'Tier-1 ROC  (Restricted AUC = %0.3f $\pm$ %0.2f) [Full AUC: %0.3f]' % (mean_auc1, std_auc1, mean_full_auc1), lw=2, alpha=.8)
    plt.legend(loc=1, prop={'size': 4})
    plt.savefig("auc_mean_tier1.png", bbox_inches='tight')

    mean_tprr = np.mean(list_interp_tprr, axis=0)
    mean_tprr[-1] = 1.0
    mean_full_aucr = metrics.auc(mean_fpr, mean_tprr)
    mean_aucr = np.mean(list_aucr)
    std_aucr = np.std(list_aucr)
    plt.plot(mean_fpr, mean_tprr, color='b', linewidth=1, label=r'Overall ROC (Restricted AUC = %0.3f $\pm$ %0.2f) [Full AUC: %0.3f]' % (mean_aucr, std_aucr, mean_full_aucr), lw=2, alpha=.8)
    plt.legend(loc=1, prop={'size': 4})

    plt.plot([0.01, 0.01], [0, 1], 'grey', linestyle=':', linewidth=1, label="Target FPR")
    plt.plot([0, 0.022], [0.9, 0.9], 'grey', linestyle='-.', linewidth=1, label="Target TPR")

    std_tpr1 = np.std(list_interp_tpr1, axis=0)
    tpr1_upper = np.minimum(mean_tpr1 + std_tpr1, 1)
    tpr1_lower = np.maximum(mean_tpr1 - std_tpr1, 0)
    plt.fill_between(mean_fpr, tpr1_lower, tpr1_upper, color='r', alpha=.2) #, label=r'$\pm$ 1 std. dev.')

    std_tprr = np.std(list_interp_tprr, axis=0)
    tprr_upper = np.minimum(mean_tprr + std_tprr, 1)
    tprr_lower = np.maximum(mean_tprr - std_tprr, 0)
    plt.fill_between(mean_fpr, tprr_lower, tprr_upper, color='b', alpha=.2) #, label=r'$\pm$ 1 std. dev.')

    plt.savefig("auc_mean_reconciled_"+str(time.time())+".png", bbox_inches='tight') 
    plt.close()
	
def plot_activation_trend(keys, bgrad_norm, mgrad_norm):
    plt.clf()
    dpi = 300
    figsize = (6, 2.7)
    plt.figure(num=None, figsize=figsize, dpi=dpi)

    plt.rcParams.update({"font.size":5})
    plt.ylabel('ACTIVATION TREND')
    plt.xlabel('FEATURE')

    plt.plot(bgrad_norm, 'black', marker='o', markersize=3, label='Benign')
    plt.plot(mgrad_norm, 'black', marker='o', markersize=3, label='Malware')

    plt.plot(bgrad_norm, 'grey', linewidth=0.3)
    plt.plot(mgrad_norm, 'red', linewidth=0.3)
    
    for i, key in enumerate(keys):
        x1, y1 = [i, i], [0, bgrad_norm[i]]
        plt.plot(x1, y1, 'black', linestyle=':', linewidth=0.75)

    for i, key in enumerate(keys):
        x1, y1 = [i, i], [0, mgrad_norm[i]]
        plt.plot(x1, y1, 'red', linestyle=':', linewidth=0.75)
    
    plt.xticks(range(0, len(keys)), keys, rotation=80, fontsize=3)
    plt.yticks(fontsize=5)
    plt.legend()    
    plt.plot([0,len(keys)],[0,0], 'black', linestyle='-.', linewidth=0.3)
    plt.savefig("offline_activation_gradient.png", bbox_inches='tight')
    plt.close()
	
def confusion_matrix_heatmap(cm, tier):
    plt.clf()
    mpl.style.use('seaborn')
    row1 = np.sum(cm[0])
    row2 = np.sum(cm[1])
    cm = np.array(cm) * 1.0
    cm[0][0] = cm[0][0] / ( 1.0 * row1 )
    cm[0][1] = cm[0][1] / ( 1.0 * row1 )
    cm[1][0] = cm[1][0] / ( 1.0 * row2 )
    cm[1][1] = cm[1][1] / ( 1.0 * row2 )

    # CM matching Huawei's plot
    hcm = np.empty_like(cm)
    hcm[0][0] = cm[1][1]
    hcm[0][1] = cm[1][0]
    hcm[1][0] = cm[0][1]
    hcm[1][1] = cm[0][0]

    df_cm = pd.DataFrame(hcm, index = [ 0, 1], columns = [0, 1])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    #ax.set_aspect(1)
    ax.set_xticklabels(labels=[0,1], fontsize=14) 
    ax.set_yticklabels(labels=[0,1], fontsize=14) 

    cmap = sns.cubehelix_palette(light=1, as_cmap=True, start=2, hue=1)
    cmap = sns.color_palette("Blues", n_colors=1000)
    res = sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}, vmin=0.0, vmax=1.0, cmap=cmap)

    plt.yticks([0.5,1.5], [0, 1], va='center')
    plt.title('Normalized Confusion Matrix', fontsize=14)
    plt.xlabel("Predicted Label", fontsize=14)
    plt.ylabel("True Label", fontsize=14)
    plt.savefig("cm_heatmap_"+tier+".png", bbox_inches='tight', dpi=100)
    plt.close()
	
def get_gradients(model, bx, mx):
    # Defining input as tensor
    bxtensor = tf.Variable(bx, dtype=tf.float32)
    mxtensor = tf.Variable(mx, dtype=tf.float32)

    # Defining and Executing Keras Gradient function
    bgrad = K.gradients(model(bxtensor), bxtensor)[0]
    mgrad = K.gradients(model(mxtensor), mxtensor)[0]
    bgrad_val = bgrad.eval(session=K.get_session())
    mgrad_val = mgrad.eval(session=K.get_session())
    print("Benign  gradients dy/dx  : ", np.max(bgrad_val), np.min(bgrad_val))
    print("Malware gradients dy/dx  : ", np.max(mgrad_val), np.min(mgrad_val))
    bgrad_norm = np.average(bgrad_val, axis=0)
    mgrad_norm = np.average(mgrad_val, axis=0)

    #bgrad_norm = np.abs(bgrad_norm)
    #mgrad_norm = np.abs(mgrad_norm)
    
    return bgrad_norm, mgrad_norm
	
def prepare_B1(xB1, yB1, drop_feature_indices):
    xtier2 = np.delete(xB1, drop_feature_indices, axis=1)
    ytier2 = yB1
    # print(len(drop_feature_indices), drop_feature_indices, np.shape(xB1))
    return xtier2, ytier2
	
def select_features(bgrad, mgrad, q_criteria):
    is_qualified = np.abs(bgrad - mgrad) > q_criteria
    is_qualified = np.all([bgrad < q_criteria, mgrad < q_criteria], axis=0)
    #is_qualified = np.any([bgrad < q_criteria, mgrad < q_criteria], axis=0) 
    selected_keys = keys[is_qualified]
    dropped_keys = keys[is_qualified == False]
    drop_feature_indices = np.where(is_qualified == False)[0]
    print("Selecting features :", len(selected_keys), len(dropped_keys), "\t\t\t\t\t [Reduction Rate: {:6.2f}% ]".format((len(dropped_keys)/len(keys))*100)) #, "\nDropped Keys: ", dropped_keys.values)  # "\nSelected Keys: ", selected_keys, 
    return selected_keys, dropped_keys, drop_feature_indices
	
def ATI(model, xB1, yB1, keys):
    # To split the collected B1 data into Benign FN files for gradient calculation
    row_indices = np.where(yB1 == 0)[0]
    xb = xB1[row_indices,:]
    yb = yB1[row_indices]
    # print(len(row_indices))

    row_indices = np.where(yB1 == 1)[0]
    xfn = xB1[row_indices,:]
    yfn = yB1[row_indices]
    # print(len(row_indices))

    bgrad_norm, mgrad_norm = get_gradients(model, xb, xfn)
    plot_activation_trend(keys, bgrad_norm, mgrad_norm)
    return bgrad_norm, mgrad_norm
	
def get_bfn_mfp(x, y, prediction_probability, thd):
    prediction = (prediction_probability > (thd / 100)).astype(int)
    # To filter the Benign FN files from prediction results
    row_indices = np.where(prediction == 0)[0]
    xB1 = x[row_indices,:]
    yB1 = y[row_indices]
    yprobB1 = prediction_probability[row_indices]

    row_indices = np.where(prediction == 1)[0]
    xM1 = x[row_indices,:]
    yM1 = y[row_indices]
    yprobM1 = prediction_probability[row_indices]

    return xB1, yB1, yprobB1, xM1, yM1, yprobM1
	
def fall_out(y_true, y_pred_proba, thd=0.5):
    y_pred = (y_pred_proba > thd).astype(int)
    cm =confusion_matrix(y_true, y_pred)
    tn = cm[0, 0]
    fp = cm[0, 1]
    fn = cm[1, 0]
    tp = cm[1, 1]
    fpr = fp / (fp + tn)
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    f1 = 2 * ((p * r)/(p + r))
    global count
    print(count, "FPR:{0:6.2f}\tRecall:{1:6.2f}\tPrecision:{2:6.2f}\tF1-score:{3:6.2f}\t\t".format(fpr*100,r*100,p*100,f1*100), np.unique(y_true), np.unique(y_pred), cm[0][0], cm[0][1], cm[1][0], cm[1][1])
    count += 1
    return fpr

def f1_score_fn(y_true, y_pred_proba, thd=0.5):
    y_pred = (y_pred_proba > thd).astype(int)
    cm =confusion_matrix(y_true, y_pred)
    tn = cm[0, 0]
    fp = cm[0, 1]
    fn = cm[1, 0]
    tp = cm[1, 1]
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    f1 = 2 * ((p * r)/(p + r))
    return f1

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
	
def predict(model, x, y):
    prediction_probability = model.predict(x)
    return prediction_probability
	
def train(x, y, num_of_features, tier, epochs, batch_size):
    if 'tier1' in tier:
        cw = class_weight.compute_class_weight('balanced', np.unique(y), y)
    else:
        #cw = class_weight.compute_class_weight('None', np.unique(y), y)
        cw = class_weight.compute_class_weight('balanced', np.unique(y), y)
        #cw = cw[::-1]
        #cw[0] = cw[0] * 3
    model = create_model(num_of_features)

    chk = ModelCheckpoint("offline_featuristic_"+tier+".h5", monitor='val_loss', save_best_only=True)
    # model = KerasClassifier(build_fn=create_model, num_of_features=num_of_features, verbose=0)
    model.fit(x, y, batch_size=batch_size, epochs=epochs, class_weight=cw, verbose=0, callbacks=[chk])
    
    save_model(model, "offline_featuristic_"+tier+".h5")
    return model
	
def do_cross_validation(x, y, folds, keys, epochs, batch_size):
    tst = time.time()
    skf = StratifiedKFold(n_splits=folds, shuffle=True)
    for index, (train_indices, val_indices) in enumerate(skf.split(x, y)):
        
        xtrain, xval = x[train_indices], x[val_indices]
        ytrain, yval = y[train_indices], y[val_indices]

        if index ==0:
            val_len = len(xval)
            train_len = len(xtrain)
        print("\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> [ CV-FOLD " + str(index+1) + "/"+str(folds)+" ]", "Training: "+str(len(xtrain)), "Testing: "+str(len(xval)))

        model1 = train(xtrain, ytrain, num_of_features=len(keys), tier="tier1", epochs=epochs, batch_size=batch_size)       # TIER-1 training
        #model1 = load_model("offline_featuristic_tier1.h5")
        pred_proba1 = predict(model1, xtrain, ytrain)                                 # TIER-1 prediction - on training data
        thd1 = select_decision_threshold(ytrain, pred_proba1, target_fpr=0.8)
        #plot_auc(ytrain, pred_proba1, thd1, "tier1")
        
        xB1, yB1, yprobB1, xM1, yM1, yprobM1 = get_bfn_mfp(xtrain, ytrain, pred_proba1, thd1)         # Filter B1 data - To feed for TIER-2 after ATI
        bg, mg = ATI(model1, xB1, yB1, keys)                                          # Perform ATI - to find feature importance based on dy/dx gradients
        select, drop, drop_indices = select_features(bg, mg, q_criteria)              # FEATURE SELECTION - Based on Qualification criteria
        xtier2, ytier2 = prepare_B1(xB1, yB1, drop_indices)

        # TIER-2 training
        model2 = train(xtier2, ytier2, num_of_features=len(select), tier="tier2", epochs=epochs, batch_size=batch_size)
        #model2 = load_model("offline_featuristic_tier2.h5")
        pred_proba2 = predict(model2, xtier2, ytier2)                                 # TIER-2 prediction - on training B1 data
        thd2 = select_decision_threshold(ytier2, pred_proba2, target_fpr=0)

        ytrain_reconciled = np.concatenate((yM1, ytier2))
        ytrain_probreconciled = np.concatenate((yprobM1, pred_proba2))

        training_time = time.time() - tst

        display_probability_chart(ytrain, pred_proba1, thd1, "TRAINING_TIER1_PROB_PLOT_" + str(index + 1))
        display_probability_chart(ytier2, pred_proba2, thd2, "TRAINING_TIER2_PROB_PLOT_" + str(index + 1))

        print("\n      TEST/VALIDATION - Confusion Matrix: tn fp fn tp")
        tst = time.time()

        # TEST/VALIDATION
        pred_proba1 = predict(model1, xval, yval)
        tpr1, fpr1, fnr1, auc1, fpr_auc1, tpr_auc1, thds_auc1 = plot_auc(yval, pred_proba1, thd1, "tier1_" + str(index + 1))
        xB1, yB1, yprobB1, xM1, yM1, yprobM1 = get_bfn_mfp(xval, yval, pred_proba1, thd1)
        xtier2, ytier2 = prepare_B1(xB1, yB1, drop_indices)
        pred_proba2 = predict(model2, xtier2, ytier2)
        plot_auc(yB1, pred_proba2, thd2, "tier2")

        # RECONCILE - xM1, yprobM1, xB1, pred_proba2
        # print(np.shape(xM1), np.shape(xB1), np.shape(yprobM1), np.shape(pred_proba2))
        xreconciled = np.concatenate((xM1, xB1))                                      # Using xB1 instead of xtier2, as xtier2 has reduced num of features. Not possible to concatenate
        yreconciled = np.concatenate((yM1, ytier2))
        yprobreconciled = np.concatenate((yprobM1, pred_proba2))
        ypredreconciled = np.concatenate((yprobM1 > (thd1/100), pred_proba2 > (thd2/100)))
        # print("Reconciled:", np.shape(xreconciled), np.shape(yreconciled), np.shape(yprobreconciled))

        tprr, fprr, fnrr, aucr, fpr_aucr, tpr_aucr, thds_aucr = plot_auc_reconciled(yreconciled, ypredreconciled, yprobreconciled, "reconciled_" + str(index + 1))
        testing_time = time.time() - tst

        display_probability_chart(yval, pred_proba1, thd1, "TESTING-_TIER1_PROB_PLOT_"+str(index+1))
        display_probability_chart(ytier2, pred_proba2, thd2, "TESTING-_TIER2_PROB_PLOT_" + str(index + 1))
        ############################### Collecting data for mean results ######################################
        mean_fpr = np.linspace(0, 1, val_len)

        # Testing
        list_interp_tpr1.append(interp(mean_fpr, fpr_auc1, tpr_auc1))
        list_interp_tpr1[-1][0] = 0.0
        list_interp_tprr.append(interp(mean_fpr, fpr_aucr, tpr_aucr))
        list_interp_tprr[-1][0] = 0.0

        list_auc1.append(auc1)
        list_tpr1.append(tpr1)
        list_fpr1.append(fpr1)

        list_aucr.append(aucr)
        list_tprr.append(tprr)
        list_fprr.append(fprr)

        list_train_time.append(training_time)
        list_pred_time.append(testing_time)
        list_num_dropped_features.append(len(drop))
        list_dropped_features.append(drop)

        list_train_y.append(ytrain_reconciled)
        list_train_pred_val.append(ytrain_probreconciled)
        list_test_y.append(yreconciled)
        list_test_pred.append(ypredreconciled)
        list_test_pred_val.append(yprobreconciled)
    
    mean_fpr = np.linspace(0, 1, val_len)
    print("\n[ Training Time per sample: {:6.3f}".format(np.mean(list_train_time)/train_len)+" ms ]\n[ Testing Time per sample : {:6.3f}".format(np.mean(list_pred_time)/val_len)+" ms ]") 
    
    global mean_tier1_precision
    global mean_tier1_recall
    global mean_tier1_f1_score

    global mean_recon_precision
    global mean_recon_recall
    global mean_recon_f1_score

    global mean_tier1_precision_recall_f1score_support_perclass
    global mean_recon_precision_recall_f1score_support_perclass

    global mean_tier1_precision_recall_f1score_support
    global mean_recon_precision_recall_f1score_support

    #mean_tier1_precision = mean_tier1_precision / folds * 100
    #mean_tier1_recall    = mean_tier1_recall    / folds * 100
    #mean_tier1_f1_score  = mean_tier1_f1_score  / folds * 100

    #mean_recon_precision = mean_recon_precision / folds * 100
    #mean_recon_recall    = mean_recon_recall    / folds * 100
    #mean_recon_f1_score  = mean_recon_f1_score  / folds * 100

    mean_tier1_perclass_f1_score = pd.DataFrame([np.average(mean_tier1_class1_f1_score), np.average(mean_tier1_class0_f1_score)])
    mean_tier1_perclass_precision = pd.DataFrame([np.average(mean_tier1_class1_precision), np.average(mean_tier1_class0_precision)])
    mean_tier1_perclass_recall = pd.DataFrame([np.average(mean_tier1_class1_recall), np.average(mean_tier1_class0_recall)])
    mean_tier1_perclass_support  = pd.DataFrame([int(np.average(mean_tier1_class1_support)), int(np.average(mean_tier1_class0_support))])

    tier1_perclass_results = pd.concat([mean_tier1_perclass_f1_score, mean_tier1_perclass_precision, mean_tier1_perclass_recall, mean_tier1_perclass_support], axis=1)

    mean_recon_perclass_f1_score = pd.DataFrame([np.average(mean_recon_class1_f1_score), np.average(mean_recon_class0_f1_score)])
    mean_recon_perclass_precision = pd.DataFrame([np.average(mean_recon_class1_precision), np.average(mean_recon_class0_precision)])
    mean_recon_perclass_recall = pd.DataFrame([np.average(mean_recon_class1_recall), np.average(mean_recon_class0_recall)])
    mean_recon_perclass_support  = pd.DataFrame([int(np.average(mean_recon_class1_support)), int(np.average(mean_recon_class0_support))])

    recon_perclass_results = pd.concat([mean_recon_perclass_f1_score, mean_recon_perclass_precision, mean_recon_perclass_recall, mean_recon_perclass_support], axis=1)
 
    tier1_f1_score = pd.DataFrame([np.average(mean_tier1_micro_f1_score), np.average(mean_tier1_macro_f1_score), np.average(mean_tier1_weighted_f1_score)], index=['micro', 'macro', 'weighted'])
    tier1_precision = pd.DataFrame([np.average(mean_tier1_micro_precision), np.average(mean_tier1_macro_precision), np.average(mean_tier1_weighted_precision)], index=['micro', 'macro', 'weighted'])
    tier1_recall = pd.DataFrame([np.average(mean_tier1_micro_recall), np.average(mean_tier1_macro_recall), np.average(mean_tier1_weighted_recall)], index=['micro', 'macro', 'weighted'])
    tier1_support = pd.DataFrame(np.full((3,1), mean_tier1_perclass_support.sum(axis=0)), index=['micro', 'macro', 'weighted'])

    recon_f1_score = pd.DataFrame([np.average(mean_recon_micro_f1_score), np.average(mean_recon_macro_f1_score), np.average(mean_recon_weighted_f1_score)], index=['micro', 'macro', 'weighted'])
    recon_precision = pd.DataFrame([np.average(mean_recon_micro_precision), np.average(mean_recon_macro_precision), np.average(mean_recon_weighted_precision)], index=['micro', 'macro', 'weighted'])
    recon_recall = pd.DataFrame([np.average(mean_recon_micro_recall), np.average(mean_recon_macro_recall), np.average(mean_recon_weighted_recall)], index=['micro', 'macro', 'weighted'])
    recon_support = pd.DataFrame(np.full((3,1), mean_recon_perclass_support.sum(axis=0)), index=['micro', 'macro', 'weighted'])

    tier1_results = pd.concat([tier1_f1_score, tier1_precision, tier1_recall, tier1_support], axis=1)
    recon_results = pd.concat([recon_f1_score, recon_precision, recon_recall, recon_support], axis=1)

    tier1_results = pd.concat([tier1_perclass_results, tier1_results], axis=0)
    recon_results = pd.concat([recon_perclass_results, recon_results], axis=0)

    cols = ['f1-score', 'precision', 'recall', 'support']
    tier1_results.columns = cols
    recon_results.columns = cols

    tier1_results = tier1_results.round(2)
    recon_results = recon_results.round(2)

    print("Tier-1:\n", tier1_results)
    print("Reconciled :\n", recon_results)

    plot_cv_auc(mean_fpr, list_interp_tpr1, list_interp_tprr, list_auc1, list_aucr, recon_results, fpr1=np.mean(list_fpr1), fprr=np.mean(list_fprr))
    
    print("\nTIER-1 Mean TPR [Recall]:{:6.2f}".format(np.mean(list_tpr1)), "\tFPR:{:6.2f}".format(np.mean(list_fpr1)))
    print("Recon  Mean TPR [Recall]:{:6.2f}".format(np.mean(list_tprr)), "\tFPR:{:6.2f}".format(np.mean(list_fprr)))

    print("\nMean num. of features dropped         :", int(np.mean(list_num_dropped_features)))
    
    all_dropped_features = []
    for lst in list_dropped_features:
        all_dropped_features.extend(lst)

    print("All dropped features from 5-folds     : "+str(len(all_dropped_features)))
    print("Unique dropped features from 5-folds  : "+str(len(np.unique(all_dropped_features))))

    import collections
    counter = collections.Counter(all_dropped_features)
    print("Feature Frequency:",counter.most_common(),"\n")
    plot_fpr_train_vs_test(list_train_y, list_train_pred_val, list_test_y, list_test_pred_val)
    #print(np.shape(np.mean(list_test_y, axis=0).astype(int)), np.shape(np.mean(list_test_pred, axis=0).astype(int)), np.mean(list_test_y, axis=0).astype(int), np.mean(list_test_pred, axis=0).astype(int))
    global recon_cm
    print(recon_cm, recon_cm / folds)
    confusion_matrix_heatmap(recon_cm / folds, "mean")
    #confusion_matrix_heatmap(metrics.confusion_matrix(np.mean(list_test_y, axis=0).astype(int), np.mean(list_test_pred, axis=0).astype(int)), "mean")
	
def create_model(num_of_features=53, reg_rate=0.01, lr = 0.001, neurons=53, weight_constraint=0, 
                 dropout_rate=0.5, momentum=0, init_mode='glorot_normal', activation='relu', optimizer='Adam'):
    inp = Input((num_of_features,))
    '''d = Dense(num_of_features * 2, activation='relu')(inp)
    d = Dense(num_of_features * 1, activation='relu')(d)
    d = Dense(32, activation='relu')(d)
    d = Dense(16, activation='relu')(d)'''

    def ann_block(d, block_name='ann_block'):
        with K.name_scope(block_name):
            d = Dense(neurons * 2, activation=activation, kernel_regularizer=l2(reg_rate), kernel_initializer=init_mode)(d) #, kernel_constraint=maxnorm(weight_constraint))(d)
            d = BatchNormalization()(d)
            #d = Dropout(rate=dropout_rate)(d)
            #d = LeakyReLU(alpha=0.3)(d)
            return d

    d = ann_block(inp)
    d = ann_block(d)
    d = ann_block(d)
    d = ann_block(d)
    d = ann_block(d)
    d = ann_block(d)
    d = ann_block(d)
    d = ann_block(d)

    # Fixed Blocks
    d = Dense(32, activation=activation, kernel_regularizer=l2(reg_rate), kernel_initializer=init_mode)(d) #, kernel_constraint=maxnorm(weight_constraint))(d)
    d = BatchNormalization()(d)
    #d = LeakyReLU(alpha=0.3)(d)
    d = Dense(16, activation=activation, kernel_regularizer=l2(reg_rate), kernel_initializer=init_mode)(d) #, kernel_constraint=maxnorm(weight_constraint))(d)
    d = BatchNormalization()(d)
    #d = LeakyReLU(alpha=0.3)(d)
    out = Dense(1, activation='sigmoid')(d)
    model = Model(inp, out)
    optimizer = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    import keras
    METRICS = [f1_m, precision_m, recall_m, 'accuracy']
    model.compile(optimizer=optimizer, loss='binary_crossentropy',metrics=METRICS)
    return model
	
folds = 5
q_criteria = 0
epochs = 200
batch_size = 16
print("[ Q-criterion:"+str(q_criteria)+" ]")

import warnings
warnings.filterwarnings("ignore")
#pd.set_option('display.float_format','{:.2f}'.format)

reset()
do_cross_validation(xtier1, ytier1, folds, keys, epochs, batch_size)