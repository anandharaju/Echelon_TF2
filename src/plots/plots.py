import numpy as np
from sklearn import metrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import config.constants as cnst
import time


class plot_vars():
    list_fprr_training = []
    list_interp_fprr_training = []

    list_auc1 = []
    list_bacc1 = []
    list_acc1 = []
    list_tpr1 = []
    list_fpr1 = []
    list_fnr1 = []
    list_tpr_auc1 = []
    list_fpr_auc1 = []
    list_interp_tpr1 = []
    list_interp_fpr1 = []

    list_train_y = []
    list_train_yhat = []
    list_train_pred_val = []
    list_test_tier1_y = []
    list_test_tier1_yhat = []
    list_test_tier1_pred_val = []
    list_test_y = []
    list_test_yhat = []
    list_test_pred_val = []

    list_aucr = []
    list_baccr = []
    list_accr = []
    list_tprr = []
    list_fprr = []
    list_fnrr = []
    list_tpr_aucr = []
    list_fpr_aucr = []
    list_interp_tprr = []
    list_interp_fprr = []

    list_pred_time = []
    list_num_selected_features = []
    list_selected_features = []


def plot_fpr_train_vs_test(list_train_y, list_train_pred_val, list_test_y, list_test_pred_val):
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
    plt.savefig(".."+cnst.ESC+"out"+cnst.ESC+"imgs"+cnst.ESC+"FPR_train_vs_test.png", bbox_inches='tight')


def plot_auc(y1, pv1, y2, pv2):
    list_tpr_1 = []
    list_tpr_2 = []
    list_fpr_1 = []
    list_fpr_2 = []

    for i in range(0, len(y1)):
        temp_fpr_list = []
        temp_tpr_list = []
        for confidence in np.arange(0, 101, 0.1):
            cm = metrics.confusion_matrix(y1[i], pv1[i] > (confidence/100))
            tn = cm[0][0]
            fp = cm[0][1]
            fn = cm[1][0]
            tp = cm[1][1]
            TPR = (tp / (tp + fn)) * 100
            FPR = (fp / (fp + tn)) * 100
            temp_fpr_list.append(FPR)
            temp_tpr_list.append(TPR)
        list_fpr_1.append(temp_fpr_list)
        list_tpr_1.append(temp_tpr_list)

    for i in range(0, len(y2)):
        temp_fpr_list = []
        temp_tpr_list = []
        for confidence in np.arange(0, 101, 0.1):
            cm = metrics.confusion_matrix(y2[i], pv2[i] > (confidence/100))
            tn = cm[0][0]
            fp = cm[0][1]
            fn = cm[1][0]
            tp = cm[1][1]
            TPR = (tp / (tp + fn)) * 100
            FPR = (fp / (fp + tn)) * 100
            temp_fpr_list.append(FPR)
            temp_tpr_list.append(TPR)
        list_fpr_2.append(temp_fpr_list)
        list_tpr_2.append(temp_tpr_list)


    #std_auc_1 = np.std(list_auc1)
    #std_auc_2 = np.std(list_auc1)

    plt.title("AUC", fontdict={'fontsize': 6});
    plt.plot([1, 1], [0, 100], 'black', linestyle=':', label="Target FPR");
    plt.plot([0, 2], [90, 90], 'black', linestyle='-.', label="Target TPR");
    plt.xlabel("FPR", fontsize=6);
    plt.ylabel("TPR", fontsize=6);
    plt.xticks(np.arange(0, 2.2, 0.2), fontsize=6);
    plt.yticks(np.arange(0, 110, 10), fontsize=6);
    plt.xlim(0, 2.2);
    plt.plot(np.mean(list_fpr_1, axis=0), np.mean(list_tpr_1, axis=0), color='r',
             label=r'Mean Tier-1 ROC (AUC = %0.5f $\pm$ %0.2f)' % (mean_auc1, std_auc1), lw=2, alpha=.8);

    plt.savefig(".."+cnst.ESC+"out"+cnst.ESC+"imgs"+cnst.ESC+"tier1_auc_1.png", bbox_inches='tight')

    plt.plot(np.mean(list_fpr_2, axis=0), np.mean(list_tpr_2, axis=0), color='b', linestyle='-.',
             label=r'Mean Reconciled ROC (AUC = %0.5f $\pm$ %0.2f)' % (mean_aucr, std_aucr), lw=2, alpha=.8);
    plt.legend(loc=1, prop={'size': 4});

    plt.savefig(".."+cnst.ESC+"out"+cnst.ESC+"imgs"+cnst.ESC+"cv_auc_1.png", bbox_inches='tight')


def plot_history(history, tier):
    plt.clf()
    dpi = 300
    figsize = (4, 2)
    plt.figure(num=None, figsize=figsize, dpi=dpi)
    accuracy = history.history['acc']
    plt.plot(range(len(accuracy)), np.nan_to_num(accuracy), label='Tier-1 Training Trend')

    try:
        validation_accuracy = history.history['val_acc']
        plt.plot(range(len(validation_accuracy)), np.nan_to_num(validation_accuracy), label='Tier-1 Validation Trend')
    except Exception as e:
        print("val_acc not available in training history")
    plt.xlabel("Epochs", fontsize=10)
    plt.ylabel("Accuracy", fontsize=10)
    plt.legend(fontsize=10)
    plt.savefig(cnst.PLOT_PATH + tier + "_Train_Val_History.png", bbox_inches='tight')  # _" + str(time.time()) + "
    # plt.show()


def plot_partition_epoch_history(trn_acc, val_acc, trn_loss, val_loss, tier):
    plt.clf()
    dpi = 300
    figsize = (4, 2)
    plt.figure(num=None, figsize=figsize, dpi=dpi)
    try:
        plt.plot(range(len(trn_acc)), np.nan_to_num(trn_acc), label=tier + ' Training Acc')
        plt.plot(range(len(val_acc)), np.nan_to_num(val_acc), label=tier + ' Validation Acc')
        plt.plot(range(len(trn_loss)), np.nan_to_num(trn_loss), label=tier + ' Training Loss')
        plt.plot(range(len(val_loss)), np.nan_to_num(val_loss), label=tier + ' Validation Loss')
    except Exception as e:
        print("Error occurred while plotting partition epoch history", str(e))
    plt.xlabel("Epochs", fontsize=10)
    plt.ylabel("Accuracy/Loss", fontsize=10)
    plt.legend(fontsize=10)
    plt.savefig(cnst.PLOT_PATH + tier + "_Train_Val_History.png", bbox_inches='tight')  # _" + str(time.time()) + "
    # plt.show()


def save_stats_as_plot(qualification_criteria):
    df = pd.read_csv(combined_stat_file)
    # df.set_index("Type", inplace=True)

    bdf = df[df["Type"] == 0]
    mdf = df[df["Type"] != 0].reset_index(drop=True)

    bdf = bdf.drop('Type', axis=1)
    mdf = mdf.drop('Type', axis=1)

    bdf_sum = bdf.sum()
    mdf_sum = mdf.sum()

    bdf_norm = bdf_sum / bdf.shape[0]
    mdf_norm = mdf_sum / mdf.shape[0]

    dpi = 300
    figsize = (4, 2.7)
    plt.figure(num=None, figsize=figsize, dpi=dpi)
    plt.ylabel('ACTIVATION TREND')
    plt.xlabel('FEATURE')

    plt.plot(bdf_norm, 'black', marker='o', markersize=3.5, label='Benign')
    plt.plot(bdf_norm, 'black', linewidth=0.5)

    for i, key in enumerate(bdf.keys()):
        x1, y1 = [i, i], [0, bdf_norm[i]]
        plt.plot(x1, y1, 'black', linestyle=':', linewidth=0.75)

    plt.plot(mdf_norm, 'red', marker='o', markersize=3.5, label='Malware')
    plt.plot(mdf_norm, 'red', linewidth=0.5)

    for i, key in enumerate(mdf.keys()):
        x1, y1 = [i, i], [0, mdf_norm[i]]
        plt.plot(x1, y1, 'red', linestyle=':', linewidth=0.75)

    plt.xticks(range(0, len(bdf.keys())), bdf.keys(), rotation=80, fontsize=3)
    plt.yticks(fontsize=7)
    plt.savefig(intuition2, bbox_inches='tight')
    plt.legend()

    tier2_keys = bdf.keys()[(np.abs(bdf_norm - mdf_norm) // qualification_criteria) > 0]
    drop_keys = bdf.keys()[(np.abs(bdf_norm - mdf_norm) // qualification_criteria) == 0]
    return np.array(tier2_keys), np.array(drop_keys)
    # return bdf.keys(), None


def save_stats_as_plot(fmaps):

    #df = pd.read_csv(combined_stat_file)
    #df.set_index("type", inplace=True)
    #print(df)

    cucs = ["text"  , "data"  , "rdata", "pdata", "edata", "idata", "bss"  , "header", "debug", "reloc", "rsrc"  ,"sdata","xdata","hdata","itext","apiset","extjmp","cdata","cfguard","guids","sdbid","extrel","ndata","detourc","shared","shdata","didat","stabstr"]
    cuc  = [160.524, 3037.258,1076.52  , 32.259 ,15.995  , 7.096  , 61.772 ,8330     ,76.686  , 0.605  ,3903.883 , 7.995, 7.995, 0, 47.959, 40, 28.431, 119.997, 19.747, 2.25, 11.997, 67.506, 59.936, 131.694, 39.971, 11.997, 0, 1.868]
    cucs = np.array(cucs)
    cuc = np.array(cuc)
    cucs = cucs[cuc > 0]
    cuc  = cuc[cuc > 0]
    #imbalance
    dpi = 300
    figsize = (4, 2.7)
    plt.figure(num=None, figsize=figsize, dpi=dpi)
    plt.ylabel('CUC STATISTIC')
    plt.xlabel('SECTIONS')
    plt.plot(cuc, 'black', marker='o', markersize=3.5)
    plt.plot(cuc, 'black', linewidth=0.5)

    for i, val in enumerate(cucs):
        x1, y1 = [i, i], [0, cuc[i]]
        plt.plot(x1, y1, 'black', linestyle=':', linewidth=0.75)

    plt.xticks(range(0, len(cucs)), cucs, rotation=80, fontsize=8)
    plt.yticks(fontsize=7)
    #plt.gci()
    plt.savefig(intuition2, bbox_inches='tight')
    plt.show()


    '''fig, ax1 = plt.subplots(1, 1)
    plt.plot(df.loc['FN'].values, 'brown')
    plt.plot(df.loc['FP'].values)
    plt.plot(df.loc['BENIGN'].values, 'g')
    plt.plot(df.loc['MALWARE'].values, 'r')
    plt.legend(['FN', 'FP', 'BENIGN', 'MALWARE'])
    ax1.set_xticklabels(['', 'header', 'text', 'data', 'rsrc', 'pdata', 'rdata', 'padding'], minor=False, rotation=45)
    plt.gcf()
    fig.savefig(plot_file, bbox_inches='tight')
    #plt.show()
    print("Stats saved as plot in file: ", plot_file)'''


def display_probability_chart(Y, prob, threshold, plot_name):
    b = Y == 0
    m = Y == 1

    dpi = 300
    figsize = (6, 2.7)
    plt.rcParams.update({'font.size': 6})
    plt.figure(num=None, figsize=figsize, dpi=dpi)

    ax = plt.subplot(111)
    ax.plot(Y[b], range(0, len(Y[b])), 'bo', label='Target-Benign', markersize=8)
    ax.plot(Y[m], range(0, len(Y[m])), 'ro', label='Target-Malware', markersize=8)

    ax.plot(prob[b], range(0, len(prob[b])), 'b1')
    ax.plot(prob[m], range(0, len(prob[m])), 'r1')

    if threshold is not None:
        ax.plot([threshold/100, threshold/100], [0, len(Y)], 'black', linestyle=':', label="Selected Threshold")

    ax.set_xticks(np.linspace(0, 1, 21))
    # ax.set_xticklabels(fontsize=12)
    # ax.set_yticklabels(fontsize=6)
    ax.set_xlabel("Probability Range", fontsize=6)
    ax.set_ylabel(plot_name[:9] + " Samples", fontsize=6)
    ax.set_xlim([-0.05, 1.05])
    ax.grid(linewidth=0.2)

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.10), ncol=3, fancybox=True, prop={'size': 5})

    # Shrink current axis's height by 10% on the bottom
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0 + box.height * 0.3, box.width, box.height * 0.7])
    # Put a legend below current axis
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=4, prop={'size': 5})
    plt.savefig(".."+cnst.ESC+"out"+cnst.ESC+"imgs"+cnst.ESC+plot_name+".png", bbox_inches='tight')
    # plt.show()


def display_results(args, Y, pred, selected_threshold):
    # DO FOR ALL CONFDENCE LEVELS
    print('Confusion Matrix:\t\t[Confidence: ' + str(selected_threshold) + '%]\n   tn   fp   fn   tp')
    '''
    for confidence in args.confidence_levels:
        # df['predicted score - rounded to ' + str(confidence) + '% confidence'] = pred // (confidence / 100)
        acc = metrics.accuracy_score(Y, pred > (confidence / 100))
        bacc = metrics.balanced_accuracy_score(Y, pred > (confidence / 100))
        cm = metrics.confusion_matrix(Y, pred > (confidence / 100))
        # print('Confusion Matrix:\t\t[Confidence: ' + str(confidence) + '%] [Acc: ' + str(acc)
        #      + "] [Balanced Acc: " + str(bacc) + ']\n   tn   fp   fn   tp')

        tn = cm[0][0]
        fp = cm[0][1]
        fn = cm[1][0]
        tp = cm[1][1]

        TPR = (tp/(tp+fn))*100
        FPR = (fp/(fp+tn))*100
        FNR = (fn/(fn+tp))*100

        print("%5s & %5s & %5s & %5s" % (str(tn), str(fp), str(fn), str(tp)), " \\\\\\hline", "TPR: {:0.2f}".format(TPR), "FPR: {:0.2f}".format(FPR), "FNR: {:0.2f}".format(FNR))
    '''
    confidence = selected_threshold if selected_threshold is not None else args.target_confidence
    # print("Confidence - ", confidence)
    # DO FOR SELECTED CONFIDENCE
    acc = metrics.accuracy_score(Y, pred > (confidence / 100))
    bacc = metrics.balanced_accuracy_score(Y, pred > (confidence / 100))
    cm = metrics.confusion_matrix(Y, pred > (confidence / 100))
    # print('[Acc: ' + str(acc)[:6] + "] [Balanced Acc: " + str(bacc)[:6] + ']')
    tn = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tp = cm[1][1]
    print("%5s & %5s & %5s & %5s" % (str(tn), str(fp), str(fn), str(tp)))

    TPR = (tp/(tp+fn))*100
    FPR = (fp/(fp+tn))*100
    FNR = (fn/(fn+tp))*100
    fpr_auc, tpr_auc, thds_auc = metrics.roc_curve(Y, pred, drop_intermediate=False)
    auc = metrics.roc_auc_score(Y, pred)
    # print("Overall ROC AUC Score     : {:0.2f}".format(auc))  # , fpr, tpr)
    print(" Testing Threshold :", selected_threshold, "  TIER-1", "TPR: {:0.5f}".format(TPR), "FPR: {:0.5f}".format(FPR))
    # aucdf = pd.DataFrame()
    # aucdf['fpr'] = fpr_auc
    # aucdf['tpr'] = tpr_auc
    # aucdf['thds'] = thds_auc
    # aucdf.to_csv(project_base_path+cnst.ESC+'out'+cnst.ESC+'result'+cnst.ESC+'auc.csv', header=None, index=False)
    return auc, acc, bacc, TPR, FPR, FNR, fpr_auc, tpr_auc, thds_auc


def display_results_hat(args, Y, Yhat, pred, selected_threshold):
    print('Confusion Matrix:\t\t[Confidence: ' + str(selected_threshold) + '%]\n   tn   fp   fn   tp')

    confidence = selected_threshold if selected_threshold is not None else args.target_confidence
    # print("Confidence - ", confidence)

    acc = metrics.accuracy_score(Y, Yhat)
    bacc = metrics.balanced_accuracy_score(Y, Yhat)
    cm = metrics.confusion_matrix(Y, Yhat)
    # print('[Acc: ' + str(acc)[:6] + "] [Balanced Acc: " + str(bacc)[:6] + ']')
    tn = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tp = cm[1][1]
    print("%5s & %5s & %5s & %5s" % (str(tn), str(fp), str(fn), str(tp)))

    TPR = (tp/(tp+fn))*100
    FPR = (fp/(fp+tn))*100
    FNR = (fn/(fn+tp))*100
    fpr_auc, tpr_auc, thds_auc = metrics.roc_curve(Y, pred, drop_intermediate=False)
    auc = metrics.roc_auc_score(Y, pred)
    # print("Overall ROC AUC Score     : {:0.2f}".format(auc))  # , fpr, tpr)
    print(" Testing Threshold :", selected_threshold, "  TIER-2", "TPR: {:0.5f}".format(TPR), "FPR: {:0.5f}".format(FPR))
    # aucdf = pd.DataFrame()
    # aucdf['fpr'] = fpr_auc
    # aucdf['tpr'] = tpr_auc
    # aucdf['thds'] = thds_auc
    # aucdf.to_csv(project_base_path+'auc.csv', header=None, index=False)
    return auc, acc, bacc, TPR, FPR, FNR, fpr_auc, tpr_auc, thds_auc


    '''
    plt.legend(prop={'size': 3})
    plt.xlabel("Training FPR", fontsize=6)
    plt.ylabel("Testing FPR", fontsize=6)
    plt.xticks(np.arange(0, 0.11, 0.01), fontsize=6)
    plt.yticks(np.arange(0, 0.11, 0.01), fontsize=6)
    plt.xlim(0, 0.11)
    plt.ylim(0, 0.11)
    plt.title("FPR - Training vs Testing")
    plt.plot(np.arange(0, 1, 0.01), mean_fprr_training, color='red', label="Training FPR")
    plt.plot(np.arange(0, 1, 0.01), mean_fprr, color='blue', label="Test FPR")
    plt.plot([0.01, 0.01], [0.0, 0.11], 'black', linestyle=':', label="Target FPR")
    plt.savefig(".."+cnst.ESC+"out"+cnst.ESC+"imgs"+cnst.ESC+"FPR_train_vs_test_1.png", bbox_inches='tight')
    '''