import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from config import settings as cnst


def plot_cv_auc():
    cvdf = pd.read_csv(cnst.PROJECT_BASE_PATH+cnst.ESC+'out'+cnst.ESC+'result'+cnst.ESC+'mean_cv.csv', header=None)
    fpr1 = cvdf.loc[0]
    tpr1 = cvdf.loc[1]
    rfpr = cvdf.loc[2]
    rtpr = cvdf.loc[3]

    scoredf = pd.read_csv(cnst.PROJECT_BASE_PATH+cnst.ESC+'out'+cnst.ESC+'result'+cnst.ESC+'score_cv.csv', header=None)
    scores = scoredf.loc[:, 0].values

    sns.set()

    dpi = 300
    figsize = (5, 3)
    font_size = 5
    plt.figure(num=None, figsize=figsize, dpi=dpi)
    sns.lineplot(fpr1, tpr1, color='r', linewidth=1, label=r'Tier-1 ROC (Restricted AUC = %0.3f $\pm$ %0.2f) ' % (scores[0], np.std(scores[0])), alpha=.8)  # , scores[1] [Full AUC: %0.3f]
    sns.lineplot(rfpr, rtpr, color='b', linewidth=1, label=r'Reconciled ROC (Restricted AUC = %0.3f $\pm$ %0.2f) ' % (scores[2], np.std(scores[2])), alpha=.8)  # , scores[3] [Full AUC: %0.3f]

    plt.xlim(0, cnst.OVERALL_TARGET_FPR * 2)
    plt.xlabel("FPR %", fontsize=font_size)
    plt.ylabel("TPR %", fontsize=font_size)
    plt.xticks(fontsize=font_size)  # , ticks=[x/10 for x in np.arange(0, 11, 1)], labels=[str(x/10)+"%" for x in np.arange(0, 11, 1)])
    plt.yticks(fontsize=font_size, ticks=[y for y in np.arange(0, 105, 10)], labels=[str(y)+"%" for y in np.arange(0, 105, 10)])
    #plt.plot([cnst.OVERALL_TARGET_FPR, cnst.OVERALL_TARGET_FPR], [0, 100], 'grey', linestyle=':', label="Target FPR")
    #plt.plot([0, 100], [90, 90], 'grey', linestyle='-.', label="Target TPR")
    plt.legend(loc=8, prop={'size': font_size})
    plt.grid(b=True, which='both')
    plt.xlim(0, 100)
    plt.savefig(cnst.PROJECT_BASE_PATH + cnst.ESC + "out" + cnst.ESC + "imgs" + cnst.ESC + "cv_auc_full.png", bbox_inches='tight')

    #plt.xlim(0, 10)
    #plt.savefig(cnst.PROJECT_BASE_PATH + cnst.ESC + "out" + cnst.ESC + "imgs" + cnst.ESC + "cv_auc_10_percent.png", bbox_inches='tight')

    #plt.xlim(0, 1)
    #plt.savefig(cnst.PROJECT_BASE_PATH + cnst.ESC + "out" + cnst.ESC + "imgs" + cnst.ESC + "cv_auc_1_percent.png", bbox_inches='tight')

    #plt.xlim(0, cnst.OVERALL_TARGET_FPR * 1.1)
    #plt.savefig(cnst.PROJECT_BASE_PATH + cnst.ESC + "out" + cnst.ESC + "imgs" + cnst.ESC + "cv_auc_by_target_fpr.png", bbox_inches='tight')
    plt.show()


