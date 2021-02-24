import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import metrics


def plot_cv_auc():
    sns.set()
    sns.set_style("whitegrid", {'axes.grid': False})
    dpi = 300
    figsize = (5, 4)
    font_size = 9
    plt.figure(num=None, figsize=figsize, dpi=dpi)

    model = ["LockBoost - Semantic Aware",
             "LockBoost - Section Based",
             "AdaBoost-CNN",
             "CNN-Xgboost",
             "Raff et al (Malconv)",
             "Model 6",
             "Model 7",
             "Krcal et al"]
    files = [# "roc_96.43_0.10.csv",
             "roc_95.18_0.096.csv",
             "roc_93.29_0.098.csv",
             # "roc_91.93_0.096.csv",
             "roc_91.12_0.10.csv",
             "roc_90.44_0.098.csv",
             # "roc_88.66_0.096.csv",
             "roc_88.37_0.098.csv"
             ]
    auc = [0.999,
           0.996,
           0.996,
           0.994,
           0.988]
    rauc = [0.02,
            0.01,
            0.13,
            0.22,
            0.07]

    for i in range(0, len(files)):
        cvdf = pd.read_csv("D:\\03_GitWorks\\Echelon_TF2\\data\\"+files[i], header=None)
        sns.lineplot(cvdf.iloc[:, 0], cvdf.iloc[:, 1], linewidth=1,
                     label=r''+model[i]+' (Restricted AUC = %0.3f $\pm$ %0.2f) ' % (auc[i], rauc[i]), alpha=.8)

    plt.xlim(0, 0.001)
    plt.ylim(0.5, 1)
    plt.xlabel("FPR %", fontsize=font_size)
    plt.ylabel("TPR %", fontsize=font_size)
    plt.xticks([0, 0.0002, 0.0004, 0.0006, 0.0008, 0.001], labels=["0%", "0.02%", "0.04%", "0.06%", "0.08%", "0.1%"], fontsize=font_size)
    plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], labels=["0%", "10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%", "100%"], fontsize=font_size)
    plt.legend(prop={'size': font_size-1})
    # plt.grid(b=True, which='both')
    plt.savefig("D:\\03_GitWorks\\Echelon_TF2\\data\\kdd_auc.png", bbox_inches='tight')
    plt.show()


plot_cv_auc()