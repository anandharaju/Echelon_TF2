import pandas as pd
import matplotlib.pyplot as plt
import config.constants as cnst
import numpy as np

# aucdf = pd.read_csv('aucmalconv.csv', header=None)
# fpr = aucdf[0]
# tpr = aucdf[1]


class cv:
    train_data = {}
    val_data = {}
    test_data = {}

    t1_tpr_folds = {}
    t1_fpr_folds = {}
    recon_tpr_folds = {}
    recon_fpr_folds = {}

    t1_tpr_list = np.array([])
    t1_fpr_list = np.array([])
    recon_tpr_list = np.array([])
    recon_fpr_list = np.array([])

    t1_mean_tpr_auc = None
    t1_mean_fpr_auc = None
    recon_mean_tpr_auc = None
    recon_mean_fpr_auc = None

    t1_mean_auc_score = np.array([])
    t1_mean_auc_score_restricted = np.array([])
    recon_mean_auc_score = np.array([])
    recon_mean_auc_score_restricted = np.array([])


def plot_cv_auc(cv_obj):
    print("CROSS VALIDATION >>> TIER1 TPR:", np.mean(cv_obj.t1_tpr_list),
          "FPR:", np.mean(cv_obj.t1_fpr_list),
          "OVERALL TPR:", np.mean(cv_obj.recon_tpr_list),
          "FPR:", np.mean(cv_obj.recon_fpr_list))
    plt.plot(cv_obj.t1_mean_fpr_auc, cv_obj.t1_mean_tpr_auc, color='r', linewidth=3,
             label=r'Tier-1 ROC (Restricted AUC = %0.3f) [Full AUC: %0.3f]' % (
                 np.mean(cv_obj.t1_mean_auc_score_restricted),
                 np.mean(cv_obj.t1_mean_auc_score)),
                 # metrics.roc_auc_score(pt1.ytrue, pt1.yprob, max_fpr=cnst.OVERALL_TARGET_FPR/100),
                 # metrics.roc_auc_score(pt1.ytrue, pt1.yprob)),
             lw=2, alpha=.8)

    plt.plot(cv_obj.recon_mean_fpr_auc, cv_obj.recon_mean_tpr_auc, color='b', linewidth=3,
             label=r'Reconciled ROC (Restricted AUC = %0.3f) [Full AUC: %0.3f]' % (
                 np.mean(cv_obj.recon_mean_auc_score_restricted),
                 np.mean(cv_obj.recon_mean_auc_score)),
                 # metrics.roc_auc_score(ytruereconciled, yprobreconciled, max_fpr=cnst.OVERALL_TARGET_FPR/100),
                 # metrics.roc_auc_score(ytruereconciled, yprobreconciled)),
                 # metrics.roc_auc_score(pt1.yM1, pt1.yprobM1, max_fpr=cnst.OVERALL_TARGET_FPR/100) + metrics.roc_auc_score(pt2.ytrue, pt2.yprob, max_fpr=cnst.OVERALL_TARGET_FPR/100),
                 # metrics.roc_auc_score(pt1.yM1, pt1.yprobM1) + metrics.roc_auc_score(pt2.ytrue, pt2.yprob)),
             lw=2, alpha=.8)
    plt.xlim(0, 2)
    plt.xlabel("FPR %", fontsize=14)
    plt.ylabel("TPR %", fontsize=14)
    plt.plot([1, 1], [0, 100], 'black', linestyle=':', label="Target FPR")
    plt.legend(loc=1, prop={'size': 14})
    plt.plot([0, 2], [90, 90], 'black', linestyle='-.', label="Target TPR")
    plt.savefig(cnst.PROJECT_BASE_PATH +cnst.ESC+ "out"+cnst.ESC+"imgs"+cnst.ESC+"cv_auc.png", bbox_inches='tight')
