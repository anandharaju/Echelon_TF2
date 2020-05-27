import pandas as pd
import numpy as np
from train import train
from predict import predict
import time
import matplotlib.pyplot as plt
from scipy import interp
from sklearn import metrics
import config.constants as cnst
import analyzers.analyze_dataset as analyzer
from plots.plots import plot_vars as pv
from sklearn.model_selection import StratifiedKFold, train_test_split
import random
from datetime import datetime
from plots.auc import plot_cv_auc
from plots.auc import cv as cv_info
from collections import OrderedDict
import os
import pickle
from analyzers.collect_exe_files import partition_pkl_files_by_count, partition_pkl_files_by_size
import sklearn.utils


def generate_cv_folds_data(dataset_path):
    mastertraindata = analyzer.ByteData()
    testdata = analyzer.ByteData()
    valdata = analyzer.ByteData()
    cv_obj = cv_info()

    # Analyze Data set - Required only for Huawei pickle files
    if cnst.GENERATE_BENIGN_MALWARE_FILES: analyzer.analyze_dataset(cnst.CHECK_FILE_SIZE)
    # Load Data set info from CSV files
    adata, _, _ = analyzer. load_dataset(dataset_path)

    # SPLIT TRAIN AND TEST DATA
    # Save them in files -- fold wise
    skf = StratifiedKFold(n_splits=cnst.CV_FOLDS, shuffle=True, random_state=cnst.RANDOM_SEED)
    for index, (master_train_indices, test_indices) in enumerate(skf.split(adata.xdf, adata.ydf)):
        if cnst.REGENERATE_DATA:
            mastertraindata.xdf, testdata.xdf = adata.xdf[master_train_indices], adata.xdf[test_indices]
            mastertraindata.ydf, testdata.ydf = adata.ydf[master_train_indices], adata.ydf[test_indices]
            mastertraindata.xdf, valdata.xdf, mastertraindata.ydf, valdata.ydf = train_test_split(mastertraindata.xdf, mastertraindata.ydf, test_size=cnst.VAL_SET_SIZE, stratify=mastertraindata.ydf)

            pd.concat([mastertraindata.xdf, mastertraindata.ydf], axis=1).to_csv(cnst.PROJECT_BASE_PATH + cnst.ESC + "data" + cnst.ESC + "master_train_"+str(index)+"_pkl.csv", header=None, index=None)
            pd.concat([valdata.xdf, valdata.ydf], axis=1).to_csv(cnst.PROJECT_BASE_PATH + cnst.ESC + "data" + cnst.ESC + "master_val_" + str(index)+ "_pkl.csv", header=None, index=None)
            pd.concat([testdata.xdf, testdata.ydf], axis=1).to_csv(cnst.PROJECT_BASE_PATH + cnst.ESC + "data" + cnst.ESC + "master_test_"+str(index)+ "_pkl.csv", header=None, index=None)

        train_csv = pd.read_csv(cnst.PROJECT_BASE_PATH + cnst.ESC + "data" + cnst.ESC + "master_train_" + str(index) + "_pkl.csv", header=None)
        val_csv = pd.read_csv(cnst.PROJECT_BASE_PATH + cnst.ESC + "data" + cnst.ESC + "master_val_" + str(index) + "_pkl.csv", header=None)
        test_csv = pd.read_csv(cnst.PROJECT_BASE_PATH + cnst.ESC + "data" + cnst.ESC + "master_test_" + str(index) + "_pkl.csv", header=None)

        mastertraindata.xdf, valdata.xdf, testdata.xdf = train_csv.iloc[:, 0], val_csv.iloc[:, 0], test_csv.iloc[:, 0]
        mastertraindata.ydf, valdata.ydf, testdata.ydf = train_csv.iloc[:, 1], val_csv.iloc[:, 1], test_csv.iloc[:, 1]

        cv_obj.train_data[index] = mastertraindata
        cv_obj.val_data[index] = valdata
        cv_obj.test_data[index] = testdata

    return cv_obj


def partition_dataset(dataset_path):
    adata, _, _ = analyzer.load_dataset(dataset_path)
    if not os.path.exists(cnst.DATA_SOURCE_PATH):
        os.makedirs(cnst.DATA_SOURCE_PATH)

    skf = StratifiedKFold(n_splits=cnst.PARTITIONS, shuffle=True, random_state=cnst.RANDOM_SEED)
    pd.DataFrame().to_csv(cnst.PROJECT_BASE_PATH + cnst.ESC + "data" + cnst.ESC + "master_partition_pkl.csv", header=None, index=None)
    for index, (_, partition_indices) in enumerate(skf.split(adata.xdf, adata.ydf)):
        temp_partitioningdata = pd.concat([adata.xdf[partition_indices], adata.ydf[partition_indices]], axis=1)
        temp_partitioningdata = sklearn.utils.shuffle(temp_partitioningdata)
        temp_partitioningdata = temp_partitioningdata.reset_index(drop=True)
        temp_partitioningdata.to_csv(cnst.PROJECT_BASE_PATH + cnst.ESC + "data" + cnst.ESC + "master_partition_pkl.csv", header=None, index=None, mode="a")

    partitioningdf = pd.read_csv(cnst.PROJECT_BASE_PATH + cnst.ESC + "data" + cnst.ESC + "master_partition_pkl.csv", header=None)
    pcount = partition_pkl_files_by_count(None, None, partitioningdf.iloc[:, 0], partitioningdf.iloc[:, 1]) if cnst.PARTITION_BY_COUNT else partition_pkl_files_by_size(None, None, partitioningdf.iloc[:, 0], partitioningdf.iloc[:, 1])
    pd.DataFrame([{"master": pcount}]).to_csv(os.path.join(cnst.DATA_SOURCE_PATH, "master_partition_tracker.csv"), index=False)
    return pcount


def train_predict(model_idx, dataset_path=None):
    tst = time.time()
    print("\nSTART TIME  [", datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "]")
    # cv_obj = generate_cv_folds_data(dataset_path)

    m_pcount = None
    if cnst.REGENERATE_DATA_AND_PARTITIONS:
        m_pcount = partition_dataset(dataset_path)
        cpt = time.time()
        print("\nTIME ELAPSED FOR GENERATING PARTITIONS:", str(int(cpt - tst) / 60), " minutes   [", datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "]")
    else:
        m_pcount = pd.read_csv(os.path.join(cnst.DATA_SOURCE_PATH, "master_partition_tracker.csv"))["master"][0]

    tst_pcount = int(round(m_pcount * cnst.TST_SET_SIZE))
    val_pcount = int(round((m_pcount - tst_pcount) * cnst.VAL_SET_SIZE))
    trn_pcount = m_pcount - (tst_pcount + val_pcount)

    print("Total Partition:", m_pcount, "\t\t\tTrain:", trn_pcount, "Val:", val_pcount, "Test:", tst_pcount)

    for fold_index in range(cnst.CV_FOLDS):
        tst_partitions = np.arange(tst_pcount) + (fold_index * tst_pcount)
        trn_val_partitions = [x for x in range(m_pcount) if x not in tst_partitions]
        val_partitions = trn_val_partitions[-1*val_pcount:]
        trn_partitions = [x for x in trn_val_partitions if x not in val_partitions]
        pd.DataFrame([{"train": trn_partitions, "val": val_partitions, "test": tst_partitions}]).to_csv(os.path.join(cnst.DATA_SOURCE_PATH, "partition_tracker_" + str(fold_index) + ".csv"), index=False)

        print("\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> [ CV-FOLD " + str(fold_index + 1) + "/" + str(cnst.CV_FOLDS) + " ]", "Training: " + str(len(trn_partitions)), "Validation: " + str(len(val_partitions)), "Testing: " + str(len(tst_partitions)))
        if fold_index not in cnst.RUN_FOLDS:
            continue

        if not cnst.SKIP_ENTIRE_TRAINING:
            cpt = time.time()
            train.init(model_idx, trn_partitions, val_partitions, fold_index)
            print("\nTIME ELAPSED FOR TRAINING:", str(int(time.time() - cpt) / 60), " minutes   [", datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "]")
        else:
            print("SKIPPED: Tier 1&2 Training Process")

        if cnst.ONLY_TIER1_TRAINING:
            continue
        print("**********************  PREDICTION TIER 1&2 - STARTED  ************************")
        cpt = time.time()
        cv_obj = cv_info()
        pred_cv_obj = predict.init(model_idx, tst_partitions, cv_obj, fold_index)
        if pred_cv_obj is not None:
            cv_obj = pred_cv_obj
        else:
            print("Problem occurred during prediction phase of current fold. Proceeding to next fold . . .")
        print("**********************  PREDICTION TIER 1&2 - ENDED    ************************")
        print("\nTIME ELAPSED FOR PREDICTION:", str(int(time.time() - cpt) / 60), " minutes   [", datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "]")

        tet = time.time() - tst
        print("\nTOTAL TIME ELAPSED :", str(int(tet) / 60), " minutes   [", datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "]")
        # return

        if cv_obj is not None:
            cvdf = pd.DataFrame([cv_obj.t1_mean_fpr_auc, cv_obj.t1_mean_tpr_auc, cv_obj.recon_mean_fpr_auc, cv_obj.recon_mean_tpr_auc])
            scoredf = pd.DataFrame([np.mean(cv_obj.t1_mean_auc_score_restricted), np.mean(cv_obj.t1_mean_auc_score), np.mean(cv_obj.recon_mean_auc_score_restricted), np.mean(cv_obj.recon_mean_auc_score)])
            cvdf.to_csv(cnst.PROJECT_BASE_PATH+cnst.ESC+"out"+cnst.ESC+"result"+cnst.ESC+"mean_cv.csv", index=False, header=None)
            scoredf.to_csv(cnst.PROJECT_BASE_PATH + cnst.ESC+"out"+cnst.ESC+"result"+cnst.ESC+"score_cv.csv", index=False, header=None)
            print("CROSS VALIDATION >>> TIER1 TPR:", np.mean(cv_obj.t1_tpr_list), "FPR:", np.mean(cv_obj.t1_fpr_list), "OVERALL TPR:", np.mean(cv_obj.recon_tpr_list), "FPR:", np.mean(cv_obj.recon_fpr_list))
            print("Tier-1 ROC (Restricted AUC = %0.3f) [Full AUC: %0.3f]" % (np.mean(cv_obj.t1_mean_auc_score_restricted), np.mean(cv_obj.t1_mean_auc_score)))
            print("Reconciled ROC (Restricted AUC = %0.3f) [Full AUC: %0.3f]" % (np.mean(cv_obj.recon_mean_auc_score_restricted), np.mean(cv_obj.recon_mean_auc_score)))
