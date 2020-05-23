from utils import utils
import numpy as np
import pandas as pd
from sklearn import metrics
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.models import load_model
from utils.filter import filter_benign_fn_files, filter_malware_fp_files
from config import constants as cnst
from .predict_args import DefaultPredictArguments, Predict as pObj
import math
from plots.plots import display_probability_chart
from analyzers.collect_exe_files import get_partition_data, partition_pkl_files_by_count
import gc
import os


def get_model_memory_usage(batch_size, model):
    import numpy as np
    from keras import backend as K

    shapes_mem_count = 0
    internal_model_mem_count = 0
    for l in model.layers:
        layer_type = l.__class__.__name__
        if layer_type == 'Model':
            internal_model_mem_count += get_model_memory_usage(batch_size, l)
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    number_size = 4.0
    if K.floatx() == 'float16':
         number_size = 2.0
    if K.floatx() == 'float64':
         number_size = 8.0

    total_memory = number_size*(batch_size*shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3) + internal_model_mem_count
    return gbytes


def predict_byte(model, partition, xfiles, args):
    xlen = len(xfiles)
    pred_steps = xlen//args.batch_size if xlen % args.batch_size == 0 else xlen//args.batch_size + 1
    pred = model.predict_generator(
        utils.data_generator(partition, xfiles, np.ones(xfiles.shape), args.max_len, args.batch_size, shuffle=False),
        steps=pred_steps,
        verbose=args.verbose
        )
    return pred


def predict_byte_by_section(model, wpartition, spartition, xfiles, q_sections, section_map, args):
    xlen = len(xfiles)
    pred_steps = xlen//args.batch_size if xlen % args.batch_size == 0 else xlen//args.batch_size + 1
    pred = model.predict_generator(
        utils.data_generator_by_section(wpartition, spartition, q_sections, section_map, xfiles, np.ones(xfiles.shape), args.max_len, args.batch_size, shuffle=False),
        steps=pred_steps,
        verbose=args.verbose
        )
    return pred


def predict_by_features(model, fn_list, label, batch_size, verbose, features_to_drop=None):
    pred = model.predict_generator(
        utils.data_generator_by_features(fn_list, np.ones(fn_list.shape), batch_size, False, features_to_drop),
        steps=len(fn_list),
        verbose=verbose
    )
    return pred


def predict_by_fusion(model, fn_list, label, batch_size, verbose):
    byte_sequence_max_len = model.input[0].shape[1]
    pred = model.predict_generator(
        utils.data_generator_by_fusion(fn_list, np.ones(fn_list.shape), byte_sequence_max_len, batch_size, shuffle=False),
        steps=len(fn_list),
        verbose=verbose
    )
    return pred


def calculate_prediction_metrics(predict_obj):
    predict_obj.ypred = (predict_obj.yprob >= (predict_obj.thd / 100)).astype(int)
    cm = metrics.confusion_matrix(predict_obj.ytrue, predict_obj.ypred, labels=[cnst.BENIGN, cnst.MALWARE])
    tn = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tp = cm[1][1]

    predict_obj.tpr = (tp / (tp + fn)) * 100
    predict_obj.fpr = (fp / (fp + tn)) * 100
    print("Before AUC score computation:", predict_obj.thd, predict_obj.tpr, predict_obj.fpr)
    try:
        predict_obj.auc = metrics.roc_auc_score(predict_obj.ytrue, predict_obj.ypred)
        predict_obj.rauc = metrics.roc_auc_score(predict_obj.ytrue, predict_obj.ypred, max_fpr=cnst.OVERALL_TARGET_FPR/100)
    except Exception as e:
        print(str(e))

    print("Threshold used for Prediction :", predict_obj.thd,
          "TPR: {:6.2f}".format(predict_obj.tpr),
          "\tFPR: {:6.2f}".format(predict_obj.fpr),
          "\tAUC: ", predict_obj.auc,
          "\tRst. AUC: ", predict_obj.rauc)
    return predict_obj


def select_decision_threshold(predict_obj):
    predict_obj.target_fpr = (np.floor((predict_obj.target_fpr / 100) * len(predict_obj.yprob[predict_obj.ytrue == 0])) * 100) / len(predict_obj.yprob[predict_obj.ytrue == 0])
    calibrated_threshold = (np.percentile(predict_obj.yprob[predict_obj.ytrue == 0], q=[100 - predict_obj.target_fpr]) * 100)[0]
    print("Initial Calibrated Threshold:", calibrated_threshold, predict_obj.target_fpr)
    pow = str(calibrated_threshold)[::-1].find('.')
    calibrated_threshold = math.ceil(calibrated_threshold * 10 ** (pow - 1)) / (10 ** (pow - 1))
    print("Ceiled Threshold:", calibrated_threshold)
    selected_threshold = calibrated_threshold if calibrated_threshold < 100.0 else 100.0

    temp_ypred = (predict_obj.yprob > (selected_threshold / 100)).astype(int)
    cm = metrics.confusion_matrix(predict_obj.ytrue, temp_ypred, labels=[cnst.BENIGN, cnst.MALWARE])
    tn = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tp = cm[1][1]
    TPR = (tp / (tp + fn)) * 100
    FPR = (fp / (fp + tn)) * 100

    print("Selected Threshold: {:6.2f}".format(selected_threshold), "TPR: {:6.2f}".format(TPR), "\tFPR: {:6.2f}".format(FPR))
    predict_obj.thd = selected_threshold
    predict_obj.ypred = temp_ypred
    predict_obj.tpr = TPR
    predict_obj.fpr = FPR
    return predict_obj


def get_bfn_mfp(pObj):
    prediction = (pObj.yprob > (pObj.thd / 100)).astype(int)

    if cnst.PERFORM_B2_BOOSTING:
        if pObj.boosting_upper_bound is None:
            fn_indices = np.all([pObj.ytrue.ravel() == cnst.MALWARE, prediction.ravel() == cnst.BENIGN], axis=0)
            pObj.boosting_upper_bound = np.min(pObj.yprob[fn_indices]) if np.sum(fn_indices) > 0 else 0
            print("Setting B2 boosting threshold:", pObj.boosting_upper_bound)

        # To filter the predicted Benign FN files from prediction results
        brow_indices = np.where(np.all([prediction == cnst.BENIGN, pObj.yprob >= pObj.boosting_upper_bound], axis=0))[0]
        pObj.xB1 = pObj.xtrue[brow_indices]
        pObj.yB1 = pObj.ytrue[brow_indices]
        pObj.yprobB1 = pObj.yprob[brow_indices]
        pObj.ypredB1 = prediction[brow_indices]

        # To filter the benign files that can be boosted directly to B2 set
        boosted_indices = np.where(np.all([prediction == cnst.BENIGN, pObj.yprob < pObj.boosting_upper_bound], axis=0))[0]
        fn_escaped_by_boosting = np.where(np.all([prediction.ravel() == cnst.BENIGN, pObj.yprob.ravel() < pObj.boosting_upper_bound, pObj.ytrue.ravel() == cnst.MALWARE], axis=0))[0]
        pObj.boosted_xB2 = pObj.xtrue[boosted_indices]
        pObj.boosted_yB2 = pObj.ytrue[boosted_indices]
        pObj.boosted_yprobB2 = pObj.yprob[boosted_indices]
        pObj.boosted_ypredB2 = prediction[boosted_indices]
        print("Number of files boosted to B2:", len(np.where(prediction == cnst.BENIGN)[0]), "-", len(brow_indices), "=", len(boosted_indices), "Boosting Bound:", pObj.boosting_upper_bound, "Escaped FNs:", len(fn_escaped_by_boosting))

    else:
        print("NO BOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOSTING")
        # To filter the predicted Benign FN files from prediction results
        brow_indices = np.where(prediction == cnst.BENIGN)[0]
        pObj.xB1 = pObj.xtrue[brow_indices]
        pObj.yB1 = pObj.ytrue[brow_indices]
        pObj.yprobB1 = pObj.yprob[brow_indices]
        pObj.ypredB1 = prediction[brow_indices]

    # print("\nPREDICT MODULE    Total B1 [{0}]\tGroundTruth [{1}:{2}]".format(len(brow_indices),
    # len(np.where(pObj.yB1 == cnst.BENIGN)[0]), len(np.where(pObj.yB1 == cnst.MALWARE)[0])))

    mrow_indices = np.where(prediction == cnst.MALWARE)[0]
    pObj.xM1 = pObj.xtrue[mrow_indices]
    pObj.yM1 = pObj.ytrue[mrow_indices]
    pObj.yprobM1 = pObj.yprob[mrow_indices]
    pObj.ypredM1 = prediction[mrow_indices]

    return pObj


def predict_tier1(model_idx, pobj, fold_index):
    predict_args = DefaultPredictArguments()
    tier1_model = load_model(predict_args.model_path + cnst.TIER1_MODELS[model_idx] + "_" + str(fold_index) + ".h5")
    # model.summary()
    print("Memory Required:", get_model_memory_usage(predict_args.batch_size, tier1_model))

    if cnst.EXECUTION_TYPE[model_idx] == cnst.BYTE:
        pobj.yprob = predict_byte(tier1_model, pobj.partition, pobj.xtrue, predict_args)
    elif cnst.EXECUTION_TYPE[model_idx] == cnst.FEATURISTIC:
        pobj.yprob = predict_by_features(tier1_model, pobj.xtrue, predict_args)
    elif cnst.EXECUTION_TYPE[model_idx] == cnst.FUSION:
        pobj.yprob = predict_by_fusion(tier1_model, pobj.xtrue, predict_args)

    del tier1_model
    gc.collect()
    return pobj


def select_thd_get_metrics_bfn_mfp(tier, pobj):
    if pobj.thd is None:
        pobj = select_decision_threshold(pobj)  # +++ returned pobj also includes ypred based on selected threshold
    else:
        pobj = calculate_prediction_metrics(pobj)

    if tier == cnst.TIER1:
        pobj = get_bfn_mfp(pobj)

    return pobj


def select_thd_get_metrics(pobj):
    if pobj.thd is None:
        pobj = select_decision_threshold(pobj)  # +++ returned pobj also includes ypred based on selected threshold
    else:
        pobj = calculate_prediction_metrics(pobj)

    return pobj


def predict_tier2(model_idx, pobj, fold_index):
    predict_args = DefaultPredictArguments()
    tier2_model = load_model(predict_args.model_path + cnst.TIER2_MODELS[model_idx] + "_" + str(fold_index) + ".h5")

    print("Memory Required:", get_model_memory_usage(predict_args.batch_size, tier2_model))

    if cnst.EXECUTION_TYPE[model_idx] == cnst.BYTE:
        # pbs.trigger_predict_by_section()
        pobj.yprob = predict_byte_by_section(tier2_model, pobj.wpartition, pobj.spartition, pobj.xtrue, pobj.q_sections, pobj.predict_section_map, predict_args)
    elif cnst.EXECUTION_TYPE[model_idx] == cnst.FEATURISTIC:
        pobj.yprob = predict_by_features(tier2_model, pobj.xtrue, predict_args)
    elif cnst.EXECUTION_TYPE[model_idx] == cnst.FUSION:
        pobj.yprob = predict_by_fusion(tier2_model, pobj.xtrue, predict_args)

    del tier2_model
    gc.collect()
    return pobj


def get_reconciled_tpr_fpr(yt1, yp1, yt2, yp2):
    cm1 = metrics.confusion_matrix(yt1, yp1, labels=[cnst.BENIGN, cnst.MALWARE])
    tn1 = cm1[0][0]
    fp1 = cm1[0][1]
    fn1 = cm1[1][0]
    tp1 = cm1[1][1]

    cm2 = metrics.confusion_matrix(yt2, yp2, labels=[cnst.BENIGN,cnst.MALWARE])
    tn2 = cm2[0][0]
    fp2 = cm2[0][1]
    fn2 = cm2[1][0]
    tp2 = cm2[1][1]

    tpr = ((tp1+tp2) / (tp1+tp2+fn1+fn2)) * 100
    fpr = ((fp1+fp2) / (fp1+fp2+tn1+tn2)) * 100

    return tpr, fpr


def get_tpr_fpr(yt, yp):
    cm = metrics.confusion_matrix(yt, yp, labels=[cnst.BENIGN, cnst.MALWARE])
    tn = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tp = cm[1][1]
    tpr = (tp / (tp+fn)) * 100
    fpr = (fp / (fp+tn)) * 100
    return tpr, fpr


def reconcile(pt1, pt2, cv_obj, fold_index):
    print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   RECONCILING DATA  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    # RECONCILE - xM1, yprobM1, xB1, pred_proba2
    print("BEFORE RECONCILIATION: [Total True Malwares:", np.sum(pt1.ytrue), "]    M1 has => "+str(np.shape(pt1.xM1)[0]),
          "\tTPs:", np.sum(np.all([pt1.ytrue.ravel() == cnst.MALWARE, pt1.ypred.ravel() == cnst.MALWARE], axis=0)),
          "\tFPs:", np.sum(np.all([pt1.ytrue.ravel() == cnst.BENIGN, pt1.ypred.ravel() == cnst.MALWARE], axis=0)))
    print("                       [Total True Benign  :", len(np.where(pt1.ytrue.ravel() == cnst.BENIGN)[0]), "]    B1 has => "+str(len(np.where(pt1.ypred == cnst.BENIGN)[0])),
          "\tTNs:", np.sum(np.all([pt1.ytrue.ravel() == cnst.BENIGN, pt1.ypred.ravel() == cnst.BENIGN], axis=0)),
          "\tFNs:", np.sum(np.all([pt1.ytrue.ravel() == cnst.MALWARE, pt1.ypred.ravel() == cnst.BENIGN], axis=0)))

    # scaled_pt1_yprobM1 = (pt1.yprobM1 - np.min(pt1.yprobM1)) / (np.max(pt1.yprobM1) - np.min(pt1.yprobM1))
    # scaled_pt1_boosted_yprobB2 = (pt1.boosted_yprobB2 - np.min(pt1.boosted_yprobB2)) / (np.max(pt1.boosted_yprobB2) - np.min(pt1.boosted_yprobB2))
    # scaled_pt2_yprob = (pt2.yprob - np.min(pt2.yprob)) / (np.max(pt2.yprob) - np.min(pt2.yprob))

    if cnst.PERFORM_B2_BOOSTING:
        print("                       Boosted:"+str(np.shape(pt1.boosted_xB2)[0]), "\tRemaining B1:"+str(np.shape(pt2.xtrue)[0]))
        xtruereconciled = np.concatenate((pt1.xM1, pt1.boosted_xB2, pt2.xtrue))  # pt2.xtrue contains xB1
        ytruereconciled = np.concatenate((pt1.yM1, pt1.boosted_yB2, pt2.ytrue))
        ypredreconciled = np.concatenate((pt1.ypredM1, pt1.boosted_ypredB2, pt2.ypred))
        yprobreconciled = np.concatenate((pt1.yprobM1, pt1.boosted_yprobB2, pt2.yprob))
        # yprobreconciled = np.concatenate((scaled_pt1_yprobM1, scaled_pt1_boosted_yprobB2, scaled_pt2_yprob))
    else:
        xtruereconciled = np.concatenate((pt1.xM1, pt2.xtrue))  # pt2.xtrue contains xB1
        ytruereconciled = np.concatenate((pt1.yM1, pt2.ytrue))
        ypredreconciled = np.concatenate((pt1.ypredM1, pt2.ypred))
        yprobreconciled = np.concatenate((pt1.yprobM1, pt2.yprob))
        # yprobreconciled = np.concatenate((scaled_pt1_yprobM1, scaled_pt2_yprob))

    print("AFTER RECONCILIATION :", "[M1+B1] => [", "M1+(M2+B2)", "+B2_Boosted" if cnst.PERFORM_B2_BOOSTING else "", "]  =", np.shape(xtruereconciled)[0],
          "New TPs found: [M2] =>", np.sum(np.all([pt2.ytrue.ravel() == cnst.MALWARE, pt2.ypred.ravel() == cnst.MALWARE], axis=0)),
          "\tNew FPs:  =>", np.sum(np.all([pt2.ytrue.ravel() == cnst.BENIGN, pt2.ypred.ravel() == cnst.MALWARE], axis=0)),
          "\n[RECON TPs] =>", np.sum(np.all([ytruereconciled.ravel() == cnst.MALWARE, ypredreconciled.ravel() == cnst.MALWARE], axis=0)),
          "\tFPs:", np.sum(np.all([ytruereconciled.ravel() == cnst.BENIGN, ypredreconciled.ravel() == cnst.MALWARE], axis=0)),
          "\n[RECON TNs] =>", np.sum(np.all([ytruereconciled.ravel() == cnst.BENIGN, ypredreconciled.ravel() == cnst.BENIGN], axis=0)),
          "\tFNs:", np.sum(np.all([ytruereconciled.ravel() == cnst.MALWARE, ypredreconciled.ravel() == cnst.BENIGN], axis=0)))

    reconciled_tpr = np.array([])
    reconciled_fpr = np.array([])
    tier1_tpr = np.array([])
    tier1_fpr = np.array([])
    probability_score = np.arange(0, 100.01, 0.2)
    for p in probability_score:
        rtpr, rfpr = None, None
        if cnst.PERFORM_B2_BOOSTING:
            ytrue_M1B2_Boosted = np.concatenate((pt1.yM1, pt1.boosted_yB2))
            yprob_M1B2_Boosted = np.concatenate((pt1.yprobM1, pt1.boosted_yprobB2))
            rtpr, rfpr = get_reconciled_tpr_fpr(ytrue_M1B2_Boosted, yprob_M1B2_Boosted > (p/100), pt2.ytrue, pt2.yprob > (p/100))
        else:
            rtpr, rfpr = get_reconciled_tpr_fpr(pt1.yM1, pt1.yprobM1 > (p/100), pt2.ytrue, pt2.yprob > (p/100))
        reconciled_tpr = np.append(reconciled_tpr, rtpr)
        reconciled_fpr = np.append(reconciled_fpr, rfpr)

        tpr1, fpr1 = get_tpr_fpr(pt1.ytrue, pt1.yprob > (p/100))
        tier1_tpr = np.append(tier1_tpr, tpr1)
        tier1_fpr = np.append(tier1_fpr, fpr1)

    cv_obj.t1_tpr_folds[fold_index] = tier1_tpr
    cv_obj.t1_fpr_folds[fold_index] = tier1_fpr
    cv_obj.recon_tpr_folds[fold_index] = reconciled_tpr
    cv_obj.recon_fpr_folds[fold_index] = reconciled_fpr

    cv_obj.t1_mean_tpr_auc = tier1_tpr if cv_obj.t1_mean_tpr_auc is None else np.sum([cv_obj.t1_mean_tpr_auc, tier1_tpr], axis=0) / 2
    cv_obj.t1_mean_fpr_auc = tier1_fpr if cv_obj.t1_mean_fpr_auc is None else np.sum([cv_obj.t1_mean_fpr_auc, tier1_fpr], axis=0) / 2
    cv_obj.recon_mean_tpr_auc = reconciled_tpr if cv_obj.recon_mean_tpr_auc is None else np.sum([cv_obj.recon_mean_tpr_auc, reconciled_tpr], axis=0) / 2
    cv_obj.recon_mean_fpr_auc = reconciled_fpr if cv_obj.recon_mean_fpr_auc is None else np.sum([cv_obj.recon_mean_fpr_auc, reconciled_fpr], axis=0) / 2

    tpr1, fpr1 = get_tpr_fpr(pt1.ytrue, pt1.ypred)
    rtpr, rfpr = None, None
    if cnst.PERFORM_B2_BOOSTING:
        ytrue_M1B2_Boosted = np.concatenate((pt1.yM1, pt1.boosted_yB2))
        ypred_M1B2_Boosted = np.concatenate((pt1.ypredM1, pt1.boosted_ypredB2))
        rtpr, rfpr = get_reconciled_tpr_fpr(ytrue_M1B2_Boosted, ypred_M1B2_Boosted, pt2.ytrue, pt2.ypred)
    else:
        rtpr, rfpr = get_reconciled_tpr_fpr(pt1.yM1, pt1.ypredM1, pt2.ytrue, pt2.ypred)
    print("FOLD:", fold_index+1, "TIER1 TPR:", tpr1, "FPR:", fpr1, "OVERALL TPR:", rtpr, "FPR:", rfpr)

    cv_obj.t1_tpr_list = np.append(cv_obj.t1_tpr_list, tpr1)
    cv_obj.t1_fpr_list = np.append(cv_obj.t1_fpr_list, fpr1)
    cv_obj.recon_tpr_list = np.append(cv_obj.recon_tpr_list, rtpr)
    cv_obj.recon_fpr_list = np.append(cv_obj.recon_fpr_list, rfpr)

    cv_obj.t1_mean_auc_score = np.append(cv_obj.t1_mean_auc_score, metrics.roc_auc_score(pt1.ytrue, pt1.yprob))
    cv_obj.t1_mean_auc_score_restricted = np.append(cv_obj.t1_mean_auc_score_restricted, metrics.roc_auc_score(pt1.ytrue, pt1.yprob, max_fpr=cnst.OVERALL_TARGET_FPR/100))
    cv_obj.recon_mean_auc_score = np.append(cv_obj.recon_mean_auc_score, metrics.roc_auc_score(ytruereconciled, yprobreconciled))
    cv_obj.recon_mean_auc_score_restricted = np.append(cv_obj.recon_mean_auc_score_restricted, metrics.roc_auc_score(ytruereconciled, yprobreconciled, max_fpr=cnst.OVERALL_TARGET_FPR/100))
    print("Tier1 Restricted AUC = %0.3f [Full AUC: %0.3f]" % (metrics.roc_auc_score(pt1.ytrue, pt1.yprob, max_fpr=cnst.OVERALL_TARGET_FPR/100), metrics.roc_auc_score(pt1.ytrue, pt1.yprob)),
          "Recon Restricted AUC = %0.3f [Full AUC: %0.3f]" % (metrics.roc_auc_score(ytruereconciled, yprobreconciled, max_fpr=cnst.OVERALL_TARGET_FPR/100), metrics.roc_auc_score(ytruereconciled, yprobreconciled)))

    return cv_obj, rfpr


def benchmark_tier1(model_idx, ptier1, fold_index, recon_fpr):
    print("\nBenchmarking on Testing Data for TIER1 with overall FPR :", recon_fpr)
    ptier1.target_fpr = recon_fpr
    ptier1.thd = None
    # predict_tier1(model_idx, ptier1, fold_index)


def init(model_idx, test_partitions, cv_obj, fold_index):
    # TIER-1 PREDICTION OVER TEST DATA
    partition_tracker_df = pd.read_csv(cnst.DATA_SOURCE_PATH + cnst.ESC + "partition_tracker_" + str(fold_index) + ".csv")
    print("\nPrediction on Testing Data - TIER1       # Partitions:", partition_tracker_df["test"][0])

    todf = pd.read_csv(os.path.join(cnst.PROJECT_BASE_PATH + cnst.ESC + "out" + cnst.ESC + "result" + cnst.ESC, "training_outcomes_" + str(fold_index) + ".csv"))
    thd1 = todf["thd1"][0]
    thd2 = todf["thd2"][0]
    boosting_bound = todf["boosting_bound"][0]
    section_map = None
    q_sections = pd.read_csv(os.path.join(cnst.PROJECT_BASE_PATH + cnst.ESC + "out" + cnst.ESC + "result" + cnst.ESC, "qualified_sections_" + str(fold_index) + ".csv"), header=None).values.squeeze()
    print("THD1", thd1, "THD2", thd2, "Boosting Bound", boosting_bound)

    pd.DataFrame().to_csv(cnst.PROJECT_BASE_PATH + cnst.ESC + "data" + cnst.ESC + "b1_test_"+str(fold_index)+"_pkl.csv", header=None, index=None)
    pd.DataFrame().to_csv(cnst.PROJECT_BASE_PATH + cnst.ESC + "data" + cnst.ESC + "m1_test_"+str(fold_index)+"_pkl.csv", header=None, index=None)

    predict_t1_test_data_all = pObj(cnst.TIER2, None, None, None)
    predict_t1_test_data_all.thd = thd1

    for pcount in test_partitions:
        tst_datadf = pd.read_csv(cnst.DATA_SOURCE_PATH + cnst.ESC + "p" + str(pcount) + ".csv", header=None)

        predict_t1_test_data_partition = pObj(cnst.TIER1, None, tst_datadf.iloc[:, 0].values, tst_datadf.iloc[:, 1].values)
        predict_t1_test_data_partition.thd = thd1
        predict_t1_test_data_partition.boosting_upper_bound = boosting_bound
        predict_t1_test_data_partition.partition = get_partition_data(None, None, pcount, "t1")
        predict_t1_test_data_partition = predict_tier1(model_idx, predict_t1_test_data_partition, fold_index)

        del predict_t1_test_data_partition.partition  # Release Memory
        gc.collect()

        predict_t1_test_data_all.xtrue = predict_t1_test_data_partition.xtrue if predict_t1_test_data_all.xtrue is None else np.concatenate([predict_t1_test_data_all.xtrue, predict_t1_test_data_partition.xtrue])
        predict_t1_test_data_all.ytrue = predict_t1_test_data_partition.ytrue if predict_t1_test_data_all.ytrue is None else np.concatenate([predict_t1_test_data_all.ytrue, predict_t1_test_data_partition.ytrue])
        predict_t1_test_data_all.yprob = predict_t1_test_data_partition.yprob if predict_t1_test_data_all.yprob is None else np.concatenate([predict_t1_test_data_all.yprob, predict_t1_test_data_partition.yprob])
        predict_t1_test_data_all.ypred = predict_t1_test_data_partition.ypred if predict_t1_test_data_all.ypred is None else np.concatenate([predict_t1_test_data_all.ypred, predict_t1_test_data_partition.ypred])

        predict_t1_test_data_partition = get_bfn_mfp(predict_t1_test_data_partition)

        predict_t1_test_data_all.xB1 = predict_t1_test_data_partition.xB1 if predict_t1_test_data_all.xB1 is None else np.concatenate([predict_t1_test_data_all.xB1, predict_t1_test_data_partition.xB1])
        predict_t1_test_data_all.yB1 = predict_t1_test_data_partition.yB1 if predict_t1_test_data_all.yB1 is None else np.concatenate([predict_t1_test_data_all.yB1, predict_t1_test_data_partition.yB1])
        predict_t1_test_data_all.yprobB1 = predict_t1_test_data_partition.yprobB1 if predict_t1_test_data_all.yprobB1 is None else np.concatenate([predict_t1_test_data_all.yprobB1, predict_t1_test_data_partition.yprobB1])
        predict_t1_test_data_all.ypredB1 = predict_t1_test_data_partition.ypredB1 if predict_t1_test_data_all.ypredB1 is None else np.concatenate([predict_t1_test_data_all.ypredB1, predict_t1_test_data_partition.ypredB1])

        predict_t1_test_data_all.xM1 = predict_t1_test_data_partition.xM1 if predict_t1_test_data_all.xM1 is None else np.concatenate([predict_t1_test_data_all.xM1, predict_t1_test_data_partition.xM1])
        predict_t1_test_data_all.yM1 = predict_t1_test_data_partition.yM1 if predict_t1_test_data_all.yM1 is None else np.concatenate([predict_t1_test_data_all.yM1, predict_t1_test_data_partition.yM1])
        predict_t1_test_data_all.yprobM1 = predict_t1_test_data_partition.yprobM1 if predict_t1_test_data_all.yprobM1 is None else np.concatenate([predict_t1_test_data_all.yprobM1, predict_t1_test_data_partition.yprobM1])
        predict_t1_test_data_all.ypredM1 = predict_t1_test_data_partition.ypredM1 if predict_t1_test_data_all.ypredM1 is None else np.concatenate([predict_t1_test_data_all.ypredM1, predict_t1_test_data_partition.ypredM1])

        predict_t1_test_data_all.boosted_xB2 = predict_t1_test_data_partition.boosted_xB2 if predict_t1_test_data_all.boosted_xB2 is None else np.concatenate([predict_t1_test_data_all.boosted_xB2, predict_t1_test_data_partition.boosted_xB2])
        predict_t1_test_data_all.boosted_yB2 = predict_t1_test_data_partition.boosted_yB2 if predict_t1_test_data_all.boosted_yB2 is None else np.concatenate([predict_t1_test_data_all.boosted_yB2, predict_t1_test_data_partition.boosted_yB2])
        predict_t1_test_data_all.boosted_yprobB2 = predict_t1_test_data_partition.boosted_yprobB2 if predict_t1_test_data_all.boosted_yprobB2 is None else np.concatenate([predict_t1_test_data_all.boosted_yprobB2, predict_t1_test_data_partition.boosted_yprobB2])
        predict_t1_test_data_all.boosted_ypredB2 = predict_t1_test_data_partition.boosted_ypredB2 if predict_t1_test_data_all.boosted_ypredB2 is None else np.concatenate([predict_t1_test_data_all.boosted_ypredB2, predict_t1_test_data_partition.boosted_ypredB2])

        test_b1datadf = pd.concat([pd.DataFrame(predict_t1_test_data_partition.xB1), pd.DataFrame(predict_t1_test_data_partition.yB1), pd.DataFrame(predict_t1_test_data_partition.yprobB1)], axis=1)
        test_b1datadf.to_csv(cnst.PROJECT_BASE_PATH + cnst.ESC + "data" + cnst.ESC + "b1_test_"+str(fold_index)+"_pkl.csv", header=None, index=None, mode='a')
        test_m1datadf = pd.concat([pd.DataFrame(predict_t1_test_data_partition.xM1), pd.DataFrame(predict_t1_test_data_partition.yM1), pd.DataFrame(predict_t1_test_data_partition.yprobM1)], axis=1)
        test_m1datadf.to_csv(cnst.PROJECT_BASE_PATH + cnst.ESC + "data" + cnst.ESC + "m1_test_"+str(fold_index)+"_pkl.csv", header=None, index=None, mode='a')

    # test_b1datadf_all = pd.read_csv(cnst.PROJECT_BASE_PATH + cnst.ESC + "data" + cnst.ESC + "b1_test_"+str(fold_index)+"_pkl.csv", header=None)
    # predict_t1_test_data_all.xB1, predict_t1_test_data_all.yB1 = test_b1datadf_all.iloc[:, 0], test_b1datadf_all.iloc[:, 1]
    # test_m1datadf_all = pd.read_csv(cnst.PROJECT_BASE_PATH + cnst.ESC + "data" + cnst.ESC + "m1_test_"+str(fold_index)+"_pkl.csv", header=None)
    # predict_t1_test_data_all.xM1, predict_t1_test_data_all.yM1 = test_m1datadf_all.iloc[:, 0], test_m1datadf_all.iloc[:, 1]

    predict_t1_test_data_all = select_thd_get_metrics(predict_t1_test_data_all)

    if len(predict_t1_test_data_all.xB1) == 0:
        print(" \n !!!!!      Skipping Tier-2 - B1 set is empty")
        return None

    test_b1_partition_count = partition_pkl_files_by_count("b1_test", fold_index, predict_t1_test_data_all.xB1, predict_t1_test_data_all.yB1)

    # TIER-2 PREDICTION
    print("Prediction on Testing Data - TIER2 [B1 data]         # Partitions", test_b1_partition_count)  # \t\t\tSection Map Length:", len(section_map))
    predict_t2_test_data_all = pObj(cnst.TIER2, None, None, None)
    predict_t2_test_data_all.thd = thd2
    if thd2 is not None and q_sections is not None:
        for pcount in range(0, test_b1_partition_count):
            b1_tst_datadf = pd.read_csv(cnst.DATA_SOURCE_PATH + cnst.ESC + "b1_test_" + str(fold_index) + "_p" + str(pcount) + ".csv", header=None)

            predict_t2_test_data_partition = pObj(cnst.TIER2, None, b1_tst_datadf.iloc[:, 0], b1_tst_datadf.iloc[:, 1])  # predict_t1_test_data.xB1, predict_t1_test_data.yB1)
            predict_t2_test_data_partition.thd = thd2
            predict_t2_test_data_partition.q_sections = q_sections
            predict_t2_test_data_partition.predict_section_map = section_map
            predict_t2_test_data_partition.wpartition = get_partition_data("b1_test", fold_index, pcount, "t1")
            predict_t2_test_data_partition.spartition = get_partition_data("b1_test", fold_index, pcount, "t2")
            predict_t2_test_data_partition = predict_tier2(model_idx, predict_t2_test_data_partition, fold_index)

            del predict_t2_test_data_partition.wpartition  # Release Memory
            del predict_t2_test_data_partition.spartition  # Release Memory
            gc.collect()

            predict_t2_test_data_all.xtrue = predict_t2_test_data_partition.xtrue if predict_t2_test_data_all.xtrue is None else np.concatenate([predict_t2_test_data_all.xtrue, predict_t2_test_data_partition.xtrue])
            predict_t2_test_data_all.ytrue = predict_t2_test_data_partition.ytrue if predict_t2_test_data_all.ytrue is None else np.concatenate([predict_t2_test_data_all.ytrue, predict_t2_test_data_partition.ytrue])
            predict_t2_test_data_all.yprob = predict_t2_test_data_partition.yprob if predict_t2_test_data_all.yprob is None else np.concatenate([predict_t2_test_data_all.yprob, predict_t2_test_data_partition.yprob])
            predict_t2_test_data_all.ypred = predict_t2_test_data_partition.ypred if predict_t2_test_data_all.ypred is None else np.concatenate([predict_t2_test_data_all.ypred, predict_t2_test_data_partition.ypred])

            print("All Tier-2 Test data Size updated:", predict_t2_test_data_all.ytrue.shape)

        predict_t2_test_data_all = select_thd_get_metrics_bfn_mfp(cnst.TIER2, predict_t2_test_data_all)

        display_probability_chart(predict_t2_test_data_all.ytrue, predict_t2_test_data_all.yprob, predict_t2_test_data_all.thd, "TESTING_TIER2_PROB_PLOT_" + str(fold_index+1))
        print("List of TPs found: ", predict_t2_test_data_all.xtrue[np.all([predict_t2_test_data_all.ytrue.ravel() == cnst.MALWARE, predict_t2_test_data_all.ypred.ravel() == cnst.MALWARE], axis=0)])
        print("List of New FPs  : ", predict_t2_test_data_all.xtrue[np.all([predict_t2_test_data_all.ytrue.ravel() == cnst.BENIGN, predict_t2_test_data_all.ypred.ravel() == cnst.MALWARE], axis=0)])
    
        # RECONCILIATION OF PREDICTION RESULTS FROM TIER - 1&2
        still_benign_indices = np.where(np.all([predict_t2_test_data_all.ytrue.ravel() == cnst.BENIGN, predict_t2_test_data_all.ypred.ravel() == cnst.BENIGN], axis=0))[0]
        predict_t2_test_data_all.yprob[still_benign_indices] = predict_t1_test_data_all.yprobB1[still_benign_indices]  # Assign Tier-1 probabilities for samples that are still benign to avoid AUC conflict

        new_tp_indices = np.where(np.all([predict_t2_test_data_all.ytrue.ravel() == cnst.MALWARE, predict_t2_test_data_all.ypred.ravel() == cnst.MALWARE], axis=0))[0]
        predict_t2_test_data_all.yprob[new_tp_indices] -= predict_t2_test_data_all.yprob[new_tp_indices] + 1

        new_fp_indices = np.where(np.all([predict_t2_test_data_all.ytrue.ravel() == cnst.BENIGN, predict_t2_test_data_all.ypred.ravel() == cnst.MALWARE], axis=0))[0]
        predict_t2_test_data_all.yprob[new_fp_indices] -= predict_t2_test_data_all.yprob[new_fp_indices]

        cvobj, benchmark_fpr = reconcile(predict_t1_test_data_all, predict_t2_test_data_all, cv_obj, fold_index)
        # benchmark_tier1(model_idx, predict_t1_test_data, fold_index, benchmark_fpr)
        return cvobj
    else:
        print("Skipping Tier-2 prediction --- Reconciliation --- Not adding fold entry to CV AUC")
        return None


if __name__ == '__main__':
    print("PREDICT MAIN")
    '''print("Prediction on Testing Data")
    #import Predict as pObj
    testdata = pd.read_csv(cnst.PROJECT_BASE_PATH + 'small_pkl_1_1.csv', header=None)
    pObj_testdata = Predict(cnst.TIER1, cnst.TIER1_TARGET_FPR, testdata.iloc[:, 0].values, testdata.iloc[:, 1].values)
    pObj_testdata.thd = 67.1
    pObj_testdata = predict_tier1(0, pObj_testdata)  # TIER-1 prediction - on training data
    print("TPR:", pObj_testdata.tpr, "FPR:", pObj_testdata.fpr)'''


'''
def get_prediction_data(result_file):
    t1 = pd.read_csv(result_file, header=None)
    y1 = t1[1]
    p1 = t1[2]
    pv1 = t1[3]
    return y1, p1, pv1
'''
