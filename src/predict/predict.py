from utils import utils
import numpy as np
import pandas as pd
from sklearn import metrics
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.models import load_model
from config import settings as cnst
from .predict_args import DefaultPredictArguments, Predict as pObj
from plots.plots import display_probability_chart
from analyzers.collect_exe_files import get_partition_data, partition_pkl_files_by_count, partition_pkl_files_by_size
import gc
import os
from trend import activation_trend_identification as ati
import logging
import math


def predict_byte(model, partition, xfiles, args):
    """
        Function to perform prediction process on entire byte sequence of PE samples in Tier-1
        Params:
            model: Trained Tier-1 model to be used for prediction
            xfiles: list of sample names to be used for prediction
            partition: t1 partition containing the data for sample list in xfiles
            args: contains various config parameters
        Returns:
            pred: object with predicted probabilities
    """
    xlen = len(xfiles)
    pred_steps = xlen//args.batch_size if xlen % args.batch_size == 0 else xlen//args.batch_size + 1
    pred = model.predict(
        utils.data_generator(partition, xfiles, np.ones(xfiles.shape), args.max_len, args.batch_size, shuffle=False),
        steps=pred_steps,
        verbose=args.verbose
        )
    return pred


def predict_byte_by_section(model, spartition, xfiles, q_sections, section_map, args):
    """
        Function to perform prediction process on PE section level byte sequence of PE samples in Tier-2
        Params:
            model: Trained Tier-2 model to be used for prediction
            xfiles: list of sample names to be used for prediction
            spartition: t2 partition containing the section data for sample list in xfiles
            q_sections: list of qualified sections to be used - remaining PE sections' data will be set to 0
            section_map: not used here
            args: contains various config parameters
        Returns:
            pred: object with predicted probabilities
    """
    xlen = len(xfiles)
    pred_steps = xlen//args.batch_size if xlen % args.batch_size == 0 else xlen//args.batch_size + 1
    pred = model.predict(
        utils.data_generator_by_section(spartition, q_sections, section_map, xfiles, np.ones(xfiles.shape), args.max_len, args.batch_size, shuffle=False),
        steps=pred_steps,
        verbose=args.verbose
        )
    return pred


def predict_nn(args):
    """Obsolete: For this block implementation"""
    try:
        p = list()
        p.append(np.array(args.t2_nn_x_predict))
        if cnst.USE_SECTION_ID_EMB_FOR_NN:
            p.append(np.array(args.sec_id_emb_predict))
        if cnst.USE_ACT_MAG_FOR_NN:
            p.append(np.array(args.act_mag_predict))
        pred = args.t2_nn_model.predict(p, verbose=1)
    except Exception as e:
        logging.exception("Error during NN prediction")
    return pred


def predict_by_features(model, fn_list, label, batch_size, verbose, features_to_drop=None):
    """Obsolete: For this block implementation"""
    pred = model.predict_generator(
        utils.data_generator_by_features(fn_list, np.ones(fn_list.shape), batch_size, False, features_to_drop),
        steps=len(fn_list),
        verbose=verbose
    )
    return pred


def predict_by_fusion(model, fn_list, label, batch_size, verbose):
    """Obsolete: For this block implementation"""
    byte_sequence_max_len = model.input[0].shape[1]
    pred = model.predict_generator(
        utils.data_generator_by_fusion(fn_list, np.ones(fn_list.shape), byte_sequence_max_len, batch_size, shuffle=False),
        steps=len(fn_list),
        verbose=verbose
    )
    return pred


def calculate_prediction_metrics(predict_obj):
    """
        Function to calculate the prediction metrics like confusion matrix, auc, restricted auc
    """
    predict_obj.ypred = (predict_obj.yprob >= (predict_obj.thd / 100)).astype(int)
    cm = metrics.confusion_matrix(predict_obj.ytrue, predict_obj.ypred, labels=[cnst.BENIGN, cnst.MALWARE])
    tn = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tp = cm[1][1]

    predict_obj.tpr = (tp / (tp + fn)) * 100
    predict_obj.fpr = (fp / (fp + tn)) * 100
    # print("Before AUC score computation:", predict_obj.thd, predict_obj.tpr, predict_obj.fpr)
    try:
        predict_obj.auc = metrics.roc_auc_score(predict_obj.ytrue, predict_obj.ypred)
        predict_obj.rauc = metrics.roc_auc_score(predict_obj.ytrue, predict_obj.ypred, max_fpr=cnst.OVERALL_TARGET_FPR/100)
    except Exception as e:
        logging.exception("Error during prediction metrics calculation.")
    return predict_obj


def select_decision_threshold(predict_obj):
    """
        Function to calibrate and find a decision threshold for classification that satisfies supplied Target FPR
        Returns:
            predict_obj: prediction object updated with selected decision threshold value ,y_pred, TPR and FPR
    """
    predict_obj.target_fpr = (np.floor((predict_obj.target_fpr / 100) * len(predict_obj.yprob[predict_obj.ytrue == 0])) * 100) / len(predict_obj.yprob[predict_obj.ytrue == 0])
    calibrated_threshold = (np.percentile(predict_obj.yprob[predict_obj.ytrue == 0], q=[100 - predict_obj.target_fpr]) * 100)[0]
    logging.debug("Initial Calibrated Threshold: %s    %s", calibrated_threshold, predict_obj.target_fpr)
    pow = str(calibrated_threshold)[::-1].find('.')
    calibrated_threshold = math.ceil(calibrated_threshold * 10 ** (pow - 1)) / (10 ** (pow - 1))
    logging.debug("Ceiled Threshold: %s", calibrated_threshold)
    selected_threshold = calibrated_threshold if calibrated_threshold < 100.0 else 100.0

    temp_ypred = (predict_obj.yprob >= (selected_threshold / 100)).astype(int)
    cm = metrics.confusion_matrix(predict_obj.ytrue, temp_ypred, labels=[cnst.BENIGN, cnst.MALWARE])
    tn = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tp = cm[1][1]
    TPR = (tp / (tp + fn)) * 100
    FPR = (fp / (fp + tn)) * 100

    logging.info("Selected Threshold: " + str(selected_threshold) + "    TPR: {:6.3f}\tFPR: {:6.3f}".format(TPR, FPR))
    predict_obj.thd = selected_threshold
    predict_obj.ypred = temp_ypred
    predict_obj.tpr = TPR
    predict_obj.fpr = FPR
    return predict_obj


def get_bfn_mfp(pObj):
    """
        Function to perfom filtering of FNs and FPs in Tier-1 results (in pObj) to form the B1 and M1 sets.
        It also performs identifying a possible boosting bound during training process, and promotes the samples
        with yprob < boosting bound to B2 set directly - as they are treated as obvious benign files.
        Returns:
            pObj: predict object updated with B1 and M1 set information
    """
    prediction = (pObj.yprob >= (pObj.thd / 100)).astype(int)

    if cnst.PERFORM_B2_BOOSTING:
        if pObj.boosting_upper_bound is None:
            fn_indices = np.all([pObj.ytrue.ravel() == cnst.MALWARE, prediction.ravel() == cnst.BENIGN], axis=0)
            pObj.boosting_upper_bound = np.min(pObj.yprob[fn_indices]) if np.sum(fn_indices) > 0 else 0
            logging.info("Setting B2 boosting threshold: %s", pObj.boosting_upper_bound)

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
        logging.info("Number of files boosted to B2=" + str(len(boosted_indices)) + " \t[ " + str(len(np.where(prediction == cnst.BENIGN)[0])) + " - " + str(len(brow_indices)) + " ]     Boosting Bound used: " + str(pObj.boosting_upper_bound) + "   Escaped FNs:" + str(len(fn_escaped_by_boosting)))

    else:
        logging.info("NO BOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOSTING")
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
    """
        Function to initiate the Tier-1 prediction process
        Params:
            model_idx: Default 0. Do not change
            pobj: initialized with parameters for prediction
            fold_index: current fold index of cross validation
        Returns:
            pobj: object containing the predict results
    """
    predict_args = DefaultPredictArguments()
    tier1_model = load_model(predict_args.model_path + cnst.TIER1_MODELS[model_idx] + "_" + str(fold_index) + ".h5")
    # model.summary()

    if cnst.EXECUTION_TYPE[model_idx] == cnst.BYTE:
        pobj.yprob = predict_byte(tier1_model, pobj.partition, pobj.xtrue, predict_args)
    elif cnst.EXECUTION_TYPE[model_idx] == cnst.FEATURISTIC:
        pobj.yprob = predict_by_features(tier1_model, pobj.xtrue, predict_args)
    elif cnst.EXECUTION_TYPE[model_idx] == cnst.FUSION:
        pobj.yprob = predict_by_fusion(tier1_model, pobj.xtrue, predict_args)

    del tier1_model
    gc.collect()
    return pobj


def predict_tier2_block(model_idx, pobj, fold_index):
    """
        Function to initiate the Tier-2's block-based prediction process
        Params:
            model_idx: Default 0. Do not change
            pobj: initialized with parameters for prediction
            fold_index: current fold index of cross validation
        Returns:
            pobj: object containing the predict results
    """
    predict_args = DefaultPredictArguments()
    predict_args.max_len = cnst.TIER2_NEW_INPUT_SHAPE
    tier2_model = load_model(predict_args.model_path + cnst.TIER2_MODELS[model_idx] + "_" + str(fold_index) + ".h5")

    if cnst.EXECUTION_TYPE[model_idx] == cnst.BYTE:
        pobj.yprob = predict_byte(tier2_model, pobj.partition, pobj.xtrue, predict_args)
    elif cnst.EXECUTION_TYPE[model_idx] == cnst.FEATURISTIC:
        pobj.yprob = predict_by_features(tier2_model, pobj.xtrue, predict_args)
    elif cnst.EXECUTION_TYPE[model_idx] == cnst.FUSION:
        pobj.yprob = predict_by_fusion(tier2_model, pobj.xtrue, predict_args)

    del tier2_model
    gc.collect()
    return pobj


def select_thd_get_metrics_bfn_mfp(tier, pobj):
    """
        Funtion used to select decision threshold during training process or
        to obtain prediction metrics (TPR, FPR) during Testing phase.

        It also perfoms filtering FNs and FPs in Tier-1 results to form the B1 and M1 sets.
        Params:
            tier: TIER1 or TIER2
            pobj: object with prediction outcomes
        Returns:
            pobj: object updated with calculated prediction metrics
    """
    if pobj.thd is None:
        pobj = select_decision_threshold(pobj)  # +++ returned pobj also includes ypred based on selected threshold
    else:
        pobj = calculate_prediction_metrics(pobj)

    if tier == cnst.TIER1:
        pobj = get_bfn_mfp(pobj)

    return pobj


def select_thd_get_metrics(pobj):
    """
        Funtion used to select decision threshold during training process or
        to obtain prediction metrics (TPR, FPR) during Testing phase.
        Params:
            tier: TIER1 or TIER2
            pobj: object with prediction outcomes
        Returns:
            pobj: object updated with calculated prediction metrics
    """
    if pobj.thd is None:
        pobj = select_decision_threshold(pobj)  # +++ returned pobj also includes ypred based on selected threshold
    else:
        pobj = calculate_prediction_metrics(pobj)

    return pobj


def predict_tier2(model_idx, pobj, fold_index):
    """
        Function to initiate the Tier-2 prediction process
        Params:
            model_idx: Default 0. Do not change
            pobj: initialized with parameters for prediction
            fold_index: current fold index of cross validation
        Returns:
            pobj: object containing the predict results
    """
    predict_args = DefaultPredictArguments()
    tier2_model = load_model(predict_args.model_path + cnst.TIER2_MODELS[model_idx] + "_" + str(fold_index) + ".h5")

    '''if cnst.EXECUTION_TYPE[model_idx] == cnst.BYTE:
        # pbs.trigger_predict_by_section()
        pobj.yprob = predict_byte_by_section(tier2_model, pobj.spartition, pobj.xtrue, pobj.q_sections, pobj.predict_section_map, predict_args)
    elif cnst.EXECUTION_TYPE[model_idx] == cnst.FEATURISTIC:
        pobj.yprob = predict_by_features(tier2_model, pobj.xtrue, predict_args)
    elif cnst.EXECUTION_TYPE[model_idx] == cnst.FUSION:
        pobj.yprob = predict_by_fusion(tier2_model, pobj.xtrue, predict_args)'''

    if cnst.EXECUTION_TYPE[model_idx] == cnst.BYTE:
        # pbs.trigger_predict_by_section()
        pobj.yprob = predict_byte_by_section(tier2_model, pobj.spartition, pobj.xtrue, pobj.q_sections, pobj.predict_section_map, predict_args)
    elif cnst.EXECUTION_TYPE[model_idx] == cnst.FEATURISTIC:
        pobj.yprob = predict_by_features(tier2_model, pobj.xtrue, predict_args)
    elif cnst.EXECUTION_TYPE[model_idx] == cnst.FUSION:
        pobj.yprob = predict_by_fusion(tier2_model, pobj.xtrue, predict_args)

    del tier2_model
    gc.collect()
    return pobj


def get_reconciled_tpr_fpr(yt1, yp1, yt2, yp2):
    """
        Function to obtain TPR and FPR of reconciled Tier-1 & Tier-2's y_true, y_pred
    """
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
    """
        Function to obtain TPR and FPR based on y_true and y_pred
    """
    cm = metrics.confusion_matrix(yt, yp, labels=[cnst.BENIGN, cnst.MALWARE])
    tn = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tp = cm[1][1]
    tpr = (tp / (tp+fn)) * 100
    fpr = (fp / (fp+tn)) * 100
    return tpr, fpr


def reconcile(pt1, pt2, cv_obj, fold_index):
    """
        Funtion to reconcile the prediction results of Tier-1 and Tier-2 for the current fold of cross validation
    """
    logging.info("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   RECONCILING DATA  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    # RECONCILE - xM1, yprobM1, xB1, pred_proba2
    logging.info("BEFORE RECONCILIATION: [Total True Malwares:" + str(np.sum(pt1.ytrue)) + "]    M1 has => " + str(np.shape(pt1.xM1)[0]) +
          "\tTPs:" + str(np.sum(np.all([pt1.ytrue.ravel() == cnst.MALWARE, pt1.ypred.ravel() == cnst.MALWARE], axis=0))) +
          "\tFPs:" + str(np.sum(np.all([pt1.ytrue.ravel() == cnst.BENIGN, pt1.ypred.ravel() == cnst.MALWARE], axis=0))))
    logging.info("                       [Total True Benign  :" + str(len(np.where(pt1.ytrue.ravel() == cnst.BENIGN)[0])) + "]    B1 has => " + str(len(np.where(pt1.ypred == cnst.BENIGN)[0])) +
          "\tTNs:" + str(np.sum(np.all([pt1.ytrue.ravel() == cnst.BENIGN, pt1.ypred.ravel() == cnst.BENIGN], axis=0))) +
          "\tFNs:" + str(np.sum(np.all([pt1.ytrue.ravel() == cnst.MALWARE, pt1.ypred.ravel() == cnst.BENIGN], axis=0))))

    if cnst.PERFORM_B2_BOOSTING:
        logging.info("                       Boosted: " + str(np.shape(pt1.boosted_xB2)[0]) + "\tRemaining B1:" + str(np.shape(pt2.xtrue)[0]))
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

    logging.info("AFTER RECONCILIATION :   [M1+B1] => [ M1+(M2+B2)" + ("+B2_Boosted" if cnst.PERFORM_B2_BOOSTING else "") + " ]  = " + str(np.shape(xtruereconciled)[0]) +
          "   New TPs found: [M2] =>" + str(np.sum(np.all([pt2.ytrue.ravel() == cnst.MALWARE, pt2.ypred.ravel() == cnst.MALWARE], axis=0))) +
          "\tNew FPs:  =>" + str(np.sum(np.all([pt2.ytrue.ravel() == cnst.BENIGN, pt2.ypred.ravel() == cnst.MALWARE], axis=0))) +
          "\n[RECON TPs] =>" + str(np.sum(np.all([ytruereconciled.ravel() == cnst.MALWARE, ypredreconciled.ravel() == cnst.MALWARE], axis=0))) +
          "\tFPs:" + str(np.sum(np.all([ytruereconciled.ravel() == cnst.BENIGN, ypredreconciled.ravel() == cnst.MALWARE], axis=0))) +
          "\n[RECON TNs] =>" + str(np.sum(np.all([ytruereconciled.ravel() == cnst.BENIGN, ypredreconciled.ravel() == cnst.BENIGN], axis=0))) +
          "\tFNs:" + str(np.sum(np.all([ytruereconciled.ravel() == cnst.MALWARE, ypredreconciled.ravel() == cnst.BENIGN], axis=0))))

    reconciled_tpr = np.array([])
    reconciled_fpr = np.array([])
    tier1_tpr = np.array([])
    tier1_fpr = np.array([])
    probability_score = np.arange(0, 100.01, 0.1)  # 0.1
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
    logging.info("FOLD: %s\tTIER1 TPR: %s\tFPR: %s\tOVERALL TPR: %s\tFPR: %s", fold_index+1, tpr1, fpr1, rtpr, rfpr)

    cv_obj.t1_tpr_list = np.append(cv_obj.t1_tpr_list, tpr1)
    cv_obj.t1_fpr_list = np.append(cv_obj.t1_fpr_list, fpr1)
    cv_obj.recon_tpr_list = np.append(cv_obj.recon_tpr_list, rtpr)
    cv_obj.recon_fpr_list = np.append(cv_obj.recon_fpr_list, rfpr)

    cv_obj.t1_mean_auc_score = np.append(cv_obj.t1_mean_auc_score, metrics.roc_auc_score(pt1.ytrue, pt1.yprob))
    cv_obj.t1_mean_auc_score_restricted = np.append(cv_obj.t1_mean_auc_score_restricted, metrics.roc_auc_score(pt1.ytrue, pt1.yprob, max_fpr=cnst.OVERALL_TARGET_FPR/100))
    cv_obj.recon_mean_auc_score = np.append(cv_obj.recon_mean_auc_score, metrics.roc_auc_score(ytruereconciled, yprobreconciled))
    cv_obj.recon_mean_auc_score_restricted = np.append(cv_obj.recon_mean_auc_score_restricted, metrics.roc_auc_score(ytruereconciled, yprobreconciled, max_fpr=cnst.OVERALL_TARGET_FPR/100))
    logging.info(r'Tier1 Restricted AUC = %0.3f ± %0.2f ' % (  # [Full AUC: %0.3f]
        metrics.roc_auc_score(pt1.ytrue, pt1.yprob, max_fpr=cnst.OVERALL_TARGET_FPR/100)
        , np.std(metrics.roc_auc_score(pt1.ytrue, pt1.yprob, max_fpr=cnst.OVERALL_TARGET_FPR/100))
        # , metrics.roc_auc_score(pt1.ytrue, pt1.yprob)
        ))
    logging.info(r'Recon Restricted AUC = %0.3f ± %0.2f ' % (  # [Full AUC: %0.3f]
        metrics.roc_auc_score(ytruereconciled, yprobreconciled, max_fpr=cnst.OVERALL_TARGET_FPR/100)
        , np.std(metrics.roc_auc_score(ytruereconciled, yprobreconciled, max_fpr=cnst.OVERALL_TARGET_FPR/100))
        # , metrics.roc_auc_score(ytruereconciled, yprobreconciled)
        ))

    return cv_obj, rfpr


def init(model_idx, test_partitions, cv_obj, fold_index, scalers):
    """ Module for performing prediction for Testing phase
        # ##################################################################################################################
        # OBJECTIVES:
        #     1) Predict using trained Tier-1 model over Testing data - using selected THD1 found during Tier-1 training process
        #     2) Obtain B1 set of test samples and collect top activation blocks based dataset using qualified sections found
        #     3) Predict using trained Tier-2 model over block dataset - using THD2 found during Tier-2 training process
        #     4) Reconcile results for Tier-1 and Tier-2 and generate overall results
        # ##################################################################################################################

        Args:
            model_idx: Default 0 for byte sequence models. Do not change.
            test_partitions: list of partition indexes to be used for testing
            cv_obj: object to hold cross validation results for current fold
            fold_index: current fold of cross-validation
            scalers: not used here
        Returns:
            cv_obj
    """
    # TIER-1 PREDICTION OVER TEST DATA
    partition_tracker_df = pd.read_csv(cnst.DATA_SOURCE_PATH + cnst.ESC + "partition_tracker_" + str(fold_index) + ".csv")
    logging.info("Prediction on Testing Data - TIER1       # Partitions: %s", partition_tracker_df["test"][0])

    todf = pd.read_csv(os.path.join(cnst.PROJECT_BASE_PATH + cnst.ESC + "out" + cnst.ESC + "result" + cnst.ESC, "training_outcomes_" + str(fold_index) + ".csv"))
    thd1 = todf["thd1"][0]
    thd2 = todf["thd2"][0]
    boosting_bound = todf["boosting_bound"][0]
    logging.info("THD1: %s    THD2: %s    Boosting Bound: %s", thd1, thd2, boosting_bound)

    if thd2 is None or math.isnan(thd2):
        logging.critical("Threshold for Tier-2 model is not available. Aborting entire prediction process.")
        return

    section_map = None
    q_sections = pd.read_csv(os.path.join(cnst.PROJECT_BASE_PATH + cnst.ESC + "out" + cnst.ESC + "result" + cnst.ESC, "qualified_sections_" + str(fold_index) + ".csv"), header=None).iloc[0]

    predict_t1_test_data_all = pObj(cnst.TIER2, None, None, None)
    predict_t1_test_data_all.thd = thd1

    if not cnst.SKIP_TIER1_PREDICTION:
        pd.DataFrame().to_csv(cnst.PROJECT_BASE_PATH + cnst.ESC + "data" + cnst.ESC + "tier1_test_"+str(fold_index)+"_pkl.csv", header=None, index=None)
        pd.DataFrame().to_csv(cnst.PROJECT_BASE_PATH + cnst.ESC + "data" + cnst.ESC + "b1_test_"+str(fold_index)+"_pkl.csv", header=None, index=None)
        pd.DataFrame().to_csv(cnst.PROJECT_BASE_PATH + cnst.ESC + "data" + cnst.ESC + "m1_test_"+str(fold_index)+"_pkl.csv", header=None, index=None)
        pd.DataFrame().to_csv(cnst.PROJECT_BASE_PATH + cnst.ESC + "data" + cnst.ESC + "b2_test_"+str(fold_index)+"_pkl.csv", header=None, index=None)
        for pcount in test_partitions:
            logging.info("Predicting partition: %s", pcount)
            tst_datadf = pd.read_csv(cnst.DATA_SOURCE_PATH + cnst.ESC + "p" + str(pcount) + ".csv", header=None)

            predict_t1_test_data_partition = pObj(cnst.TIER1, None, tst_datadf.iloc[:, 0].values, tst_datadf.iloc[:, 1].values)
            predict_t1_test_data_partition.thd = thd1
            predict_t1_test_data_partition.boosting_upper_bound = boosting_bound
            predict_t1_test_data_partition.partition = get_partition_data(None, None, pcount, "t1")
            predict_t1_test_data_partition = predict_tier1(model_idx, predict_t1_test_data_partition, fold_index)

            del predict_t1_test_data_partition.partition  # Release Memory
            gc.collect()

            # predict_t1_test_data_partition = get_bfn_mfp(predict_t1_test_data_partition)
            predict_t1_test_data_partition = select_thd_get_metrics_bfn_mfp("TIER1", predict_t1_test_data_partition)

            predict_t1_test_data_all.xtrue = predict_t1_test_data_partition.xtrue if predict_t1_test_data_all.xtrue is None else np.concatenate([predict_t1_test_data_all.xtrue, predict_t1_test_data_partition.xtrue])
            predict_t1_test_data_all.ytrue = predict_t1_test_data_partition.ytrue if predict_t1_test_data_all.ytrue is None else np.concatenate([predict_t1_test_data_all.ytrue, predict_t1_test_data_partition.ytrue])
            predict_t1_test_data_all.yprob = predict_t1_test_data_partition.yprob if predict_t1_test_data_all.yprob is None else np.concatenate([predict_t1_test_data_all.yprob, predict_t1_test_data_partition.yprob])
            predict_t1_test_data_all.ypred = predict_t1_test_data_partition.ypred if predict_t1_test_data_all.ypred is None else np.concatenate([predict_t1_test_data_all.ypred, predict_t1_test_data_partition.ypred])

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

            test_tier1datadf = pd.concat([pd.DataFrame(predict_t1_test_data_partition.xtrue), pd.DataFrame(predict_t1_test_data_partition.ytrue), pd.DataFrame(predict_t1_test_data_partition.yprob), pd.DataFrame(predict_t1_test_data_partition.ypred)], axis=1)
            test_tier1datadf.to_csv(cnst.PROJECT_BASE_PATH + cnst.ESC + "data" + cnst.ESC + "tier1_test_"+str(fold_index)+"_pkl.csv", header=None, index=None, mode='a')
            test_b1datadf = pd.concat([pd.DataFrame(predict_t1_test_data_partition.xB1), pd.DataFrame(predict_t1_test_data_partition.yB1), pd.DataFrame(predict_t1_test_data_partition.yprobB1), pd.DataFrame(predict_t1_test_data_partition.ypredB1)], axis=1)
            test_b1datadf.to_csv(cnst.PROJECT_BASE_PATH + cnst.ESC + "data" + cnst.ESC + "b1_test_"+str(fold_index)+"_pkl.csv", header=None, index=None, mode='a')
            test_m1datadf = pd.concat([pd.DataFrame(predict_t1_test_data_partition.xM1), pd.DataFrame(predict_t1_test_data_partition.yM1), pd.DataFrame(predict_t1_test_data_partition.yprobM1), pd.DataFrame(predict_t1_test_data_partition.ypredM1)], axis=1)
            test_m1datadf.to_csv(cnst.PROJECT_BASE_PATH + cnst.ESC + "data" + cnst.ESC + "m1_test_"+str(fold_index)+"_pkl.csv", header=None, index=None, mode='a')
            test_b2datadf = pd.concat([pd.DataFrame(predict_t1_test_data_partition.boosted_xB2), pd.DataFrame(predict_t1_test_data_partition.boosted_yB2), pd.DataFrame(predict_t1_test_data_partition.boosted_yprobB2), pd.DataFrame(predict_t1_test_data_partition.boosted_ypredB2)], axis=1)
            test_b2datadf.to_csv(cnst.PROJECT_BASE_PATH + cnst.ESC + "data" + cnst.ESC + "b2_test_"+str(fold_index)+"_pkl.csv", header=None, index=None, mode='a')

        test_b1_partition_count = partition_pkl_files_by_count("b1_test", fold_index, predict_t1_test_data_all.xB1, predict_t1_test_data_all.yB1) if cnst.PARTITION_BY_COUNT else partition_pkl_files_by_size("b1_test", fold_index, predict_t1_test_data_all.xB1, predict_t1_test_data_all.yB1)

    else:
        logging.info("Skipped TIER-1 prediction")

        test_tier1datadf_all = pd.read_csv(cnst.PROJECT_BASE_PATH + cnst.ESC + "data" + cnst.ESC + "tier1_test_"+str(fold_index)+"_pkl.csv", header=None)
        test_b1datadf_all = pd.read_csv(cnst.PROJECT_BASE_PATH + cnst.ESC + "data" + cnst.ESC + "b1_test_"+str(fold_index)+"_pkl.csv", header=None)
        test_m1datadf_all = pd.read_csv(cnst.PROJECT_BASE_PATH + cnst.ESC + "data" + cnst.ESC + "m1_test_"+str(fold_index)+"_pkl.csv", header=None)
        test_b2datadf_all = pd.read_csv(cnst.PROJECT_BASE_PATH + cnst.ESC + "data" + cnst.ESC + "b2_test_"+str(fold_index)+"_pkl.csv", header=None)

        predict_t1_test_data_all.xtrue = test_tier1datadf_all.iloc[:, 0]
        predict_t1_test_data_all.ytrue = test_tier1datadf_all.iloc[:, 1]
        predict_t1_test_data_all.yprob = test_tier1datadf_all.iloc[:, 2]
        predict_t1_test_data_all.ypred = test_tier1datadf_all.iloc[:, 3]

        predict_t1_test_data_all.xB1 = test_b1datadf_all.iloc[:, 0]
        predict_t1_test_data_all.yB1 = test_b1datadf_all.iloc[:, 1]
        predict_t1_test_data_all.yprobB1 = test_b1datadf_all.iloc[:, 2]
        predict_t1_test_data_all.ypredB1 = test_b1datadf_all.iloc[:, 3]

        predict_t1_test_data_all.xM1 = test_m1datadf_all.iloc[:, 0]
        predict_t1_test_data_all.yM1 = test_m1datadf_all.iloc[:, 1]
        predict_t1_test_data_all.yprobM1 = test_m1datadf_all.iloc[:, 2]
        predict_t1_test_data_all.ypredM1 = test_m1datadf_all.iloc[:, 3]

        predict_t1_test_data_all.boosted_xB2 = test_b2datadf_all.iloc[:, 0]
        predict_t1_test_data_all.boosted_yB2 = test_b2datadf_all.iloc[:, 1]
        predict_t1_test_data_all.boosted_yprobB2 = test_b2datadf_all.iloc[:, 2]
        predict_t1_test_data_all.boosted_ypredB2 = test_b2datadf_all.iloc[:, 3]

        logging.info("Loaded old TIER-1 prediction data")

    predict_t1_test_data_all = select_thd_get_metrics(predict_t1_test_data_all)
    logging.info("Threshold used for Prediction : " + str(predict_t1_test_data_all.thd) + "\t\tTPR: {:6.3f}\tFPR: {:6.3f}\tAUC: {:6.3f}\tRst. AUC: {:6.3f}".format(
        predict_t1_test_data_all.tpr, predict_t1_test_data_all.fpr, predict_t1_test_data_all.auc, predict_t1_test_data_all.rauc))

    if len(predict_t1_test_data_all.xB1) == 0:
        logging.info("!!!!!      Skipping Tier-2 - B1 set is empty")
        return None

    if thd2 is not None and q_sections is not None:
        p_args = DefaultPredictArguments()
        p_args.q_sections = q_sections
        p_args.t1_model_name = cnst.TIER1_MODELS[model_idx] + "_" + str(fold_index) + ".h5"
        p_args.t2_model_name = cnst.TIER2_MODELS[model_idx] + "_" + str(fold_index) + ".h5"

        if not cnst.SKIP_TIER1_PREDICTION:
            logging.info("Collecting Block Data for fold-%s partitions-%s type-%s", fold_index, test_b1_partition_count, "test")
            ati.collect_b1_block_dataset(p_args, fold_index, test_b1_partition_count, "test")

        logging.info("Retrieving stored B1 Data for Block Prediction with selected THD2")
        '''p_args.t2_nn_model = load_model(cnst.MODEL_PATH + "nn_t2_" + str(fold_index) + ".h5")
        nn_test_data = pd.read_csv(cnst.PROJECT_BASE_PATH + cnst.ESC + 'data' + cnst.ESC + 'b1_test_'+str(fold_index)+'_qX_nn_dataset.csv', header=None)
        all_nn_x_predict, p_args.t2_nn_y_predict = nn_test_data.iloc[:, 0:-2], nn_test_data.iloc[:, -1]
        # p_args.t2_nn_x_predict = pd.DataFrame(scaler.transform(p_args.t2_nn_x_predict), columns=p_args.t2_nn_x_predict.columns)
        p_args.t2_nn_x_predict = pd.DataFrame(scalers[0].transform(all_nn_x_predict.iloc[:, 0:128]), columns=all_nn_x_predict.columns[0:128])
        if cnst.USE_SECTION_ID_EMB_FOR_NN:
            if cnst.SCALE_SECTION_ID_EMB_FOR_NN:
                p_args.sec_id_emb_predict = pd.DataFrame(scalers[1].transform(all_nn_x_predict.iloc[:, 128:256]), columns=all_nn_x_predict.columns[128:256])
            else:
                p_args.sec_id_emb_predict = all_nn_x_predict.iloc[:, 128:256]
        if cnst.USE_ACT_MAG_FOR_NN:
            if cnst.SCALE_ACT_MAG_FOR_NN:
                p_args.act_mag_predict = pd.DataFrame(scalers[2].transform(all_nn_x_predict.iloc[:, 256:384]), columns=all_nn_x_predict.columns[256:384])
            else:
                p_args.act_mag_predict = all_nn_x_predict.iloc[:, 256:384]
        # TIER-2 PREDICTION
        print("Prediction on Testing Data - TIER2 [B1 data]         # Partitions", test_b1_partition_count)  # \t\t\tSection Map Length:", len(section_map))
        t2_nn_pred = predict_nn(p_args)
        predict_t2_test_data_all = pObj(cnst.TIER2, None, nn_test_data.iloc[:, -2], p_args.t2_nn_y_predict)
        predict_t2_test_data_all.thd = thd2
        predict_t2_test_data_all.yprob = t2_nn_pred
        '''
        # TIER-2 PREDICTION
        logging.info("Prediction on Testing Data - TIER2 [B1 data]         # Partitions: %s", test_b1_partition_count)  # \t\t\tSection Map Length:", len(section_map))
        predict_t2_test_data_all = pObj(cnst.TIER2, None, None, None)
        predict_t2_test_data_all.thd = thd2

        for pcount in range(0, test_b1_partition_count):
            b1_tst_datadf = pd.read_csv(cnst.DATA_SOURCE_PATH + cnst.ESC + "b1_test_" + str(fold_index) + "_p" + str(pcount) + ".csv", header=None)

            predict_t2_test_data_partition = pObj(cnst.TIER2, None, b1_tst_datadf.iloc[:, 0], b1_tst_datadf.iloc[:, 1])  # predict_t1_test_data.xB1, predict_t1_test_data.yB1)
            predict_t2_test_data_partition.thd = thd2
            predict_t2_test_data_partition.q_sections = q_sections
            predict_t2_test_data_partition.predict_section_map = section_map
            # predict_t2_test_data_partition.wpartition = get_partition_data("b1_test", fold_index, pcount, "t1")
            predict_t2_test_data_partition.spartition = get_partition_data("b1_test", fold_index, pcount, "t2")
            predict_t2_test_data_partition = predict_tier2(model_idx, predict_t2_test_data_partition, fold_index)

            # del predict_t2_test_data_partition.wpartition  # Release Memory
            del predict_t2_test_data_partition.spartition  # Release Memory
            gc.collect()

            predict_t2_test_data_all.xtrue = predict_t2_test_data_partition.xtrue if predict_t2_test_data_all.xtrue is None else np.concatenate([predict_t2_test_data_all.xtrue, predict_t2_test_data_partition.xtrue])
            predict_t2_test_data_all.ytrue = predict_t2_test_data_partition.ytrue if predict_t2_test_data_all.ytrue is None else np.concatenate([predict_t2_test_data_all.ytrue, predict_t2_test_data_partition.ytrue])
            predict_t2_test_data_all.yprob = predict_t2_test_data_partition.yprob if predict_t2_test_data_all.yprob is None else np.concatenate([predict_t2_test_data_all.yprob, predict_t2_test_data_partition.yprob])
            predict_t2_test_data_all.ypred = predict_t2_test_data_partition.ypred if predict_t2_test_data_all.ypred is None else np.concatenate([predict_t2_test_data_all.ypred, predict_t2_test_data_partition.ypred])

            logging.info("Overall Tier-2 Test data Size updated: %s", predict_t2_test_data_all.ytrue.shape)

        predict_t2_test_data_all = select_thd_get_metrics_bfn_mfp(cnst.TIER2, predict_t2_test_data_all)
        display_probability_chart(predict_t2_test_data_all.ytrue, predict_t2_test_data_all.yprob, predict_t2_test_data_all.thd, "TESTING_TIER2_PROB_PLOT_F" + str(fold_index))
        logging.info("List of TPs found: %s", predict_t2_test_data_all.xtrue[np.all([predict_t2_test_data_all.ytrue.ravel() == cnst.MALWARE, predict_t2_test_data_all.ypred.ravel() == cnst.MALWARE], axis=0)])
        logging.info("List of New FPs  : %s", predict_t2_test_data_all.xtrue[np.all([predict_t2_test_data_all.ytrue.ravel() == cnst.BENIGN, predict_t2_test_data_all.ypred.ravel() == cnst.MALWARE], axis=0)])
    
        # RECONCILIATION OF PREDICTION RESULTS FROM TIER - 1&2
        predict_t1_test_data_all.yprobB1 = np.array(predict_t1_test_data_all.yprobB1).ravel()
        predict_t1_test_data_all.yprobM1 = np.array(predict_t1_test_data_all.yprobM1).ravel()
        predict_t1_test_data_all.boosted_yprobB2 = np.array(predict_t1_test_data_all.boosted_yprobB2).ravel()

        predict_t1_test_data_all.ypredB1 = np.array(predict_t1_test_data_all.ypredB1).ravel()
        predict_t1_test_data_all.ypredM1 = np.array(predict_t1_test_data_all.ypredM1).ravel()
        predict_t1_test_data_all.boosted_ypredB2 = np.array(predict_t1_test_data_all.boosted_ypredB2).ravel()

        predict_t2_test_data_all.yprob = np.array(predict_t2_test_data_all.yprob).ravel()
        predict_t2_test_data_all.ypred = np.array(predict_t2_test_data_all.ypred).ravel()

        still_benign_indices = np.where(np.all([predict_t2_test_data_all.ytrue.ravel() == cnst.BENIGN, predict_t2_test_data_all.ypred.ravel() == cnst.BENIGN], axis=0))[0]
        predict_t2_test_data_all.yprob[still_benign_indices] = predict_t1_test_data_all.yprobB1[still_benign_indices]  # Assign Tier-1 probabilities for samples that are still benign to avoid AUC conflict

        new_tp_indices = np.where(np.all([predict_t2_test_data_all.ytrue.ravel() == cnst.MALWARE, predict_t2_test_data_all.ypred.ravel() == cnst.MALWARE], axis=0))[0]
        #predict_t2_test_data_all.yprob[new_tp_indices] -= predict_t2_test_data_all.yprob[new_tp_indices] + 1

        new_fp_indices = np.where(np.all([predict_t2_test_data_all.ytrue.ravel() == cnst.BENIGN, predict_t2_test_data_all.ypred.ravel() == cnst.MALWARE], axis=0))[0]
        predict_t2_test_data_all.yprob[new_fp_indices] = predict_t1_test_data_all.yprob[new_fp_indices]

        cvobj, benchmark_fpr = reconcile(predict_t1_test_data_all, predict_t2_test_data_all, cv_obj, fold_index)
        # benchmark_tier1(model_idx, predict_t1_test_data, fold_index, benchmark_fpr)
        return cvobj
    else:
        logging.info("Skipping Tier-2 prediction --- Reconciliation --- Not adding fold entry to CV AUC")
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
