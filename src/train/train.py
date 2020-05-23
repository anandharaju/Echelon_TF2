import warnings
warnings.filterwarnings("ignore")

import time
import os
from os.path import join
import argparse
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model, save_model
from keras.utils import multi_gpu_model
from utils import utils
import model_skeleton.featuristic as featuristic
import model_skeleton.malfusion as malfusion
import model_skeleton.echelon as echelon
import model_skeleton.echelon_multi as echelon_multi
from keras import optimizers
from trend import activation_trend_identification as ati
import config.constants as cnst
from .train_args import DefaultTrainArguments
from plots.plots import plot_partition_epoch_history
from predict import predict
from predict.predict_args import Predict as pObj, DefaultPredictArguments, QStats
import numpy as np
from sklearn.utils import class_weight
import pandas as pd
from analyzers.collect_available_sections import collect_sections
import random
from plots.plots import display_probability_chart
from analyzers.collect_exe_files import get_partition_data, partition_pkl_files_by_count
import gc


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


def train(args):
    # print("Memory Required:", get_model_memory_usage(args.t1_batch_size, args.t1_model_base))
    train_steps = len(args.t1_x_train) // args.t1_batch_size
    args.t1_train_steps = train_steps - 1 if len(args.t1_x_train) % args.t1_batch_size == 0 else train_steps + 1

    if args.t1_x_val is not None:
        val_steps = len(args.t1_x_val) // args.t1_batch_size
        args.t1_val_steps = val_steps - 1 if len(args.t1_x_val) % args.t1_batch_size == 0 else val_steps + 1

    args.t1_ear = EarlyStopping(monitor='acc', patience=3)
    args.t1_mcp = ModelCheckpoint(join(args.save_path, args.t1_model_name),
                               monitor="acc", save_best_only=args.save_best, save_weights_only=False)

    history = args.t1_model_base.fit_generator(
        utils.data_generator(args.train_partition, args.t1_x_train, args.t1_y_train, args.t1_max_len, args.t1_batch_size, args.t1_shuffle),
        class_weight=args.t1_class_weights,
        steps_per_epoch=args.t1_train_steps,
        epochs=args.t1_epochs,
        verbose=args.t1_verbose,
        callbacks=[args.t1_ear, args.t1_mcp]
        # , validation_data=utils.data_generator(args.t1_x_val, args.t1_y_val, args.t1_max_len, args.t1_batch_size,
        # args.t1_shuffle) , validation_steps=val_steps
    )

    # plot_history(history, cnst.TIER1)
    return history


def train_by_section(args):
    train_steps = len(args.t2_x_train)//args.t2_batch_size
    args.t2_train_steps = train_steps - 1 if len(args.t2_x_train) % args.t2_batch_size == 0 else train_steps + 1

    if args.t2_x_val is not None:
        val_steps = len(args.t2_x_val) // args.t2_batch_size
        args.t2_val_steps = val_steps - 1 if len(args.t2_x_val) % args.t2_batch_size == 0 else val_steps + 1

    args.t2_ear = EarlyStopping(monitor='acc', patience=3)
    args.t2_mcp = ModelCheckpoint(join(args.save_path, args.t2_model_name),
                               monitor="acc", save_best_only=args.save_best, save_weights_only=False)

    # Check MAX_LEN modification is needed - based on proportion of section vs whole file size
    # args.max_len = cnst.MAX_FILE_SIZE_LIMIT + (cnst.CONV_WINDOW_SIZE * len(args.q_sections))
    history = args.t2_model_base.fit_generator(
        utils.data_generator_by_section(args.whole_b1_train_partition, args.section_b1_train_partition, args.q_sections, args.train_section_map, args.t2_x_train, args.t2_y_train, args.t2_max_len, args.t2_batch_size, args.t2_shuffle),
        class_weight=args.t2_class_weights,
        steps_per_epoch=len(args.t2_x_train)//args.t2_batch_size + 1,
        epochs=args.t2_epochs,
        verbose=args.t2_verbose,
        callbacks=[args.t2_ear, args.t2_mcp]
        # , validation_data=utils.data_generator_by_section(args.q_sections, args.t2_x_val, args.t2_y_val
        # , args.t2_max_len, args.t2_batch_size, args.t2_shuffle)
        # , validation_steps=args.val_steps
    )
    # plot_history(history, cnst.TIER2)
    return history


def get_model1(args):
    # prepare TIER-1 model
    model1 = None
    optimizer = optimizers.Adam(lr=0.001)  # , beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    if args.resume:
        if cnst.USE_PRETRAINED_FOR_TIER1:
            print("[ CAUTION ] : Resuming with pretrained model for TIER1 - "+args.pretrained_t1_model_name)
            model1 = load_model(args.model_path + args.pretrained_t1_model_name)
            model1.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
            if cnst.NUM_GPU > 1:
                multi_gpu_model1 = multi_gpu_model(model1, gpus=cnst.NUM_GPU)
                multi_gpu_model1.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
                return multi_gpu_model1
        else:
            print("[ CAUTION ] : Resuming with old model")
            model1 = load_model(args.model_path + args.t1_model_name)
            model1.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
            if cnst.NUM_GPU > 1:
                multi_gpu_model1 = multi_gpu_model(model1, gpus=cnst.NUM_GPU)
                multi_gpu_model1.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=["accuracy"])
                return multi_gpu_model1
    else:
        if args.byte:
            model1 = echelon.model(args.t1_max_len, args.t1_win_size)
        elif args.featuristic:
            model1 = featuristic.model(args.total_features)
        elif args.fusion:
            model1 = malfusion.model(args.max_len, args.win_size)
        # optimizer = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model1.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        # param_dict = {'lr': [0.00001, 0.0001, 0.001, 0.1]}
        # model_gs = GridSearchCV(model, param_dict, cv=10)

    # model1.summary()
    return model1


def get_model2(args):
    # prepare TIER-2 model
    model2 = None
    if args.resume:
        if cnst.USE_PRETRAINED_FOR_TIER2:
            print("[ CAUTION ] : Resuming with pretrained model for TIER2 - "+args.pretrained_t2_model_name)
            model2 = load_model(args.model_path + args.pretrained_t2_model_name)
            if cnst.NUM_GPU > 1:
                model2 = multi_gpu_model(model2, gpus=cnst.NUM_GPU)
            optimizer = optimizers.Adam(lr=0.001)  # , beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
            model2.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        else:
            print("[ CAUTION ] : Resuming with old model")
            model2 = load_model(args.model_path + args.t2_model_name)
            # if cnst.NUM_GPU > 1:
            #    model2 = multi_gpu_model(model2, gpus=cnst.NUM_GPU)
            optimizer = optimizers.Adam(lr=0.001)
            model2.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=["accuracy"])

    else:
        # print("*************************** CREATING new model *****************************")
        if args.byte:
            model2 = echelon.model(args.t2_max_len, args.t2_win_size)
        elif args.featuristic:
            model2 = featuristic.model(len(args.selected_features))
        elif args.fusion:
            model2 = malfusion.model(args.max_len, args.win_size)

        # optimizer = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        optimizer = optimizers.Adam(lr=0.001)  # , beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model2.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # model2.summary()
    return model2


def train_tier1(args):
    # print("************************ TIER 1 TRAINING - STARTED ****************************
    # Samples:", len(args.t1_x_train))
    if args.tier1:
        if args.byte:
            return train(args)
    # print("************************ TIER 1 TRAINING - ENDED   ****************************")


def train_tier2(args):
    # print("************************ TIER 2 TRAINING - STARTED ****************************")
    if args.tier2:
        if args.byte:
            return train_by_section(args)
    # print("************************ TIER 2 TRAINING - ENDED   ****************************")


def evaluate_tier1(args):
    print("Memory Required:", get_model_memory_usage(args.t1_batch_size, args.t1_model_base))
    eval_steps = len(args.t1_x_val) // args.t1_batch_size
    args.t1_val_steps = eval_steps - 1 if len(args.t1_x_val) % args.t1_batch_size == 0 else eval_steps + 1

    history = args.t1_model_base.evaluate_generator(
        utils.data_generator(args.val_partition, args.t1_x_val, args.t1_y_val, args.t1_max_len, args.t1_batch_size, args.t1_shuffle),
        steps=args.t1_val_steps,
        verbose=args.t1_verbose
    )
    # plot_history(history, cnst.TIER1)
    return history


def evaluate_tier2(args):
    print("Memory Required:", get_model_memory_usage(args.t2_batch_size, args.t2_model_base))
    eval_steps = len(args.t2_x_val) // args.t2_batch_size
    args.t2_val_steps = eval_steps - 1 if len(args.t2_x_val) % args.t2_batch_size == 0 else eval_steps + 1

    history = args.t2_model_base.evaluate_generator(
        utils.data_generator_by_section(args.wpartition, args.spartition, args.q_sections, None, args.t2_x_val, args.t2_y_val, args.t2_max_len, args.t2_batch_size, args.t2_shuffle),
        steps=args.t2_val_steps,
        verbose=args.t2_verbose
    )
    # plot_history(history, cnst.TIER2)
    return history


# ######################################################################################################################
# OBJECTIVES:
#     1) Train Tier-1 and select its decision threshold for classification using Training data
#     2) Perform ATI over training data and select influential sections to be used by Tier-2
#     3) Train Tier-2 on selected features
#     4) Save trained models for Tier-1 and Tier-2
# ######################################################################################################################


def init(model_idx, train_partitions, val_partitions, fold_index):
    t_args = DefaultTrainArguments()

    if cnst.EXECUTION_TYPE[model_idx] == cnst.BYTE:                 t_args.byte = True
    elif cnst.EXECUTION_TYPE[model_idx] == cnst.FEATURISTIC:        t_args.featuristic = True
    elif cnst.EXECUTION_TYPE[model_idx] == cnst.FUSION:             t_args.fusion = True

    t_args.t1_model_name = cnst.TIER1_MODELS[model_idx] + "_" + str(fold_index) + ".h5"
    t_args.t2_model_name = cnst.TIER2_MODELS[model_idx] + "_" + str(fold_index) + ".h5"

    # print("######################################   TRAINING TIER-1  ###############################################")
    # partition_tracker_df = pd.read_csv(cnst.DATA_SOURCE_PATH + cnst.ESC + "partition_tracker_"+str(fold_index)+".csv")

    if not cnst.SKIP_TIER1_TRAINING:
        print("************************ TIER 1 TRAINING - STARTED ****************************")
        t_args.t1_model_base = get_model1(t_args)
        best_val_loss = float('inf')
        best_val_acc = 0
        epochs_since_best = 0
        mean_trn_loss = []
        mean_trn_acc = []
        mean_val_loss = []
        mean_val_acc = []
        for epoch in range(cnst.EPOCHS):  # External Partition Purpose
            print("[OUTER PARTITIONS TIER-1 EPOCH]", epoch+1)
            cur_trn_loss = []
            cur_trn_acc = []
            for tp_idx in train_partitions:
                print("Tier-1 Training over partition index:", tp_idx)
                tr_datadf = pd.read_csv(cnst.DATA_SOURCE_PATH + cnst.ESC + "p" + str(tp_idx) + ".csv", header=None)
                t_args.t1_x_train, t_args.t1_x_val, t_args.t1_y_train, t_args.t1_y_val = tr_datadf.iloc[:, 0].values, None, tr_datadf.iloc[:, 1].values, None
                t_args.t1_class_weights = class_weight.compute_class_weight('balanced', np.unique(t_args.t1_y_train), t_args.t1_y_train)  # Class Imbalance Tackling - Setting class weights
                t_args.train_partition = get_partition_data(None, None, tp_idx, "t1")
                t_history = train_tier1(t_args)
                cur_trn_loss.append(t_history.history['accuracy'][0])
                cur_trn_acc.append(t_history.history['loss'][0])
                del t_args.train_partition
                gc.collect()
                cnst.USE_PRETRAINED_FOR_TIER1 = False

            cur_val_loss = []
            cur_val_acc = []
            # Evaluating after each epoch for early stopping over validation loss
            for vp_idx in val_partitions:
                val_datadf = pd.read_csv(cnst.DATA_SOURCE_PATH + cnst.ESC + "p" + str(vp_idx) + ".csv", header=None)
                t_args.t1_x_train, t_args.t1_x_val, t_args.t1_y_train, t_args.t1_y_val = None, val_datadf.iloc[:, 0].values, None, val_datadf.iloc[:, 1].values
                t_args.val_partition = get_partition_data(None, None, vp_idx, "t1")
                v_history = evaluate_tier1(t_args)
                cur_val_loss.append(v_history[0])
                cur_val_acc.append(v_history[1])
                del t_args.val_partition
                gc.collect()

            mean_trn_loss.append(np.mean(cur_trn_loss))
            mean_trn_acc.append(np.mean(cur_trn_acc))
            mean_val_loss.append(np.mean(cur_val_loss))
            mean_val_acc.append(np.mean(cur_val_acc))

            print("Current Epoch Loss:", mean_val_loss[epoch], "Current Epoch Acc:", mean_val_acc[epoch])
            if mean_val_loss[epoch] < best_val_loss:
                best_val_loss = mean_val_loss[epoch]
                # save_model(model, filename)
                epochs_since_best = 0
                print("Updating best loss:", best_val_loss)
            else:
                epochs_since_best += 1
                print('{} epochs passed since best val loss of '.format(epochs_since_best), best_val_loss)
                if 0 < cnst.EARLY_STOPPING_PATIENCE <= epochs_since_best:
                    print('Triggering early stopping as no improvement found since last {} epochs!'.format(epochs_since_best), "Best Loss:", best_val_loss, "\n\n")
                    break
        del t_args.t1_model_base
        gc.collect()
        plot_partition_epoch_history(mean_trn_acc, mean_val_acc, mean_trn_loss, mean_val_loss, "Tier1")
        print("************************ TIER 1 TRAINING - ENDED ****************************")
    else:
        cnst.USE_PRETRAINED_FOR_TIER1 = False  # Use model trained through Echelon
        print("SKIPPED: Tier-1 Training process")

    if cnst.ONLY_TIER1_TRAINING:
        return

    # TIER-1 PREDICTION OVER TRAINING DATA [Select THD1]
    min_boosting_bound = None
    max_thd1 = None
    b1val_partition_count = 0

    if not cnst.SKIP_TIER1_VALIDATION:
        print("*** Prediction over Validation data in TIER-1 to select THD1 and Boosting Bound")
        pd.DataFrame().to_csv(cnst.PROJECT_BASE_PATH + cnst.ESC + "data" + cnst.ESC + "b1_val_" + str(fold_index) + "_pkl.csv", header=None, index=None)
        for vp_idx in val_partitions:
            val_datadf = pd.read_csv(cnst.DATA_SOURCE_PATH + cnst.ESC + "p"+str(vp_idx)+".csv", header=None)
            predict_t1_val_data = pObj(cnst.TIER1, cnst.TIER1_TARGET_FPR, val_datadf.iloc[:, 0].values, val_datadf.iloc[:, 1].values)
            predict_t1_val_data.partition = get_partition_data(None, None, vp_idx, "t1")
            predict_t1_val_data = predict.predict_tier1(model_idx, predict_t1_val_data, fold_index)
            predict_t1_val_data = predict.select_thd_get_metrics_bfn_mfp(cnst.TIER1, predict_t1_val_data)

            min_boosting_bound = predict_t1_val_data.boosting_upper_bound if min_boosting_bound is None or predict_t1_val_data.boosting_upper_bound < min_boosting_bound else min_boosting_bound
            max_thd1 = predict_t1_val_data.thd if max_thd1 is None or predict_t1_val_data.thd > max_thd1 else max_thd1

            del predict_t1_val_data.partition  # Release Memory
            gc.collect()

            val_b1datadf = pd.concat([pd.DataFrame(predict_t1_val_data.xB1), pd.DataFrame(predict_t1_val_data.yB1)], axis=1)
            val_b1datadf.to_csv(cnst.PROJECT_BASE_PATH + cnst.ESC + "data" + cnst.ESC + "b1_val_"+str(fold_index)+"_pkl.csv", header=None, index=None, mode='a')

        val_b1datadf = pd.read_csv(cnst.PROJECT_BASE_PATH + cnst.ESC + "data" + cnst.ESC + "b1_val_"+str(fold_index)+"_pkl.csv", header=None)
        b1val_partition_count = partition_pkl_files_by_count("b1_val", fold_index, val_b1datadf.iloc[:, 0], val_b1datadf.iloc[:, 1])
        pd.DataFrame([{"b1_train": None, "b1_val": b1val_partition_count, "b1_test": None}]).to_csv(os.path.join(cnst.DATA_SOURCE_PATH, "b1_partition_tracker_" + str(fold_index) + ".csv"), index=False)
        pd.DataFrame([{"thd1": max_thd1, "thd2": None, "boosting_bound": min_boosting_bound}]).to_csv(os.path.join(cnst.PROJECT_BASE_PATH + cnst.ESC + "out" + cnst.ESC + "result" + cnst.ESC, "training_outcomes_" + str(fold_index) + ".csv"), index=False)
    else:
        print("SKIPPED: Prediction over Validation data in TIER-1 to select THD1 and Boosting Bound")

    tier1_val_outcomes = pd.read_csv(os.path.join(cnst.PROJECT_BASE_PATH + cnst.ESC + "out" + cnst.ESC + "result" + cnst.ESC, "training_outcomes_" + str(fold_index) + ".csv"))
    max_val_thd1 = tier1_val_outcomes["thd1"][0]
    min_val_boosting_bound = tier1_val_outcomes["boosting_bound"][0]

    if not cnst.SKIP_TIER1_TRAINING_PRED:
        print("\n*** Prediction over Training data in TIER-1 to generate B1 data for TIER-2 Training")
        pd.DataFrame().to_csv(cnst.PROJECT_BASE_PATH + cnst.ESC + "data" + cnst.ESC + "b1_train_" + str(fold_index) + "_pkl.csv", header=None, index=None)
        for tp_idx in train_partitions:
            tr_datadf = pd.read_csv(cnst.DATA_SOURCE_PATH + cnst.ESC + "p" + str(tp_idx) + ".csv", header=None)
            predict_t1_train_data = pObj(cnst.TIER1, cnst.TIER1_TARGET_FPR, tr_datadf.iloc[:, 0].values, tr_datadf.iloc[:, 1].values)
            predict_t1_train_data.thd = max_val_thd1
            predict_t1_train_data.boosting_upper_bound = min_val_boosting_bound
            predict_t1_train_data.partition = get_partition_data(None, None, tp_idx, "t1")
            predict_t1_train_data = predict.predict_tier1(model_idx, predict_t1_train_data, fold_index)
            predict_t1_train_data = predict.select_thd_get_metrics_bfn_mfp(cnst.TIER1, predict_t1_train_data)

            del predict_t1_train_data.partition  # Release Memory
            gc.collect()

            train_b1data_partition_df = pd.concat([pd.DataFrame(predict_t1_train_data.xB1), pd.DataFrame(predict_t1_train_data.yB1)], axis=1)
            train_b1data_partition_df.to_csv(cnst.PROJECT_BASE_PATH + cnst.ESC + "data" + cnst.ESC + "b1_train_" + str(fold_index) + "_pkl.csv", header=None, index=None, mode='a')

        train_b1data_all_df = pd.read_csv(cnst.PROJECT_BASE_PATH + cnst.ESC + "data" + cnst.ESC + "b1_train_" + str(fold_index) + "_pkl.csv", header=None)
        b1_partition_tracker = pd.read_csv(os.path.join(cnst.DATA_SOURCE_PATH, "b1_partition_tracker_" + str(fold_index) + ".csv"))
        b1_partition_tracker["b1_train"][0] = partition_pkl_files_by_count("b1_train", fold_index, train_b1data_all_df.iloc[:, 0], train_b1data_all_df.iloc[:, 1])
        b1_partition_tracker.to_csv(os.path.join(cnst.DATA_SOURCE_PATH, "b1_partition_tracker_" + str(fold_index) + ".csv"), index=False)
    else:
        print("SKIPPED: Prediction over Training data in TIER-1 to generate B1 data for TIER-2 Training")

    print("Loading stored B1 Data from Training set to train Tier-2 model")
    train_b1data_all_df = pd.read_csv(cnst.PROJECT_BASE_PATH + cnst.ESC + "data" + cnst.ESC + "b1_train_"+str(fold_index)+"_pkl.csv", header=None)
    b1_all_file_cnt = len(train_b1data_all_df.iloc[:, 1])
    b1b_all_truth_cnt = len(np.where(train_b1data_all_df.iloc[:, 1] == cnst.BENIGN)[0])
    b1m_all_truth_cnt = len(np.where(train_b1data_all_df.iloc[:, 1] == cnst.MALWARE)[0])

    b1_partition_tracker = pd.read_csv(os.path.join(cnst.DATA_SOURCE_PATH, "b1_partition_tracker_" + str(fold_index) + ".csv"))
    b1tr_partition_count = b1_partition_tracker["b1_train"][0].astype(int)
    b1val_partition_count = b1_partition_tracker["b1_val"][0].astype(int)

    # ATI PROCESS - SELECTING QUALIFIED_SECTIONS - ### Pass B1 data
    if not cnst.SKIP_ATI_PROCESSING:
        print("\nATI - PROCESSING BENIGN AND MALWARE FILES\t\t", "B1 FILES COUNT:", np.shape(train_b1data_all_df.iloc[:, 1])[0], "[# Partitions: "+str(b1tr_partition_count)+"]")
        print("-----------------------------------------")
        ati.init(t_args, fold_index, b1tr_partition_count, b1_all_file_cnt, b1b_all_truth_cnt, b1m_all_truth_cnt) if t_args.ati else None
    else:
        print("SKIPPED: Performing ATI over B1 data of Training set")

    q_sections_by_q_criteria = {}
    ati_qsections = pd.read_csv(cnst.PROJECT_BASE_PATH + cnst.ESC + "out" + cnst.ESC + "result" + cnst.ESC + "qsections_by_qcriteria_" + str(fold_index) + ".csv", header=None)
    for i, row in ati_qsections.iterrows():
        row.dropna(inplace=True)
        q_sections_by_q_criteria[row[0]] = row[1:]  # [section for section in row[1:] if section is not 'nan']

    # print("Collecting section map over B1 data of Training set")
    # t_args.train_section_map = collect_sections(t_args.t2_x_train, t_args.t2_y_train)
    # print("Train section map:\n", t_args.train_section_map)

    print("************************ TIER 2 TRAINING - STARTED ****************************       # Samples:", len(train_b1data_all_df.iloc[:, 0]))
    if cnst.DO_SUBSAMPLING:
        ben_idx = t_args.t2_y_train.index[t_args.t2_y_train == cnst.BENIGN].tolist()
        mal_idx = t_args.t2_y_train.index[t_args.t2_y_train == cnst.MALWARE].tolist()
        t_args.t2_x_train = pd.concat([t_args.t2_x_train.loc[random.sample(ben_idx, len(mal_idx))], t_args.t2_x_train.loc[mal_idx]], ignore_index=True)
        t_args.t2_y_train = pd.concat([t_args.t2_y_train.loc[random.sample(ben_idx, len(mal_idx))], t_args.t2_y_train.loc[mal_idx]], ignore_index=True)
        print("Sub-sampling complete for Tier-2 training data")

    # Need to decide the TRAIN:VAL ratio for tier2
    t_args.t2_x_val, t_args.t2_y_val = None, None
    t2_fpr = cnst.TIER2_TARGET_FPR
    print("Updated Tier-2 target FPR:", t2_fpr)
    gc.collect()

    thd2 = 0
    maxdiff = 0
    q_sections_selected = None
    q_criterion_selected = None
    best_t2_model = None
    predict_args = DefaultPredictArguments()

    qstats = QStats(cnst.PERCENTILES, q_sections_by_q_criteria.keys(), q_sections_by_q_criteria.values())
    for q_criterion in q_sections_by_q_criteria:
        # TIER-2 TRAINING & PREDICTION OVER B1 DATA for current set of q_sections
        if not cnst.SKIP_TIER2_TRAINING:
            t_args.t2_model_base = get_model2(t_args)
            print("\n", list(q_sections_by_q_criteria.keys()), "\nChecking Q_Criterion:", q_criterion, q_sections_by_q_criteria[q_criterion].values)
            # print("************************ Q_Criterion ****************************", q_criterion)
            t_args.q_sections = q_sections_by_q_criteria[q_criterion]

            print("Tier-2 Training over Train B1 [# Partitions: "+str(b1tr_partition_count)+"]")
            best_val_loss = float('inf')
            best_val_acc = 0
            epochs_since_best = 0
            mean_trn_loss = []
            mean_trn_acc = []
            mean_val_loss = []
            mean_val_acc = []
            for epoch in range(cnst.EPOCHS):
                print("[OUTER PARTITIONS TIER-2 EPOCH]", epoch + 1)
                cur_trn_loss = []
                cur_trn_acc = []
                for pcount in range(0, b1tr_partition_count):
                    print("Tier-2 Training over Train B1 partition:", pcount)
                    b1traindatadf = pd.read_csv(cnst.DATA_SOURCE_PATH + cnst.ESC + "b1_train_" + str(fold_index) + "_p" + str(pcount) + ".csv", header=None)
                    t_args.t2_x_train, t_args.t2_y_train = b1traindatadf.iloc[:, 0], b1traindatadf.iloc[:, 1]
                    t_args.whole_b1_train_partition = get_partition_data("b1_train", fold_index, pcount, "t1")
                    t_args.section_b1_train_partition = get_partition_data("b1_train", fold_index, pcount, "t2")
                    t_args.t2_class_weights = class_weight.compute_class_weight('balanced', np.unique(b1traindatadf.iloc[:, 1]), b1traindatadf.iloc[:, 1])  # Class Imbalance Tackling - Setting class weights

                    t2_history = train_tier2(t_args)

                    cur_trn_loss.append(t2_history.history['accuracy'][0])
                    cur_trn_acc.append(t2_history.history['loss'][0])
                    del t_args.whole_b1_train_partition  # Release Memory
                    del t_args.section_b1_train_partition  # Release Memory
                    gc.collect()
                    cnst.USE_PRETRAINED_FOR_TIER2 = False

                cur_val_loss = []
                cur_val_acc = []
                # Evaluating after each epoch for early stopping over validation loss
                for pcount in range(0, b1val_partition_count):
                    b1valdatadf = pd.read_csv(cnst.DATA_SOURCE_PATH + cnst.ESC + "b1_val_" + str(fold_index) + "_p" + str(pcount) + ".csv", header=None)
                    t_args.t2_x_val, t_args.t2_y_val = b1valdatadf.iloc[:,0].values, b1valdatadf.iloc[:,1].values
                    t_args.wpartition = get_partition_data("b1_val", fold_index, pcount, "t1")
                    t_args.spartition = get_partition_data("b1_val", fold_index, pcount, "t2")
                    t_args.q_sections = q_sections_by_q_criteria[q_criterion]
                    v_history = evaluate_tier2(t_args)
                    cur_val_loss.append(v_history[0])
                    cur_val_acc.append(v_history[1])
                    del t_args.wpartition  # Release Memory
                    del t_args.spartition  # Release Memory
                    gc.collect()

                mean_trn_loss.append(np.mean(cur_trn_loss))
                mean_trn_acc.append(np.mean(cur_trn_acc))
                mean_val_loss.append(np.mean(cur_val_loss))
                mean_val_acc.append(np.mean(cur_val_acc))

                print("Current Tier-2 Epoch Loss:", mean_val_loss[epoch], "Current Tier-2 Epoch Acc:", mean_val_acc[epoch])
                if mean_val_loss[epoch] < best_val_loss:
                    best_val_loss = mean_val_loss[epoch]
                    # save_model(model, filename)
                    epochs_since_best = 0
                    print("Updating best loss:", best_val_loss)
                else:
                    epochs_since_best += 1
                    print('{} epochs passed since best val loss of '.format(epochs_since_best), best_val_loss)
                    if 0 < cnst.EARLY_STOPPING_PATIENCE <= epochs_since_best:
                        print('Tier-2 Triggering early stopping as no improvement found since last {} epochs!'.format(epochs_since_best), "Best Loss:", best_val_loss, "\n\n")
                        break
            del t_args.t2_model_base
            gc.collect()
            plot_partition_epoch_history(mean_trn_acc, mean_val_acc, mean_trn_loss, mean_val_loss, "Tier2"+str(q_criterion)[:4])
            print("************************ TIER 2 TRAINING - ENDED ****************************")
        else:
            cnst.USE_PRETRAINED_FOR_TIER2 = False  # Use model trained through Echelon
            print("SKIPPED: Tier-2 Training Process")

        # print("Loading stored B1 Data from Validation set for THD2 selection")
        # val_b1datadf_all = pd.read_csv(cnst.PROJECT_BASE_PATH + cnst.ESC + "data" + cnst.ESC + "b1_val_"+str(fold_index)+"_pkl.csv", header=None)  # xxs.csv
        predict_t2_val_data_all = pObj(cnst.TIER2, t2_fpr, None, None)

        print("Tier-2 Validation over Val B1 [# Partitions: "+str(b1val_partition_count)+"]")
        for pcount in range(0, b1val_partition_count):
            print("Tier-2 Validation over Val B1 partition:", pcount)
            b1valdatadf = pd.read_csv(cnst.DATA_SOURCE_PATH + cnst.ESC + "b1_val_" + str(fold_index) + "_p" + str(pcount) + ".csv", header=None)
            predict_t2_val_data_partition = pObj(cnst.TIER2, t2_fpr, b1valdatadf.iloc[:, 0], b1valdatadf.iloc[:, 1])
            predict_t2_val_data_partition.wpartition = get_partition_data("b1_val", fold_index, pcount, "t1")
            predict_t2_val_data_partition.spartition = get_partition_data("b1_val", fold_index, pcount, "t2")
            predict_t2_val_data_partition.thd = None
            predict_t2_val_data_partition.q_sections = q_sections_by_q_criteria[q_criterion]
            # predict_t2_val_data_partition.predict_section_map = t_args.train_section_map

            predict_t2_val_data_partition = predict.predict_tier2(model_idx, predict_t2_val_data_partition, fold_index)

            del predict_t2_val_data_partition.wpartition  # Release Memory
            del predict_t2_val_data_partition.spartition  # Release Memory
            gc.collect()

            predict_t2_val_data_all.xtrue = predict_t2_val_data_partition.xtrue if predict_t2_val_data_all.xtrue is None else np.concatenate([predict_t2_val_data_all.xtrue, predict_t2_val_data_partition.xtrue])
            predict_t2_val_data_all.ytrue = predict_t2_val_data_partition.ytrue if predict_t2_val_data_all.ytrue is None else np.concatenate([predict_t2_val_data_all.ytrue, predict_t2_val_data_partition.ytrue])
            predict_t2_val_data_all.yprob = predict_t2_val_data_partition.yprob if predict_t2_val_data_all.yprob is None else np.concatenate([predict_t2_val_data_all.yprob, predict_t2_val_data_partition.yprob])
            predict_t2_val_data_all.ypred = predict_t2_val_data_partition.ypred if predict_t2_val_data_all.ypred is None else np.concatenate([predict_t2_val_data_all.ypred, predict_t2_val_data_partition.ypred])

            print("All Tier-2 Test data Size updated:", predict_t2_val_data_all.ytrue.shape)

        predict_t2_val_data_all = predict.select_thd_get_metrics_bfn_mfp(cnst.TIER2, predict_t2_val_data_all)

        display_probability_chart(predict_t2_val_data_all.ytrue, predict_t2_val_data_all.yprob, predict_t2_val_data_all.thd, "Training_TIER2_PROB_PLOT_" + str(fold_index + 1) + "{:6.2f}".format(q_criterion))
        # print("FPR: {:6.2f}".format(predict_t2_val_data_all.fpr), "TPR: {:6.2f}".format(predict_t2_val_data_all.tpr), "\tTHD2: {:6.2f}".format(predict_t2_val_data_all.thd))

        curdiff = predict_t2_val_data_all.tpr - predict_t2_val_data_all.fpr
        if curdiff != 0 and curdiff > maxdiff:
            maxdiff = curdiff
            q_criterion_selected = q_criterion
            best_t2_model = load_model(predict_args.model_path + cnst.TIER2_MODELS[model_idx] + "_" + str(fold_index) + ".h5")
            thd2 = predict_t2_val_data_all.thd
            q_sections_selected = q_sections_by_q_criteria[q_criterion]
            print("Best Q-criterion so far . . . ", q_criterion_selected, predict_t2_val_data_all.thd, predict_t2_val_data_all.fpr, predict_t2_val_data_all.tpr)

        qstats.thds.append(predict_t2_val_data_all.thd)
        qstats.fprs.append(predict_t2_val_data_all.fpr)
        qstats.tprs.append(predict_t2_val_data_all.tpr)

    # Save the best model found
    try:
        best_t2_model.save(predict_args.model_path + cnst.TIER2_MODELS[model_idx] + "_" + str(fold_index) + ".h5")
        # save_model(model=best_t2_model, filepath=predict_args.model_path + cnst.TIER2_MODELS[model_idx],
        # save_weights_only=False, overwrite=True)
        # for section in q_sections_selected:
        #    print(section)
    except Exception as e:
        print("Best Model not available to save - ", str(e))

    print("Percentile\t#Sections\tQ-Criterion\tTHD\t\tFPR\t\tTPR\t\t[TPR-FPR]")
    for i,p in enumerate(cnst.PERCENTILES):
        print(str(qstats.percentiles[i])+"\t\t"+str(len(list(qstats.sections)[i]))+"\t\t{:6.6f}\t{:6.2f}\t\t{:6.2f}\t\t{:6.2f}\t\t{:6.2f}".format(list(qstats.qcriteria)[i], qstats.thds[i], qstats.fprs[i], qstats.tprs[i], qstats.tprs[i]-qstats.fprs[i]))

    # Get the sections that had maximum TPR and low FPR over B1 training data as Final Qualified sections
    print("\n\tBest Q_Criterion:", q_criterion_selected, "Related Q_Sections:", q_sections_selected.values)

    print("************************ TIER 2 TRAINING - ENDED   ****************************")
    # return None, None, thd2, q_sections_selected, t_args.train_section_map
    pd.DataFrame([{"thd1": max_val_thd1, "thd2": thd2, "boosting_bound": min_val_boosting_bound}]).to_csv(os.path.join(cnst.PROJECT_BASE_PATH + cnst.ESC + "out" + cnst.ESC + "result" + cnst.ESC, "training_outcomes_" + str(fold_index) + ".csv"), index=False)
    pd.DataFrame(q_sections_selected).to_csv(os.path.join(cnst.PROJECT_BASE_PATH + cnst.ESC + "out" + cnst.ESC + "result" + cnst.ESC, "qualified_sections_" + str(fold_index) + ".csv"), header=None, index=False)
    return  # predict_t1_train_data.thd, predict_t1_train_data.boosting_upper_bound, thd2, q_sections_selected, t_args.train_section_map


if __name__ == '__main__':
    init()

'''with open(join(t_args.save_path, 'history\\history'+section+'.pkl'), 'wb') as f:
    print(join(t_args.save_path, 'history\\history'+section+'.pkl'))
    pickle.dump(section_history.history, f)'''
