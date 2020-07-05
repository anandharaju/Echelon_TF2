import warnings
warnings.filterwarnings("ignore")
import logging
import os
from os.path import join
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model, save_model, model_from_json
from keras.utils import multi_gpu_model
from utils import utils
import model_skeleton.featuristic as featuristic
import model_skeleton.malfusion as malfusion
import model_skeleton.echelon as echelon
from keras import optimizers
from trend import activation_trend_identification as ati
import config.settings as cnst
from .train_args import DefaultTrainArguments
from plots.plots import plot_partition_epoch_history, plot_nn_history
from predict import predict
from predict.predict_args import Predict as pObj, DefaultPredictArguments, QStats
import numpy as np
from sklearn.utils import class_weight
import pandas as pd
from plots.plots import display_probability_chart
from analyzers.collect_exe_files import get_partition_data, partition_pkl_files_by_count, partition_pkl_files_by_size
import gc
from shutil import copyfile


def train(args):
    """ Function for training Tier-1 model with whole byte sequence data
        Args:
            args: An object containing all the required parameters for training
        Returns:
            history: Returns history object from keras training process
    """
    train_steps = len(args.t1_x_train) // args.t1_batch_size
    args.t1_train_steps = train_steps - 1 if len(args.t1_x_train) % args.t1_batch_size == 0 else train_steps + 1

    if args.t1_x_val is not None:
        val_steps = len(args.t1_x_val) // args.t1_batch_size
        args.t1_val_steps = val_steps - 1 if len(args.t1_x_val) % args.t1_batch_size == 0 else val_steps + 1

    args.t1_ear = EarlyStopping(monitor='acc', patience=3)
    args.t1_mcp = ModelCheckpoint(join(args.save_path, args.t1_model_name),
                               monitor="acc", save_best_only=args.save_best, save_weights_only=False)

    data_gen = utils.data_generator(args.train_partition, args.t1_x_train, args.t1_y_train, args.t1_max_len, args.t1_batch_size, args.t1_shuffle)
    history = args.t1_model_base.fit(
        data_gen,
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


def train_by_blocks(args):
    """ Function for training Tier-2 model with top activation blocks data
        Args:
            args: An object containing all the required parameters for training
        Returns:
            history: Returns history object from keras training process
    """
    train_steps = len(args.t2_x_train) // args.t2_batch_size
    args.t2_train_steps = train_steps - 1 if len(args.t2_x_train) % args.t2_batch_size == 0 else train_steps + 1

    if args.t2_x_val is not None:
        val_steps = len(args.t2_x_val) // args.t2_batch_size
        args.t2_val_steps = val_steps - 1 if len(args.t2_x_val) % args.t2_batch_size == 0 else val_steps + 1

    args.t2_ear = EarlyStopping(monitor='acc', patience=3)
    args.t2_mcp = ModelCheckpoint(join(args.save_path, args.t2_model_name),
                                  monitor="acc", save_best_only=args.save_best, save_weights_only=False)
    data_gen = utils.data_generator(args.train_partition, args.t2_x_train, args.t2_y_train, args.t2_max_len, args.t2_batch_size, args.t2_shuffle)
    history = args.t2_model_base.fit(
        data_gen,
        class_weight=args.t2_class_weights,
        steps_per_epoch=args.t2_train_steps,
        epochs=args.t2_epochs,
        verbose=args.t2_verbose,
        callbacks=[args.t2_ear, args.t2_mcp]
        # , validation_data=utils.data_generator_by_section(args.q_sections, args.t2_x_val, args.t2_y_val
        # , args.t2_max_len, args.t2_batch_size, args.t2_shuffle)
        # , validation_steps=args.val_steps
    )
    # plot_history(history, cnst.TIER2)
    return history


def train_by_section(args):
    ''' Obsolete: For block-based implementation'''
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
    data_gen = utils.data_generator_by_section(args.section_b1_train_partition, args.q_sections, args.train_section_map, args.t2_x_train, args.t2_y_train, args.t2_max_len, args.t2_batch_size, args.t2_shuffle)
    history = args.t2_model_base.fit(
        data_gen,
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
    """ Function to prepare model required for Tier-1's training/prediction.
        Args:
            args: An object with required parameters/hyper-parameters for loading, configuring and compiling
        Returns:
            model1: Returns a Tier-1 model
    """
    model1 = None
    optimizer = optimizers.Adam(lr=0.001)  # , beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    if args.resume:
        if cnst.USE_PRETRAINED_FOR_TIER1:
            logging.info("[ CAUTION ] : Resuming with pretrained model for TIER1 - " + args.pretrained_t1_model_name)
            model1 = load_model(args.model_path + args.pretrained_t1_model_name, compile=False)
            model1.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
            if cnst.NUM_GPU > 1:
                multi_gpu_model1 = multi_gpu_model(model1, gpus=cnst.NUM_GPU)
                # multi_gpu_model1.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
                return multi_gpu_model1
        else:
            logging.info("[ CAUTION ] : Resuming with old model")
            model1 = load_model(args.model_path + args.t1_model_name, compile=False)
            model1.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
            if cnst.NUM_GPU > 1:
                multi_gpu_model1 = multi_gpu_model(model1, gpus=cnst.NUM_GPU)
                # multi_gpu_model1.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=["accuracy"])
                return multi_gpu_model1
    else:
        logging.info("[CAUTION]: Proceeding training with custom model skeleton")
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
    '''Obsolete: For block-based implementation'''
    model2 = None
    optimizer = optimizers.Adam(lr=0.001)
    if args.resume:
        if cnst.USE_PRETRAINED_FOR_TIER2:
            logging.info("[ CAUTION ] : Resuming with pretrained model for TIER2 - " + args.pretrained_t2_model_name)
            model2 = load_model(args.model_path + args.pretrained_t2_model_name, compile=False)
            model2.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=["accuracy"])
            if cnst.NUM_GPU > 1:
                model2 = multi_gpu_model(model2, gpus=cnst.NUM_GPU)
        else:
            logging.info("[ CAUTION ] : Resuming with old model")
            model2 = load_model(args.model_path + args.t2_model_name, compile=False)
            model2.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=["accuracy"])
            if cnst.NUM_GPU > 1:
                model2 = multi_gpu_model(model2, gpus=cnst.NUM_GPU)

    else:
        # logging.info("*************************** CREATING new model *****************************")
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


def get_block_model2(args):
    """ Function to prepare model required for Tier-2's training/prediction - For top activation block implementation.
        Model's input shape is set to a reduced value specified in TIER2_NEW_INPUT_SHAPE parameter in settings.
        Args:
            args: An object with required parameters/hyper-parameters for loading, configuring and compiling
        Returns:
            model2: Returns a Tier-2 model
    """
    model2 = None
    optimizer = optimizers.Adam(lr=0.001)
    if args.resume:
        if cnst.USE_PRETRAINED_FOR_TIER2:
            logging.info("[ CAUTION ] : Resuming with pretrained model for TIER2 - " + args.pretrained_t2_model_name)
            model2 = load_model(args.model_path + args.pretrained_t2_model_name, compile=False)
            model2.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=["accuracy"])
            if cnst.NUM_GPU > 1:
                model2 = multi_gpu_model(model2, gpus=cnst.NUM_GPU)
        else:
            logging.info("[ CAUTION ] : Resuming with old model")
            model2 = load_model(args.model_path + args.t1_model_name, compile=False)
            logging.info(str(model2.summary()))
            model2 = change_model(model2, new_input_shape=(None, cnst.TIER2_NEW_INPUT_SHAPE))
            logging.info(str(model2.summary()))
            model2.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=["accuracy"])
            if cnst.NUM_GPU > 1:
                model2 = multi_gpu_model(model2, gpus=cnst.NUM_GPU)

    else:
        # logging.info("*************************** CREATING new model *****************************")
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


def change_model(model, new_input_shape=(None, cnst.TIER2_NEW_INPUT_SHAPE)):
    """ Function to transfer weights of pre-trained Malconv to the block based model with reduced input shape.
        Args:
            model: An object with required parameters/hyper-parameters for loading, configuring and compiling
            new_input_shape: a value <= Tier-1 model's input shape. Typically, ( Num of Conv. Filters * Size of Conv. Stride )
        Returns:
            new_model: new model with reduced input shape and weights updated
    """
    model._layers[0].batch_input_shape = new_input_shape
    new_model = model_from_json(model.to_json())
    for layer in new_model.layers:
        try:
            layer.set_weights(model.get_layer(name=layer.name).get_weights())
            logging.info("Loaded and weights set for layer {}".format(layer.name))
        except Exception as e:
            logging.exception("Could not transfer weights for layer {}".format(layer.name))
    return new_model


def train_tier1(args):
    # logging.info("************************ TIER 1 TRAINING - STARTED ****************************
    # Samples:", len(args.t1_x_train))
    if args.tier1:
        if args.byte:
            return train(args)
    # logging.info("************************ TIER 1 TRAINING - ENDED   ****************************")


def train_tier2(args):
    # logging.info("************************ TIER 2 TRAINING - STARTED ****************************")
    if args.tier2:
        if args.byte:
            return train_by_section(args)
    # print("************************ TIER 2 TRAINING - ENDED   ****************************")


def evaluate_tier1(args):
    """ Function to evaluate the Tier-1 model being trained at the end of each Epoch. (Not after completing each partition !)
        Args:
            args: An object with evaluation parameters and data.
        Returns:
            history: Returns a history object with Tier-1 evaluation loss and accuracy
    """
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
    """ Function to evaluate the Tier-2 model being trained at the end of each Epoch. (Not after completing each partition !)
        Args:
            args: An object with evaluation parameters and data.
        Returns:
            history: Returns a history object with Tier-2 evaluation loss and accuracy
    """
    eval_steps = len(args.t2_x_val) // args.t2_batch_size
    args.t2_val_steps = eval_steps - 1 if len(args.t2_x_val) % args.t2_batch_size == 0 else eval_steps + 1

    history = args.t2_model_base.evaluate_generator(
        utils.data_generator_by_section(args.spartition, args.q_sections, None, args.t2_x_val, args.t2_y_val, args.t2_max_len, args.t2_batch_size, args.t2_shuffle),
        steps=args.t2_val_steps,
        verbose=args.t2_verbose
    )
    # plot_history(history, cnst.TIER2)
    return history


def evaluate_tier2_block(args):
    """ Function to evaluate the Tier-2 block-based model being trained at the end of each Epoch. (Not after completing each partition !)
        Args:
            args: An object with evaluation parameters and data.
        Returns:
            history: Returns a history object with block-model evaluation loss and accuracy
    """
    eval_steps = len(args.t2_x_val) // args.t2_batch_size
    args.t2_val_steps = eval_steps - 1 if len(args.t2_x_val) % args.t2_batch_size == 0 else eval_steps + 1

    history = args.t2_model_base.evaluate_generator(
        utils.data_generator(args.val_partition, args.t2_x_val, args.t2_y_val, args.t2_max_len, args.t2_batch_size, args.t2_shuffle),
        steps=args.t2_val_steps,
        verbose=args.t2_verbose
    )
    # plot_history(history, cnst.TIER2)
    return history


def init(model_idx, train_partitions, val_partitions, fold_index):
    """ Module for Training and Validation
    # ##################################################################################################################
    # OBJECTIVES:
    #     1) Train Tier-1 and select its decision threshold for classification using Training data
    #     2) Perform ATI over training data and select influential (Qualified) sections to be used by Tier-2
    #     3) Train Tier-2 on selected PE sections' top activation blocks
    #     4) Save trained models for Tier-1 and Tier-2
    # ##################################################################################################################

    Args:
        model_idx: Default 0 for byte sequence models. Do not change.
        train_partitions: list of partition indexes to be used for Training
        val_partitions: list of partition indexes to be used for evaluation and validation
        fold_index: current fold of cross-validation

    Returns:
        None (Resultant data are stored in CSV for further use)
    """
    t_args = DefaultTrainArguments()

    if cnst.EXECUTION_TYPE[model_idx] == cnst.BYTE:                 t_args.byte = True
    elif cnst.EXECUTION_TYPE[model_idx] == cnst.FEATURISTIC:        t_args.featuristic = True
    elif cnst.EXECUTION_TYPE[model_idx] == cnst.FUSION:             t_args.fusion = True

    t_args.t1_model_name = cnst.TIER1_MODELS[model_idx] + "_" + str(fold_index) + ".h5"
    t_args.t2_model_name = cnst.TIER2_MODELS[model_idx] + "_" + str(fold_index) + ".h5"

    t_args.t1_best_model_name = cnst.TIER1_MODELS[model_idx] + "_" + str(fold_index) + "_best.h5"
    t_args.t2_best_model_name = cnst.TIER2_MODELS[model_idx] + "_" + str(fold_index) + "_best.h5"

    # logging.info("##################################   TRAINING TIER-1  ###########################################")
    # partition_tracker_df = pd.read_csv(cnst.DATA_SOURCE_PATH + cnst.ESC + "partition_tracker_"+str(fold_index)+".csv")

    if not cnst.SKIP_TIER1_TRAINING:
        logging.info("************************ TIER 1 TRAINING - STARTED ****************************")
        t_args.t1_model_base = get_model1(t_args)
        best_val_loss = float('inf')
        best_val_acc = 0
        epochs_since_best = 0
        mean_trn_loss = []
        mean_trn_acc = []
        mean_val_loss = []
        mean_val_acc = []
        cwy = []
        for tp_idx in train_partitions:
            cwdf = pd.read_csv(cnst.DATA_SOURCE_PATH + cnst.ESC + "p" + str(tp_idx) + ".csv", header=None)
            cwy = np.concatenate([cwy, cwdf.iloc[:, 1].values])
        t_args.t1_class_weights = class_weight.compute_class_weight('balanced', np.unique(cwy), cwy)
        for epoch in range(cnst.EPOCHS):  # External Partition Purpose
            logging.info("[ PARTITION LEVEL TIER-1 EPOCH  : %s ]", epoch+1)
            cur_trn_loss = []
            cur_trn_acc = []
            for tp_idx in train_partitions:
                logging.info("Training on partition: %s", tp_idx)
                tr_datadf = pd.read_csv(cnst.DATA_SOURCE_PATH + cnst.ESC + "p" + str(tp_idx) + ".csv", header=None)
                t_args.t1_x_train, t_args.t1_x_val, t_args.t1_y_train, t_args.t1_y_val = tr_datadf.iloc[:, 0].values, None, tr_datadf.iloc[:, 1].values, None
                # t_args.t1_class_weights = class_weight.compute_class_weight('balanced',
                # np.unique(t_args.t1_y_train), t_args.t1_y_train)  # Class Imbalance Tackling - Setting class weights
                t_args.train_partition = get_partition_data(None, None, tp_idx, "t1")
                t_history = train_tier1(t_args)
                cur_trn_loss.append(t_history.history['loss'][0])
                cur_trn_acc.append(t_history.history['accuracy'][0])
                del t_args.train_partition
                gc.collect()
                cnst.USE_PRETRAINED_FOR_TIER1 = False

            cur_val_loss = []
            cur_val_acc = []
            # Evaluating after each epoch for early stopping over validation loss
            logging.info("Evaluating on validation data . . .")
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

            if mean_val_loss[epoch] < best_val_loss:
                best_val_loss = mean_val_loss[epoch]
                try:
                    copyfile(join(t_args.save_path, t_args.t1_model_name), join(t_args.save_path, t_args.t1_best_model_name))
                except Exception as e:
                    logging.exception("Saving EPOCH level best model failed for Tier1")
                epochs_since_best = 0
                logging.info("Current Epoch Loss: %s\tCurrent Epoch Acc: %s\tUpdating best loss: %s", str(mean_val_loss[epoch]).ljust(25), str(mean_val_acc[epoch]).ljust(25), best_val_loss)
            else:
                logging.info("Current Epoch Loss: %s\tCurrent Epoch Acc: %s", mean_val_loss[epoch], mean_val_acc[epoch])
                epochs_since_best += 1
                logging.info('{} epochs passed since best val loss of {}'.format(epochs_since_best, best_val_loss))
                if cnst.EARLY_STOPPING_PATIENCE_TIER1 <= epochs_since_best:
                    logging.info('Triggering early stopping as no improvement found since last {} epochs!  Best Loss: {}'.format(epochs_since_best, best_val_loss))
                    try:
                        copyfile(join(t_args.save_path, t_args.t1_best_model_name), join(t_args.save_path, t_args.t1_model_name))
                    except Exception as e:
                        logging.exception("Retrieving EPOCH level best model failed for Tier1")
                    break
            if epoch + 1 == cnst.EPOCHS:
                try:
                    copyfile(join(t_args.save_path, t_args.t1_best_model_name), join(t_args.save_path, t_args.t1_model_name))
                except Exception as e:
                    logging.exception("Retrieving EPOCH level best model failed for Tier1.")
        del t_args.t1_model_base
        gc.collect()
        plot_partition_epoch_history(mean_trn_acc, mean_val_acc, mean_trn_loss, mean_val_loss, "Tier1_F" + str(fold_index+1))
        logging.info("************************ TIER 1 TRAINING - ENDED ****************************")
    else:
        cnst.USE_PRETRAINED_FOR_TIER1 = False  # Use model trained through Echelon
        logging.info("SKIPPED: Tier-1 Training process")

    if cnst.ONLY_TIER1_TRAINING:
        return

    # TIER-1 PREDICTION OVER TRAINING DATA [Select THD1]
    min_boosting_bound = None
    max_thd1 = None
    b1val_partition_count = 0

    if not cnst.SKIP_TIER1_VALIDATION:
        logging.info("*** Prediction over Validation data in TIER-1 to select THD1 and Boosting Bound")
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
        b1val_partition_count = partition_pkl_files_by_count("b1_val", fold_index, val_b1datadf.iloc[:, 0], val_b1datadf.iloc[:, 1]) if cnst.PARTITION_BY_COUNT else partition_pkl_files_by_size("b1_val", fold_index, val_b1datadf.iloc[:, 0], val_b1datadf.iloc[:, 1])
        pd.DataFrame([{"b1_train": None, "b1_val": b1val_partition_count, "b1_test": None}]).to_csv(os.path.join(cnst.DATA_SOURCE_PATH, "b1_partition_tracker_" + str(fold_index) + ".csv"), index=False)
        pd.DataFrame([{"thd1": max_thd1, "thd2": None, "boosting_bound": min_boosting_bound}]).to_csv(os.path.join(cnst.PROJECT_BASE_PATH + cnst.ESC + "out" + cnst.ESC + "result" + cnst.ESC, "training_outcomes_" + str(fold_index) + ".csv"), index=False)
    else:
        logging.info("SKIPPED: Prediction over Validation data in TIER-1 to select THD1 and Boosting Bound")

    tier1_val_outcomes = pd.read_csv(os.path.join(cnst.PROJECT_BASE_PATH + cnst.ESC + "out" + cnst.ESC + "result" + cnst.ESC, "training_outcomes_" + str(fold_index) + ".csv"))
    max_val_thd1 = tier1_val_outcomes["thd1"][0]
    min_val_boosting_bound = tier1_val_outcomes["boosting_bound"][0]

    if not cnst.SKIP_TIER1_TRAINING_PRED:
        logging.info("*** Prediction over Training data in TIER-1 to generate B1 data for TIER-2 Training")
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
        b1_partition_tracker["b1_train"][0] = partition_pkl_files_by_count("b1_train", fold_index, train_b1data_all_df.iloc[:, 0], train_b1data_all_df.iloc[:, 1]) if cnst.PARTITION_BY_COUNT else partition_pkl_files_by_size("b1_train", fold_index, train_b1data_all_df.iloc[:, 0], train_b1data_all_df.iloc[:, 1])
        b1_partition_tracker.to_csv(os.path.join(cnst.DATA_SOURCE_PATH, "b1_partition_tracker_" + str(fold_index) + ".csv"), index=False)
    else:
        logging.info("SKIPPED: Prediction over Training data in TIER-1 to generate B1 data for TIER-2 Training")

    logging.info("Loading stored B1 Data from Training set to train Tier-2 model")
    train_b1data_all_df = pd.read_csv(cnst.PROJECT_BASE_PATH + cnst.ESC + "data" + cnst.ESC + "b1_train_"+str(fold_index)+"_pkl.csv", header=None)
    b1_all_file_cnt = len(train_b1data_all_df.iloc[:, 1])
    b1b_all_truth_cnt = len(np.where(train_b1data_all_df.iloc[:, 1] == cnst.BENIGN)[0])
    b1m_all_truth_cnt = len(np.where(train_b1data_all_df.iloc[:, 1] == cnst.MALWARE)[0])

    b1_partition_tracker = pd.read_csv(os.path.join(cnst.DATA_SOURCE_PATH, "b1_partition_tracker_" + str(fold_index) + ".csv"))
    b1tr_partition_count = b1_partition_tracker["b1_train"][0].astype(int)
    b1val_partition_count = b1_partition_tracker["b1_val"][0].astype(int)

    # ATI PROCESS - SELECTING QUALIFIED_SECTIONS - ### Pass B1 data
    if not cnst.SKIP_ATI_PROCESSING:
        logging.info("ATI - PROCESSING BENIGN AND MALWARE FILES\t\t\tB1 FILES COUNT: %s [# Partitions: %s ]", np.shape(train_b1data_all_df.iloc[:, 1])[0], b1tr_partition_count)
        logging.info("-----------------------------------------")
        ati.init(t_args, fold_index, b1tr_partition_count, b1_all_file_cnt, b1b_all_truth_cnt, b1m_all_truth_cnt) if t_args.ati else None
    else:
        logging.info("SKIPPED: Performing ATI over B1 data of Training set")

    q_sections_by_q_criteria = {}
    ati_qsections = pd.read_csv(cnst.PROJECT_BASE_PATH + cnst.ESC + "out" + cnst.ESC + "result" + cnst.ESC + "qsections_by_qcriteria_" + str(fold_index) + ".csv", header=None)
    for i, row in ati_qsections.iterrows():
        row.dropna(inplace=True)
        q_sections_by_q_criteria[row[0]] = row[1:]  # [section for section in row[1:] if section is not 'nan']

    # logging.info("Collecting section map over B1 data of Training set")
    # t_args.train_section_map = collect_sections(t_args.t2_x_train, t_args.t2_y_train)
    # logging.info("Train section map:\n", t_args.train_section_map)

    logging.info("************************ TIER 2 TRAINING - STARTED ****************************       # Samples: %s", len(train_b1data_all_df.iloc[:, 0]))
    '''if cnst.DO_SUBSAMPLING:
        ben_idx = t_args.t2_y_train.index[t_args.t2_y_train == cnst.BENIGN].tolist()
        mal_idx = t_args.t2_y_train.index[t_args.t2_y_train == cnst.MALWARE].tolist()
        t_args.t2_x_train = pd.concat([t_args.t2_x_train.loc[random.sample(ben_idx, len(mal_idx))], t_args.t2_x_train.loc[mal_idx]], ignore_index=True)
        t_args.t2_y_train = pd.concat([t_args.t2_y_train.loc[random.sample(ben_idx, len(mal_idx))], t_args.t2_y_train.loc[mal_idx]], ignore_index=True)
        logging.info("Sub-sampling complete for Tier-2 training data")'''

    # Need to decide the TRAIN:VAL ratio for tier2
    t_args.t2_x_val, t_args.t2_y_val = None, None
    t2_fpr = cnst.TIER2_TARGET_FPR
    logging.info("Updated Tier-2 target FPR: %s", t2_fpr)
    gc.collect()

    thd2 = 0
    maxdiff = 0
    q_sections_selected = None
    q_criterion_selected = None
    qcnt_selected = None
    best_t2_model = None
    best_scaler = None
    predict_args = DefaultPredictArguments()
    scaler = None
    t_args.t2_class_weights = class_weight.compute_class_weight('balanced', np.unique(train_b1data_all_df.iloc[:, 1]), train_b1data_all_df.iloc[:, 1])
    qstats = QStats(cnst.PERCENTILES, q_sections_by_q_criteria.keys(), q_sections_by_q_criteria.values())
    for qcnt, q_criterion in enumerate(q_sections_by_q_criteria):
        # TIER-2 TRAINING & PREDICTION OVER B1 DATA for current set of q_sections
        if not cnst.SKIP_TIER2_TRAINING:
            t_args.t2_model_base = get_model2(t_args)
            logging.info("%s\nChecking Q_Criterion: %s \n %s", list(q_sections_by_q_criteria.keys()), q_criterion, q_sections_by_q_criteria[q_criterion].values)
            t_args.q_sections = list(q_sections_by_q_criteria[q_criterion])

            logging.info("Tier-2 CNN Training over B1_Train set [Total # Partitions: " + str(b1tr_partition_count) + "]")
            best_val_loss = float('inf')
            best_val_acc = 0
            epochs_since_best = 0
            mean_trn_loss = []
            mean_trn_acc = []
            mean_val_loss = []
            mean_val_acc = []
            for epoch in range(cnst.EPOCHS):
                logging.info("[ PARTITIONS LEVEL TIER-2 EPOCH : %s ]", epoch + 1)
                cur_trn_loss = []
                cur_trn_acc = []
                for pcount in range(0, b1tr_partition_count):
                    logging.info("Training on partition: %s", pcount)
                    b1traindatadf = pd.read_csv(cnst.DATA_SOURCE_PATH + cnst.ESC + "b1_train_" + str(fold_index) + "_p" + str(pcount) + ".csv", header=None)
                    t_args.t2_x_train, t_args.t2_y_train = b1traindatadf.iloc[:, 0], b1traindatadf.iloc[:, 1]
                    t_args.whole_b1_train_partition = get_partition_data("b1_train", fold_index, pcount, "t1")
                    t_args.section_b1_train_partition = get_partition_data("b1_train", fold_index, pcount, "t2")

                    t2_history = train_tier2(t_args)

                    cur_trn_loss.append(t2_history.history['loss'][0])
                    cur_trn_acc.append(t2_history.history['accuracy'][0])
                    del t_args.whole_b1_train_partition  # Release Memory
                    del t_args.section_b1_train_partition  # Release Memory
                    gc.collect()
                    cnst.USE_PRETRAINED_FOR_TIER2 = False

                cur_val_loss = []
                cur_val_acc = []
                # Evaluating after each epoch for early stopping over validation loss
                logging.info("Evaluating on B1 validation data. . .")
                for pcount in range(0, b1val_partition_count):
                    b1valdatadf = pd.read_csv(cnst.DATA_SOURCE_PATH + cnst.ESC + "b1_val_" + str(fold_index) + "_p" + str(pcount) + ".csv", header=None)
                    t_args.t2_x_val, t_args.t2_y_val = b1valdatadf.iloc[:,0].values, b1valdatadf.iloc[:,1].values
                    # t_args.wpartition = get_partition_data("b1_val", fold_index, pcount, "t1")
                    t_args.spartition = get_partition_data("b1_val", fold_index, pcount, "t2")
                    t_args.q_sections = q_sections_by_q_criteria[q_criterion]
                    v_history = evaluate_tier2(t_args)
                    cur_val_loss.append(v_history[0])
                    cur_val_acc.append(v_history[1])
                    # del t_args.wpartition  # Release Memory
                    del t_args.spartition  # Release Memory
                    gc.collect()

                mean_trn_loss.append(np.mean(cur_trn_loss))
                mean_trn_acc.append(np.mean(cur_trn_acc))
                mean_val_loss.append(np.mean(cur_val_loss))
                mean_val_acc.append(np.mean(cur_val_acc))

                if mean_val_loss[epoch] < best_val_loss:
                    best_val_loss = mean_val_loss[epoch]
                    try:
                        copyfile(join(t_args.save_path, t_args.t2_model_name), join(t_args.save_path, t_args.t2_best_model_name))
                    except Exception as e:
                        logging.exception("Saving EPOCH level best model failed for Tier2.")
                    epochs_since_best = 0
                    logging.info("Current Tier-2 Epoch Loss: %s\t Epoch Acc: %s\tUpdating best loss: %s", str(mean_val_loss[epoch]).ljust(25), str(mean_val_acc[epoch]).ljust(25), best_val_loss)
                else:
                    logging.info("Current Tier-2 Epoch Loss: %s\t Current Tier-2 Epoch Acc: %s", str(mean_val_loss[epoch]).ljust(25), str(mean_val_acc[epoch]).ljust(25))
                    epochs_since_best += 1
                    logging.info('{} epochs passed since best val loss of {}'.format(epochs_since_best, best_val_loss))
                    if cnst.EARLY_STOPPING_PATIENCE_TIER2 <= epochs_since_best:
                        logging.info('Tier-2 Triggering early stopping as no improvement found since last {} epochs!    Best Loss:'.format(epochs_since_best, best_val_loss))
                        try:
                        	copyfile(join(t_args.save_path, t_args.t2_best_model_name), join(t_args.save_path, t_args.t2_model_name))
                        except Exception as e:
                            logging.exception("Retrieving EPOCH level best model failed for Tier2.")
                        break

                if epoch + 1 == cnst.EPOCHS:
        	        try:
    	                copyfile(join(t_args.save_path, t_args.t2_best_model_name), join(t_args.save_path, t_args.t2_model_name))
                    except Exception as e:
                        logging.exception("Retrieving EPOCH level best model failed for Tier2.")
            del t_args.t2_model_base
            gc.collect()
            plot_partition_epoch_history(mean_trn_acc, mean_val_acc, mean_trn_loss, mean_val_loss, "Tier2_F" + str(fold_index+1) + "_Q" + str(qcnt))  # str(q_criterion)[:4])
            logging.info("************************ TIER 2 TRAINING - ENDED ****************************")
        else:
            cnst.USE_PRETRAINED_FOR_TIER2 = False  # Use model trained through Echelon
            logging.info("SKIPPED: Tier-2 Training Process")

        # print("Loading stored B1 Data from Validation set for THD2 selection")
        # val_b1datadf_all = pd.read_csv(cnst.PROJECT_BASE_PATH + cnst.ESC + "data" + cnst.ESC + "b1_val_"+str(fold_index)+"_pkl.csv", header=None)  # xxs.csv
        predict_t2_val_data_all = pObj(cnst.TIER2, t2_fpr, None, None)

        logging.info("Tier-2 Validation over B1_Val set [# Partitions: " + str(b1val_partition_count) + "]")
        for pcount in range(0, b1val_partition_count):
            logging.info("Validating partition: %s", pcount)
            b1valdatadf = pd.read_csv(
                cnst.DATA_SOURCE_PATH + cnst.ESC + "b1_val_" + str(fold_index) + "_p" + str(pcount) + ".csv",
                header=None)
            predict_t2_val_data_partition = pObj(cnst.TIER2, t2_fpr, b1valdatadf.iloc[:, 0], b1valdatadf.iloc[:, 1])
            # predict_t2_val_data_partition.wpartition = get_partition_data("b1_val", fold_index, pcount, "t1")
            predict_t2_val_data_partition.spartition = get_partition_data("b1_val", fold_index, pcount, "t2")
            predict_t2_val_data_partition.thd = None
            predict_t2_val_data_partition.q_sections = q_sections_by_q_criteria[q_criterion]
            # predict_t2_val_data_partition.predict_section_map = t_args.train_section_map

            predict_t2_val_data_partition = predict.predict_tier2(model_idx, predict_t2_val_data_partition, fold_index)

            # del predict_t2_val_data_partition.wpartition  # Release Memory
            del predict_t2_val_data_partition.spartition  # Release Memory
            gc.collect()

            predict_t2_val_data_all.xtrue = predict_t2_val_data_partition.xtrue if predict_t2_val_data_all.xtrue is None else np.concatenate(
                [predict_t2_val_data_all.xtrue, predict_t2_val_data_partition.xtrue])
            predict_t2_val_data_all.ytrue = predict_t2_val_data_partition.ytrue if predict_t2_val_data_all.ytrue is None else np.concatenate(
                [predict_t2_val_data_all.ytrue, predict_t2_val_data_partition.ytrue])
            predict_t2_val_data_all.yprob = predict_t2_val_data_partition.yprob if predict_t2_val_data_all.yprob is None else np.concatenate(
                [predict_t2_val_data_all.yprob, predict_t2_val_data_partition.yprob])
            predict_t2_val_data_all.ypred = predict_t2_val_data_partition.ypred if predict_t2_val_data_all.ypred is None else np.concatenate(
                [predict_t2_val_data_all.ypred, predict_t2_val_data_partition.ypred])

            logging.debug("All Tier-2 Test data Size updated: %s", predict_t2_val_data_all.ytrue.shape)

        predict_t2_val_data_all = predict.select_thd_get_metrics_bfn_mfp(cnst.TIER2, predict_t2_val_data_all)

        display_probability_chart(predict_t2_val_data_all.ytrue, predict_t2_val_data_all.yprob, predict_t2_val_data_all.thd, "Training_TIER2_PROB_PLOT_F" + str(fold_index) + "_" + str(qcnt))
        curdiff = predict_t2_val_data_all.tpr - predict_t2_val_data_all.fpr
        if curdiff != 0 and curdiff > maxdiff:
            # Save the PE sections that had maximum TPR and low FPR over B1 training data as Final Qualified sections
            maxdiff = curdiff
            q_criterion_selected = q_criterion
            qcnt_selected = qcnt
            best_t2_model = load_model(cnst.MODEL_PATH + cnst.TIER2_MODELS[model_idx] + "_" + str(fold_index) + ".h5")
            # best_t2_nn_model = load_model(cnst.MODEL_PATH + "nn_t2_" + str(fold_index) + "_q"+str(qcnt)+".h5")
            # best_scalers = scalers
            thd2 = predict_t2_val_data_all.thd
            q_sections_selected = q_sections_by_q_criteria[q_criterion]
            logging.info("Best Q-criterion so far . . . %s\t\t%s\t%s\t%s", q_criterion_selected, predict_t2_val_data_all.thd, predict_t2_val_data_all.fpr, predict_t2_val_data_all.tpr)

        qstats.thds.append(predict_t2_val_data_all.thd)
        qstats.fprs.append(predict_t2_val_data_all.fpr)
        qstats.tprs.append(predict_t2_val_data_all.tpr)

    try:
        best_t2_model.save(cnst.MODEL_PATH + cnst.TIER2_MODELS[model_idx] + "_" + str(fold_index) + ".h5")
        # best_t2_nn_model.save(cnst.MODEL_PATH + "nn_t2_" + str(fold_index) + ".h5")

        logging.info("Best Q_Criterion: %s\nRelated Q_Sections: %s", q_criterion_selected, q_sections_selected.values)
        logging.info("************************ TIER 2 TRAINING - ENDED   ****************************")
        pd.DataFrame([{"thd1": max_val_thd1, "thd2": thd2, "boosting_bound": min_val_boosting_bound}]).to_csv(
            os.path.join(cnst.PROJECT_BASE_PATH + cnst.ESC + "out" + cnst.ESC + "result" + cnst.ESC, "training_outcomes_" + str(fold_index) + ".csv"), index=False)
        pd.DataFrame(q_sections_selected).to_csv(
            os.path.join(cnst.PROJECT_BASE_PATH + cnst.ESC + "out" + cnst.ESC + "result" + cnst.ESC, "qualified_sections_" + str(fold_index) + ".csv"), header=None, index=False)

        logging.info("Percentile\t#Sections\tQ-Criterion\tTHD\t\tFPR\t\tTPR\t\t[TPR-FPR]")
        for i, p in enumerate(cnst.PERCENTILES):
            try:
                logging.info(str(qstats.percentiles[i])+"\t\t"+str(len(list(qstats.sections)[i]))+"\t\t{:6.6f}\t{:6.2f}\t\t{:6.3f}\t\t{:6.3f}\t\t{:6.3f}".format(list(qstats.qcriteria)[i], qstats.thds[i], qstats.fprs[i], qstats.tprs[i], qstats.tprs[i]-qstats.fprs[i]))
            except:
                logging.error("Percentile list specified in 'settings' is not processed by ATI yet. Set SKIP_ATI_PROCESSING = False in settings and re-run.")
    except Exception as e:
        logging.critical("Best Model not available to save. Try re-training tier-2 with different parameters / Q-criterion percentiles.\n\tExiting Training Process . . .")
    return  # best_scalers


if __name__ == '__main__':
    init()

'''with open(join(t_args.save_path, 'history\\history'+section+'.pkl'), 'wb') as f:
    logging.info(join(t_args.save_path, 'history\\history'+section+'.pkl'))
    pickle.dump(section_history.history, f)'''
