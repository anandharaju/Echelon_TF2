import config.constants as cnst


class DefaultTrainArguments:
    q_sections = None
    byte = True
    fusion = False
    featuristic = False

    train_section_map = None

    ati = True
    tier1 = True
    tier2 = True
    t1_ear = None  # Early stopping
    t1_ear = None  # Early stopping
    t2_mcp = None  # Model Checkpoint
    t2_mcp = None  # Model Checkpoint
    t1_x_val = None
    t1_y_val = None
    t2_x_val = None
    t2_y_val = None
    t1_x_test = None
    t1_y_test = None
    t2_x_test = None
    t2_y_test = None
    t1_x_train = None
    t1_y_train = None
    t2_x_train = None
    t2_y_train = None
    t1_shuffle = True
    t2_shuffle = True
    t1_val_steps = None
    t2_val_steps = None
    t1_model_name = None
    t2_model_name = None
    t1_model_base = None  # Hold model base skeleton for tier-1
    t2_model_base = None  # Hold model base skeleton for tier-2
    t1_train_steps = None
    t2_train_steps = None
    t1_class_weight = None
    t2_class_weight = None
    t1_verbose = cnst.T1_VERBOSE
    t2_verbose = cnst.T2_VERBOSE
    t1_epochs = cnst.TIER1_EPOCHS
    t2_epochs = cnst.TIER2_EPOCHS
    t1_batch_size = cnst.T1_TRAIN_BATCH_SIZE
    t2_batch_size = cnst.T2_TRAIN_BATCH_SIZE
    t1_win_size = cnst.CONV_WINDOW_SIZE
    t2_win_size = cnst.CONV_WINDOW_SIZE
    t1_max_len = cnst.MAX_FILE_SIZE_LIMIT  # help="model input length"
    t2_max_len = cnst.MAX_FILE_SIZE_LIMIT  # help="model input length"
    pretrained_t1_model_name = cnst.TIER1_PRETRAINED_MODEL
    pretrained_t2_model_name = cnst.TIER2_PRETRAINED_MODEL

    resume = cnst.RESUME  # Set to True to load already trained model
    save_best = False  # help="Save model with best validation accuracy"

    save_path = cnst.SAVE_PATH
    model_path = cnst.MODEL_PATH
    csv = cnst.PROJECT_BASE_PATH + cnst.ESC + "data" + cnst.ESC + "training.csv"

    train_partition = None
