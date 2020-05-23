import config.constants as cnst


class Predict:
    def __init__(self, label, fpr, x, y):
        self.tier = label
        self.target_fpr = fpr
        self.xtrue = x
        self.ytrue = y
    tier = None
    target_fpr = None
    xtrue = None
    ytrue = None
    ypred = None
    yprob = None
    thd = None
    tpr = None
    fpr = None
    auc = None
    rauc = None

    xB1 = None
    yB1 = None
    ypredB1 = None
    yprobB1 = None
    xM1 = None
    yM1 = None
    ypredM1 = None
    yprobM1 = None

    boosting_upper_bound = None
    boosted_xB2 = None
    boosted_yB2 = None
    boosted_ypredB2 = None
    boosted_yprobB2 = None

    q_sections = None
    predict_section_map = None

    partition = None


class DefaultPredictArguments:
    batch_size = cnst.PREDICT_BATCH_SIZE
    verbose = cnst.PREDICT_VERBOSE
    shuffle = True
    tier1 = True
    tier2 = True
    fmap = True
    byte = True
    featuristic = False
    fusion = False
    tier1_epochs = 1
    tier2_epochs = 1
    total_features = 53
    max_len = cnst.MAX_FILE_SIZE_LIMIT  # help="model input length"
    win_size = 500
    save_path = cnst.SAVE_PATH
    model_path = cnst.MODEL_PATH


class QStats:
    def __init__(self, p, q, s):
        self.percentiles = p
        self.qcriteria = q
        self.sections = s
        self.thds = []
        self.tprs = []
        self.fprs = []
    percentiles = None
    qcriteria = None
    sections = None
    thds = []
    tprs = []
    fprs = []
