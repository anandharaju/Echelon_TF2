import os
import sys

RESUME = True
print("Detected Platform:",sys.platform)
LINUX_ENV = False if 'win' in sys.platform else True
ESC = "/" if LINUX_ENV else "\\"
# 42 :Answer to the Ultimate Question of Life, the Universe, and Everything
# ~ The Hitchhiker's Guide to the Galaxy
RANDOM_SEED = 42
VERBOSE_0 = 0
VERBOSE_1 = 1
TIER1_PRETRAINED_MODEL = "ember_malconv.h5"
TIER2_PRETRAINED_MODEL = "ember_malconv.h5"
BENIGN = 0
MALWARE = 1
T1_TRAIN_BATCH_SIZE = 64
T2_TRAIN_BATCH_SIZE = 32
PREDICT_BATCH_SIZE = 128

T1_VERBOSE = VERBOSE_1
T2_VERBOSE = VERBOSE_1
PREDICT_VERBOSE = VERBOSE_1
ATI_PREDICT_VERBOSE = VERBOSE_1

#####################################################################################
USE_GPU = True
NUM_GPU = 1
GPU_MEM_LIMIT = 0

REGENERATE_DATA_AND_PARTITIONS = False
DO_SUBSAMPLING = False

PROJECT_ROOT = os.getcwdb().decode("utf-8").split("/")[-2] if LINUX_ENV else os.getcwdb().decode("utf-8").split("\\")[-2]
USE_PRETRAINED_FOR_TIER1 = True  # True:Malconv False:Echelon
USE_PRETRAINED_FOR_TIER2 = True
PERFORM_B2_BOOSTING = True
VAL_SET_SIZE = 0.2
TST_SET_SIZE = 0.2

EPOCHS = 10
EARLY_STOPPING_PATIENCE = 2
# TIER-1
TIER1 = "TIER1"
TIER1_EPOCHS = 1
TIER1_TARGET_FPR = 0.1

SKIP_ENTIRE_TRAINING = False
ONLY_TIER1_TRAINING = False

SKIP_TIER1_TRAINING = False
SKIP_TIER1_VALIDATION = False           # Generates Val B1
SKIP_TIER1_TRAINING_PRED = False        # Generates Train B1
SKIP_ATI_PROCESSING = False

SKIP_TIER2_TRAINING = False


# TIER-2
TIER2 = "TIER2"
TIER2_EPOCHS = 1
TIER2_TARGET_FPR = 0

OVERALL_TARGET_FPR = 0.1
#####################################################################################

# CROSS VALIDATION
CV_FOLDS = 5
PARTITIONS = 100
INITIAL_FOLD = 0
RANDOMIZE = False  # Set Random seed for True

#  DATA SOURCE
MAX_PARTITION_SIZE = int(1 * 2 ** 30)
MAX_FILES_PER_PARTITION = 512
MAX_SECT_BYTE_MAP_SIZE = 2000
MAX_FILE_SIZE_LIMIT = 2**20  # 204800
MAX_FILE_COUNT_LIMIT = None
CONV_WINDOW_SIZE = 500
CONV_STRIDE_SIZE = 500
MAX_FILE_CONVOLUTED_SIZE = int(MAX_FILE_SIZE_LIMIT / CONV_STRIDE_SIZE)
USE_POOLING_LAYER = True
PROJECT_BASE_PATH = '/home/aduraira/projects/def-wangk/aduraira/' + PROJECT_ROOT if LINUX_ENV else 'D:\\03_GitWorks\\'+PROJECT_ROOT
DATA_SOURCE_PATH = '/home/aduraira/projects/def-wangk/aduraira/partitions/' + PROJECT_ROOT if LINUX_ENV else 'D:\\08_Dataset\\Internal\\mar2020\\partitions\\xs_partition\\'
PKL_SOURCE_PATH = '/home/aduraira/projects/def-wangk/aduraira/pickles/' if LINUX_ENV else 'D:\\08_Dataset\\Internal\\mar2020\\pickles\\'
ALL_FILE = PROJECT_BASE_PATH  + ESC + 'data' + ESC + 'ds1_pkl.csv'  # 'balanced_pkl.csv'  # small_pkl_1_1.csv'
BENIGN_FILE = PROJECT_BASE_PATH + ESC + 'data' + ESC + 'medium_benign_pkl.csv'
MALWARE_FILE = PROJECT_BASE_PATH + ESC + 'data' + ESC + 'medium_malware_pkl.csv'
TRAINING_FILE = PROJECT_BASE_PATH + ESC + 'data' + ESC + 'training.csv'
TESTING_FILE = PROJECT_BASE_PATH + ESC + 'data' + ESC + 'testing.csv'
GENERATE_BENIGN_MALWARE_FILES = False
CHECK_FILE_SIZE = False

TIER1_MODELS = ['echelon_byte', 'echelon_featuristic', 'echelon_fusion']
TIER2_MODELS = ['echelon_byte_2', 'echelon_featuristic_2', 'echelon_fusion_2']
EXECUTION_TYPE = ['BYTE', 'FEATURISTIC', 'FUSION']
PLOT_TITLE = ['Byte Sequence', 'Superficial Features', 'MalFusion']
BYTE = 'BYTE'
FEATURISTIC = 'FEATURISTIC'
FUSION = 'FUSION'

TENSORBOARD_LOG_PATH = PROJECT_BASE_PATH + ESC + "log" + ESC + "tensorboard" + ESC
PLOT_PATH = PROJECT_BASE_PATH + ESC + "out" + ESC + "imgs" + ESC

SAVE_PATH = PROJECT_BASE_PATH  + ESC + 'model' + ESC   # help='Directory to save model and log'
MODEL_PATH = PROJECT_BASE_PATH + ESC + 'model' + ESC  # help="model to resume"


# #####################################################################################################################
# FEATURE MAP VISUALIZATION
# #####################################################################################################################
LAYER_NUM_TO_STUNT = 4 # 6 for echelon
PERCENTILES = [92, 94]
RUN_FOLDS = [0]

COMBINED_FEATURE_MAP_STATS_FILE = PROJECT_BASE_PATH + ESC + 'out' + ESC + 'result' + ESC + 'combined_stats.csv'
COMMON_COMBINED_FEATURE_MAP_STATS_FILE = PROJECT_BASE_PATH + ESC + 'out' + ESC + 'result' + ESC + 'combined_stats_common.csv'
SECTION_SUPPORT = PROJECT_BASE_PATH + ESC + "out" + ESC + "result" + ESC + "section_support_by_samples.csv"

TAIL = "OVERLAY"
PADDING = "PADDING"
LEAK = "SECTIONLESS"