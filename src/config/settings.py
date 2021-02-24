import os
import sys
from config import helper
import logging

RESUME = False
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

BSIZE = 16

T1_TRAIN_BATCH_SIZE = 64 if LINUX_ENV else 32  # 216
T2_TRAIN_BATCH_SIZE = 64 if LINUX_ENV else 128  # 64
PREDICT_BATCH_SIZE = 0 if LINUX_ENV else BSIZE  # 248

T1_VERBOSE = VERBOSE_0
T2_VERBOSE = VERBOSE_0
PREDICT_VERBOSE = VERBOSE_0
ATI_PREDICT_VERBOSE = VERBOSE_0

#####################################################################################
USE_GPU = False
NUM_GPU = 1
GPU_MEM_LIMIT = 0

SKIP_CROSS_VALIDATION = False
# DO_SUBSAMPLING = False

REGENERATE_DATA = REGENERATE_PARTITIONS = False
#  List of folders containing PE samples & is_benign?
RAW_SAMPLE_DIR = 'D:\\08_Dataset\\Internal\\mar2020\\dummy\\less_than_10kb\\'
PKL_SOURCE_PATH = '/home/aduraira/projects/def-wangk/aduraira/pickles/' if LINUX_ENV else RAW_SAMPLE_DIR # 'D:\\08_Dataset\\Internal\\mar2020\\dummy\\'
DATASET_BACKUP_FILE = 'D:\\08_Dataset\\Internal\\mar2020\\dummy\\dummy.csv'
SAMPLE_SIZE = int((1024 * 1024)//500*500)+500

BROUND = 200  # 0: Initial round
TOTAL_SAMPLES = 200000
BASE_WEIGHT = 1 / (TOTAL_SAMPLES * 0.64)
SAMPLE_WEIGHTS = [BASE_WEIGHT*i for i in range(1, 1000)]

PROJECT_ROOT = os.getcwdb().decode("utf-8").split("/")[-2] if LINUX_ENV else os.getcwdb().decode("utf-8").split("\\")[-2]
USE_PRETRAINED_FOR_TIER1 = False  # True:Malconv False:Echelon
USE_PRETRAINED_FOR_TIER2 = True
PERFORM_B2_BOOSTING = True
VAL_SET_SIZE = 0.2
TST_SET_SIZE = 0.3

EPOCHS = 1
EARLY_STOPPING_PATIENCE_TIER1 = 5  # 5
EARLY_STOPPING_PATIENCE_TIER2 = 3  # 5

# TIER-1
TIER1 = "TIER1"
TIER1_EPOCHS = 1
TIER1_TARGET_FPR = OVERALL_TARGET_FPR = 0.1

SKIP_ENTIRE_TRAINING = True
ONLY_TIER1_TRAINING = False

TIER2_NEW_INPUT_SHAPE = SAMPLE_SIZE

SKIP_TIER1_TRAINING = False
SKIP_TIER1_VALIDATION = False           # Generates Val B1
SKIP_TIER1_TRAINING_PRED = False        # Generates Train B1
SKIP_ATI_PROCESSING = False

SKIP_TIER2_TRAINING = False


# TIER-2
TIER2 = "TIER2"
TIER2_EPOCHS = 1
TIER2_TARGET_FPR = 0
SKIP_TIER1_PREDICTION = False
#####################################################################################

# CROSS VALIDATION
CV_FOLDS = 5
GRANULARITY_FOR_UNIFORMNESS = 10
INITIAL_FOLD = 0

#  DATA SOURCE
PARTITION_BY_COUNT = True
MAX_PARTITION_SIZE = int(2 * 2 ** 30)
MAX_FILES_PER_PARTITION = BSIZE
MAX_SECT_BYTE_MAP_SIZE = 2000
MAX_FILE_SIZE_LIMIT = SAMPLE_SIZE  # 204800

MAX_FILE_COUNT_LIMIT = None
CONV_WINDOW_SIZE = CONV_STRIDE_SIZE = 1024 * 50
NUM_FILTERS = 128
MAX_FILE_CONVOLUTED_SIZE = int(MAX_FILE_SIZE_LIMIT / CONV_STRIDE_SIZE)
USE_POOLING_LAYER = True
# DATA_SOURCE_PATH       ds2_original
# PKL_SOURCE_PATH        pickles_sfu
# ALL_FILE               ds2_original_pkl.csv
# DATASET_BACKUP_FILE    Raw_To_Pickle_DS2_from_DS1.csv

PROJECT_BASE_PATH = '/home/aduraira/projects/def-wangk/aduraira/' + PROJECT_ROOT if LINUX_ENV else 'D:\\03_GitWorks\\'+PROJECT_ROOT
DATA_SOURCE_PATH = '/home/aduraira/projects/def-wangk/aduraira/partitions/ds1/' if LINUX_ENV else 'D:\\08_Dataset\\Internal\\mar2020\\partitions\\xs_partition\\'
ALL_FILE = PROJECT_BASE_PATH + ESC + 'data' + ESC + 'ds1_pkl.csv'  # 'balanced_pkl.csv'  # small_pkl_1_1.csv'
BENIGN_FILE = PROJECT_BASE_PATH + ESC + 'data' + ESC + 'medium_benign_pkl.csv'
MALWARE_FILE = PROJECT_BASE_PATH + ESC + 'data' + ESC + 'medium_malware_pkl.csv'
TRAINING_FILE = PROJECT_BASE_PATH + ESC + 'data' + ESC + 'training.csv'
TESTING_FILE = PROJECT_BASE_PATH + ESC + 'data' + ESC + 'testing.csv'
GENERATE_BENIGN_MALWARE_FILES = False
CHECK_FILE_SIZE = False

FASTTEXT_PATH = '/home/aduraira/projects/def-wangk/aduraira/fasttext/cc.en.1.bin' if LINUX_ENV else 'D:\\03_GitWorks\\fastText\\cc.en.1.bin'

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
PERCENTILES = [60]  # [0, 25, 50, 75, 80, 85, 88, 90, 91, 92, 94, 95, 96]
RUN_FOLDS = list(range(INITIAL_FOLD, CV_FOLDS))
ARG1 = None
LOG_FILE_NAME = '../log/'
if 'main.py' in sys.argv[0]:
    try:
        ARG1 = sys.argv[1]
    except Exception as e:
        print("No fold number passed through CLI. Running all folds")
        LOG_FILE_NAME += 'all.log'

    if ARG1:
        if ARG1 == "-h" or ARG1 == "--help":
            helper.GlobalHelper().show_help()
            exit()
        elif 1 <= int(ARG1) <= CV_FOLDS:
            RUN_FOLDS = [int(ARG1)-1]
            LOG_FILE_NAME += 'fold_'+ARG1+'.log'
        else:
            print("\nInvalid fold number. Valid fold numbers are from 1 to " + str(CV_FOLDS) + ", as per current setting.")
            print("Do not pass any argument if running all folds (or) specify a single number to run respective fold.")
            exit()

# Transfer Learning
TIER2_PRETRAINED_MODEL = "echelon_byte_2_"+ARG1+".h5"


COMBINED_FEATURE_MAP_STATS_FILE = PROJECT_BASE_PATH + ESC + 'out' + ESC + 'result' + ESC + 'combined_stats.csv'
COMMON_COMBINED_FEATURE_MAP_STATS_FILE = PROJECT_BASE_PATH + ESC + 'out' + ESC + 'result' + ESC + 'combined_stats_common.csv'
SECTION_SUPPORT = PROJECT_BASE_PATH + ESC + "out" + ESC + "result" + ESC + "section_support_by_samples.csv"

TAIL = "OVERLAY"
PADDING = "PADDING"
LEAK = "SECTIONLESS"

# Initialize Logging
# logger = logging.getLogger(__name__)
ENABLE_LOGGER = False                    # Set to False if using Jupyter Notebook
OVERWRITE_LOG_ON_EACH_RUN = True
if ENABLE_LOGGER:
    logging.basicConfig(level=logging.INFO, filename=LOG_FILE_NAME, filemode='w' if OVERWRITE_LOG_ON_EACH_RUN else 'a', format='%(asctime)s :: %(levelname)s :: %(message)s')
else:
    logging.basicConfig(level=logging.INFO, format='%(message)s')  # %(asctime)s :: %(levelname)s :: %(message)s')
