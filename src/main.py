import logging
import time
import os
import glob
from config import settings as cnst
from config.echelon_meta import EchelonMeta
import core.generate_train_predict as gtp
from keras import backend as K
import gc
from numba import cuda
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)
logging.info("Tensorflow Version: %s", tf.__version__)
from shutil import copyfile


# ####################################
# Set USE_GPU=FALSE to RUN IN CPU ONLY
# ####################################
# logging.info('GPU found') if tf.test.gpu_device_name() else logging.info("No GPU found")
if not cnst.USE_GPU:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
else:
    # ***** FOR TENSORFLOW 1.12 *****
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # if cnst.GPU_MEM_LIMIT > 0:
    #    config.gpu_options.per_process_gpu_memory_fraction = cnst.GPU_MEM_LIMIT
    #    logging.info(">>> Retricting GPU Memory Usage:", cnst.GPU_MEM_LIMIT)
    # set_session(tf.Session(config=config))

    # ***** FOR TENSORFLOW 2.1.x *****
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                # tf.config.experimental.set_virtual_device_configuration(
                # gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8092)])
        except RuntimeError as e:
            logging.exception("GPU related error occurred.")


def main():
    """ Entry point for the two-tier framework
    Command line switches:
        None: To run all folds of cross validation
        int : A valid integer for running a specific fold of cross validation
        -h or --help: For help related to src/config/settings
    Returns:
        None
    """
    gc.collect()
    K.clear_session()
    metaObj = EchelonMeta()
    # metaObj.project_details()
    # utils.initiate_tensorboard_logging(cnst.TENSORBOARD_LOG_PATH)              # -- TENSOR BOARD LOGGING
    tst = time.time()
    model = 0  # model index
    try:
        metaObj.run_setup()
        gtp.train_predict(model, cnst.ALL_FILE)
        metaObj.run_setup()
    except Exception as e:
        logging.exception("Exiting . . . Due to below error:")
        # K.clear_session()
        # cuda.select_device(0)
        # cuda.close()
        # logging.debug("Clearing keras session.")
        # logging.debug("Closing cuda")
    return


if __name__ == '__main__':
    try:
        copyfile(cnst.DATASET_BACKUP_FILE, cnst.ALL_FILE)
        #copyfile(cnst.PKL_SOURCE_PATH + cnst.ESC + 'available_sections.csv', cnst.PROJECT_BASE_PATH + cnst.ESC + "data" + cnst.ESC + 'available_sections.csv')
        #copyfile(cnst.PKL_SOURCE_PATH + cnst.ESC + 'section_embeddings.csv', cnst.PROJECT_BASE_PATH + cnst.ESC + "data" + cnst.ESC + 'section_embeddings.csv')
    except:
        logging.exception("Error occurred while copying data pre-processing outcomes. Exiting . . .")
        exit()
    main()
