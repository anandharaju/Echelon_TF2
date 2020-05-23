import time
import os
import glob
from config import constants as cnst
from config.echelon_meta import EchelonMeta
import core.generate_train_predict as gtp
from keras.backend.tensorflow_backend import set_session
from keras.backend.tensorflow_backend import clear_session
from keras.backend.tensorflow_backend import get_session
from keras import backend as K
import gc
from numba import cuda
import tensorflow as tf


# ####################################
# Set USE_GPU=FALSE to RUN IN CPU ONLY
# ####################################
# print('GPU found') if tf.test.gpu_device_name() else print("No GPU found")
if not cnst.USE_GPU:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
else:
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    if cnst.GPU_MEM_LIMIT > 0:
        config.gpu_options.per_process_gpu_memory_fraction = cnst.GPU_MEM_LIMIT
        print(">>> Retricting GPU Memory Usage:", cnst.GPU_MEM_LIMIT)
    set_session(tf.Session(config=config))


def clean_files():
    project_path = cnst.PROJECT_BASE_PATH
    for model_file in glob.glob(project_path + cnst.ESC + "model" + cnst.ESC + "echelon_byte*"):
        os.remove(model_file)
    for img_file in glob.glob(project_path + + cnst.ESC + "out" + cnst.ESC + "imgs" + cnst.ESC + "*.png"):
        os.remove(img_file)


def main():
    print(gc.collect())
    sess = get_session()
    clear_session()
    sess.close()
    K.clear_session()
    metaObj = EchelonMeta()
    # metaObj.project_details()
    # utils.initiate_tensorboard_logging(cnst.TENSORBOARD_LOG_PATH)              # -- TENSOR BOARD LOGGING
    tst = time.time()
    ###################################################################################################################
    # ********************            SET PARAMETERS FOR TRAINING & TESTING             *******************************
    ###################################################################################################################

    # clean_files()

    model = 0  # model index
    try:
        metaObj.run_setup()
        gtp.train_predict(model, cnst.ALL_FILE)
        metaObj.run_setup()
    except Exception as e:
        print(e)
        K.clear_session()
        cuda.select_device(0)
        cuda.close()
    return

    '''cust_data = ['small_pkl_6_1.csv', 'small_pkl_7_1.csv', 'small_pkl_8_1.csv', 'small_pkl_9_1.csv', 'small_pkl_10_1.csv']
    for file in cust_data:
        count = 3
        path = cnst.PROJECT_BASE_PATH + file
        while count > 0:
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>", file)
            gtp.train_predict(MODEL, path)
            count -= 1'''
    ###################################################################################################################


if __name__ == '__main__':
    main()
