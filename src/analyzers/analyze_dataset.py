import os
import pickle
import numpy as np
import config.constants as cnst
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


class ByteData:
    xdf = None
    ydf = None


def load_dataset(path=None):
    # LOAD DATA
    af = ByteData()
    if path is None:
        adf = pd.read_csv(cnst.ALL_FILE, header=None)
    else:
        adf = pd.read_csv(path, header=None)
    af.xdf = adf[0]
    af.ydf = adf[1]

    '''bf = ByteData()
    bdf = pd.read_csv(cnst.BENIGN_FILE, header=None)
    bf.xdf = bdf[0]
    bf.ydf = bdf[1]

    mf = ByteData()
    mdf = pd.read_csv(cnst.MALWARE_FILE, header=None)
    mf.xdf = mdf[0]
    mf.ydf = mdf[1]'''

    return af, None, None # bf, mf


'''
File: Analyze_Data set
Functionality: Generate stats about data set
1) Malware, Benign Count
2) File sizes
3) Max byte length of sections - [.code]
4) Baseline accuracy
'''


def analyze_dataset(check_file_size=False):
    with open(cnst.all_file, 'w+') as a, open(cnst.benign_file, 'w+') as b, open(cnst.malware_file, 'w+') as m:
        for path in cnst.paths:
            print(" >>>", path)
            unprocessed = 0
            max_size = 0
            max_file = None
            file_count = 0
            limit = cnst.MAX_FILE_SIZE_LIMIT # 2Mb
            large_file_count = 0
            os.chdir(path)
            for file in glob.glob("*.pkl"):
                file = path + cnst.ESC + file
                with open(file, 'rb') as f:
                    file_count += 1
                    if check_file_size:
                        fjson = pickle.load(f)
                        size = np.shape(fjson["image_256w"])[0] * np.shape(fjson["image_256w"])[1]
                        if size > max_size:
                            max_size = size
                            max_file = file
                        if size > limit:
                            large_file_count += 1
                    try:
                        if "clean" in file:
                            a.write(file + ",0\n")
                            b.write(file + ",0\n")
                        else:
                            a.write(file + ",1\n")
                            m.write(file + ",1\n")
                    except Exception as e:
                        unprocessed += 1
                        print("FAIL [ Unprocessed: ", str(unprocessed), "] [ Err: " + str(e) + " ] [ FILE ID - ", file,
                              "] ")

            print("File Count:", file_count, "Large Files:", large_file_count)
            print("Max. file size:", str(max_size / (1024 * 1024)) + " Mb", "File Name:", max_file)
