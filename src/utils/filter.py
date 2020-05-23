import os
import pandas as pd
import shutil

fp_file = '..\\out\\result\\FP.csv'
fn_file = '..\\out\\result\\FN.csv'
fpfn_file = '..\\out\\result\\FPandFN.csv'
benign_file = '..\\out\\result\\benign.csv'
malware_file = '..\\out\\result\\malware.csv'
benign_fn_file = '..\\out\\result\\benign_fn.csv'
malware_fp_file = '..\\out\\result\\malware_fp.csv'


def filter_fp_fn_files(result_file):
    df = pd.read_csv(result_file, header=None)
    if os.path.exists(fp_file):
        os.remove(fp_file)

    if os.path.exists(fn_file):
        os.remove(fn_file)

    if os.path.exists(fpfn_file):
        os.remove(fpfn_file)

    fpc, fnc = 0, 0
    print("Combining FPs and FNs from all confidence levels [70, 80, 90, 95]% ")

    with open(fpfn_file, "w+") as ffpfn:
        with open(fp_file, "w+") as ffp:
            for i, record in df.iterrows():
                if (record[1] == 0) and ((record[2] + record[3] + record[4] + record[5]) > record[1]):
                    ffp.write(record[0] + ",0," + str('{:2.10f}'.format(record[6])) + "\n")
                    ffpfn.write(record[0] + ",0," + str('{:2.10f}'.format(record[6])) + "\n")
                    fpc += 1
        print("\nFP Count : ", fpc)

        with open(fn_file, "w+") as ffn:
            for i, record in df.iterrows():
                if (record[1] == 1) and ((record[2] + record[3] + record[4] + record[5]) < (record[1] * 4)):
                    ffn.write(record[0] + ",1," + str('{:2.10f}'.format(record[6])) + "\n")
                    ffpfn.write(record[0] + ",1," + str('{:2.10f}'.format(record[6])) + "\n")
                    fnc += 1
        print("FN Count : ", fnc, "\n")


def filter_benign_fn_files(result_file):
    df = pd.read_csv(result_file, header=None)
    '''
    if os.path.exists(fp_file):
        os.remove(fp_file)

    if os.path.exists(fn_file):
        os.remove(fn_file)

    if os.path.exists(fpfn_file):
        os.remove(fpfn_file)

    fpc, fnc = 0, 0
    
    with open(fpfn_file, "w+") as ffpfn:
        with open(fp_file, "w+") as ffp:
            for i, record in df.iterrows():
                #if (record[1] == 0) and ((record[2] + record[3] + record[4] + record[5]) > record[1]):
                if (record[1] == 0) and (record[2] > record[1]):
                    ffp.write(record[0] + ",0," + str('{:2.10f}'.format(record[2])) + "\n")
                    ffpfn.write(record[0] + ",0," + str('{:2.10f}'.format(record[2])) + "\n")
                    fpc += 1
        print("\nFP Count : ", fpc)

        with open(fn_file, "w+") as ffn:
            for i, record in df.iterrows():
                #if (record[1] == 1) and ((record[2] + record[3] + record[4] + record[5]) < (record[1] * 4)):
                if (record[1] == 1) and (record[2] < record[1]):
                    ffn.write(record[0] + ",1," + str('{:2.10f}'.format(record[2])) + "\n")
                    ffpfn.write(record[0] + ",1," + str('{:2.10f}'.format(record[2])) + "\n")
                    fnc += 1
        print("FN Count : ", fnc, "\n")
    '''
    if os.path.exists(benign_fn_file):
        os.remove(benign_fn_file)

    with open(benign_fn_file, "w+") as bfn:
        for i, record in df.iterrows():
            if record[1] == 0 and record[2] == 0:
                bfn.write(str(record[0]) + "," + str(record[1]) + ",0," + str('{:2.10f}'.format(record[3])) + "\n")
            if record[1] == 1 and record[2] == 0:
                bfn.write(str(record[0]) + "," + str(record[1]) + ",0," + str('{:2.10f}'.format(record[3])) + "\n")
    return benign_fn_file


def filter_malware_fp_files(result_file):
    df = pd.read_csv(result_file, header=None)

    if os.path.exists(malware_fp_file):
        os.remove(malware_fp_file)

    with open(malware_fp_file, "w+") as mfp:
        for i, record in df.iterrows():
            if record[1] == 1 and record[2] > 0:
                mfp.write(str(record[0]) + "," + str(record[1]) + ",1," + str('{:2.10f}'.format(record[3])) + "\n")
            if record[1] == 0 and record[2] > 0:
                mfp.write(str(record[0]) + "," + str(record[1]) + ",1," + str('{:2.10f}'.format(record[3])) + "\n")
    return malware_fp_file


if __name__ == '__main__':
    filter_benign_fn_files('..\\out\\result\\echelon.result.csv')

