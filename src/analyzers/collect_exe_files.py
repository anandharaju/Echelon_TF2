import os, shutil
import pefile
import pandas as pd
from shutil import copyfile
import config.constants as cnst
import pickle


def partition_pkl_files_by_size(type, fold, files, labels):
    if type is not None:
        partition_label = type + "_" + str(fold) + "_"
    else:
        partition_label = ""

    # csv = pd.read_csv(csv_path, header=None)
    print("Total number of files to partition:", len(files))
    partition_count = 0
    file_count = 0
    t1_partition_data = {}
    t2_partition_data = {}
    cur_t1_partition_size = 0
    cur_t2_partition_size = 0
    start = 0
    end = 0

    for i, file in enumerate(files):  # iloc[:, 0]:
        t1_pkl_src_path = os.path.join(cnst.PKL_SOURCE_PATH + cnst.ESC + "t1" + cnst.ESC, file)
        t2_pkl_src_path = os.path.join(cnst.PKL_SOURCE_PATH + cnst.ESC + "t2" + cnst.ESC, file)

        t1_src_file_size = os.stat(t1_pkl_src_path).st_size
        t2_src_file_size = os.stat(t2_pkl_src_path).st_size

        if cur_t1_partition_size > cnst.MAX_PARTITION_SIZE or cur_t2_partition_size > cnst.MAX_PARTITION_SIZE:
            t1_partition_path = os.path.join(cnst.DATA_SOURCE_PATH, partition_label + "t1_p" + str(partition_count))
            t2_partition_path = os.path.join(cnst.DATA_SOURCE_PATH, partition_label + "t2_p" + str(partition_count))

            with open(t1_partition_path+".pkl", "wb") as pt1handle:
                pickle.dump(t1_partition_data, pt1handle)
            with open(t2_partition_path+".pkl", "wb") as pt2handle:
                pickle.dump(t2_partition_data, pt2handle)

            end = i
            pd.DataFrame(list(zip(files[start:end], labels[start:end]))).to_csv(cnst.DATA_SOURCE_PATH + cnst.ESC + partition_label + "p" + str(partition_count) + ".csv", header=None, index=False)
            print("Created Partition", partition_label+"p"+str(partition_count), "with", file_count, "files and tracker csv with " + str(len(files[start:end])) + " files.")
            file_count = 0
            partition_count += 1
            t1_partition_data = {}
            t2_partition_data = {}
            cur_t1_partition_size = 0
            cur_t2_partition_size = 0
            start = end

        with open(t1_pkl_src_path, 'rb') as f1, open(t2_pkl_src_path, 'rb') as f2:
            cur_t1pkl = pickle.load(f1)
            cur_t2pkl = pickle.load(f2)
            t1_partition_data[file[:-4]] = cur_t1pkl
            t2_partition_data[file[:-4]] = cur_t2pkl
            cur_t1_partition_size += t1_src_file_size
            cur_t2_partition_size += t2_src_file_size
            file_count += 1

    if cur_t1_partition_size > 0 or cur_t2_partition_size > 0:
        t1_partition_path = os.path.join(cnst.DATA_SOURCE_PATH, partition_label+"t1_p"+str(partition_count))
        t2_partition_path = os.path.join(cnst.DATA_SOURCE_PATH, partition_label+"t2_p"+str(partition_count))
        with open(t1_partition_path + ".pkl", "wb") as pt1handle:
            pickle.dump(t1_partition_data, pt1handle)
        with open(t2_partition_path + ".pkl", "wb") as pt2handle:
            pickle.dump(t2_partition_data, pt2handle)
        pd.DataFrame(list(zip(files[start:], labels[start:]))).to_csv(cnst.DATA_SOURCE_PATH + cnst.ESC + partition_label + "p" + str(partition_count) + ".csv", header=None, index=False)
        print("Created Partition", partition_label+"p"+str(partition_count), "with", file_count, "files and tracker csv with " + str(len(files[start:])) + " files.")
        partition_count += 1
    return partition_count


def partition_pkl_files_by_count(type, fold, files, labels):
    if type is not None:
        partition_label = type + "_" + str(fold) + "_"
    else:
        partition_label = ""

    # csv = pd.read_csv(csv_path, header=None)
    print("Total number of files to partition:", len(files))
    partition_count = 0
    t1_partition_data = {}
    t2_partition_data = {}
    cur_partition_file_count = 0
    start = 0
    end = 0

    for i, file in enumerate(files):  # iloc[:, 0]:
        t1_pkl_src_path = os.path.join(cnst.PKL_SOURCE_PATH + cnst.ESC + "t1" + cnst.ESC, file)
        t2_pkl_src_path = os.path.join(cnst.PKL_SOURCE_PATH + cnst.ESC + "t2" + cnst.ESC, file)

        if cur_partition_file_count == cnst.MAX_FILES_PER_PARTITION:
            t1_partition_path = os.path.join(cnst.DATA_SOURCE_PATH, partition_label + "t1_p" + str(partition_count))
            t2_partition_path = os.path.join(cnst.DATA_SOURCE_PATH, partition_label + "t2_p" + str(partition_count))

            with open(t1_partition_path+".pkl", "wb") as pt1handle:
                pickle.dump(t1_partition_data, pt1handle)
            with open(t2_partition_path+".pkl", "wb") as pt2handle:
                pickle.dump(t2_partition_data, pt2handle)

            end = i
            pd.DataFrame(list(zip(files[start:end], labels[start:end]))).to_csv(cnst.DATA_SOURCE_PATH + cnst.ESC + partition_label + "p" + str(partition_count) + ".csv", header=None, index=False)
            print("Created Partition", partition_label+"p"+str(partition_count), "with", cur_partition_file_count, "files and tracker csv with " + str(len(files[start:end])) + " files.")
            partition_count += 1
            t1_partition_data = {}
            t2_partition_data = {}
            cur_partition_file_count = 0
            start = end

        with open(t1_pkl_src_path, 'rb') as f1, open(t2_pkl_src_path, 'rb') as f2:
            cur_t1pkl = pickle.load(f1)
            cur_t2pkl = pickle.load(f2)
            t1_partition_data[file[:-4]] = cur_t1pkl
            t2_partition_data[file[:-4]] = cur_t2pkl
            cur_partition_file_count += 1

    if cur_partition_file_count > 0:
        t1_partition_path = os.path.join(cnst.DATA_SOURCE_PATH, partition_label+"t1_p"+str(partition_count))
        t2_partition_path = os.path.join(cnst.DATA_SOURCE_PATH, partition_label+"t2_p"+str(partition_count))
        with open(t1_partition_path + ".pkl", "wb") as pt1handle:
            pickle.dump(t1_partition_data, pt1handle)
        with open(t2_partition_path + ".pkl", "wb") as pt2handle:
            pickle.dump(t2_partition_data, pt2handle)
        pd.DataFrame(list(zip(files[start:], labels[start:]))).to_csv(cnst.DATA_SOURCE_PATH + cnst.ESC + partition_label + "p" + str(partition_count) + ".csv", header=None, index=False)
        print("Created Partition", partition_label+"p"+str(partition_count), "with", cur_partition_file_count, "files and tracker csv with " + str(len(files[start:])) + " files.")
        partition_count += 1
    return partition_count


def get_partition_data(type, fold, partition_count, tier):
    if type is not None:
        partition_label = type + "_" + str(fold) + "_" + tier + "_p" + str(partition_count)
    else:
        partition_label = tier + "_p" + str(partition_count)

    print("Loading partition:", partition_label)
    partition_path = os.path.join(cnst.DATA_SOURCE_PATH, partition_label + ".pkl")

    if not os.path.isfile(partition_path):
        print("Partition file ", partition_path, ' does not exist.')
    else:
        with open(partition_path, "rb") as pkl_handle:
            return pickle.load(pkl_handle)


def group_files_by_pkl_list():
    csv = pd.read_csv("D:\\03_GitWorks\\Project\\data\\xs_pkl.csv", header=None)
    dst_folder = "D:\\03_GitWorks\\Project\\data\\xs_pkl\\"
    for file in csv.iloc[:, 0]:
        src_path = os.path.join("D:\\08_Dataset\\Internal\\mar2020\\pickle_files\\", file)
        dst_Path = os.path.join(dst_folder, file)
        copyfile(src_path, dst_Path)


def sep_files_by_pkl_list():
    csv = pd.read_csv(cnst.ALL_FILE, header=None)
    t1_dst_folder = cnst.PKL_SOURCE_PATH + cnst.ESC + "t1" + cnst.ESC
    t2_dst_folder = cnst.PKL_SOURCE_PATH + cnst.ESC + "t2" + cnst.ESC
    if not os.path.exists(t1_dst_folder):
        os.makedirs(t1_dst_folder)
    if not os.path.exists(t2_dst_folder):
        os.makedirs(t2_dst_folder)
    for file in csv.iloc[:, 0]:
        src_path = os.path.join('/home/aduraira/projects/def-wangk/aduraira/pickle_files/', file)
        with open(src_path, 'rb') as f:
            t1_pkl = {}
            t2_pkl = {}
            cur_pkl = pickle.load(f)
            t1_pkl = {"whole_bytes": cur_pkl["whole_bytes"], "benign": cur_pkl["benign"]}
            del cur_pkl["whole_bytes"]
            t2_pkl = cur_pkl
            with open(t1_dst_folder + file, "wb") as t1handle:
                pickle.dump(t1_pkl, t1handle)
            with open(t2_dst_folder + file, "wb") as t2handle:
                pickle.dump(t2_pkl, t2handle)


def copy_files(src_path, dst_path, ext, max_size):
    total_count = 0
    total_size = 0
    unprocessed = 0
    dst_dir = dst_path
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    for src_dir, dirs, files in os.walk(src_path):
        for file_ in files:
            if total_count >= max_files: break
            try:
                src_file = os.path.join(src_dir, file_)
                dst_file = os.path.join(dst_dir, file_)

                #For Benign
                #if not fnmatch.fnmatch(src_file, ext):
                #    continue

                #For Malware
                #if fnmatch.fnmatch(src_file, ext):
                    #continue

                src_file_size = os.stat(src_file).st_size
                if src_file_size > max_size:
                    continue

                try:
                    # check if file can be processed by pefile module
                    pe = pefile.PE(src_file)
                    if pe._PE__warnings is not None and len(pe._PE__warnings) > 0 \
                            and pe._PE__warnings[0] == 'Invalid section 0. Contents are null-bytes.':
                        raise Exception(pe._PE__warnings[0]+" "+pe._PE__warnings[1])
                    for item in pe.sections:
                        # Check if all sections are parse-able without error
                        _ = item.Name.rstrip(b'\x00').decode("utf-8").strip()
                        _ = item.get_data()
                except Exception as e:
                    unprocessed += 1
                    print("parse failed . . . [ Unprocessed Count: ", str(unprocessed), "] [ Error: " + str(e)
                          + " ] [ FILE ID - ", src_file, "] ")
                    continue

                shutil.copy(src_file, dst_dir)
                print(total_count, "      ", src_file, dst_file)
            except Exception as e1:
                print("Copy failed ", src_file)
            total_count += 1
            total_size += src_file_size
    return total_count, total_size


def collect_sha():
    df_info = pd.DataFrame()
    df = pd.read_csv('D:\\03_GitWorks\\ATI_changes\\data\\ds1_pkl_unique.csv', header=None)

    for i, file in enumerate(df.iloc[:, 0]):
        if (i+1) % 1000 == 0:
            print(i+1, "files processed.")
        try:
            # if df.iloc[:, 1][i] == 0:
            #    continue
            src_path = os.path.join('D:\\08_Dataset\\Internal\\mar2020\\pickle_files', 'ee8c29cc3fb611fb6149cd769871cb9b33e024e9.pkl')
            with open(src_path, 'rb') as f:
                cur_pkl = pickle.load(f)
                df_info = pd.concat([df_info, pd.DataFrame([[cur_pkl["md5"], cur_pkl["sha1"], cur_pkl["sha256"]]], columns=['md5', 'sha1', 'sha256'])])
        except Exception as e:
            print(file)
            print(str(e))
        break
    df_info.to_csv('D:\\08_Dataset\\Internal\\mar2020\\Malware_Info.csv', index=False)
    return


if __name__ == '__main__':
    src_path = "D:\\08_Dataset\\VirusTotal\\repo\\all"
    dst_path = "D:\\08_Dataset\\aug24_malware\\"

    ext = '*.exe'
    max_size = 512000  # bytes 500KB
    max_files = 110000
    # total_count, total_size = copy_files(src_path, dst_path, ext, max_size)

    #sep_files_by_pkl_list()
    collect_sha()
    '''

    # group_files_by_pkl_list()
    for fold in range(0, 5):
        partition_pkl_files("master_train_"+str(fold), "D:\\03_GitWorks\\Project\\data\\master_train_"+str(fold)+"_pkl.csv")
        partition_pkl_files("master_val_"+str(fold), "D:\\03_GitWorks\\Project\\data\\master_val_"+str(fold)+"_pkl.csv")
        partition_pkl_files("master_test_"+str(fold), "D:\\03_GitWorks\\Project\\data\\master_test_"+str(fold)+"_pkl.csv")
    print("\nCompleted.")

    '''
