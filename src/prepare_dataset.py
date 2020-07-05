import os
import pefile
import hashlib
import pickle
import time
import pandas as pd
from config import settings as cnst
from collections import OrderedDict
from utils import embedder


all_sections = OrderedDict({".header": 0})


def raw_pe_to_pkl(path, is_benign, unprocessed, processed):
    list_idx = []
    for src_dir, dirs, files in os.walk(path):
        for file_ in files:
            file_data = {}
            try:
                src_file = os.path.join(src_dir, file_)
                src_file_size = os.stat(src_file).st_size
                if src_file_size > cnst.MAX_FILE_SIZE_LIMIT:
                    print("Skipping as file size exceeds ", cnst.MAX_FILE_SIZE_LIMIT, "[ Unprocessed / Skipped Count: "+str(unprocessed)+"]")
                    unprocessed += 1
                    continue
                else:
                    file_data["size_byte"] = src_file_size

                pe = pefile.PE(src_file)
                pe_name = "pe_" + str(processed) + ".pkl"
                with open(src_file, 'rb') as fhandle:
                    file_byte_data = fhandle.read()
                    fid = [pe_name
                           , 0 if is_benign else 1
                           , file_
                           , hashlib.md5(file_byte_data).hexdigest()
                           , hashlib.sha1(file_byte_data).hexdigest()
                           , hashlib.sha256(file_byte_data).hexdigest()]
                    file_data["whole_bytes"] = list(file_byte_data)

                wb_size = len(file_data["whole_bytes"])
                file_data["whole_bytes_size"] = wb_size
                file_data["benign"] = is_benign
                # file_data["num_of_sections"] = pe.FILE_HEADER.NumberOfSections
                file_data["section_info"] = {}
                for section in pe.sections:
                    section_name = section.Name.strip(b'\x00').decode("utf-8").strip()
                    section_data = {}
                    section_data["section_data"] = list(section.get_data())
                    section_data["section_size_byte"] = section.SizeOfRawData
                    section_data["section_bounds"] = {}
                    section_data["section_bounds"]["start_offset"] = section.PointerToRawData
                    section_data["section_bounds"]["end_offset"] = section.PointerToRawData + section.SizeOfRawData - 1
                    file_data["section_info"][section_name] = section_data

                file_data["section_info"][".header"] = {
                    "section_data": list(pe.header),
                    "section_size_byte": len(pe.header),
                    "section_bounds": {
                        "start_offset": 0,
                        "end_offset": len(pe.header)
                    }}

                t1_pkl = {"whole_bytes": file_data["whole_bytes"], "benign": file_data["benign"]}
                sections_end = 0
                keys = file_data["section_info"].keys()
                for key in keys:
                    if file_data["section_info"][key]['section_bounds']["end_offset"] > sections_end:
                        sections_end = file_data["section_info"][key]['section_bounds']["end_offset"]

                if sections_end <= 0:
                    print("[OVERLAY DATA NOT ADDED] Invalid section end found - ", sections_end)
                elif sections_end < wb_size - 1:
                    data = file_data["whole_bytes"][sections_end + 1:wb_size]
                    section_data = dict()
                    section_data["section_data"] = data
                    section_data["section_size_byte"] = len(data)
                    # section_bounds
                    section_data["section_bounds"] = {}
                    section_data["section_bounds"]["start_offset"] = sections_end + 1
                    section_data["section_bounds"]["end_offset"] = wb_size - 1
                    file_data["section_info"][cnst.TAIL] = section_data

                del file_data["whole_bytes"]
                t2_pkl = file_data
                with open(t1_dst_folder + pe_name, "wb") as t1handle:
                    pickle.dump(t1_pkl, t1handle)
                with open(t2_dst_folder + pe_name, "wb") as t2handle:
                    pickle.dump(t2_pkl, t2handle)
                list_idx.append(fid)
                processed += 1
                for section in file_data["section_info"].keys():
                    if section in all_sections:
                        all_sections[section] += 1
                    else:
                        all_sections[section] = 1
                    all_sections['.header'] += 1
                print("Total Count:", processed, "Unprocessed/Skipped:", unprocessed)

                # Test saved data
                # with open(pkl_file, "rb") as pkl:
                #    print(pickle.load(pkl)["num_of_sections"])
            except Exception as e:
                unprocessed += 1
                print("parse failed . . . [ Unprocessed #:", str(unprocessed), "] [ ERROR: " + str(e) + " ] [ FILE: ", src_file, "] ")

            if processed % 1000 == 0:
                print("# files processed:", processed)
    pd.DataFrame(list_idx).to_csv(cnst.DATASET_BACKUP_FILE, index=False, header=None, mode='a')
    return unprocessed, processed


if __name__ == '__main__':
    total_processed = 0
    total_unprocessed = 0
    start_time = time.time()
    t1_dst_folder = cnst.PKL_SOURCE_PATH + "t1" + cnst.ESC
    t2_dst_folder = cnst.PKL_SOURCE_PATH + "t2" + cnst.ESC

    if not os.path.exists(t1_dst_folder):
        os.makedirs(t1_dst_folder)
    if not os.path.exists(t2_dst_folder):
        os.makedirs(t2_dst_folder)

    if os.path.exists(cnst.DATASET_BACKUP_FILE):
        os.remove(cnst.DATASET_BACKUP_FILE)

    for dir in cnst.RAW_SAMPLE_DIRS.keys():
        total_unprocessed, total_processed = raw_pe_to_pkl(dir, cnst.RAW_SAMPLE_DIRS[dir], total_unprocessed, total_processed)
    end_time = time.time()

    print("\nData collection completed for all given paths.")
    print("\nTotal:", total_processed+total_unprocessed, "\tprocessed: ", total_processed, "unprocessed:", total_unprocessed)
    print("Time elapsed: {0:.3f}".format((end_time - start_time) / 60), "minute(s)")

    # collect list of available sections from pkl and store mapping to their embedding
    pd.DataFrame.from_dict([all_sections.keys()]).to_csv(cnst.PKL_SOURCE_PATH + cnst.ESC + 'available_sections.csv', index=False, header=None)

    embedder.embed_section_names()

