import os
import time
import pickle
import numpy as np
import argparse
import pandas as pd
from config import constants as cnst
from analyzers.parse_pe import parse_pe_section_data
from keras.preprocessing.sequence import pad_sequences
from collections import OrderedDict


def preprocess(partition, file_list, max_len):
    '''
    Return processed data (ndarray) and original file length (list)
    '''
    corpus = []
    pkeys = partition.keys()
    for fn in file_list:
        fn = fn[:-4]
        if fn not in pkeys:
            print(fn, 'not exists in partition')
        else:
            data = partition[fn]["whole_bytes"]  # image_256w
            corpus.append(data)
            # For reading a executable file directly
            # corpus.append(f.read())

    # corpus = [[byte for byte in doc] for doc in corpus]
    len_list = None  # [len(doc) for doc in corpus]
    seq = pad_sequences(corpus, maxlen=max_len, truncating='post', padding='post')  # , value=b'\x00')
    return seq, len_list


def preprocess_by_section(wpartition, spartition, file_list, max_len, sections, section_map):
    '''
    Return processed data (ndarray) and original section length (list)
    '''
    if sections is None:
        print("No sections supplied to process. Check if Q-criterion based selection completed successfully.")

    corpus = []
    # section_byte_map = OrderedDict.fromkeys(section_map, value=0)
    pkeys = spartition.keys()
    for fn in file_list:
        fn = fn[:-4]
        if fn not in pkeys:
            print(fn, 'not exists in partition')
        else:
            combined = []
            fjson = spartition[fn]
            try:
                keys = fjson["section_info"].keys()
                '''
                # Update byte map with sections that are present in current file
                for key in keys:
                    if key in section_map:
                        section_byte_map[key] = 1
                    #else:
                    #    print("Unknown Section in B1 samples:", key)

                byte_map = []
                byte_map_input = section_byte_map.values()  # ordered dict preserves the order of byte map
                for x in byte_map_input:
                    byte_map.append(x)

                combined = np.concatenate([combined, byte_map, np.zeros(cnst.CONV_WINDOW_SIZE)])
                '''

                for section in sections:
                    if section in keys:
                        combined = np.concatenate(
                            [combined, fjson["section_info"][section]["section_data"], np.zeros(cnst.CONV_WINDOW_SIZE)])

                if cnst.TAIL in sections:
                    whole_bytes = wpartition[fn]["whole_bytes"]
                    fsize = len(whole_bytes)
                    sections_end = 0
                    for key in keys:
                        if fjson["section_info"][key]['section_bounds']["end_offset"] > sections_end:
                            sections_end = fjson["section_info"][key]['section_bounds']["end_offset"]

                    if sections_end < 0:
                        print("[OVERLAY DATA NOT ADDED] Invalid section end found - ", sections_end)

                    if sections_end < fsize - 1:
                        combined = np.concatenate([combined, whole_bytes[sections_end:fsize], np.zeros(cnst.CONV_WINDOW_SIZE)])

                    if len(combined) > max_len:
                        print("[CAUTION: LOSS_OF_DATA] Combined sections exceeded max sample length by " + str(len(combined) - max_len) + " bytes. FileSize:"+str(fsize)+" sections_end:"+str(sections_end))

                corpus.append(combined)

            except Exception as e:
                print("Module: process_by_section. Error:", str(e))

    corpus = [[byte for byte in doc] for doc in corpus]
    len_list = [len(doc) for doc in corpus]
    seq = pad_sequences(corpus, maxlen=max_len, padding='post', truncating='post')
    return seq, len_list


if __name__ == '__main__':
    '''args = parser.parse_args()

    df = pd.read_csv(args.csv, header=None)
    fn_list = df[0].values
    
    print('Preprocessing ...... this may take a while ...')
    st = time.time()
    processed_data = preprocess_by_section(fn_list, args.max_len, '.data')[0]
    print('Finished ...... %d sec' % int(time.time()-st))
    
    with open(args.save_path, 'wb') as f:
        pickle.dump(processed_data, f)
    print('Preprocessed data store in', args.save_path)'''
    st = time.time()
    #get_offline_features()
    print("Time taken:", time.time() - st)
