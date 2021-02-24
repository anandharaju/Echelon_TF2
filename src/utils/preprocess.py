import time
import numpy as np
from config import settings as cnst
from keras.preprocessing.sequence import pad_sequences
import logging
import os
import pickle
import pefile


def train_preprocess(partition, file_list, max_len):
    '''
    Return processed data (ndarray) and original file length (list)
    '''
    corpus = []
    pkeys = partition.keys()
    for fn in file_list:
        # fn = fn[:-4]
        if fn not in pkeys:
            logging.critical(fn + ' not exists in partition')
        else:
            data = partition[fn]["whole_bytes"]  # image_256w
            corpus.append(data)
            # For reading a executable file directly
            # corpus.append(f.read())

    # corpus = [[byte for byte in doc] for doc in corpus]
    len_list = None  # [len(doc) for doc in corpus]
    seq = pad_sequences(corpus, maxlen=max_len, truncating='post', padding='post')  # , value=b'\x00')
    return seq, len_list


def train_preprocess_by_section(spartition, file_list, max_len, sections):
    '''
    Return processed data (ndarray) and original section length (list)
    '''
    if sections is None:
        logging.critical("No sections supplied to process. Check if Q-criterion based selection completed successfully.")

    corpus = []
    # section_byte_map = OrderedDict.fromkeys(section_map, value=0)
    pkeys = spartition.keys()
    for fn in file_list:
        # fn = fn[:-4]
        if fn not in pkeys:
            logging.critical(fn + ' not exists in partition')
        else:
            fjson = spartition[fn]
            combined = np.zeros(fjson["whole_bytes_size"] if cnst.LINUX_ENV else 2 ** 20)
            try:
                keys = fjson["section_info"].keys()
                for section in sections:
                    if section in keys:
                        # print(np.shape(combined))
                        start = fjson["section_info"][section]["section_bounds"]["start_offset"]
                        # end = fjson["section_info"][section]["section_bounds"]["end_offset"] + 1
                        data = fjson["section_info"][section]["section_data"]
                        combined[start:start+len(data)] = data
                #if len(combined) > max_len:
                #    logging.info("[CAUTION: LOSS_OF_DATA] Combined sections exceeded max sample length by " + str(len(combined) - max_len) + " bytes.")
                corpus.append(combined)

            except Exception as e:
                logging.exception("Error in Module: preprocess/process_by_section.")

    #corpus = [[byte for byte in doc] for doc in corpus]
    #len_list = [len(doc) for doc in corpus]
    seq = pad_sequences(corpus, maxlen=max_len, padding='post', truncating='post')
    return seq, None  # len_list


def preprocess(partition, file_list):
    '''
    Return processed data (ndarray) and original file length (list)
    '''
    # corpus = []
    # pkeys = partition.keys()
    # for fn in file_list:
        # fn = fn[:-4]
        # if fn not in pkeys:
        #    logging.critical(fn + ' not exists in partition')
        # else:
        # data = partition[fn[:-4]]["whole_bytes"]  # image_256w
        # corpus.append(partition[fn[:-4]]["whole_bytes"])
        # For reading a executable file directly
        # corpus.append(f.read())

    # corpus = [[byte for byte in doc] for doc in corpus]
    # len_list = None  # [len(doc) for doc in corpus]
    a1 = time.time()
    corpus = [partition[fn[:-4]]["whole_bytes"] for fn in file_list]
    b1 = time.time()
    seq = pad_sequences(corpus, maxlen=cnst.MAX_FILE_SIZE_LIMIT, truncating='post', padding='post')  # , value=b'\x00')
    c1 = time.time()
    # print("File Count:", len(file_list), "Corpus load time:", (b1 - a1)/len(file_list), "Padding:", (c1 - b1)/len(file_list))
    return seq  # , len_list


def preprocess_by_section_by_samples(file_list, max_len, sections):
    '''
    Return processed data (ndarray) and original section length (list)
    '''
    if sections is None:
        logging.critical("No sections supplied to process. Check if Q-criterion based selection completed successfully.")

    corpus = []
    for fn in file_list:
        if not os.path.isfile(cnst.PKL_SOURCE_PATH + 't2' + cnst.ESC + fn):
            print(fn, 'not exist')
        else:
            try:
                with open(cnst.PKL_SOURCE_PATH + 't2' + cnst.ESC + fn, 'rb') as f:
                    fjson = pickle.load(f)
                    keys = fjson["section_info"].keys()
                    combined = np.zeros(cnst.MAX_FILE_SIZE_LIMIT)  # np.zeros(fjson["whole_bytes_size"] if cnst.LINUX_ENV else 2 ** 20)
                    for section in sections:
                        if section in keys:
                            start = fjson["section_info"][section]["section_bounds"]["start_offset"]
                            # end = fjson["section_info"][section]["section_bounds"]["end_offset"] + 1
                            data = fjson["section_info"][section]["section_data"]
                            combined[start:start + len(data)] = data
                    #if len(combined) > max_len:
                    #    logging.info("[CAUTION: LOSS_OF_DATA] Combined sections exceeded max sample length by " + str(len(combined) - max_len) + " bytes.")
                    corpus.append(combined)
            except Exception as e:
                logging.exception("Error in Module: preprocess/process_by_section.")

    corpus = [[byte for byte in doc] for doc in corpus]
    # len_list = [len(doc) for doc in corpus]
    seq = pad_sequences(corpus, maxlen=max_len, padding='post', truncating='post')
    return seq  #, len_list


def preprocess_by_section(spartition, file_list, sections):
    '''
    Return processed data (ndarray) and original section length (list)
    '''
    # if sections is None:
    #    logging.critical("No sections supplied to process. Check if Q-criterion based selection completed successfully.")

    corpus = []
    # section_byte_map = OrderedDict.fromkeys(section_map, value=0)
    # pkeys = spartition.keys()
    for fn in file_list:
        # fn = fn[:-4]
        #if fn not in pkeys:
        #    logging.critical(fn + ' not exists in partition')
        #else:
        fjson = spartition[fn[:-4]]
        # Zero Replacement logic [Maintains same 1MB length]
        combined = np.zeros(cnst.MAX_FILE_SIZE_LIMIT)  # np.zeros(fjson["whole_bytes_size"] if cnst.LINUX_ENV else 2 ** 20)
        #try:
        keys = fjson["section_info"].keys()
        for section in sections:
            if section in keys:
                # print(np.shape(combined))
                start = fjson["section_info"][section]["section_bounds"]["start_offset"]
                # end = fjson["section_info"][section]["section_bounds"]["end_offset"] + 1
                data = fjson["section_info"][section]["section_data"]
                combined[start:start + len(data)] = data
        # if len(combined) > max_len:
        #    logging.info("[CAUTION: LOSS_OF_DATA] Combined sections exceeded max sample length by " + str(len(combined) - max_len) + " bytes.")
        corpus.append(combined)
        '''
                        # Concatenation logic - non-uniform section ends are padded to meet nearest conv. window multiple
                        combined = []
                        try:
                            keys = fjson["section_info"].keys()
                            for section in sections:
                                if section in keys:
                                    # print(np.shape(combined))
                                    # start = fjson["section_info"][section]["section_bounds"]["start_offset"]
                                    # end = fjson["section_info"][section]["section_bounds"]["end_offset"] + 1
                                    data = fjson["section_info"][section]["section_data"]
                                    # print(np.shape(data), np.shape(combined), np.shape(len(data)))
                                    combined = np.concatenate((combined, data, np.zeros(cnst.CONV_WINDOW_SIZE - (len(data) % cnst.CONV_WINDOW_SIZE))), axis=None)
                                    if len(data) % cnst.CONV_WINDOW_SIZE > 0:
                                        combined = np.concatenate((combined, data, np.zeros(cnst.CONV_WINDOW_SIZE - (len(data) % cnst.CONV_WINDOW_SIZE))), axis=None)
                                    else:
                                        combined = np.concatenate((combined, data), axis=None)
                            if len(combined) > max_len:
                                logging.debug("[CAUTION: LOSS_OF_DATA] Combined sections exceeded max sample length by " + str(
                                    len(combined) - max_len) + " bytes. #Sections:"+str(len(sections)))
                            corpus.append(combined)
        '''
        #except Exception as e:
        #    logging.exception("Error in Module: preprocess/process_by_section.")

    # corpus = [[byte for byte in doc] for doc in corpus]
    # len_list = [len(doc) for doc in corpus]
    seq = pad_sequences(corpus, maxlen=cnst.MAX_FILE_SIZE_LIMIT, padding='post', truncating='post')
    return seq  #, len_list


def direct_preprocess(file_list):
    corpus = []
    #  a1 = time.time()
    for fn in file_list:
        with open(cnst.RAW_SAMPLE_DIR + fn, 'rb') as f:
            corpus.append(list(f.read()))
    seq = pad_sequences(corpus, maxlen=cnst.MAX_FILE_SIZE_LIMIT, truncating='post', padding='post')
    #  b1 = time.time()
    #  print("Files:", len(file_list), "Corpus load time:", (b1 - a1)/len(file_list)*1000, "ms", np.shape(seq))
    return seq


def direct_preprocess_by_section(file_list, sections):
    corpus = []
    #  a1 = time.time()
    for fn in file_list:
        '''combined = np.zeros(cnst.MAX_FILE_SIZE_LIMIT)
        pe = pefile.PE(cnst.RAW_SAMPLE_DIR + fn)
        if ".header" in sections:
            combined[0: len(pe.header)] = list(pe.header)
        for section in pe.sections:
            if section.Name.strip(b'\x00').decode("utf-8").strip() in sections:
                combined[section.PointerToRawData:(section.PointerToRawData+section.SizeOfRawData)] = list(section.get_data())
        corpus.append(combined)'''
        #  combined = ''  # []
        pe = pefile.PE(cnst.RAW_SAMPLE_DIR + fn)
        # if ".header" in sections:
        combined = pe.header  # .extend(list(pe.header))

        '''sdict = dict()
        for section in pe.sections:
            sdict[section.Name.strip(b'\x00').decode("utf-8").strip()] = section.get_data()

        sdictkeys = sdict.keys()
        for section in sections:
            if section in sdictkeys:
                combined += sdict['section']
        '''
        for section in pe.sections:
            if section.Name.strip(b'\x00').decode("utf-8").strip() in sections:
                # combined.extend(np.zeros(cnst.CONV_WINDOW_SIZE - (len(combined)%cnst.CONV_WINDOW_SIZE)))
                combined += section.get_data()  # .extend(list(section.get_data()))
                # if len(combined) > cnst.MAX_FILE_SIZE_LIMIT:  break
        corpus.append(list(combined))
    seq = pad_sequences(corpus, maxlen=cnst.MAX_FILE_SIZE_LIMIT, padding='post', truncating='post')
    #  b1 = time.time()
    #  print("Files:", len(file_list), "Corpus load time:", (b1 - a1)/len(file_list)*1000, "ms", np.shape(seq))
    return seq


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
