import time
import numpy as np
from config import settings as cnst
from keras.preprocessing.sequence import pad_sequences
import logging


def preprocess(partition, file_list, max_len):
    '''
    Return processed data (ndarray) and original file length (list)
    '''
    corpus = []
    pkeys = partition.keys()
    for fn in file_list:
        fn = fn[:-4]
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


def preprocess_by_section(spartition, file_list, max_len, sections, section_map):
    '''
    Return processed data (ndarray) and original section length (list)
    '''
    if sections is None:
        logging.critical("No sections supplied to process. Check if Q-criterion based selection completed successfully.")

    corpus = []
    # section_byte_map = OrderedDict.fromkeys(section_map, value=0)
    pkeys = spartition.keys()
    for fn in file_list:
        fn = fn[:-4]
        if fn not in pkeys:
            logging.critical(fn + ' not exists in partition')
        else:
            fjson = spartition[fn]
            # Zero Replacement logic [Maintains same 1MB length]
            '''combined = np.zeros(fjson["whole_bytes_size"] if cnst.LINUX_ENV else 2 ** 20)
            try:
                keys = fjson["section_info"].keys()
                for section in sections:
                    if section in keys:
                        # print(np.shape(combined))
                        start = fjson["section_info"][section]["section_bounds"]["start_offset"]
                        end = fjson["section_info"][section]["section_bounds"]["end_offset"] + 1
                        data = fjson["section_info"][section]["section_data"]
                        combined[start:start+len(data)] = data
                if len(combined) > max_len:
                    logging.info("[CAUTION: LOSS_OF_DATA] Combined sections exceeded max sample length by " + str(len(combined) - max_len) + " bytes.")
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
                        combined = np.concatenate(combined, data, np.zeros(cnst.CONV_WINDOW_SIZE - (len(data) % cnst.CONV_WINDOW_SIZE)))
                if len(combined) > max_len:
                    logging.info("[CAUTION: LOSS_OF_DATA] Combined sections exceeded max sample length by " + str(
                        len(combined) - max_len) + " bytes.")
                corpus.append(combined)

            except Exception as e:
                logging.exception("Error in Module: preprocess/process_by_section.")

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
