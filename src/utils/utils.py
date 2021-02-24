import numpy as np
import pandas as pd
#from keras.backend.tensorflow_backend import get_session
from utils.preprocess import preprocess, preprocess_by_section, preprocess_by_section_by_samples, train_preprocess, train_preprocess_by_section, direct_preprocess, direct_preprocess_by_section
import datetime, time
import config.settings as cnst
from keras.preprocessing.sequence import pad_sequences
import pefile


def initiate_tensorboard_logging(tf, log_path):
    writer = tf.summary.FileWriter(log_path)
    writer.add_graph(get_session().graph)


def train_test_split(data, label, val_size, seed):
    idx = np.arange(len(data))
    np.random.seed(seed)
    np.random.shuffle(idx)
    split = int(len(data)*val_size)
    x_train, x_test = data[idx[split:]], data[idx[:split]]
    y_train, y_test = label[idx[split:]], label[idx[:split]]
    return x_train, x_test, y_train, y_test


def train_data_generator(partition, data, labels, max_len, batch_size, shuffle):
    idx = np.arange(len(data))
    if shuffle:
        np.random.shuffle(idx)
    batches = [idx[range(batch_size*i, min(len(data), batch_size*(i+1)))] for i in range(len(data)//batch_size+1)]
    if partition is not None:
        while True:
            for i in batches:
                try:
                    xx = direct_preprocess(data[i])
                    yy = labels[i]
                    yield (xx, yy)
                except Exception as e:
                    print(str(e), "TIER-1 Error during PRE-PROCESSING . . . ")  # [", labels[i], data[i], "]")
    else:
        print("TIER1 : No partition supplied. Check if partition loaded correctly with correct path")


def train_data_generator_by_section(spartition, sections, data, labels, max_len, batch_size, shuffle=False):
    idx = np.arange(len(data))
    if shuffle:
        np.random.shuffle(idx)
    batches = [idx[range(batch_size*i, min(len(data), batch_size*(i+1)))] for i in range(len(data)//batch_size+1)]
    if spartition is not None:
        while True:
            for i in batches:
                try:
                    xx = train_preprocess_by_section(spartition, data[i], max_len, sections)[0]
                    yy = labels[i]
                    yield (xx, yy)
                except Exception as e:
                    print("TIER-2 Error during PRE-PROCESSING SECTIONS. . .   [", labels[i], data[i], "]", str(e))
    else:
        print("TIER2 : No partitions supplied. Check if partition loaded correctly with correct path")


def data_generator(partition, data):
    # idx = np.arange(len(data))
    # if shuffle:
    #    np.random.shuffle(idx)
    # batches = [idx[range(batch_size*i, min(len(data), batch_size*(i+1)))] for i in range(len(data)//batch_size+1)]
    # total = 0
    # if partition is not None:
    #while True:
    a1 = time.time()
    corpus = [partition[fn[:-4]]["whole_bytes"] for fn in data]
    seq = pad_sequences(corpus, maxlen=cnst.MAX_FILE_SIZE_LIMIT, truncating='post', padding='post')
    b1 = time.time()
    yield (seq, np.ones(data.shape))
    #yield (preprocess(partition, data), np.ones(data.shape))
    c1 = time.time()
    #  print("File Count:", len(data), "Corpus load time:", (b1 - a1) / len(data) * 1000 , "Yield:", (c1 - a1) * 1000 / len(data))
    #print("Data Gen:", (c1 - b1) / len(data.shape))
        # for i in [0]:  # batches:
        # tst = time.time()
        #try:
            # xx = preprocess(partition, data[i], max_len)  # [0]
            # yy = labels[i]
        # xx = preprocess(partition, data, max_len)  # [0]
        # yy = labels
        # yield (xx, yy)
        #except Exception as e:
        #    print(str(e), "TIER-1 Error during PRE-PROCESSING . . . ")  # [", labels[i], data[i], "]")
        # tet = time.time()
        # total += int((tet - tst) * 1000)
        # print("[][][][][][][][][][][][][][][][][][][][][][][1]   DATA GEN TIME 1:", total / cnst.PREDICT_BATCH_SIZE, "ms")
    #else:
    #    print("TIER1 : No partition supplied. Check if partition loaded correctly with correct path")


def data_generator_by_section_by_samples(sections, data, labels, max_len, batch_size, shuffle):
    idx = np.arange(len(data))
    if shuffle:
        np.random.shuffle(idx)
    batches = [idx[range(batch_size*i, min(len(data), batch_size*(i+1)))] for i in range(len(data)//batch_size+1)]
    total = 0
    while True:
        for i in batches:
            tst = time.time()
            try:
                xx = preprocess_by_section_by_samples(data[i], max_len, sections)  #[0]
                yy = labels[i]
                yield (xx, yy)
            except Exception as e:
                print("TIER-2 Error during PRE-PROCESSING SECTIONS. . .   [", labels[i], data[i], "]", str(e))
            tet = time.time()
            total += int((tet - tst)*1000)
        print("[][][][][][][][][][][][][][][][][][][][][][][2] XXX  DATA GEN TIME 2:", total/8, "ms")


def data_generator_by_section(spartition, sections, data):
    #idx = np.arange(len(data))
    #if shuffle:
    #    np.random.shuffle(idx)
    #batches = [idx[range(batch_size*i, min(len(data), batch_size*(i+1)))] for i in range(len(data)//batch_size+1)]
    # total = 0
    # if spartition is not None:
    # while True:
    yield (preprocess_by_section(data, sections), np.ones(data.shape))
        # for i in [0]:  # batches:
        # tst = time.time()
        # try:
            # xx = preprocess_by_section(spartition, data[i], max_len, sections, section_map)  #[0]
            # yy = labels[i]
        # xx = preprocess_by_section(spartition, data, max_len, sections, section_map)  # [0]
        # yy = labels
        # yield (xx, yy)
        #except Exception as e:
        #    print("TIER-2 Error during PRE-PROCESSING SECTIONS. . .   [", labels[i], data[i], "]", str(e))
        #tet = time.time()
        #total += int((tet - tst) * 1000)
    #print("[][][][][][][][][][][][][][][][][][][][][][][3]   DATA GEN TIME 2:", total / cnst.PREDICT_BATCH_SIZE, "ms")
    # else:
    #    print("TIER2 : No partitions supplied. Check if partition loaded correctly with correct path")


def direct_data_generator(data, labels):
    # a1 = time.time()
    x = 0
    y = cnst.PREDICT_BATCH_SIZE
    l = len(data)

    file_path = cnst.PKL_SOURCE_PATH + "sample_weights.csv"
    swdf = pd.read_csv(file_path, header=None)

    sw = []
    for d in data:
        lct = swdf.index[swdf.iloc[:, 0] == d]
        if len(lct) > 0:
            sw.append(swdf.iloc[:, 4][lct].values[0])
        else:
            sw.append(cnst.BASE_WEIGHT)

    sw = np.array(sw)

    while y < l:
        yield (direct_preprocess(data[x:y]), labels[x:y], sw[x:y])
        x = y
        y += cnst.PREDICT_BATCH_SIZE
    yield (direct_preprocess(data[x:l]), labels[x:l], sw[x:l])
    #  b1 = time.time()
    #  print("dd Files:", len(data), "Corpus load time:", (b1 - a1)/len(data)*1000, "ms")


def direct_data_generator_by_section(sections, data, labels):
    #  a1 = time.time()
    x = 0
    y = cnst.PREDICT_BATCH_SIZE
    l = len(data)
    while y < l:
        yield (direct_preprocess_by_section(data[x:y], sections), labels[x:y])
        x = y
        y += cnst.PREDICT_BATCH_SIZE
    yield (direct_preprocess_by_section(data[x:l], sections), labels[x:l])
    #  b1 = time.time()
    #  print("dd Files:", len(data), "Corpus load time:", (b1 - a1)/len(data)*1000, "ms")


class logger():
    def __init__(self):
        self.fn = []
        self.len = []
        self.pad_len = []
        self.loss = []
        self.pred = []
        self.org = []
    def write(self, fn, org_score, file_len, pad_len, loss, pred):
        self.fn.append(fn.split('/')[-1])
        self.org.append(org_score)
        self.len.append(file_len)
        self.pad_len.append(pad_len)
        self.loss.append(loss)
        self.pred.append(pred)
        
        print('\nFILE:', fn)
        if pad_len > 0:
            print('\tfile length:', file_len)
            print('\tpad length:', pad_len)
            #if not np.isnan(loss):
            print('\tloss:', loss)
            print('\tscore:', pred)
        else:
            print('\tfile length:', file_len, ', Exceed max length ! Ignored !')
        print('\toriginal score:', org_score)
        
    def save(self, path):
        d = {'filename':self.fn, 
             'original score':self.org, 
             'file length':self.len,
             'pad length':self.pad_len, 
             'loss':self.loss, 
             'predict score':self.pred}
        df = pd.DataFrame(data=d)
        df.to_csv(path, index=False, columns=['filename', 'original score', 
                                              'file length', 'pad length', 
                                              'loss', 'predict score'])
        print('\nLog saved to "%s"\n' % path)
