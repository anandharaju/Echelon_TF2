import numpy as np
import pandas as pd
from keras.backend.tensorflow_backend import get_session
from utils.preprocess import preprocess, preprocess_by_section


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


def data_generator(partition, data, labels, max_len, batch_size, shuffle):
    idx = np.arange(len(data))
    if shuffle:
        np.random.shuffle(idx)
    batches = [idx[range(batch_size*i, min(len(data), batch_size*(i+1)))] for i in range(len(data)//batch_size+1)]
    if partition is not None:
        while True:
            for i in batches:
                try:
                    xx = preprocess(partition, data[i], max_len)[0]
                    yy = labels[i]
                    yield (xx, yy)
                except Exception as e:
                    print(str(e), "TIER-1 Error during PRE-PROCESSING . . . ")  # [", labels[i], data[i], "]")
    else:
        print("TIER1 : No partition supplied. Check if partition loaded correctly with correct path")


def data_generator_by_section(wpartition, spartition, sections, section_map, data, labels, max_len, batch_size, shuffle):
    idx = np.arange(len(data))
    if shuffle:
        np.random.shuffle(idx)
    batches = [idx[range(batch_size*i, min(len(data), batch_size*(i+1)))] for i in range(len(data)//batch_size+1)]
    if wpartition is not None and spartition is not None:
        while True:
            for i in batches:
                try:
                    xx = preprocess_by_section(wpartition, spartition, data[i], max_len, sections, section_map)[0]
                    yy = labels[i]
                    yield (xx, yy)
                except Exception as e:
                    print("TIER-2 Error during PRE-PROCESSING SECTIONS. . .   [", labels[i], data[i], "]", str(e))
    else:
        print("TIER2 : No partitions supplied. Check if partition loaded correctly with correct path")


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
