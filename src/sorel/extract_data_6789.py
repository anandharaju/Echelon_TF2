import os
import sys
import pefile
import hashlib
import pickle
import time
import pandas as pd
from collections import OrderedDict
import zlib


DATASET_BACKUP_FILE = 'D:\\08_Dataset\\Sorel\\file_stats_6789.csv'
Error_Files = 'D:\\08_Dataset\\Sorel\\erroneous_files_6789.csv'
RAW_SAMPLE_DIRS = {
    'F:\\Sorel\\6789': False              # Malware-False
}
print("parameters set")


def get_statistics(path, is_benign, unprocessed, processed):
    list_idx = []
    src_file_bytes = None
    for src_dir, dirs, files in os.walk(path):
        for file_ in files:
            if file_ <= "63e6163d8b4cd5d91c0a4bb1d16ea1b144960e04649ed51a913ff6c7ed093642":
                processed += 1
                continue
            try:
                src_file = os.path.join(src_dir, file_)
                with open(src_file, 'rb') as rhandle:
                    src_file_bytes = rhandle.read()
                if not src_file_bytes.startswith(b'MZ'):
                    src_file_bytes = zlib.decompress(src_file_bytes)
                    with open(src_file, 'wb') as whandle:
                        whandle.write(src_file_bytes)

                pe = pefile.PE(src_file)
                list_idx.append([processed
                                    , 1
                                    , file_
                                    , hashlib.md5(src_file_bytes).hexdigest()
                                    , hashlib.sha1(src_file_bytes).hexdigest()
                                    , hashlib.sha256(src_file_bytes).hexdigest()
                                    , os.stat(src_file).st_size])
                processed += 1

            except Exception as e:
                unprocessed += 1
                pd.DataFrame([file_]).to_csv(Error_Files, index=False, header=None, mode='a')

            if processed % 10000 == 0:
                print("Total Count:", processed, "Unprocessed/Skipped:", unprocessed, file_)
                pd.DataFrame(list_idx).to_csv(DATASET_BACKUP_FILE, index=False, header=None, mode='a')
                list_idx = []
    return unprocessed, processed


total_processed = 0
total_unprocessed = 0
start_time = time.time()

for dirc in RAW_SAMPLE_DIRS.keys():
    print(dirc)
    total_unprocessed, total_processed = get_statistics(dirc, RAW_SAMPLE_DIRS[dirc], total_unprocessed, total_processed)

end_time = time.time()

print("\nData collection completed for all given paths.")
print("\nTotal:", total_processed+total_unprocessed, "\tprocessed: ", total_processed, "unprocessed:", total_unprocessed)
print("Time elapsed: {0:.3f}".format((end_time - start_time) / (60 * 60)), "hours")

