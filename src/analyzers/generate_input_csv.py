import os
import pefile
import config.constants as cnst


SKIP = False
processed_bfile_count = 0
processed_mfile_count = 0
target_paths = ['Raw malware files path',
                'Raw benign files path']
all_file = cnst.ALL_FILE
benign_file = cnst.BENIGN_FILE
malware_file = cnst.MALWARE_FILE

unprocessed = 0
if not SKIP:
    print("\nProcessing files at - ", str(target_paths))
    with open(all_file, 'a+') as a, open(benign_file, 'a+') as b, open(malware_file, 'a+') as m:
        for target_path in target_paths:
            # if processed_file_count > cnst.MAX_FILE_SIZE_LIMIT:
            #    break
            for file in os.listdir(target_path):
                try:
                    # check if file can be processed by pefile module
                    pe = pefile.PE(target_path+file)
                    for item in pe.sections:
                        # Check if all sections are parse-able without error
                        _ = item.Name.rstrip(b'\x00').decode("utf-8").strip()
                        _ = item.get_data()

                    src_file_size = os.stat(target_path + file).st_size
                    if src_file_size > cnst.MAX_FILE_SIZE_LIMIT:
                        print("Skipping as file size exceeds ", cnst.MAX_FILE_SIZE_LIMIT,
                              "[ Unprocessed / Skipped Count: "+str(unprocessed)+"]")
                        unprocessed += 1
                        continue
                    if not file.endswith('.csv') and not file.endswith('.png'):
                        # print(file)
                        if "benign" in target_path:  # file.strip().lower().count('.') != 0:
                            a.write(target_path + file + ",0\n")
                            b.write(target_path + file + ",0\n")
                            processed_bfile_count += 1
                        else:
                            a.write(target_path + file + ",1\n")
                            m.write(target_path + file + ",1\n")
                            processed_mfile_count += 1
                except Exception as e:
                    unprocessed += 1
                    print("parse failed . . . [ Unprocessed / Skipped Count: ", str(unprocessed),
                          "] [ Error: " + str(e) + " ] [ FILE ID - ", file, "] ")
                print("[BENIGN: "+str(processed_bfile_count)+"]\t[MALWARE: "+str(processed_mfile_count)+"]")

    print("File collection completed.")
