import os, shutil, errno, fnmatch
import pefile


def collect_pe_warnings(path):
    warnings_set = set()
    count = 0
    for src_dir, dirs, files in os.walk(path):
        for file_ in files:
            print(count)
            try:
                # check if file can be processed by pefile module
                pe = pefile.PE(src_dir+file_)
                if pe._PE__warnings is not None and len(pe._PE__warnings) > 0:
                    for i in range(0, len(pe._PE__warnings)):
                        warnings_set.add(str(pe._PE__warnings[i]))
                        print(pe._PE__warnings[i])
            except Exception as e:
                print("parse failed . . . [ Error: " + str(e) + " ] [ FILE ID - ", file_, "] ")
            count += 1
    return warnings_set


if __name__ == '__main__':
    src_path = "give path to 500kb raw files"
    warnings = sorted(collect_pe_warnings(src_path))
    print(warnings)
    with open('warnings.txt', 'w+') as f:
        f.write(str(warnings))
