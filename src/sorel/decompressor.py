import os


for src_dir, dirs, files in os.walk('F:\\Sorel\\binaries'):
    for file_ in files:
        if file_.startswith("6"):
            os.remove(os.path.join(src_dir, file_))
        print(file_)



