import shutil
import os
# create new directory
old_root_path = "D://AIP490_data//tst_fall_detection"
new_root_path = "D://AIP490_data//tst_fall_detection_space_copy"
os.mkdir(new_root_path)

for (root,dirs,files) in os.walk(old_root_path, topdown=True):
    print(root,dirs,files)
    if "FileskeletonSkSpace.csv" in files:
        old_abs_path = os.path.join(root,"FileskeletonSkSpace.csv")
        lst = old_abs_path.split("\\")
        subject = int(lst[1][4:])
        isfall = (lst[2] == 'Fall')
        action = lst[3]+lst[4]
        new_filename = '-'.join(lst[1:5])+".csv"

        new_abs_path = os.path.join(new_root_path, new_filename)
        shutil.copyfile(old_abs_path, new_abs_path)