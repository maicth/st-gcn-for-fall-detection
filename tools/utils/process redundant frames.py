import os
import shutil
import pandas as pd
dir_path = "C:/Users/FPT/Downloads"
tst_file_lst = [
'Data1-ADL-lay-1.csv',
'Data1-ADL-lay-2.csv',
'Data1-ADL-lay-3.csv',
'Data1-ADL-walk-1.csv',
'Data1-ADL-walk-3.csv',
'Data10-ADL-walk-1.csv',
'Data2-ADL-lay-2.csv',
'Data2-ADL-lay-3.csv',
'Data2-ADL-walk-1.csv',
'Data2-ADL-walk-2.csv',
'Data2-ADL-walk-3.csv',
'Data2-Fall-EndUpSit-1.csv',
'Data2-Fall-EndUpSit-2.csv',
'Data2-Fall-EndUpSit-3.csv',
'Data5-ADL-walk-1.csv'
]
fallfree_file_lst = [
    'Fall-1-19-4_Skeleton.csv',
    'Non-fall-1-70-3_Skeleton.csv',
    'Non-fall-1-70-4_Skeleton.csv'
]

# cut last frames
for f in fallfree_file_lst:
    file_path = os.path.join(dir_path,f)
    df = pd.read_csv(file_path, header=None)
    # df_2 = df[:(300*25*6)]  # for tst
    df_2 = df[:(300 * 25)]  # for fallfree
    df_2.to_csv(file_path, index=False, header=False)