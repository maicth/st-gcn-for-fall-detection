import os
import shutil
import pandas as pd
dir_path = "D://AIP490_data//tst_space_cut_last_frames"
file_lst = [
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
# for f in file_lst:
#     os.remove(os.path.join(dir_path,f))

# cut last frames
for f in file_lst:
    file_path = os.path.join(dir_path,f)
    df = pd.read_csv(file_path, header=None)
    df_2 = df[:(300*25*6)]
    df_2.to_csv(file_path, index=False, header=False)