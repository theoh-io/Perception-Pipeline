import pandas as pd
import numpy as np
import os

path_folder="../../Results/"
method="siam"
path=path_folder+method

def filter_files(folder_path, filter, sep, nb_split=1):
    list_files=os.listdir(folder_path)
    remove_list=[]
    for i in range(len(list_files)):
        try:
            name, type=list_files[i].rsplit(sep, nb_split)
            if type != filter:
                remove_list.append(list_files[i])
        except:
            remove_list.append(list_files[i])
        
    for j in range(len(remove_list)):
        list_files.remove(remove_list[j])
    list_files.sort()
    return list_files


list_files=filter_files(path, "tracker.txt", "_", 1)
#print(list_files)

mean_list=[]
for i in range(len(list_files)):
    times=np.array([])
    df=pd.read_csv(path+"/"+list_files[i], header=None)
    #print(df)
    times=np.squeeze(df.to_numpy())
    #print(times)
    mean_time=np.mean(times, axis=0)
    print(f"mean time for file {i} is : {mean_time}")
    mean_list.append(mean_time)

mean_list=np.array(mean_list)
print(f"average run time across all vids is : {np.mean(mean_list , axis=0)}")
