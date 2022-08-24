import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset,IterableDataset,DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
import time
from sklearn import preprocessing
from os import walk
from os import path
from sklearn.utils import shuffle

# combine one batch of npz files together into a small dataset
def combine_dataset_generator(files: list, file_load_size: int, reedom_seed: int):
    total_chunks = int(np.ceil(len(files) / float(file_load_size))) #ceil mean round up
    files = shuffle(files, random_state=reedom_seed) #shuffle the data files order
    for i in range(total_chunks):
        # In this method, we do all the preprocessing.
        # First read data from files in a chunk. Preprocess it. Extract labels. Then return combined dataset.
        file_batch = files[i * file_load_size:(i + 1) * file_load_size]  # This extracts one batch of file names from the list `filenames`.
        first_file = file_batch.pop(0)
        npzfile = np.load(first_file)
        data , GT = npzfile["X"], npzfile["Y"]

        for file in file_batch:
            npzfile = np.load(file) # np.array
            nparray_X, nparray_Y = npzfile["X"], npzfile["Y"]
            data = np.concatenate([data,nparray_X],0)
            GT = np.concatenate([GT,nparray_Y],0)
        data, GT = shuffle(data, GT, random_state = reedom_seed)
        data = torch.from_numpy(data).double()
        GT = torch.from_numpy(GT).double()
        yield data, GT

# get one batch data form combined dataset
def batch_generator(files: list, file_load_size: int , batch_size:int, reedom_seed:int):
    combined_dataset = combine_dataset_generator(files, file_load_size, reedom_seed)
    total_chunks = int(np.ceil(len(files) / float(file_load_size)))
    for i in range(total_chunks):
        database, GT_database = next(combined_dataset)
        for local_index in range(0, database.shape[0], batch_size):
            input_local = database[local_index:(local_index + batch_size)]
            GT_local = GT_database[local_index:(local_index + batch_size)]
            yield input_local, GT_local

class BreatheDataset(IterableDataset):
    def __init__(self, files: list, file_load_size: int , batch_size:int, random_seed:int = 123):
        super(BreatheDataset).__init__()
        # `filenames` is a list of strings the contains all file names.
        # `file_load_size` is the determines the number of files that we want to combine in a small dataset.
        self.batch_generator = batch_generator(files, file_load_size, batch_size, random_seed)

    def __iter__(self):
        return self.batch_generator

if __name__ == '__main__': #程序在运行时启用了多线程，而多线程的使用用到了freeze_support()函数。freeze_support()函数在linux和类unix系统上可直接运行，在windows系统中需要跟在main后边。
    train_files = []
    for dirPath, dirNames, fileNames in walk("processed_data\\train"):
        for f in fileNames:
            train_files.append((path.join(dirPath, f)))
    train_dataset = BreatheDataset(train_files, 2, 16)
    # for data,GT in train_dataset:
    #     print(data.shape)
    #     print(GT.shape)
    #     break
    train_data_loader = DataLoader(dataset=train_dataset, num_workers=0, batch_size=None, shuffle=False)
    times = 0
    for i, (data, GT) in enumerate(train_data_loader):
        times +=1
    print(times)
        # print(data.shape) #torch.Size([16, 200, 1])
        # print(data)
        # break

