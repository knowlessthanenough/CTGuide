from os import walk
from os import path
import pandas as pd
import time
import numpy as np
from pathlib import Path
import torch
from sklearn import preprocessing
#usage: turn a continue time series in to time period
#x_width: how many time step is use as input
#y_width: how many time step ahead from X (e.g. predicted 1 time step ahead this =1, if just transform =0)
#x_feature: how many input feature per time step (*all input must put in front, e.g. 0~2 this 3 colume is input (x_feature=3), the rest will identify as output feature)
#step: slicing window step
def time_series_data_slicing(DataSet, x_width: int, y_width: int, x_feature: int, step:int = 1):
    row,colume = DataSet.shape
    y_feature = colume-x_feature
    y_last_element_index = x_width + y_width
    total_slide_times = (row - x_width - y_width)//step

    # create storage for X(input feaure) (StandardScaler)
    X = np.reshape(DataSet.iloc[:x_width, :x_feature].values, (x_width, x_feature))
    scaler = preprocessing.StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    X = np.reshape(X_scaled, (1, x_width, x_feature))

    # create storage for Y(output feaure)
    Y = np.reshape(DataSet.iloc[y_width:y_last_element_index, x_feature:].values,(1, x_width,y_feature)) #use x_width because same length as input
    for i in range(1,total_slide_times+1):
        j = i * step
        y_temp = np.reshape(DataSet.iloc[j + y_width:j + y_last_element_index, x_feature:].values,(1,x_width,y_feature))

        # StandardScaler
        x_temp = np.reshape(DataSet.iloc[j:j + x_width, :x_feature].values, (x_width, x_feature))
        scaler = preprocessing.StandardScaler().fit(x_temp)
        x_scaled = scaler.transform(x_temp)
        x_scaled = np.reshape(x_scaled, (1, x_width, x_feature))

        X = np.concatenate([X,x_scaled],0)
        Y = np.concatenate([Y,y_temp],0)
    # print(X.shape,Y.shape)
    return X,Y
# data = {"X1":[0,1,2,3,4,5,6,7,8,9,10],
#         'X2':[10,11,12,13,14,15,16,17,18,19,110],
#         "X3":[20,21,22,23,24,25,26,27,28,29,210],
#         "GT":[30,31,32,33,34,35,36,37,38,39,310]}
# df=pd.DataFrame(data)
# print(df.shape)
# input,GT=time_series_data_slicing(df,3,1,2)
# print(input.shape)
# print(GT.shape)
# print(input)
#--------------------------------------

def data_preprocess(files: list):
    counter = 0

    for i in range(len(files)):
        fname = files[counter]
        counter = counter + 1
        df = pd.read_csv(fname, sep=" ", header=None)
        data = pd.DataFrame(columns=["sensor", "GT"])
        data['sensor'] = df[2]
        data['GT'] = df[3]
        start = time.time()
        print("-------start create dataset-------")
        X, Y = time_series_data_slicing(data, 200, 40, 1)# here is preprocessing (also do standscaler)
        end = time.time()
        print('-------dataset created, it take:', end - start)
        save_data_location = ('processed_data/' + Path(fname).parts[-2] + '\\' + path.splitext(path.basename(fname))[0])
        np.savez((save_data_location + '.npz'), X=X, Y=Y)

        #to load data: X,Y = np.load('x.npy'),np.load('y.npy')

if __name__ == '__main__':
    files = [] # a list to save all file direction
    for dirPath, dirNames, fileNames in walk("data\\"):
        for f in fileNames:
            files.append((path.join(dirPath, f)))
    print(files)
    data_preprocess(files) # do slicing window for all data and save it in to .npz files

    # # Way to load data
    # npzfile = np.load("processed_data\\train\\Aligned_Array.npz")
    # X = npzfile["X"]
    # X = torch.from_numpy(X)
    # print(X.shape)

