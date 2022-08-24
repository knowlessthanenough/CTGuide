import pandas as pd
from datetime import datetime, timedelta
from os import walk
from os import path
from pathlib import Path



def data_aline(files: list):
    Belt_files = []
    GT_files = []
    for file in files:
        if (path.splitext(path.basename(file))[0][0:4]) == 'Belt':
            Belt_files.append(file)
        else:
            GT_files.append(file)
    print(Belt_files)
    print(GT_files)


    for i in range(len(Belt_files)):
        Belt_file_name = Belt_files[i]
        GT_file_name = GT_files[i]
        #the aline function
        belt = pd.read_csv(Belt_file_name, header = None, sep=' ')
        GT = pd.read_csv(GT_file_name, header=None)

        #get belt start time
        start_time = belt.iat[0,1]
        datatime_belt_begin_time = datetime.strptime(start_time, "%H:%M:%S.%f").time()
        # print(datatime_belt_begin_time)

        #get GT start time
        begin_time = GT.iloc[1].item().split()[-1]
        datatime_GT_begin_time = datetime.strptime(begin_time, "%H:%M:%S.%f").time()
        # print(datatime_GT_begin_time)

        #calculate start time different (belt start first)
        if datatime_belt_begin_time < datatime_GT_begin_time:
            duration_belt = timedelta(hours=datatime_belt_begin_time.hour, minutes=datatime_belt_begin_time.minute, seconds=datatime_belt_begin_time.second, microseconds=datatime_belt_begin_time.microsecond)
            duration_GT = timedelta(hours=datatime_GT_begin_time.hour, minutes=datatime_GT_begin_time.minute, seconds=datatime_GT_begin_time.second, microseconds=datatime_GT_begin_time.microsecond)
            duration = (duration_GT - duration_belt).total_seconds()
            alined_belt_start_number = int(duration//0.05 + 1) #equal to round up

            #cut and reset the belt begin time
            belt = belt.iloc[alined_belt_start_number:]
            belt.reset_index(drop=True, inplace=True)
            start_time = belt.iat[0,1]
            datatime_belt_begin_time = datetime.strptime(start_time, "%H:%M:%S.%f").time()
            duration_belt = timedelta(hours=datatime_belt_begin_time.hour, minutes=datatime_belt_begin_time.minute, seconds=datatime_belt_begin_time.second, microseconds=datatime_belt_begin_time.microsecond)

        #calculate start time different (GT start first)
        duration = (duration_belt - duration_GT).total_seconds()
        alined_GT_start_number = int(duration//0.001 + 1)
        #cut the GT begin time
        GT = GT.iloc[alined_GT_start_number + 6:]
        GT.reset_index(drop=True, inplace=True)
        GT = GT[0].str.split('\t', expand=True)
        # print(GT.iloc[0]) #print specific row
        # print(duration) #<class 'float'>
        # print(alined_belt_start_time)
        # print(belt.iat[alined_belt_start_time,1]) #return position of element in specific [row,column]


        #get GT data by the freq of belt(20Hz)
        useful_GT_index = []
        for i in range(0,len(GT),50):
            useful_GT_index.append(i)
        useful_GT = pd.DataFrame(GT.iloc[useful_GT_index][2])
        useful_GT.reset_index(drop=True, inplace=True)
        # print(useful_GT.head())
        # print('useful_GT:',len(useful_GT))
        # print('useful_belt:',len(belt))
        if len(useful_GT) > len(belt):
            useful_GT = useful_GT[:len(belt)]
        elif len(useful_GT) < len(belt):
            belt = belt[:len(useful_GT)]
        else:
            pass
        # print('useful_GT:',len(useful_GT))
        # print('useful_belt:',len(belt))

        # dataframe combine
        belt[3] = useful_GT[2]

        # write to csv ### need change
        save_data_location = ('Alined_Final_data\\' + Path(Belt_file_name).parts[1] + '\\' + Path(Belt_file_name).parts[2] + '\\' +
                                  path.splitext(path.basename(Belt_file_name))[0])
        # print(save_data_location)
        belt.to_csv(save_data_location + ".csv", encoding='utf-8', sep=" ", index = None, header=None)

if __name__ == '__main__':
    files = []  # a list to save all file direction
    for dirPath, dirNames, fileNames in walk("Final_data\\"):
        for f in fileNames:
            files.append((path.join(dirPath, f)))
    data_aline(files)