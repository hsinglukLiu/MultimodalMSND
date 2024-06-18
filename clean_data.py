import math
from copy import deepcopy

import pandas as pd
import numpy as np

# 筛选出一天的数据进行保存
data = pd.read_csv('cleaned_yellow_data.csv', header=None)

for i in range(1, 8):
    data.columns = ['tpep_pickup_datetime'
                    ,'tpep_dropoff_datetime'
                    ,'passenger_count'
                    ,'trip_distance'
                    ,'pickup_longitude'
                    ,'pickup_latitude'
                    ,'dropoff_longitude'
                    ,'dropoff_latitude'
                    ,'fare_amount']
    date = '2016-01-0'+str(i)
    new_data = pd.DataFrame(np.empty(shape = [0,9]), columns = ['tpep_pickup_datetime'
                                                                ,'tpep_dropoff_datetime'
                                                                ,'passenger_count'
                                                                ,'trip_distance'
                                                                ,'pickup_longitude'
                                                                ,'pickup_latitude'
                                                                ,'dropoff_longitude'
                                                                ,'dropoff_latitude'
                                                                ,'fare_amount'])
    for i in range(data.shape[0]):
        if date == data.iloc[i][0][:10]:
            new_data = new_data.append(data.iloc[i])

    new_data.to_csv("dataset/data"+str(date)+".csv",index=False,sep=',')

    # 筛选出一天第一个时段的数据进行保存
    endtime = 8
    new_data1 = np.empty(shape=[0, 7])
    for i in range(new_data.shape[0]):
        if int(new_data.iloc[i][1][11:13]) < endtime:
            dt = int(new_data.iloc[i][0][11:13]) * 12 + int(int(new_data.iloc[i][0][14:16])/5)
            at = int(new_data.iloc[i][1][11:13]) * 12 + int(int(new_data.iloc[i][1][14:16])/5)
            price = new_data.iloc[i][8]/max(0.5, new_data.iloc[i][3])
            new_data1 = np.append(new_data1, [[dt, at, new_data.iloc[i][4], new_data.iloc[i][5], new_data.iloc[i][6], new_data.iloc[i][7], price]], axis = 0)
    # np.savetxt('dataset/temp16010108.csv',new_data1,delimiter=',')


    # 处理每个需求对应的坐标到station维
    new_data2 = np.empty(shape=[0, 5])
    station = np.zeros(shape=[72, 1])
    latitude0 = 40.700221
    laStep = 0.02
    longtitude0 = -74.012905
    longStep = 0.013
    for i in range(new_data1.shape[0]):
        longIndex = int((new_data1[i,2] - longtitude0) / longStep)
        laIndex = int((new_data1[i,3] - latitude0) / laStep)
        currStation1 = longIndex * 8 + laIndex
        longIndex = int((new_data1[i, 4] - longtitude0) / longStep)
        laIndex = int((new_data1[i, 5] - latitude0) / laStep)
        currStation2 = longIndex * 8 + laIndex

        if currStation1 != currStation2:
            station[currStation1] = 1
            station[currStation2] = 1
            new_data2 = np.append(new_data2, [[new_data1[i,0],new_data1[i,1],currStation1,currStation2,new_data1[i,6]]], axis=0)
    #
    # np.savetxt('dataset/temp010108.csv', new_data2, delimiter=',')
    #
    # # 处理station
    # station_data = pd.DataFrame(np.empty(shape = [0,7]))
    # for i in range(station.shape[0]):
    #     if station[i,0] == 1:
    #         longIndex = int(i/8)
    #         laIndex = i % 8
    #         minlong = longtitude0 + longIndex * longStep
    #         maxlong = min(longtitude0 + (longIndex + 1) * longStep, -73.909708)
    #         minla = latitude0 + laIndex * laStep
    #         maxla = min(latitude0 + (laIndex + 1) * laStep, 40.877199)
    #         station_data = station_data.append([[i, minlong, maxlong, minla, maxla, (minlong+maxlong)/2, (minla+maxla)/2]])
    # station_data.columns = ['station_id', 'min_longitude','max_longitude'
    #                                                             ,'min_latitude','max_latitude'
    #                                                             ,'pointx','pointy']
    # station_data.to_csv("dataset/station_data01.csv",index=False,sep=',')

    # 处理rider
    rider_data = pd.DataFrame(np.empty(shape = [0,7]))
    for i in range(new_data2.shape[0]):
        if new_data2[i,0] < 96:
            trueT = max(1, new_data2[i, 1] - new_data2[i, 0])
            ed = max(0, new_data2[i, 0] - max(int(trueT/2), 1))
            la = min(96, new_data2[i, 1] + max(int(trueT/2), 1))
            tb = int(trueT * 1.5)
            vr = 1 + int(3 * np.random.random())
            rider_data = rider_data.append([[i, ed, la, tb, new_data2[i, 2], new_data2[i, 3], vr]])
    rider_data.columns = ['id','ed','la','tb','os','ds','vr']
    rider_data.to_csv("dataset/rider_data"+str(i)+".csv",index=False,sep=',')

# velo = []
# for i in range(new_data2.shape[0]):
#     if new_data2[i,0] < 96:
#         trueT = max(1, new_data2[i, 1] - new_data2[i, 0])
#         sta1 = np.flatnonzero(station_data['station_id'] == new_data2[i, 2])[0]
#         sta2 = np.flatnonzero(station_data['station_id'] == new_data2[i, 3])[0]
#         velo.append(calDistance(station_data.iloc[sta1][5],station_data.iloc[sta1][6],station_data.iloc[sta2][5],station_data.iloc[sta2][6])/trueT)

def calDistance(long1, la1, long2, la2):
    R = 6317
    R2 = math.cos(math.pi*(la2/180)) * R
    dist = 2 * math.pi * R2 * abs(la2 - la1) / 360 + 2 * math.pi * R * abs(long2 - long1) / 360
    return dist