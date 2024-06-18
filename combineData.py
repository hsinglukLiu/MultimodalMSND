import pandas as pd
import numpy as np

# 筛选出一天的数据进行保存
data = {}
data[1] = pd.read_csv('dataset/rider_data######.csv').values

alldata = np.empty(shape=[0, 7])
for i in data.keys():
    alldata = np.append(alldata, data[i], axis=0)
all_rider_data = pd.DataFrame(alldata)
all_rider_data.columns = ['id', 'ed', 'la', 'tb', 'os', 'ds', 'vr']
all_rider_data.to_csv("dataset/all_rider_data.csv", index=False, sep=',')
