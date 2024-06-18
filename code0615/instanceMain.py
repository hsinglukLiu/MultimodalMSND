
from time import time

import generateInstance
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    stationDict, dist, mode = generateInstance.initInstance("instance0615.xlsx")
    originriderList = generateInstance.initRiderInstance("dataset/all_rider_data.csv",
                                                         stationDict, dist, mode, 0, 10)
    # originriderList = originriderList[0:5000]

    # -----------------------循环生成instance_num个算例 -----------------------
    instance_num = 1
    CPU_time_subProblem = pd.DataFrame(np.zeros([instance_num, 4]))
    for i in range(instance_num):  # 生成B类算例，就一个
        instanceIndex = i + 1  # 算例index

        print('\n\n\n\n ------------ instance ID: {}'.format(i))
        # originriderList = generateInstance("dataset/rider_data"+str(instanceIndex)+".csv", stationDict, dist, mode, 0, 9)
        # 随机筛选一定规模的rider
        riderNum = 5000

        # instanceIndex = 1  # 算例index
        np.random.shuffle(originriderList)
        begin_time = time()

        linkSet, selectRider, available_rate = generateInstance.riderRouteSearch1(dist, stationDict, originriderList, mode, riderNum)

        end_time = time()
        print("主体部分计算时间：", end_time - begin_time)

        CPU_time_subProblem.iloc[i, 0] = instanceIndex
        CPU_time_subProblem.iloc[i, 1] = end_time - begin_time

        # 最终路径矩阵，第一行是每条路的cost！
        np.savetxt('instance/A_'+str(riderNum) + '_' + str(instanceIndex) + '_linkSet.csv', linkSet, delimiter=',')

        instanceList = np.array(shape=[0, 9])
        for r in originriderList:
            if r.ID in selectRider:
                instanceList = np.append(instanceList, [[r.ID, r.origion.pointX, r.origion.pointY, r.destination.pointX
                                                            , r.destination.pointY, r.earliestDepartTime, r.latestArriveTime
                                                            , r.maxRideTime_subway, r.maxTransferNum]], axis=0)

        np.savetxt('dataset/instance_January_' + str(instanceIndex) + '.csv', instanceList, delimiter=',')
