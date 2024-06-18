# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from time import time

import instance
from copy import deepcopy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    stationDict, originriderList, dist, mode = instance.initInstance("instance1.xlsx")

    # 随机筛选一定规模的rider
    riderNum = 1000
    riderList = np.random.choice(originriderList[0:10000], riderNum, replace=False).tolist()
    np.random.shuffle(originriderList)
    begin_time = time()
    linkSet, selectRider = instance.riderRouteSearch(dist, stationDict, riderList, mode, riderNum)

    end_time = time()
    print("主体部分计算时间：", end_time - begin_time)
    # 最终路径矩阵，第一行是每条路的cost！
    np.savetxt('code0615/Instance/A_1000_1_linkSet.csv', linkSet, delimiter=',')
    riderIndexList = []

    totalRouteList = None
    for r in originriderList:
        if r.ID in selectRider:
            currRouteList = np.append([r.routePrice], r.routeList, axis=0)
            totalRouteList = instance.addFinalRoute(totalRouteList, currRouteList)
            riderIndexList.append(totalRouteList.shape[1])
    np.savetxt('code0615/Instance/A_1000_1_riderRoute.csv', totalRouteList, delimiter=',')

    totalRouteList1 = None
    for r in originriderList:
        if r.ID in selectRider:
            currRouteList = np.append([r.routePrice[0]], r.routeList[0], axis=0)
            totalRouteList1 = instance.addFinalRoute(totalRouteList1, np.array([currRouteList]))

    print(riderIndexList[-1])
    end_time = time()
    print("全部处理完成时间：", end_time - begin_time)
    #
    df = pd.DataFrame(data=riderIndexList)
    df.to_csv('Instance/A_1000_1_riderIndex.csv', encoding='utf-8')
    df = pd.DataFrame(data=selectRider)
    df.to_csv('Instance/A_1000_1_selectRider.csv', encoding='utf-8')
    station_data = pd.read_excel("code0615/instance0615.xlsx", sheet_name='Sheet1')

    # plt.figure(figsize=(24,8))
    # ax1 = plt.subplot(1, 2, 1)
    # # 第一行第二列图形
    # ax2 = plt.subplot(1, 2, 2)
    #
    # plt.sca(ax1)
    # for i in range(totalRouteList.shape[1]):
    #     for k in range(1, totalRouteList.shape[0]):
    #         linkRow = int(totalRouteList[k, i])
    #         if linkRow == -1:
    #             break
    #         if i % 7 == 0:
    #             color = 'mediumaquamarine'
    #         elif i % 7 == 1:
    #             color = 'teal'
    #         elif i % 7 == 2:
    #             color = 'royalblue'
    #         elif i % 7 == 3:
    #             color = 'slategrey'
    #         elif i % 7 == 4:
    #             color = 'skyblue'
    #         elif i % 7 == 5:
    #             color = 'mediumturquoise'
    #         elif i % 7 == 6:
    #             color = 'steelblue'
    #         station1 = int(station_data[station_data["station_id"] == linkSet[linkRow, 1]].index.tolist()[0])
    #         station2 = int(station_data[station_data["station_id"] == linkSet[linkRow, 3]].index.tolist()[0])
    #         if linkSet[linkRow, 0] < 2:
    #             linkSet[linkRow, 0] = 2
    #             if linkSet[linkRow, 2] < 2:
    #                 linkSet[linkRow, 2] = 2
    #         if linkSet[linkRow, 0] != linkSet[linkRow, 2]:
    #             plt.plot((linkSet[linkRow, 0], linkSet[linkRow, 2]), (station1, station2), color=color)#
    # # plt.show()
    # # plt.close(0)
    # # plt.savefig('.fig1.png')
    # # plt.close(0)
    #
    # # plt.figure(1)
    # plt.sca(ax2)
    # np.random.shuffle(totalRouteList)
    # totalRouteList1 = totalRouteList[:, 0: int(totalRouteList.shape[1]/2.5)]
    # for i in range(totalRouteList1.shape[1]):
    #     for k in range(1, totalRouteList1.shape[0]):
    #         linkRow = int(totalRouteList1[k, i])
    #         if linkRow == -1:
    #             break
    #         if i % 7 == 0:
    #             color = 'mediumaquamarine'
    #         elif i % 7 == 1:
    #             color = 'teal'
    #         elif i % 7 == 2:
    #             color = 'royalblue'
    #         elif i % 7 == 3:
    #             color = 'slategrey'
    #         elif i % 7 == 4:
    #             color = 'skyblue'
    #         elif i % 7 == 5:
    #             color = 'mediumturquoise'
    #         elif i % 7 == 6:
    #             color = 'steelblue'
    #         station1 = int(station_data[station_data["station_id"] == linkSet[linkRow, 1]].index.tolist()[0])
    #         station2 = int(station_data[station_data["station_id"] == linkSet[linkRow, 3]].index.tolist()[0])
    #         if linkSet[linkRow, 0] < 2:
    #             linkSet[linkRow, 0] = 2
    #             if linkSet[linkRow, 2] < 2:
    #                 linkSet[linkRow, 2] = 2
    #         if linkSet[linkRow, 0] != linkSet[linkRow, 2]:
    #             plt.plot((linkSet[linkRow, 0], linkSet[linkRow, 2]), (station1, station2), color=color)#
    # # plt.savefig('.fig1.png')
    # plt.show()

    # plt.figure()
    # for i in range(linkSet.shape[0]):
    #      plt.plot((linkSet[i,0],linkSet[i,2]), (linkSet[i,1],linkSet[i,3]))
    # plt.show()
