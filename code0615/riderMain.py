from time import time

import generateInstance
from copy import deepcopy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    stationDict, originriderList, dist, mode = generateInstance.initInstance("instance0615.xlsx", 7, 70)
    price = 0.7

    print(len(originriderList))
    # 随机筛选一定规模的rider
    riderNum = 1000
    instanceIndex = 4 # 1: 0-3, 2:3-5, 3:5-7, 4:7-
    np.random.shuffle(originriderList)
    begin_time = time()

    linkSet, selectRider, available_rate = generateInstance.riderRouteSearch(dist, stationDict, originriderList, mode, riderNum)

    end_time = time()
    print("主体部分计算时间：", end_time - begin_time)

    # 最终路径矩阵，第一行是每条路的cost！
    np.savetxt('taxiPrice=' + str(price) + '/A_'+str(riderNum) + '_' + str(instanceIndex) + '_linkSet.csv', linkSet, delimiter=',')

    riderIndexList = {}
    totalRouteList = {}

    caseList = [2]
    for case in caseList:
        for t in range(1, 2):
            riderIndexList[case, t] = []
            totalRouteList[case, t] = None
            for r in originriderList:
                if r.ID in selectRider:
                    if r.routeList[case, t] is not None:
                        currRouteList = np.append([r.routePrice[case, t]], r.routeList[case, t], axis=0)
                        totalRouteList[case, t] = generateInstance.addFinalRoute(totalRouteList[case, t], currRouteList)
                    if totalRouteList[case, t] is None:
                        riderIndexList[case, t].append(0)
                    else:
                        riderIndexList[case, t].append(totalRouteList[case, t].shape[1])
            np.savetxt('taxiPrice=' + str(price) + '/A_'+str(riderNum) + '_' + str(instanceIndex)
                       + '_riderRoute_case' + str(case) + '_type_' + str(t) + '.csv', totalRouteList[case, t], delimiter=',')

            print(riderIndexList[case, t][-1])

            df = pd.DataFrame(data=riderIndexList[case, t])
            df.to_csv('taxiPrice=' + str(price) + '/A_'+str(riderNum) + '_' + str(instanceIndex)
                      + '_riderIndex_case' + str(case) + '_type_' + str(t) + '.csv', encoding='utf-8')


    # # case4
    # riderIndexList[4] = []
    # totalRouteList[4] = None
    # for r in originriderList:
    #     if r.ID in selectRider:
    #         currRouteList = np.append([r.routePrice_case4], r.routeList_case4, axis=0)
    #         totalRouteList[4] = generateInstance.addFinalRoute(totalRouteList[4], currRouteList)
    #         riderIndexList[4].append(totalRouteList[4].shape[1])
    # np.savetxt('Instance/A_'+str(riderNum) + '_' + str(instanceIndex) + '_riderRoute_case4.csv', totalRouteList[4], delimiter=',')
    # print(riderIndexList[4][-1])
    #
    #
    # df = pd.DataFrame(data=riderIndexList[4])
    # df.to_csv('Instance/A_'+str(riderNum) + '_' + str(instanceIndex) + '_riderIndex_case4.csv', encoding='utf-8')
    #
    # df = pd.DataFrame(data=selectRider)
    # df.to_csv('Instance/A_'+str(riderNum) + '_' + str(instanceIndex) + '_selectRider.csv', encoding='utf-8')


    # plt.figure()  # 创建一个以 axes 为单位的 2x2 网格的 figure
    # for k in rlist:
    #     for i in range(len(k.nodeList)-1):
    #         plt.plot((k.nodeList[i][0],k.nodeList[i+1][0]), (k.nodeList[i][1],k.nodeList[i+1][1]))
    # plt.show()
    # plt.figure()
    # for i in range(linkSet[0].shape[0]):
    #      plt.plot((linkSet[0][i,0],linkSet[0][i,2]), (linkSet[0][i,1],linkSet[0][i,3]))
    # plt.show()

    # station_data = pd.read_excel("instance0615.xlsx", sheet_name='Sheet1')
    #
    # plt.figure(1)
    # for c in totalRouteList.keys():
    #     for i in range(totalRouteList[c].shape[1]):
    #         for k in range(1, totalRouteList[c].shape[0]):
    #             linkRow = int(totalRouteList[c][k, i])
    #             if linkRow == -1:
    #                 break
    #             if linkSet[linkRow, 4] == 0:
    #                 color = 'orange'
    #                 linestyle = '-'
    #             elif linkSet[linkRow, 4] == 1:
    #                 if linkSet[linkRow, 0]==7 and  linkSet[linkRow, 2]==29:
    #                     print("aaa")
    #                 color = 'green'
    #                 linestyle = '--'
    #             else:
    #                 color = 'purple'
    #                 linestyle = '-.'
    #             # station1 = station_data[station_data["station_id"] == linkSet[linkRow, 1]].index.tolist()[0]
    #             # station2 = station_data[station_data["station_id"] == linkSet[linkRow, 3]].index.tolist()[0]
    #             plt.plot((linkSet[linkRow, 0], linkSet[linkRow, 2]), (linkSet[linkRow, 1], linkSet[linkRow, 3]), color,
    #                      linestyle=linestyle)
    #     plt.show()
