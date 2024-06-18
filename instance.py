import math
from copy import deepcopy

from DTO import *
import pandas as pd
import numpy as np
from time import time
import xlrd

v_max = 2.3
tau = 0.9

'''读取函数'''


def initInstance(path):
    station_data = pd.read_excel(path, sheet_name='Sheet1')
    stationDict = {}
    for i in range(station_data.shape[0]):
        newStation = Station(station_data.loc[i, 'station_id']
                             , station_data.loc[i, 'pointx'], station_data.loc[i, 'pointy'], 0)
        stationDict[station_data.loc[i, 'station_id']] = newStation

    dist = pd.DataFrame(np.zeros([len(stationDict.keys()), len(stationDict.keys())]), index=stationDict.keys(),
                        columns=stationDict.keys())
    for i in stationDict.keys():
        for j in stationDict.keys():
            dist.loc[i, j] = calDistance(stationDict[i].pointX, stationDict[i].pointY, stationDict[j].pointX,
                                         stationDict[j].pointY)

    mode_data = pd.read_excel(path, sheet_name='Sheet3')
    mode = Mode()
    mode.ID = mode_data['id']
    mode.modeVehicleNum = mode_data['num']
    mode.minRiderNum = mode_data['minrider']
    mode.modeVelocity = mode_data['velo']
    mode.modePrice = mode_data['price']

    rider_data = pd.read_excel(path, sheet_name='Sheet2')
    riderList = []
    for i in range(rider_data.shape[0]):
        riderList.append(Rider(rider_data.loc[i, 'id'], rider_data.loc[i, 'ed'], rider_data.loc[i, 'la']
                               , rider_data.loc[i, 'tb'], rider_data.loc[i, 'vr'], 0))
        riderList[i].origion = stationDict[rider_data.loc[i, 'os']]
        riderList[i].destination = stationDict[rider_data.loc[i, 'ds']]
        riderList[i].maxPrice = tau * mode.modePrice[0] * (dist[riderList[i].origion.ID][riderList[i].destination.ID]
                                                           / mode.modeVelocity[0])
        riderList[i].maxRideTime_taxi = rider_data.loc[i, 'tb']

    return stationDict, riderList, dist, mode


'''按经纬度计算曼哈顿距离'''


def calDistance(long1, la1, long2, la2):
    R = 6317
    R2 = math.cos(math.pi * (la2 / 180)) * R
    dist = 2 * math.pi * R2 * abs(la2 - la1) / 360 + 2 * math.pi * R * abs(long2 - long1) / 360
    return dist


'''rider路径穷举预处理'''


def riderRouteSearch(dist, stationDict, riderList, mode, riderNum):
    # vehicle需要服务的route片段
    selectRider = []
    linkSet = np.empty(shape=[0, 7])  # ti,si,tj,sj,m,cnt,p
    count = 0
    for rider in riderList:
        begin_time = time()
        # 字典
        usedTimeS = {}
        # 首先划定范围，确定椭圆内station
        a = rider.maxRideTime_taxi * v_max * 0.01  # 按1公里-0.01度进行转化
        b = ((rider.origion.pointX - rider.destination.pointX) ** 2
             + (rider.origion.pointY - rider.destination.pointY) ** 2) ** 0.5
        x0 = (rider.origion.pointX + rider.destination.pointX) / 2
        y0 = (rider.origion.pointY + rider.destination.pointY) / 2
        for key in stationDict.keys():
            if ((stationDict[key].pointX - x0) ** 2) / (a ** 2) + ((stationDict[key].pointY - y0) ** 2) / (b ** 2) <= 1:
                rider.availableSList.append(stationDict[key])
                usedTimeS[stationDict[key].ID] = []
        timeWindow = rider.latestArriveTime - rider.earliestDepartTime
        if (len(rider.availableSList) >= 11 and timeWindow > 7) or (
                len(rider.availableSList) >= 15 and timeWindow > 5) or (
                len(rider.availableSList) >= 18 and timeWindow > 4) or (
                len(rider.availableSList) >= 21 and timeWindow > 3):
            continue
        print(len(rider.availableSList))

        # 开始扩展route
        routeList = []
        finalrouteList = None
        finalroutePrice = []
        riderLink = []  # 用于记录某link对于某rider是否已经重复过，以便对于link上经过的rider数进行唯一计次
        latestDepartTime = rider.latestArriveTime - (dist[rider.origion.ID][rider.destination.ID]
                                                     / mode.modeVelocity[0])
        # 对于从origin出发的路处理
        for s in rider.availableSList:
            if s != rider.origion:
                for t in range(int(rider.earliestDepartTime), int(latestDepartTime)):
                    for m in mode.ID:
                        dynaTime = dynamicTime(dist, rider.origion, s, mode.modeVelocity[m])
                        currTime = t + dynaTime
                        price = dynaTime * mode.modePrice[m]
                        if dynaTime <= rider.maxRideTime_taxi and currTime <= rider.latestArriveTime \
                                and price <= rider.maxPrice:
                            newroute = Route(dynaTime, 0, 0,0, m, price, price, (t + dynaTime, s))
                            newroute.nodeList.append((t, rider.origion.ID))
                            newroute.nodeList.append((t + dynaTime, s.ID))
                            newroute.isVisited.append(rider.origion.ID)
                            newroute.isVisited.append(s.ID)
                            if s == rider.destination:
                                linkSet, routeColumn = addLink(linkSet, newroute.nodeList, newroute.subRouteIndex
                                                               , [m], riderLink, dist, stationDict, mode)
                                finalrouteList = addFinalRoute(finalrouteList, routeColumn)
                                finalroutePrice.append(price)
                            else:
                                routeList.append(newroute)

        # 对于每条可行路扩展处理
        while len(routeList) > 0:
            oldroute = routeList[0]
            routeList.remove(oldroute)
            for s in rider.availableSList:
                if s.ID not in oldroute.isVisited:
                    for m in mode.ID:
                        # 计算可行条件
                        dynaTime = dynamicTime(dist, oldroute.currentNode[1], s, mode.modeVelocity[m])
                        usedTime = oldroute.usedTime + dynaTime
                        currTime = oldroute.currentNode[0] + dynaTime
                        price = oldroute.actualPrice + dynaTime * mode.modePrice[m]
                        if m != oldroute.currentMode:
                            transNum = oldroute.usedTransferNum + 1
                        else:
                            transNum = oldroute.usedTransferNum
                        if usedTime <= rider.maxRideTime_taxi and currTime <= rider.latestArriveTime \
                                and price <= rider.maxPrice and transNum <= rider.maxTransferNum:
                            newroute = Route(usedTime, transNum,0,0, m, price,price, (currTime, s))
                            routeCopy(newroute, oldroute)
                            if m != oldroute.currentMode:
                                if newroute.subRouteIndex[-1] != len(newroute.nodeList) - 1:
                                    newroute.subRouteMode.append(oldroute.currentMode)
                                    newroute.subRouteIndex.append(len(newroute.nodeList) - 1)
                            newroute.nodeList.append((currTime, s.ID))
                            newroute.isVisited.append(s.ID)
                            # 若达到终点，保存
                            if s == rider.destination:
                                newroute.subRouteMode.append(m)
                                linkSet, routeColumn = addLink(linkSet, newroute.nodeList, newroute.subRouteIndex
                                                               , newroute.subRouteMode, riderLink, dist, stationDict, mode)
                                finalrouteList = addFinalRoute(finalrouteList, routeColumn)
                                finalroutePrice.append(price)
                            else:
                                routeList.append(newroute)
        rider.routeList = finalrouteList
        rider.routePrice = finalroutePrice
        end_time = time()
        if rider.routeList is None:
            continue
        else:
            selectRider.append(rider.ID)
            print("rider" + str(count) + 'route' + str(rider.routeList.shape[1]) + "计算时间：" + str(end_time - begin_time))
            count += 1
            if count >= riderNum:
                break
    return linkSet, selectRider


'''拷贝route类'''


def routeCopy(newroute, oldroute):
    newroute.subRouteMode = deepcopy(oldroute.subRouteMode)
    newroute.subRouteIndex = deepcopy(oldroute.subRouteIndex)
    newroute.nodeList = deepcopy(oldroute.nodeList)
    newroute.isVisited = deepcopy(oldroute.isVisited)
    newroute.waited = oldroute.waited


'''不重复的添加link'''
def addLink(linkSet, newroute, mIndex, mList, riderLink, dist, stationDict, mode):
    routeColumn = np.zeros([len(newroute) - 1, 1])
    for i in range(len(newroute) - 1):
        m = mList[-1]
        for j in range(1, len(mIndex)):
            if i < mIndex[j]:
                m = mList[j - 1]

        p = dynamicTime(dist, stationDict[newroute[i][1]], stationDict[newroute[i + 1][1]]
                        , mode.modeVelocity[m]) * mode.modePrice[m]
        newLink = np.array([[newroute[i][0], newroute[i][1]
                                , newroute[i + 1][0], newroute[i + 1][1], m, 1, p]])  # # ti,si,tj,sj,m,cnt,p
        check = (linkSet[:, 0] == newLink[0, 0]) & (linkSet[:, 1] == newLink[0, 1]) \
                & (linkSet[:, 2] == newLink[0, 2]) & (linkSet[:, 3] == newLink[0, 3]) & (linkSet[:, 4] == newLink[0, 4])
        flag = any(check)
        if not flag:
            linkSet = np.append(linkSet, newLink, axis=0)
            routeColumn[i, 0] = linkSet.shape[0] - 1  # link最后一行
            riderLink.append(linkSet.shape[0] - 1)
        else:
            row = np.where(check == True)[0][0]
            routeColumn[i, 0] = row  # 找到对应的link行加入
            if row not in riderLink:  # 若该link之前尚未被该rider经过，则计次
                linkSet[row, 5] += 1
                riderLink.append(row)
    return linkSet, routeColumn


'''添加最终路径'''


def addFinalRoute(finalrouteList, routeColumn):
    if finalrouteList is None:  # 初始化
        finalrouteList = deepcopy(routeColumn)
    else:
        if finalrouteList.shape[0] < routeColumn.shape[0]:  # 若行数不相等，给左端补齐
            filler = -1 * np.ones(shape=[routeColumn.shape[0] - finalrouteList.shape[0],
                                         finalrouteList.shape[1]])
            finalrouteList = np.append(finalrouteList, filler, axis=0)
        elif finalrouteList.shape[0] > routeColumn.shape[0]:
            filler = -1 * np.ones(shape=[finalrouteList.shape[0] - routeColumn.shape[0],
                                         routeColumn.shape[1]])
            routeColumn = np.append(routeColumn, filler, axis=0)
        finalrouteList = np.append(finalrouteList, routeColumn, axis=1)
    return finalrouteList


'''计算运行时间'''


def dynamicTime(dist, s1, s2, v):
    if s1 != s2:
        time = max(int(dist.loc[s1.ID, s2.ID] / v), 1)
    else:
        time = 0
    return time


'''vehicle路径穷举'''

def vehicleRouteSearch(linkSet, mode, k):
    begin_time = time()
    vehicleRouteList = {}
    for m in mode.ID:
        vehicleRouteList[m] = None
        linkSetM = linkSet[np.where(linkSet[:, 4] == m)]
        linkIndex = np.where(linkSet[:, 4] == m)[0]
        if mode.minRiderNum[m] == 0:  # 不进行扩展，一个link为一条路
            for i in range(linkSetM.shape[0]):
                newVehicleRoute = np.array([[linkIndex[i]]])
                vehicleRouteList[m] = addFinalRoute(vehicleRouteList[m], newVehicleRoute)
        else:  # 进行扩展，但路线开通需符合最小乘客数限制
            linkSetM1 = linkSetM[np.where(linkSetM[:, 5] > mode.minRiderNum[m])]
            linkIndex1 = np.where(linkSetM[:, 5] > mode.minRiderNum[m])[0]
            linkFlag = np.zeros([linkSetM1.shape[0]], dtype=bool)
            for i in range(linkSetM1.shape[0]):
                # 若该link的头不是任何link的尾，则将是路径的开端
                if not linkFlag[i]:
                    linkFlag[i] = True
                    newVehicleRoute = np.array([[linkIndex[linkIndex1[i]]]])
                    tempList = [newVehicleRoute]
                    # 寻找link链接的下一条link
                    while len(tempList) > 0:
                        currRoute = tempList[0]
                        tempList.remove(currRoute)
                        tailLink = currRoute[-1]
                        # 筛选下一个link并整理index
                        subLinkIndex0 = np.where(linkSetM1[:, 0] == linkSet[tailLink, 2])[0]
                        subLinkIndex1 = np.where(linkSetM1[subLinkIndex0, 1] == linkSet[tailLink, 3])[0]
                        subLinkIndex = subLinkIndex0[subLinkIndex1]

                        if subLinkIndex.shape[0] == 0:
                            # 对该路的拓展结束
                            vehicleRouteList[m] = addFinalRoute(vehicleRouteList[m], currRoute)
                        else:
                            # 从link筛选前k个
                            subLinkSet = linkSetM1[subLinkIndex]
                            subLinkIndex0 = np.argsort(-subLinkSet[:, 5])
                            k = min(k, subLinkIndex0.shape[0])
                            subLinkIndex1 = subLinkIndex[subLinkIndex0[:k]]
                            for j in subLinkIndex1:
                                newVehicleRoute = deepcopy(currRoute)
                                newVehicleRoute = np.append(newVehicleRoute, [[linkIndex[linkIndex1[j]]]], axis=0)
                                linkFlag[j] = True
                                tempList.append(newVehicleRoute)
        print(vehicleRouteList[m].shape[1])
    end_time = time()
    print("vehicle计算时间：", end_time - begin_time)
    return vehicleRouteList


def vehicleRouteSearch0(linkSet, mode):
    begin_time = time()
    vehicleRouteList = {}
    for m in range(1, 2):
        vehicleRouteList[m] = None
        linkSetM = linkSet[np.where(linkSet[:, 4] == m)]
        linkIndex = np.where(linkSet[:, 4] == m)[0]
        for i in range(linkSetM.shape[0]):
            print("current: {} | total: {}".format(i, linkSetM.shape[0]))
            # 若该link的头不是任何link的尾，则将是路径的开端
            flag = any((linkSetM[:, 2] == linkSetM[i, 0]) & (linkSetM[:, 3] == linkSetM[i, 1]))
            if not flag:
                newVehicleRoute = np.array([[linkIndex[i]]])
                tempList = [newVehicleRoute]
                # 寻找link链接的下一条link
                while len(tempList) > 0:
                    currRoute = tempList[0]
                    tempList.remove(currRoute)
                    tailLink = currRoute[-1]
                    subLinkIndex0 = np.where(linkSetM[:, 0] == linkSet[tailLink, 2])[0]
                    subLinkIndex1 = np.where(linkSetM[subLinkIndex0, 1] == linkSet[tailLink, 3])[0]
                    subLinkIndex = subLinkIndex0[subLinkIndex1]
                    if subLinkIndex.shape[0] == 0:
                        # 对该路的拓展结束
                        vehicleRouteList[m] = addFinalRoute(vehicleRouteList[m], currRoute)
                    else:
                        for j in subLinkIndex:
                            newVehicleRoute = deepcopy(currRoute)
                            newVehicleRoute = np.append(newVehicleRoute, [[linkIndex[j]]], axis=0)
                            tempList.append(newVehicleRoute)
    end_time = time()
    print("vehicle计算时间：", end_time - begin_time)
    return vehicleRouteList


'''vehicle路径beam search'''


def vehicleBeamSearch(linkSet, mode, k):
    begin_time = time()
    vehicleRouteList = {}

    for m in mode.ID:
        vehicleRouteList[m] = None
        linkSetM = linkSet[np.where(linkSet[:, 4] == m)]
        linkIndex = np.where(linkSet[:, 4] == m)[0]
        linkFlag = np.zeros([linkSetM.shape[0]], dtype=bool)
        for i in range(linkSetM.shape[0]):
            # 若该link的头不是任何link的尾，则将是路径的开端
            if not linkFlag[i]:
                linkFlag[i] = True
                newVehicleRoute = np.array([[linkIndex[i]]])
                tempList = [newVehicleRoute]
                # 寻找link链接的下一条link
                while len(tempList) > 0:
                    currRoute = tempList[0]
                    tempList.remove(currRoute)
                    tailLink = currRoute[-1]
                    # 筛选下一个link并整理index
                    subLinkIndex0 = np.where(linkSetM[:, 0] == linkSet[tailLink, 2])[0]
                    subLinkIndex1 = np.where(linkSetM[subLinkIndex0, 1] == linkSet[tailLink, 3])[0]
                    subLinkIndex = subLinkIndex0[subLinkIndex1]

                    if subLinkIndex.shape[0] == 0:
                        # 对该路的拓展结束
                        vehicleRouteList[m] = addFinalRoute(vehicleRouteList[m], currRoute)
                    else:
                        # 从link筛选前k个
                        subLinkSet = linkSetM[subLinkIndex]
                        subLinkIndex0 = np.argsort(-subLinkSet[:, 5])
                        k = min(k, subLinkIndex0.shape[0])
                        subLinkIndex1 = subLinkIndex[subLinkIndex0[:k]]
                        for j in subLinkIndex1:
                            newVehicleRoute = deepcopy(currRoute)
                            newVehicleRoute = np.append(newVehicleRoute, [[linkIndex[j]]], axis=0)
                            linkFlag[j] = True
                            tempList.append(newVehicleRoute)

    end_time = time()
    print("vehicle计算时间：", end_time - begin_time)
    return vehicleRouteList


def findVehicleRoute(linkSet, linkSetM, linkIndex, currentLink, linkFlag, k):
    newVehicleRoute = np.array([[currentLink]])
    tempList = [newVehicleRoute]
    # 寻找link链接的下一条link
    while len(tempList) > 0:
        currRoute = tempList[0]
        tempList.remove(currRoute)
        tailLink = currRoute[-1]
        # 筛选下一个link并整理index
        subLinkIndex0 = np.where(linkSetM[:, 0] == linkSet[tailLink, 2])[0]
        subLinkIndex1 = np.where(linkSetM[subLinkIndex0, 1] == linkSet[tailLink, 3])[0]
        subLinkIndex = subLinkIndex0[subLinkIndex1]

        if subLinkIndex.shape[0] == 0:
            # 对该路的拓展结束
            vehicleRouteList = addFinalRoute(vehicleRouteList, currRoute)
        else:
            # 从link筛选前k个
            subLinkSet = linkSetM[subLinkIndex]
            subLinkIndex0 = np.argsort(-subLinkSet[:, 5])
            k = min(k, subLinkIndex0.shape[0])
            subLinkIndex1 = subLinkIndex[subLinkIndex0[:k]]
            for j in subLinkIndex1:
                newVehicleRoute = deepcopy(currRoute)
                newVehicleRoute = np.append(newVehicleRoute, [[linkIndex[j]]], axis=0)
                linkFlag[j] = True
                tempList.append(newVehicleRoute)
    return vehicleRouteList


def preProcess0(dist, stationDict, riderList, mode):
    # vehicle需要服务的route片段
    deleteRider = []
    linkSet = {}
    for m in mode.ID:
        linkSet[m] = np.empty(shape=[0, 4])

    count = 0
    for rider in riderList[:10000]:
        begin_time = time()
        # 字典
        usedTimeS = {}
        # 首先划定范围，确定椭圆内station
        a = rider.maxRideTime * v_max * 0.01  # 按1公里-0.01度进行转化
        b = ((rider.origion.pointX - rider.destination.pointX) ** 2
             + (rider.origion.pointY - rider.destination.pointY) ** 2) ** 0.5
        x0 = (rider.origion.pointX + rider.destination.pointX) / 2
        y0 = (rider.origion.pointY + rider.destination.pointY) / 2
        for key in stationDict.keys():
            if ((stationDict[key].pointX - x0) ** 2) / (a ** 2) + ((stationDict[key].pointY - y0) ** 2) / (b ** 2) <= 1:
                rider.availableSList.append(stationDict[key])
                usedTimeS[stationDict[key].ID] = []
        timeWindow = rider.latestArriveTime - rider.earliestDepartTime
        if (len(rider.availableSList) >= 11 and timeWindow > 7) or (len(rider.availableSList) >= 15 and timeWindow > 4) \
                or (len(rider.availableSList) >= 18 and timeWindow > 3):
            deleteRider.append(rider.ID)
            continue
        print(len(rider.availableSList))
        # 开始扩展route
        routeList = []
        finalrouteList = []

        latestDepartTime = rider.latestArriveTime - (dist[rider.origion.ID][rider.destination.ID]
                                                     / mode.modeVelocity[0])

        for s in rider.availableSList:
            if s != rider.origion:
                for t in range(int(rider.earliestDepartTime), int(latestDepartTime)):
                    for m in mode.ID:
                        dynaTime = dynamicTime(dist, rider.origion, s, mode.modeVelocity[m])
                        currTime = t + dynaTime
                        price = dynaTime * mode.modePrice[m]
                        if dynaTime <= rider.maxRideTime and currTime <= rider.latestArriveTime \
                                and price <= rider.maxPrice:
                            newroute = Route(dynaTime, 0, m, price, (t + dynaTime, s))
                            newroute.nodeList.append((t, rider.origion.ID))
                            newroute.nodeList.append((t + dynaTime, s.ID))
                            newroute.isVisited.append(rider.origion.ID)
                            newroute.isVisited.append(s.ID)
                            if s == rider.destination:
                                finalrouteList.append(newroute)
                                addLink(linkSet, newroute.nodeList, m)
                            else:
                                routeList.append(newroute)

        while len(routeList) > 0:
            oldroute = routeList[0]
            routeList.remove(oldroute)
            for s in rider.availableSList:
                # if s == oldroute.currentNode[1]: # 在原地等待
                #     if not oldroute.waited:
                #         dynaTime = dynamicTime(dist, oldroute.currentNode[1], s, mode.modeVelocity[m])
                #         usedTime = oldroute.usedTime + dynaTime
                #         currTime = oldroute.currentNode[0] + dynaTime
                #         if usedTime <= rider.maxRideTime and currTime <= rider.latestArriveTime:
                #             newroute = Route(usedTime, oldroute.usedTransferNum, oldroute.currentMode
                #                              , oldroute.usedCost, (currTime, s))
                #             routeCopy(newroute, oldroute)
                #             newroute.waited = True
                #             newroute.nodeList.append((currTime, s.ID))
                #             routeList.append(newroute)
                # el
                if s.ID not in oldroute.isVisited:
                    for m in mode.ID:
                        dynaTime = dynamicTime(dist, oldroute.currentNode[1], s, mode.modeVelocity[m])
                        usedTime = oldroute.usedTime + dynaTime
                        currTime = oldroute.currentNode[0] + dynaTime
                        price = oldroute.actualPrice + dynaTime * mode.modePrice[m]
                        if m != oldroute.currentMode:
                            if oldroute.subRouteIndex[-1] != len(oldroute.nodeList) - 1:
                                oldroute.subRouteMode.append(oldroute.currentMode)
                                oldroute.subRouteIndex.append(len(oldroute.nodeList) - 1)
                                transNum = oldroute.usedTransferNum + 1
                        else:
                            transNum = oldroute.usedTransferNum
                        if usedTime <= rider.maxRideTime and currTime <= rider.latestArriveTime \
                                and price <= rider.maxPrice and transNum <= rider.maxTransferNum:
                            newroute = Route(usedTime, transNum,0,0, m, price, price, (currTime, s))
                            routeCopy(newroute, oldroute)
                            newroute.nodeList.append((currTime, s.ID))
                            newroute.isVisited.append(s.ID)
                            if s == rider.destination:
                                finalrouteList.append(newroute)
                                for i in range(len(newroute.subRouteMode)):
                                    addLink(linkSet,
                                            newroute.nodeList[oldroute.subRouteIndex[i]:oldroute.subRouteIndex[i + 1]],
                                            newroute.subRouteMode[i])
                                addLink(linkSet, newroute.nodeList[oldroute.subRouteIndex[-1]:], m)
                            else:
                                routeList.append(newroute)
        rider.routeList = finalrouteList
        end_time = time()
        print("rider" + str(count) + 'route' + str(len(rider.routeList)) + "计算时间：" + str(end_time - begin_time))
        count += 1
        if len(rider.routeList) == 0:
            deleteRider.append(rider.ID)
    print("删掉rider数", len(deleteRider))

    # 处理车辆路径
    begin_time = time()
    vehicleRouteList = {}
    # for m in mode.ID:
    #     vehicleRouteList[m] = []
    #     linkSet[m] = np.unique(linkSet[m], axis=0)
    #     for i in range(linkSet[m].shape[0]):
    #         # 若该link的头不是任何link的尾，则将是路径的开端
    #         flag = any((linkSet[m][:, 2] == linkSet[m][i, 0])&(linkSet[m][:, 3] == linkSet[m][i, 1]))
    #         if not flag:
    #             newVehicleRoute = [(linkSet[m][i, 0], linkSet[m][i, 1]), (linkSet[m][i, 2], linkSet[m][i, 3])]
    #             tempList = [newVehicleRoute]
    #             # 寻找link链接的下一条link
    #             while len(tempList) > 0:
    #                 currRoute = tempList[0]
    #                 tempList.remove(currRoute)
    #                 subLinkList = linkSet[m][np.where(linkSet[m][:, 0] == currRoute[-1][0])]
    #                 subLinkList = subLinkList[np.where(subLinkList[:, 1] == currRoute[-1][1])]
    #                 if subLinkList.shape[0] == 0:
    #                     # 对该路的拓展结束
    #                     vehicleRouteList[m].append(currRoute)
    #                 else:
    #                     for j in range(subLinkList.shape[0]):
    #                         newVehicleRoute = deepcopy(currRoute)
    #                         newVehicleRoute.append((subLinkList[j, 2], subLinkList[j, 3]))
    #                         tempList.append(newVehicleRoute)
    #     print(len(linkSet[m]))
    # end_time = time()
    # print("vehicle计算时间：", end_time - begin_time)

    return vehicleRouteList, linkSet


'''添加子路径'''


def addSubRoute(routeToNode, routeFromNode, newroute, m):
    headNode = newroute[0]
    tailNode = newroute[-1]
    if headNode not in routeFromNode[m].keys():
        routeFromNode[m][headNode] = []
    if tailNode not in routeToNode[m].keys():
        routeToNode[m][tailNode] = []
    # 判断路是否存在
    for k in routeFromNode[m][headNode]:
        if len(newroute) != len(k):
            continue
        else:
            flag = True
            for i in range(len(k)):
                if k[i] != newroute[i]:
                    flag = False
            if flag:
                return
    routeFromNode[m][headNode].append(newroute)
    routeToNode[m][tailNode].append(newroute)


'''添加不重复的list到字典'''


def addDistict(newlist, list1, list2):
    for i in range(newlist.shape[0]):
        f1 = any((list1[:, 0] == newlist[i, 0]) & (list1[:, 1] == newlist[i, 1])
                 & (list1[:, 2] == newlist[i, 2]) & (list1[:, 3] == newlist[i, 3])
                 & (list1[:, 4] == newlist[i, 4]) & (list1[:, 5] == newlist[i, 5]))
        f2 = any((list2[:, 0] == newlist[i, 0]) & (list2[:, 1] == newlist[i, 1])
                 & (list2[:, 2] == newlist[i, 2]) & (list2[:, 3] == newlist[i, 3])
                 & (list2[:, 4] == newlist[i, 4]) & (list2[:, 5] == newlist[i, 5]))
        if not f1 and not f2:
            list1 = np.append(list1, [newlist[i, :]], axis=0)


def preProcess1(dist, stationDict, riderList, mode):
    deleteRider = []
    count = 0
    modeLinkSet = {}
    for m in mode.ID:
        modeLinkSet[m] = np.empty(shape=[0, 6])
    for rider in riderList[:10000]:
        # 字典
        timeS = {}
        usedTimeS = {}
        begin_time = time()
        # 首先划定范围，确定椭圆内station
        a = rider.maxRideTime * v_max * 0.01
        b = ((rider.origion.pointX - rider.destination.pointX) ** 2
             + (rider.origion.pointY - rider.destination.pointY) ** 2) ** 0.5
        x0 = (rider.origion.pointX + rider.destination.pointX) / 2
        y0 = (rider.origion.pointY + rider.destination.pointY) / 2
        for key in stationDict.keys():
            if ((stationDict[key].pointX - x0) ** 2) / (a ** 2) + ((stationDict[key].pointY - y0) ** 2) / (b ** 2) <= 1:
                rider.availableSList.append(stationDict[key])
                timeS[stationDict[key].ID] = []
                usedTimeS[stationDict[key].ID] = []

        timeWindow = rider.latestArriveTime - rider.earliestDepartTime
        if (len(rider.availableSList) >= 11 and timeWindow > 7) or (
                len(rider.availableSList) >= 15 and timeWindow > 6) or (
                len(rider.availableSList) >= 18 and timeWindow > 5) \
                or (len(rider.availableSList) >= 21 and timeWindow > 3):
            deleteRider.append(rider.ID)
            continue
        print(len(rider.availableSList))

        # 开始扩展link
        originLinkSet = np.empty(shape=[0, 6])
        actS = [rider.origion]
        latestDepartTime = rider.latestArriveTime - (dist[rider.origion.ID][rider.destination.ID]
                                                     / mode.modeVelocity[0])
        for i in range(int(rider.earliestDepartTime), int(latestDepartTime)):
            timeS[rider.origion.ID].append(i)
        while len(actS) > 0:
            s1 = actS[0]
            s1time = deepcopy(timeS[s1.ID])
            usedTimeS[s1.ID] = usedTimeS[s1.ID] + timeS[s1.ID][:]
            timeS[s1.ID] = []
            actS.remove(s1)
            for s2 in rider.availableSList:
                if s2 != s1:
                    flag = False  # 判断是否需要对s2进行再扩展
                    for t in s1time:
                        if t < rider.latestArriveTime:
                            for m in mode.ID:
                                dynamicT = dynamicTime(dist, s1, s2, mode.modeVelocity[m])
                                price = dynamicT * mode.modePrice[m]
                                newLink = np.array([[t, s1.ID, t + dynamicT, s2.ID, m, price]])
                                originLinkSet = np.append(originLinkSet, newLink, axis=0)
                                if newLink[0, 2] not in timeS[s2.ID] and newLink[0, 3] not in usedTimeS[s2.ID]:
                                    timeS[s2.ID].append(t + dynamicT)
                                    flag = True
                    if s2 not in actS and s2 != rider.destination and flag:
                        actS.append(s2)
                # else: # 在原地等待，一律使用mode-1
                #     flag = False
                #     for t in s1time:
                #         if t < rider.latestArriveTime:
                #             newLink = np.array([[t, s1.ID, t + 1, s2.ID, -1, 0]])
                #             originLinkSet = np.append(originLinkSet, newLink, axis=0)
                #             if newLink[0,2] not in timeS[s2.ID] and newLink[0,3] not in usedTimeS[s2.ID]:
                #                 timeS[s2.ID].append(t + 1)
                #                 flag = True
                #     if s2 not in actS and s2 != rider.destination and flag:
                #         actS.append(s2)

        # 回溯，找出可行的点和弧,建图
        tempSet = originLinkSet[np.where(originLinkSet[:, 3] == rider.destination.ID)]
        linkSet = tempSet[np.where(tempSet[:, 2] <= rider.latestArriveTime)]
        finalLinkSet = np.empty(shape=[0, 6])
        while linkSet.shape[0] > 0:
            currentL = linkSet[0, :]
            linkSet = np.delete(linkSet, 0, axis=0)
            finalLinkSet = np.append(finalLinkSet, [currentL], axis=0)
            tempSet = originLinkSet[np.where(originLinkSet[:, 2] == currentL[0])]
            tempSet = tempSet[np.where(tempSet[:, 3] == currentL[1])]
            addDistict(tempSet, linkSet, finalLinkSet)
        end_time = time()
        print("rider" + str(count) + 'route' + str(len(finalLinkSet)) + "计算时间：" + str(end_time - begin_time))
        count += 1
        if len(finalLinkSet) == 0:
            deleteRider.append(rider.ID)
        else:
            rider.linkSet = finalLinkSet
            for m in mode.ID:
                modeLinkSet[m] = np.append(modeLinkSet[m]
                                           , finalLinkSet[np.where(finalLinkSet[:, 4] == m)], axis=0)
    for m in mode.ID:
        modeLinkSet[m] = np.unique(modeLinkSet[m], axis=0)
        print(modeLinkSet[m].shape[0])
    print("删掉rider数", len(deleteRider))


def preProcess2(dist, stationDict, riderList):
    for rider in riderList:
        # Link字典
        linkToS = {}
        linkFromS = {}
        timeS = {}
        usedTimeS = {}

        # 首先划定范围，确定椭圆内station
        a = rider.maxRideTime * v_max
        b = ((rider.origion.pointX - rider.destination.pointX) ** 2
             + (rider.origion.pointY - rider.destination.pointY) ** 2) ** 0.5
        x0 = (rider.origion.pointX + rider.destination.pointX) / 2
        y0 = (rider.origion.pointY + rider.destination.pointY) / 2
        for key in stationDict.keys():
            if ((stationDict[key].pointX - x0) ** 2) / (a ** 2) + ((stationDict[key].pointY - y0) ** 2) / (b ** 2) <= 1:
                rider.availableSList.append(stationDict[key])
                linkFromS[stationDict[key].ID] = []
                linkToS[stationDict[key].ID] = []
                timeS[stationDict[key].ID] = []
                usedTimeS[stationDict[key].ID] = []

        # 开始扩展link
        actS = [rider.origion]
        latestDepartTime = rider.latestArriveTime - (dist[rider.origion.ID][rider.destination.ID] / v_max)
        for i in range(int(rider.earliestDepartTime), int(latestDepartTime)):
            timeS[rider.origion.ID].append(i)
        while len(actS) > 0:
            s1 = actS[0]
            s1time = deepcopy(timeS[s1.ID])
            usedTimeS[s1.ID] = usedTimeS[s1.ID] + timeS[s1.ID][:]
            timeS[s1.ID] = []
            actS.remove(s1)
            for s2 in rider.availableSList:
                # if s2 != s1:
                flag = False  # 判断是否需要对s2进行再扩展
                for t in s1time:
                    if t < rider.latestArriveTime:
                        newLink = Link(t, s1.ID, t + dynamicTime(dist, t, s1, s2), s2.ID)
                        linkFromS[s1.ID].append(newLink)
                        linkToS[s2.ID].append(newLink)
                        if newLink.tailT not in timeS[s2.ID] and newLink.tailT not in usedTimeS[s2.ID]:
                            timeS[s2.ID].append(t + dynamicTime(dist, t, s1, s2))
                            flag = True
                if s2 not in actS and s2 != rider.destination and flag == True:
                    actS.append(s2)

        # 回溯，保留可行的link
        linkSet = []

        for l in linkToS[rider.destination.ID]:
            if l.tailT <= rider.latestArriveTime:
                linkSet.append(l)
        while len(linkSet) > 0:
            currentL = linkSet[0]
            rider.linkSet.append(currentL)
            linkSet.remove(currentL)
            for l in linkToS[currentL.headS]:
                if currentL.headT == l.tailT and l not in linkSet:
                    linkSet.append(l)
