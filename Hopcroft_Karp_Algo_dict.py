from time import time

import numpy as np
import pandas as pd
from code0615 import generateInstance
import matplotlib.pyplot as plt


class HKAlgo:
    def __init__(self, NUM, bmap):
        self.NUM = NUM
        self.bmap = bmap
        self.cLeft = -1 * np.ones(shape=[NUM], dtype=int)  # 表示左集合i顶点所匹配的右集合的顶点序号
        self.cRight = -1 * np.ones(shape=[NUM], dtype=int)  # 表示右集合i顶点所匹配的左集合的顶点序号
        self.dLeft = -1 * np.ones(shape=[NUM], dtype=int)  # 是广度搜增广路径时候用来存 距离 左侧未匹配点的距离
        self.dRight = -1 * np.ones(shape=[NUM], dtype=int)
        self.dis = np.inf  # 为增广路径的长度
        self.bmask = np.zeros(shape=[NUM], dtype=bool)
        self.nLeft = NUM  # nx,ny分别是左集合点个数，右集合点个数
        self.nRight = NUM

    def search_path(self):
        Q = []
        self.dis = np.inf
        self.dLeft = -1 * np.ones(shape=[self.NUM], dtype=int)
        self.dRight = -1 * np.ones(shape=[self.NUM], dtype=int)
        for i in range(self.nLeft):
            if self.cLeft[i] == -1:
                Q.append(i)
                self.dLeft[i] = 0
        # 广度搜索
        while len(Q) > 0:
            u = Q[0]
            Q.remove(u)
            if self.dLeft[u] > self.dis:  # 有增广路径就结束
                break
            for v in range(self.nRight):
                if (u, v) in self.bmap and self.dRight[v] == -1:
                    self.dRight[v] = self.dLeft[u] + 1
                    if self.cRight[v] == -1:
                        self.dis = self.dRight[v]
                    else:
                        self.dLeft[self.cRight[v]] = self.dRight[v] + 1
                        Q.append(self.cRight[v])
        return self.dis != np.inf

    def find_path(self, u):
        for v in range(self.nRight):
            if not self.bmask[v] and (u, v) in self.bmap and self.dRight[v] == self.dLeft[u] + 1:
                self.bmask[v] = True
                if self.cRight[v] != -1 and self.dRight[v] == self.dis:  # 若点已被匹配且不在增广路径上（dis不对）
                    continue
                if self.cRight[v] == -1 or self.find_path(self.cRight[v]):
                    self.cRight[v] = u
                    self.cLeft[u] = v
                    return True
        return False

    def MaxMatch(self):
        res = 0
        self.cLeft = -1 * np.ones(shape=[self.NUM], dtype=int)  # 表示左集合i顶点所匹配的右集合的顶点序号
        self.cRight = -1 * np.ones(shape=[self.NUM], dtype=int)  # 表示右集合i顶点所匹配的左集合的顶点序号
        while self.search_path():
            self.bmask = np.zeros(shape=[self.NUM], dtype=bool)
            for i in range(self.nLeft):
                if self.cLeft[i] == -1:
                    res += int(self.find_path(i))
        return res  # 返回的就是最大匹配数


def fleet_no_sharing(riderArray, linkSet, timeDist, station_data, emptyTime):
    # 每个顾客都
    linkCount = []
    selectedLink = np.empty(shape=[0, 6])
    for i in range(riderArray.shape[1]):
        firstLink = int(riderArray[2, i])
        s1 = linkSet[firstLink, 1]
        t1 = linkSet[firstLink, 0]
        s2 = 0
        for j in range(2, riderArray.shape[0]):
            if riderArray[j, i] == -1:
                break
            s2 = int(linkSet[int(riderArray[j, i]), 3])
        t2 = t1 + timeDist[2][int(s1), int(s2)]
        selectedLink = np.append(selectedLink, [[t1, s1, t2, s2, 2, 1]], axis=0)

    mapIndex = []
    for i in range(selectedLink.shape[0]):
        mapIndex.append(selectedLink[:i, 5].sum())
    print("finish prepare link")
    # 最小车辆数计算调度
    begin_time = time()
    NUM = int(selectedLink[:, 5].sum())
    bmap = {}
    # 求序列和，即index
    # 初始化各个link之间的连接关系
    for i in range(selectedLink.shape[0]):
        # print("current rider:", i)
        for j in range(selectedLink.shape[0]):
            runTime = timeDist[2][int(selectedLink[i, 3]), int(selectedLink[j, 1])]
            depart_i = selectedLink[i, 2]
            arrive_j = selectedLink[j, 0]
            # 如果调度在10分钟以内，且满足时间可行性，则更新bmap为true
            if arrive_j - depart_i <= emptyTime and arrive_j >= depart_i + runTime:
                # 找到i占据的行数
                bmap[i, j] = True
    end_time = time()
    print("邻接表计算时间：" + str(end_time - begin_time))

    # 计算匹配数
    begin_time = time()
    hk = HKAlgo(NUM, bmap)
    matchNum = hk.MaxMatch()
    print(matchNum)
    count = NUM - matchNum

    end_time = time()
    print("计算时间：" + str(end_time - begin_time))
    CPUTime = end_time - begin_time

    routeList, emptyDistance, travelDistance, travelTime = generateRoute(mapIndex, selectedLink, hk, station_data)

    print("总使用数: {} | emptyDis = {}  | travelDis = {} | travelTime = {}".format(count, emptyDistance,
                                                                                travelDistance, travelTime))
    return CPUTime, count, emptyDistance, travelDistance, travelTime

def fleet_no_transfer(riderArray, linkSet, timeDist, station_data, emptyTime):
    # 每个顾客都
    linkCount = []
    selectedLink = np.empty(shape=[0, 6])
    for i in range(riderArray.shape[1]):
        firstLink = int(riderArray[2, i])
        s1 = linkSet[firstLink, 1]
        t1 = linkSet[firstLink, 0]
        s2 = 0
        for j in range(2, riderArray.shape[0]):
            if riderArray[j, i] == -1:
                break
            s2 = linkSet[int(riderArray[j, i]), 3]
        t2 = t1 + timeDist[2][int(s1), int(s2)]

        check = (selectedLink[:, 0] == t1) & (selectedLink[:, 1] == s1) \
                & (selectedLink[:, 2] == t2) & (selectedLink[:, 3] == s2)
        flag = any(check)

        if not flag:
            selectedLink = np.append(selectedLink, [[t1, s1, t2, s2, 2, 1]], axis=0)
        else:
            row = np.where(check == True)[0][0]
            selectedLink[row, 5] += 1

    capacityList = [-1, 20, 2]
    for i in range(selectedLink.shape[0]):
        m = int(selectedLink[i, 4])
        selectedLink[i, 5] = int(selectedLink[i, 5] / capacityList[m]) + 1
    print("finish prepare link")

    # 最小车辆数计算调度
    begin_time = time()
    NUM = int(selectedLink[:, 5].sum())
    bmap = {}
    # 求序列和，即index
    mapIndex = []
    for i in range(selectedLink.shape[0]):
        mapIndex.append(selectedLink[:i, 5].sum())

    # 求序列和，即index
    # 初始化各个link之间的连接关系
    for i in range(selectedLink.shape[0]):
        # print("current rider:", i)
        for j in range(selectedLink.shape[0]):
            runTime = timeDist[2][int(selectedLink[i, 3]), int(selectedLink[j, 1])]
            depart_i = selectedLink[i, 2]
            arrive_j = selectedLink[j, 0]
            # 如果调度在15分钟以内，且满足时间可行性，则更新bmap为true
            if arrive_j - depart_i <= emptyTime and arrive_j >= depart_i + runTime:
                # 找到i占据的行数
                minrow = int(mapIndex[i])
                maxrow = int(minrow + selectedLink[i, 5])
                mincol = int(mapIndex[j])
                maxcol = int(mincol + selectedLink[j, 5])
                for a in range(minrow, maxrow):
                    for b in range(mincol, maxcol):
                        bmap[a, b] = True

    end_time = time()
    print("邻接表计算时间：" + str(end_time - begin_time))

    # 计算匹配数
    begin_time = time()
    hk = HKAlgo(NUM, bmap)
    matchNum = hk.MaxMatch()
    print(matchNum)
    count = NUM - matchNum

    end_time = time()
    print("计算时间：" + str(end_time - begin_time))
    CPUTime = end_time - begin_time

    routeList, emptyDistance, travelDistance, travelTime = generateRoute(mapIndex, selectedLink, hk, station_data)

    print("总使用数: {} | emptyDis = {}  | travelDis = {} | travelTime = {}".format(count, emptyDistance,
                                                                                travelDistance, travelTime))

    return CPUTime, count, emptyDistance, travelDistance, travelTime

def fleet_maas(riderArray, linkSet, timeDist, station_data, emptyTime):  # flag代表是否需要画图
    linkIndex = []
    linkCount = []
    for i in range(riderArray.shape[1]):
        for j in range(2, riderArray.shape[0]):  # 从第三行开始为link index
            if riderArray[j, i] == -1:
                break
            if riderArray[j, i] not in linkIndex:
                linkIndex.append(int(riderArray[j, i]))
                linkCount.append(1)
            else:
                p = linkIndex.index(int(riderArray[j, i]))
                linkCount[p] += 1

    selectedLink = linkSet[linkIndex, 0:6]
    # 把第五列作为计数列

    capacityList = [-1, 20, 2]
    for i in range(selectedLink.shape[0]):
        m = int(selectedLink[i, 4])
        selectedLink[i, 5] = int(linkCount[i] / capacityList[m]) + 1

    totalEmptyDistance = 0
    totalTravelDistance = 0
    totalTravelTime = 0
    totalCPUTime = 0
    # 分模式进行最小车辆数计算调度
    count = 0
    countM = {}
    for m in range(3):
        if m == 0:  # 地铁不计算
            continue
        begin_time = time()
        linkM = selectedLink[np.where(selectedLink[:, 4] == m)]
        NUM = int(linkM[:, 5].sum())
        bmap = {}
        # 求序列和，即index
        mapIndex = []
        for i in range(linkM.shape[0]):
            mapIndex.append(linkM[:i, 5].sum())
        # 初始化各个link之间的连接关系
        for i in range(linkM.shape[0]):
            # print("current rider:", i)
            for j in range(linkM.shape[0]):
                runTime = timeDist[m][int(linkM[i, 3]), int(linkM[j, 1])]
                depart_i = linkM[i, 2]
                arrive_j = linkM[j, 0]
                # 如果调度在15分钟以内，且满足时间可行性，则更新bmap为true
                # 更改空载时间的容忍度进行测试
                if arrive_j - depart_i <= emptyTime and arrive_j >= depart_i + runTime:
                    # 找到i占据的行数
                    minrow = int(mapIndex[i])
                    maxrow = int(minrow + linkM[i, 5])
                    mincol = int(mapIndex[j])
                    maxcol = int(mincol + linkM[j, 5])
                    for a in range(minrow, maxrow):
                        for b in range(mincol, maxcol):
                            bmap[a, b] = True
        end_time = time()
        print("邻接表计算时间：" + str(end_time - begin_time))

        # 计算匹配数
        begin_time = time()
        hk = HKAlgo(NUM, bmap)
        matchNum = hk.MaxMatch()
        print("车型"+str(m)+"使用数:" + str(NUM - matchNum))
        countM[m] = NUM - matchNum
        count += NUM - matchNum

        end_time = time()
        print("计算时间：" + str(end_time - begin_time))
        totalCPUTime += end_time - begin_time

        routeList, emptyDistance, travelDistance, travelTime = generateRoute(mapIndex, linkM, hk, station_data)
        totalEmptyDistance += emptyDistance
        totalTravelDistance += travelDistance
        totalTravelTime += travelTime

    print("总使用数: {} | emptyDis = {}  | travelDis = {} | travelTime = {}".format(count, totalEmptyDistance, totalTravelDistance, totalTravelTime))

    return totalCPUTime, count, countM, totalEmptyDistance, totalTravelDistance, totalTravelTime


def findOriginLink(mapIndex, index):
    count = 0
    for i in range(1, len(mapIndex)):
        if index < mapIndex[i]:
            break
        count += 1
    return count


def generateRoute(mapIndex, linkSet, hk, station_data):
    routeDict = {}
    routeList = {}
    usedNode = []
    for i in range(hk.nLeft):  # 遍历左边点
        if i not in usedNode:
            usedNode.append(i)
            if mapIndex is not None:
                link = findOriginLink(mapIndex, i)
            else:
                link = i
            newRoute = [link]

            currenti = hk.cLeft[i]
            while currenti != -1:
                if currenti not in routeDict.keys():
                    usedNode.append(currenti)
                    if mapIndex is not None:
                        link = findOriginLink(mapIndex, currenti)
                    else:
                        link = currenti
                    newRoute.append(link)
                    currenti = hk.cLeft[currenti]
                else:
                    newRoute = newRoute + routeDict[currenti]
                    routeDict.pop(currenti, None)
                    break
            routeDict[i] = newRoute

    emptyDistance = 0
    travelDistance = 0
    travelTime = 0
    # plt.figure(1)
    for k in routeDict.keys():
        lastTime = None
        lastStation = None
        routeList[k] = []
        for l in routeDict[k]:
            if linkSet[l, 4] == 0:
                color = 'orange'
            elif linkSet[l, 4] == 1:
                color = 'green'
            else:
                color = 'purple'
            station1 = station_data[station_data["station_id"] == linkSet[l, 1]].index.tolist()[0]
            station2 = station_data[station_data["station_id"] == linkSet[l, 3]].index.tolist()[0]
            if lastTime is not None: # 空乘
                # plt.plot((lastTime, linkSet[l, 0]), (lastStation, station1), color, linestyle='--')
                emptyDistance += generateInstance.calDistance(station_data['pointx'][lastStation], station_data['pointy'][lastStation]
                                             ,station_data['pointx'][station1], station_data['pointy'][station1])
            # plt.plot((linkSet[l, 0], linkSet[l, 2]), (station1, station2), color)
            travelDistance += generateInstance.calDistance(station_data['pointx'][station1], station_data['pointy'][station1]
                                         , station_data['pointx'][station2], station_data['pointy'][station2])
            travelTime += 5 * abs(linkSet[l, 2] - linkSet[l, 0])
            routeList[k].append(station1)
            routeList[k].append(station2)
            lastTime = linkSet[l, 2]
            lastStation = station2
    # plt.show()

    print("emptyDis = {}  | travelDis = {} | travelTime = {}".format(emptyDistance, travelDistance, travelTime))
    return routeList, emptyDistance, travelDistance, travelTime

