from time import time

import pandas as pd
import numpy as np
from gurobipy import *
import matplotlib.pyplot as plt

riderNum = 10000
capacityList = [1, 20, 2]
# Nm = [1, 0, 500]  # 按时间进行
# Nm = [1, 8, 10]  # 小算例资源
# Nm = [1, 40, 10]  # 小算例资源 5000
Nm = [1, 55, 10]  # 小算例资源 10000
# Nm = [1, 80, 15]  # 小算例资源 15000
# Nm = [1, 85, 18]  # 小算例资源 20000
# Nm = [1, 90, 18]  # 小算例资源 30000
# Nm = [1, 95, 20]  # 小算例资源 50000
# Nm = [1, 30, 5]  # 较紧资源 for Pi

instance_num = 1
caseList = [3, 4]

for case in caseList:
    CPU_time_masterProblem = pd.DataFrame(np.zeros([instance_num, 6]))
    if case == 4:
        Nm = [1, 0, 1000]
    for instanceIndex in range(1, instance_num + 1):

        print('------------ instance ID: {} ------------'.format(instanceIndex))
        # price = 0.3
        # riderIndex = pd.read_csv('code0615/taxiPrice='+str(price)+'/A_'+str(riderNum) + '_' + str(instanceIndex) + '_riderIndex_case' + str(case) + '_type_1.csv', index_col=[0]).values
        # linkSet = pd.read_csv('code0615/taxiPrice='+str(price)+'/A_'+str(riderNum) + '_' + str(instanceIndex) + '_linkSet.csv', header=None).values
        # routeArray = pd.read_csv('code0615/taxiPrice='+str(price)+'/A_'+str(riderNum) + '_' + str(instanceIndex) + '_riderRoute_case' + str(case) + '_type_1.csv', header=None).values

        riderIndex = pd.read_csv('code0615/Instance/A_'+str(riderNum) + '_' + str(instanceIndex) + '_riderIndex_case' + str(case) + '.csv', index_col=[0]).values
        linkSet = pd.read_csv('code0615/Instance/A_'+str(riderNum) + '_' + str(instanceIndex) + '_linkSet.csv', header=None).values
        routeArray = pd.read_csv('code0615/Instance/A_'+str(riderNum) + '_' + str(instanceIndex) + '_riderRoute_case' + str(case) + '.csv', header=None).values

        MP = Model('MP')

        x_rk = {}

        x_rk[0] = MP.addVar(obj=routeArray[0, 0], vtype=GRB.INTEGER, ub=1, name='r_' +str(0)+'_'+str(0))
        # 定义约束1
        capacityCons = {}
        for i in range(linkSet.shape[0]):
            expr = LinExpr()
            flag = any(routeArray[1:, 0] == i)
            if flag:
                expr.addTerms(1, x_rk[0])
            else:
                expr.addTerms(0, x_rk[0])
            capacityCons[i] = MP.addConstr(expr <= 0)

        # 定义约束2
        riderCons = {}
        riderArray = np.zeros(shape=[riderIndex.shape[0], 1])
        riderArray[0, 0] = 1
        for i in range(riderIndex.shape[0]):
            expr = LinExpr()
            expr.addTerms(riderArray[i, 0], x_rk[0])
            riderCons[i] = MP.addConstr(expr <= 1)

        # # 定义约束3：mode

        modeCons = {}
        minT = int(np.min(linkSet[:, 0]))  # 整个问题时间窗范围
        maxT = int(np.max(linkSet[:, 2]))
        for i in range(minT, maxT):
            for j in range(1, 3):
                expr = LinExpr()
                expr.addTerms(0, x_rk[0])
                modeCons[i, j] = MP.addConstr(expr <= Nm[j])

        # 加列
        for i in range(1, routeArray.shape[1]):
            col = Column()
            # 先确定是哪一个rider
            for j in range(riderIndex.shape[0]):
                if i < riderIndex[j, 0]:
                    col.addTerms(1, riderCons[j])
                    break
            # 关于linkset
            for k in range(1, routeArray.shape[0]):
                if routeArray[k, i] == -1:
                    break
                col.addTerms(1, capacityCons[routeArray[k, i]])
                price = routeArray[0, i]
            x_rk[i] = MP.addVar(obj=price, vtype=GRB.INTEGER, column=col, ub=1, name='r_'+str(j)+'_'+str(i))

        # 加link列
        x_vk = {}

        for i in range(linkSet.shape[0]):
            col = Column()
            # mode
            m = int(linkSet[i, 4])
            for t in range(int(linkSet[i, 0]), int(linkSet[i, 2])):
                if m == 0:  # 地铁不搞
                    continue
                col.addTerms(1, modeCons[t, m])
            # 关于linkset
            col.addTerms(-capacityList[m], capacityCons[i])
            # ub = math.ceil(linkSet[i, 5] / capacityList[m])
            x_vk[i] = MP.addVar(obj=0, vtype=GRB.INTEGER, column=col, name='v_' + str(m) + '_' + str(i))  #, ub=ub


        # 定义优化方向
        start_time = time()
        MP.setAttr('ModelSense', GRB.MAXIMIZE)
        # MP.setParam('MIPGap', 0.02)

        MP.write('master.lp')
        MP.optimize()
        end_time = time()

        #输出结果
        riderResult = np.empty(shape=[routeArray.shape[0] + 1, 0])  # rider + link
        linkNum = np.zeros(shape=[linkSet.shape[0]])  # 数量
        count1 = 0
        for var in MP.getVars():
            if var.x > 0.5:
                a = var.varName.split('_')
                type = a[0]
                index0 = int(a[1])
                index1 = int(a[2])
                # print(str(var.varName)+"  "+str(var.x))
                if type == 'r':
                    count1 += 1
                    riderCol = np.append([index0], routeArray[:, index1], axis=0)
                    riderResult = np.append(riderResult, riderCol.reshape(riderResult.shape[0], 1), axis=1)
                    for k in range(1, routeArray.shape[0]):
                        if routeArray[k, index1] == -1:
                            break
                        linkNum[int(routeArray[k, index1])] += 1

        # 计算占座率
        occupancyRate = []
        for i in range(linkSet.shape[0]):
            if linkNum[i] != 0:
                m = int(linkSet[i, 4])
                rate = linkNum[i] / (math.ceil(linkNum[i] / capacityList[m]) * capacityList[m])
                occupancyRate.append(rate)

        meanVal = np.mean(occupancyRate)
        stdVal = np.std(occupancyRate, ddof=1)

        print("求解时间："+ str(end_time-start_time))
        print("rider: {} | meanVal: {} | stdVal: {}".format(count1, meanVal, stdVal))
        np.savetxt('code0615/Instance/A_'+str(riderNum) + '_' + str(instanceIndex) + '_riderResult_case' + str(case) + '.csv', riderResult, delimiter=',')

        CPU_time_masterProblem.iloc[instanceIndex - 1, 0] = end_time - start_time  # CPU time
        CPU_time_masterProblem.iloc[instanceIndex - 1, 1] = MP.objVal  # obj
        CPU_time_masterProblem.iloc[instanceIndex - 1, 2] = count1  # riderNum
        CPU_time_masterProblem.iloc[instanceIndex - 1, 3] = meanVal  # occupancy meanVal
        CPU_time_masterProblem.iloc[instanceIndex - 1, 4] = stdVal  # occupancy stdVal
        CPU_time_masterProblem.iloc[instanceIndex - 1, 5] = MP.getParamInfo("MIPGap")[2]  # gap

    CPU_time_masterProblem.columns = ['CPUTime', 'OBJ', 'riderNum', 'meanVal', 'stdVal', 'GAP']
    CPU_time_masterProblem.to_csv('result/A_' + str(riderNum) + '_master_CPU_time_case' + str(case) + '.csv', encoding='utf-8')