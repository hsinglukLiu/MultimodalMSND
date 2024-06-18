from time import time

import pandas as pd
import numpy as np
from gurobipy import *
import matplotlib.pyplot as plt

def addRouteResource(routeResource, r, l):
    if r not in routeResource:
        routeResource[r] = []
    m = l[4]
    for t in range(l[0], l[2]):
        routeResource[r].append((m,t))



if __name__ == '__main__':
    # 5000-1-[1, 30, 5]: 1.522140161428e+04 r:4705

    riderNum = 5000
    instanceIndex = 3
    case = 3
    riderIndex = pd.read_csv(
        'code0615/Instance/A_' + str(riderNum) + '_' + str(instanceIndex) + '_riderIndex_case' + str(case) + '.csv',
        index_col=[0]).values
    linkSet = pd.read_csv('code0615/Instance/A_' + str(riderNum) + '_' + str(instanceIndex) + '_linkSet.csv',
                          header=None).values
    routeArray = pd.read_csv(
        'code0615/Instance/A_' + str(riderNum) + '_' + str(instanceIndex) + '_riderRoute_case' + str(case) + '.csv',
        header=None).values

    capacityList = [1, 20, 2]
    Nm = [1, 30, 5]  # 较紧资源 for Pi
    # Nm = [30, 10, 5]  # 按时间进行

    minT = np.min(linkSet[:, 0])  # 整个问题时间窗范围
    maxT = np.max(linkSet[:, 2])
    lamb = np.zeros(shape=[linkSet.shape[0]])

    MP = Model('MP')

    #每条路占用资源的数目
    routeResource = {}

    x_rk = {}
    x_rk[0] = MP.addVar(obj=routeArray[0, 0], vtype=GRB.INTEGER, ub=1, name='r_' +str(0)+'_'+str(0))
    # # 定义约束1
    # capacityCons = {}
    # for i in range(linkSet.shape[0]):
    #     expr = LinExpr()
    #     flag = any(routeArray[1:, 0] == i)
    #     if flag:
    #         expr.addTerms(1, x_rk[0])
    #         # addRouteResource(routeResource, 0, linkSet[i, :])
    #     else:
    #         expr.addTerms(0, x_rk[0])
    #     capacityCons[i] = MP.addConstr(expr <= 0)

    # 定义约束2
    riderCons = {}
    riderArray = np.zeros(shape=[riderIndex.shape[0], 1])
    riderArray[0, 0] = 1
    for i in range(riderIndex.shape[0]):
        expr = LinExpr()
        expr.addTerms(riderArray[i, 0], x_rk[0])
        riderCons[i] = MP.addConstr(expr <= 1)

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
        r = 0
        for j in range(riderIndex.shape[0]):
            if i < riderIndex[j, 0]:
                r = j
                col.addTerms(1, riderCons[r])
                break
        # 关于linkset
        price = routeArray[0, i]
        # for k in range(1, routeArray.shape[0]):
        #     if routeArray[k, i] == -1:
        #         break
        #     col.addTerms(1, capacityCons[routeArray[k, i]])
        #     addRouteResource(routeResource, i, linkSet[routeArray[k, i], :])
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
        object = 0
        # 关于linkset
        # col.addTerms(-capacityList[m], capacityCons[i])
        x_vk[i] = MP.addVar(obj=-object, vtype=GRB.INTEGER, column=col, name='v_' + str(m) + '_' + str(i))

    # 定义优化方向
    start_time = time()
    MP.setAttr('ModelSense', GRB.MAXIMIZE)
    # MP.setParam('MIPGap', 0.005)
    MP.write('master.lp')
    MP.setParam('OutputFlag', 0)

    beta = 2
    UB = 1e6
    LB = 15382.12978506
    countLB = 0 #下界为提高的次数
    while abs(UB - LB)/LB > 1e-4:
        MP.optimize()
        print("MP is solved")

        tmpUB = MP.objVal
        if tmpUB < UB:
            UB = tmpUB  # 更新下界

        # # 判断是否违反约束
        # useRider = {}  # 记录使用的顾客,以link为index
        # riderPrice = {}  # 记录顾客所对应的价格
        # useResource = {} # 记录每个截面经过的link
        #
        # tmpLB = 0
        # for i in x_rk.keys():
        #     if x_rk[i].x > 1e-4:
        #         a = x_rk[i].varName.split('_')
        #         rider = int(a[1])
        #         riderPrice[rider] = routeArray[0, i]
        #         tmpLB += routeArray[0, i]
        #         for k in range(1, routeArray.shape[0]):
        #             if routeArray[k, i] == -1:
        #                 break
        #             if routeArray[k, i] not in useRider.keys():
        #                 useRider[routeArray[k, i]] = []
        #             useRider[routeArray[k, i]].append(rider)
        #
        #             m = linkSet[routeArray[k, i], 4]
        #             for t in range(linkSet[routeArray[k, i], 0], linkSet[routeArray[k, i], 2]):
        #                 if (m, t) not in useResource.keys():
        #                     useResource[m, t] = []
        #                 if routeArray[k, i] not in useResource[m, t]:
        #                     useResource[m, t].append(routeArray[k, i])
        # tmpLB = 1.522140161428e+04
        # # 判断下界是否无变化，若4次无变化改变beta
        # if abs(LB - tmpLB) <= 1e-4 or LB > tmpLB:
        #     countLB += 1
        # else:
        #     countLB = 0
        # if countLB >= 4:
        #     beta = 0.85 * beta
        #     countLB = 0
        # if tmpLB > LB:
        #     LB = tmpLB  # 更新下界
        print('UB:' + str(UB) + ' | LB:' + str(LB) + ' | gap:' + str((UB-LB)/LB) + ' | beta:' + str(beta))

        # 计算每个乘子的梯度
        gradient = np.ones(shape=[linkSet.shape[0]])

        for i in x_vk.keys():
            m = int(linkSet[i, 4])
            if x_vk[i].x > 1e-4:
                gradient[i] += x_vk[i].x * capacityList[m]

        for i in x_rk.keys():
            for k in range(1, routeArray.shape[0]):
                if routeArray[k, i] == -1:
                    break
                gradient[int(routeArray[k, i])] -= 1

        sumGradient = 0
        for i in range(gradient.shape[0]):
            sumGradient += gradient[i]**2

        # 计算步长
        z = beta * (UB - LB) / sumGradient

        # 更新拉格朗日乘子
        for i in range(lamb.shape[0]):
            lamb[i] = max(0, lamb[i] - z * gradient[i])


        # 更新模型
        for i in x_rk.keys():
            price = routeArray[0, i]
            for k in range(1, routeArray.shape[0]):
                if routeArray[k, i] == -1:
                    break
                price -= lamb[int(routeArray[k, i])]
            x_rk[i].setAttr('obj', price)

        for i in x_vk.keys():
            m = int(linkSet[i, 4])
            x_vk[i].setAttr('obj', lamb[i] * capacityList[m])

    end_time = time()


    # #输出结果
    # riderResult = np.empty(shape=[routeArray.shape[0] + 1, 0])  # rider + link
    # linkResult = np.empty(shape=[0, 1])  # 数量 + mode + link
    # count1 = 0
    # for var in MP.getVars():
    #     if var.x > 0:
    #         a = var.varName.split('_')
    #         type = a[0]
    #         index0 = int(a[1])
    #         index1 = int(a[2])
    #         if type == 'r':
    #             count1 += 1
    #             riderCol = np.append([index0], routeArray[:, index1], axis=0)
    #             riderResult = np.append(riderResult, riderCol.reshape(riderResult.shape[0], 1), axis=1)
    # print("求解时间：" + str(end_time-start_time))
    # print("rider:", count1)
    # np.savetxt('result/data2_riderResult_v3.csv', riderResult, delimiter=',')

    np.savetxt('result/A_' + str(riderNum) + '_' + str(instanceIndex) + '_PiValue.csv', lamb, delimiter=',')

    # print(MP.getAttr("Pi", modeCons))