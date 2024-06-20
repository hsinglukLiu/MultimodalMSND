from time import time

import pandas as pd
import numpy as np
from gurobipy import *
import matplotlib.pyplot as plt

# riderIndex = pd.read_csv('data_set/data2_riderIndex.csv',index_col=[0]).values
# linkSet = pd.read_csv('data_set/data2_link.csv', header=None, dtype=int).values
# routeArray = pd.read_csv('data_set/data2_riderRoute.csv', header=None, dtype=int).values
riderIndex = pd.read_csv('data_set/smalldata4_riderIndex.csv',index_col=[0]).values
linkSet = pd.read_csv('data_set/smalldata4_link.csv', header=None, dtype=int).values
routeArray = pd.read_csv('data_set/smalldata4_riderRoute.csv', header=None).values
# Pi = {(0, 0): 0.0, (0, 1): 12.0, (0, 2): 0.0, (1, 0): 8.0, (1, 1): 24.0, (1, 2): 30.0, (2, 0): 8.0, (2, 1): 24.0, (2, 2): 30.0, (3, 0): 8.0, (3, 1): 24.0, (3, 2): 30.0, (4, 0): 7.9999999999999964, (4, 1): 24.0, (4, 2): 29.999999999999986, (5, 0): 8.0, (5, 1): 24.0, (5, 2): 30.000000000000014, (6, 0): 8.0, (6, 1): 24.0, (6, 2): 30.0, (7, 0): 8.0, (7, 1): 24.0, (7, 2): 30.000000000000004, (8, 0): 8.0, (8, 1): 24.0, (8, 2): 30.000000000000004, (9, 0): 8.0, (9, 1): 24.0, (9, 2): 30.0, (10, 0): 8.000000000000002, (10, 1): 24.0, (10, 2): 30.000000000000007, (11, 0): 7.999999999999995, (11, 1): 24.0, (11, 2): 29.99999999999998, (12, 0): 8.000000000000004, (12, 1): 23.999999999999993, (12, 2): 30.0, (13, 0): 8.000000000000005, (13, 1): 24.00000000000002, (13, 2): 30.0, (14, 0): 8.0, (14, 1): 23.99999999999998, (14, 2): 30.0, (15, 0): 8.0, (15, 1): 24.0, (15, 2): 30.0, (16, 0): 8.0, (16, 1): 24.0, (16, 2): 30.0, (17, 0): 8.0, (17, 1): 24.0, (17, 2): 30.0, (18, 0): 8.0, (18, 1): 24.0, (18, 2): 30.0, (19, 0): 8.0, (19, 1): 24.0, (19, 2): 30.0, (20, 0): 8.0, (20, 1): 24.0, (20, 2): 30.0, (21, 0): 8.0, (21, 1): 24.0, (21, 2): 30.0, (22, 0): 8.0, (22, 1): 12.0, (22, 2): 30.0, (23, 0): 0.0, (23, 1): -0.0, (23, 2): -0.0, (24, 0): -0.0, (24, 1): 0.0, (24, 2): -0.0, (25, 0): -0.0, (25, 1): -0.0, (25, 2): -0.0, (26, 0): 0.0, (26, 1): -0.0, (26, 2): 0.0, (27, 0): 0.0, (27, 1): -0.0, (27, 2): -0.0, (28, 0): -0.0, (28, 1): -0.0, (28, 2): -0.0, (29, 0): 0.0, (29, 1): -0.0, (29, 2): -0.0, (30, 0): 0.0, (30, 1): -0.0, (30, 2): -0.0, (31, 0): 0.0, (31, 1): -0.0, (31, 2): -0.0, (32, 0): 0.0, (32, 1): -0.0, (32, 2): -0.0, (33, 0): 0.0, (33, 1): -0.0, (33, 2): -0.0, (34, 0): 0.0, (34, 1): -0.0, (34, 2): -0.0, (35, 0): 0.0, (35, 1): 0.0, (35, 2): -0.0, (36, 0): 0.0, (36, 1): -0.0, (36, 2): -0.0, (37, 0): 0.0, (37, 1): 0.0, (37, 2): -0.0, (38, 0): 0.0, (38, 1): -0.0, (38, 2): -0.0, (39, 0): 0.0, (39, 1): -0.0, (39, 2): 0.0, (40, 0): 0.0, (40, 1): 0.0, (40, 2): -0.0, (41, 0): 0.0, (41, 1): -0.0, (41, 2): -0.0, (42, 0): 0.0, (42, 1): -0.0, (42, 2): -0.0, (43, 0): 0.0, (43, 1): 0.0, (43, 2): -0.0, (44, 0): 0.0, (44, 1): -0.0, (44, 2): 0.0, (45, 0): 0.0, (45, 1): 0.0, (45, 2): 0.0, (46, 0): 0.0, (46, 1): -0.0, (46, 2): -0.0, (47, 0): -0.0, (47, 1): 0.0, (47, 2): 0.0, (48, 0): 0.0, (48, 1): 0.0, (48, 2): -0.0, (49, 0): 0.0, (49, 1): 0.0, (49, 2): 0.0, (50, 0): 0.0, (50, 1): 0.0, (50, 2): 0.0, (51, 0): 0.0, (51, 1): 0.0, (51, 2): 0.0, (52, 0): 0.0, (52, 1): 0.0, (52, 2): 0.0, (53, 0): 0.0, (53, 1): 0.0, (53, 2): 0.0, (54, 0): 0.0, (54, 1): 0.0, (54, 2): 0.0, (55, 0): 0.0, (55, 1): 0.0, (55, 2): 0.0, (56, 0): 0.0, (56, 1): 0.0, (56, 2): 0.0, (57, 0): 0.0, (57, 1): 0.0, (57, 2): 0.0, (58, 0): 0.0, (58, 1): 0.0, (58, 2): 0.0, (59, 0): 0.0, (59, 1): 0.0, (59, 2): 0.0, (60, 0): 0.0, (60, 1): 0.0, (60, 2): 0.0, (61, 0): 0.0, (61, 1): 0.0, (61, 2): 0.0, (62, 0): 0.0, (62, 1): 0.0, (62, 2): 0.0, (63, 0): 0.0, (63, 1): 0.0, (63, 2): 0.0, (64, 0): 0.0, (64, 1): 0.0, (64, 2): 0.0, (65, 0): 0.0, (65, 1): 0.0, (65, 2): 0.0, (66, 0): 0.0, (66, 1): 0.0, (66, 2): 0.0, (67, 0): 0.0, (67, 1): 0.0, (67, 2): 0.0, (68, 0): 0.0, (68, 1): 0.0, (68, 2): 0.0, (69, 0): 0.0, (69, 1): 0.0, (69, 2): 0.0, (70, 0): 0.0, (70, 1): 0.0, (70, 2): 0.0, (71, 0): 0.0, (71, 1): 0.0, (71, 2): 0.0, (72, 0): 0.0, (72, 1): 0.0, (72, 2): 0.0, (73, 0): 0.0, (73, 1): 0.0, (73, 2): 0.0, (74, 0): 0.0, (74, 1): 0.0, (74, 2): 0.0, (75, 0): 0.0, (75, 1): 0.0, (75, 2): 0.0, (76, 0): 0.0, (76, 1): 0.0, (76, 2): 0.0, (77, 0): 0.0, (77, 1): 0.0, (77, 2): 0.0, (78, 0): 0.0, (78, 1): 0.0, (78, 2): 0.0, (79, 0): 0.0, (79, 1): 0.0, (79, 2): 0.0, (80, 0): 0.0, (80, 1): 0.0, (80, 2): 0.0, (81, 0): 0.0, (81, 1): 0.0, (81, 2): 0.0, (82, 0): 0.0, (82, 1): 0.0, (82, 2): 0.0, (83, 0): 0.0, (83, 1): 0.0, (83, 2): 0.0, (84, 0): 0.0, (84, 1): 0.0, (84, 2): 0.0, (85, 0): 0.0, (85, 1): 0.0, (85, 2): 0.0, (86, 0): 0.0, (86, 1): 0.0, (86, 2): 0.0, (87, 0): 0.0, (87, 1): 0.0, (87, 2): 0.0, (88, 0): 0.0, (88, 1): 0.0, (88, 2): 0.0, (89, 0): 0.0, (89, 1): 0.0, (89, 2): 0.0, (90, 0): 0.0, (90, 1): 0.0, (90, 2): 0.0, (91, 0): 0.0, (91, 1): 0.0, (91, 2): 0.0, (92, 0): 0.0, (92, 1): 0.0, (92, 2): 0.0, (93, 0): 0.0, (93, 1): 0.0, (93, 2): 0.0, (94, 0): 0.0, (94, 1): 0.0, (94, 2): 0.0, (95, 0): 0.0, (95, 1): 0.0, (95, 2): 0.0}
Pi = None
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
capacityList = [4, 6, 15]
# Nm = [30, 20, 20]  # 按时间进行
Nm = [5, 2, 1]  # 小算例资源
# Nm = [25, 14, 6]  #  用于计算拉格朗日乘子，上一行配置除以capacity
modeCons = {}
minT = np.min(linkSet[:, 0])  # 整个问题时间窗范围
maxT = np.max(linkSet[:, 2])
for i in range(minT, maxT):
    for j in range(3):
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
        if Pi is not None:
            curLink = linkSet[routeArray[k, i], :]
            curPi = 0
            for t in range(curLink[0], curLink[2]):
                curPi += Pi[t, curLink[4]]/capacityList[curLink[4]]
            price = max(0, routeArray[0, i] - curPi)
        else:
            price = routeArray[0, i]
    x_rk[i] = MP.addVar(obj=price, vtype=GRB.INTEGER, column=col, ub=1, name='r_'+str(j)+'_'+str(i))

# 加link列
x_vk = {}

for i in range(linkSet.shape[0]):
    col = Column()
    # mode
    m = linkSet[i, 4]
    for t in range(linkSet[i, 0], linkSet[i, 2]):
        col.addTerms(1, modeCons[t, m])
    # 关于linkset
    col.addTerms(-capacityList[m], capacityCons[i])
    x_vk[i] = MP.addVar(obj=0, vtype=GRB.INTEGER, column=col, name='v_' + str(m) + '_' + str(i))


# 定义优化方向
start_time = time()
MP.setAttr('ModelSense', GRB.MAXIMIZE)
MP.setParam('MIPGap', 0.005)
MP.write('master.lp')
MP.optimize()
end_time = time()
#输出结果
riderResult = np.empty(shape=[routeArray.shape[0] + 1, 0])  # rider + link
linkResult = np.empty(shape=[0, 1])  # 数量 + mode + link
count1 = 0
for var in MP.getVars():
    if var.x > 0:
        a = var.varName.split('_')
        type = a[0]
        index0 = int(a[1])
        index1 = int(a[2])
        if type == 'r':
            count1 += 1
            riderCol = np.append([index0], routeArray[:, index1], axis=0)
            riderResult = np.append(riderResult, riderCol.reshape(riderResult.shape[0], 1), axis=1)
print("求解时间："+ str(end_time-start_time))
print("rider:", count1)
np.savetxt('result/data2_riderResult_v3.csv', riderResult, delimiter=',')

# print(MP.getAttr("Pi", modeCons))