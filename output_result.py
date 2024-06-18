import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from code0615 import generateInstance
from Hopcroft_Karp_Algo_dict import *

riderNum = 20000
instance_num = 1
station_data = pd.read_excel("code0615/Instance0615.xlsx", sheet_name='Sheet1')
stationDict, dist, mode = generateInstance.initInstance("code0615/Instance0615.xlsx")

timeDist = {}
timeDist[1] = np.zeros([65, 65])
timeDist[2] = np.zeros([65, 65])
for i in stationDict.keys():
    for j in stationDict.keys():
        timeDist[1][i, j] = generateInstance.dynamicTime(dist, stationDict[i], stationDict[j], mode.modeVelocity[1])
        timeDist[2][i, j] = generateInstance.dynamicTime(dist, stationDict[i], stationDict[j], mode.modeVelocity[2])

for timeID in range(4):
    print('# time ID: {} #'.format(timeID))
    emptyTime = timeID * 2
    CPU_time_fleetSchedule = pd.DataFrame(np.zeros([10, 4 * 7]))
    for instanceIndex in range(1, instance_num + 1):
        print('$$$$$$$$$$$$$$$$$$$$ instance ID: {} $$$$$$$$$$$$$$$$$$$$'.format(instanceIndex))
        linkSet = pd.read_csv('code0615/Instance/A_' + str(riderNum) + '_' + str(instanceIndex) + '_linkSet.csv',
                              header=None).values

        #
        print("-----------------------------------MaaS---------------------------------------")
        case = 3
        riderResult = pd.read_csv(
            'code0615/Instance/A_' + str(riderNum) + '_' + str(instanceIndex) + '_riderResult_case' + str(
                case) + '.csv', header=None).values
        totalCPUTime, count, countM, totalEmptyDistance, totalTravelDistance, totalTravelTime = fleet_maas(riderResult,
                                                                                                           linkSet,
                                                                                                           timeDist,
                                                                                                           station_data,
                                                                                                           emptyTime)
        CPU_time_fleetSchedule.iloc[instanceIndex - 1, 0] = totalCPUTime
        CPU_time_fleetSchedule.iloc[instanceIndex - 1, 1] = count
        CPU_time_fleetSchedule.iloc[instanceIndex - 1, 2] = countM[1]
        CPU_time_fleetSchedule.iloc[instanceIndex - 1, 3] = countM[2]
        CPU_time_fleetSchedule.iloc[instanceIndex - 1, 4] = totalEmptyDistance
        CPU_time_fleetSchedule.iloc[instanceIndex - 1, 5] = totalTravelDistance + totalEmptyDistance
        CPU_time_fleetSchedule.iloc[instanceIndex - 1, 6] = totalTravelTime

        print("-------------------------------ride sharing-----------------------------------")
        case = 4
        riderResult = pd.read_csv(
            'code0615/Instance/A_' + str(riderNum) + '_' + str(instanceIndex) + '_riderResult_case' + str(
                case) + '.csv', header=None).values
        totalCPUTime, count, countM, totalEmptyDistance, totalTravelDistance, totalTravelTime = fleet_maas(riderResult,
                                                                                                           linkSet,
                                                                                                           timeDist,
                                                                                                           station_data,
                                                                                                           emptyTime)
        CPU_time_fleetSchedule.iloc[instanceIndex - 1, 0 + 7] = totalCPUTime
        CPU_time_fleetSchedule.iloc[instanceIndex - 1, 1 + 7] = count
        CPU_time_fleetSchedule.iloc[instanceIndex - 1, 2 + 7] = 0
        CPU_time_fleetSchedule.iloc[instanceIndex - 1, 3 + 7] = countM[2]
        CPU_time_fleetSchedule.iloc[instanceIndex - 1, 4 + 7] = totalEmptyDistance
        CPU_time_fleetSchedule.iloc[instanceIndex - 1, 5 + 7] = totalTravelDistance + totalEmptyDistance
        CPU_time_fleetSchedule.iloc[instanceIndex - 1, 6 + 7] = totalTravelTime

        print("------------------------ride sharing no transfer------------------------------")
        totalCPUTime, count, totalEmptyDistance, totalTravelDistance, totalTravelTime = fleet_no_transfer(riderResult,
                                                                                                          linkSet,
                                                                                                          timeDist,
                                                                                                          station_data,
                                                                                                          emptyTime)

        CPU_time_fleetSchedule.iloc[instanceIndex - 1, 0 + 7 * 2] = totalCPUTime
        CPU_time_fleetSchedule.iloc[instanceIndex - 1, 1 + 7 * 2] = count
        CPU_time_fleetSchedule.iloc[instanceIndex - 1, 2 + 7 * 2] = 0
        CPU_time_fleetSchedule.iloc[instanceIndex - 1, 3 + 7 * 2] = count
        CPU_time_fleetSchedule.iloc[instanceIndex - 1, 4 + 7 * 2] = totalEmptyDistance
        CPU_time_fleetSchedule.iloc[instanceIndex - 1, 5 + 7 * 2] = totalTravelDistance + totalEmptyDistance
        CPU_time_fleetSchedule.iloc[instanceIndex - 1, 6 + 7 * 2] = totalTravelTime

        print("-------------------------------ride hailing-----------------------------------")
        totalCPUTime, count, totalEmptyDistance, totalTravelDistance, totalTravelTime = fleet_no_sharing(riderResult,
                                                                                                         linkSet,
                                                                                                         timeDist,
                                                                                                         station_data,
                                                                                                         emptyTime)
        CPU_time_fleetSchedule.iloc[instanceIndex - 1, 0 + 7 * 3] = totalCPUTime
        CPU_time_fleetSchedule.iloc[instanceIndex - 1, 1 + 7 * 3] = count
        CPU_time_fleetSchedule.iloc[instanceIndex - 1, 2 + 7 * 3] = 0
        CPU_time_fleetSchedule.iloc[instanceIndex - 1, 3 + 7 * 3] = count
        CPU_time_fleetSchedule.iloc[instanceIndex - 1, 4 + 7 * 3] = totalEmptyDistance
        CPU_time_fleetSchedule.iloc[instanceIndex - 1, 5 + 7 * 3] = totalTravelDistance + totalEmptyDistance
        CPU_time_fleetSchedule.iloc[instanceIndex - 1, 6 + 7 * 3] = totalTravelTime

    CPU_time_fleetSchedule.columns = ["c1CPUTime", 'c1totalFleet', "c1SubFleet", "c1TaxiFleet", "c1EmptyDis", "c1TotalDis", "c1TravelTime"
                                      , "c2CPUTime", 'c2totalFleet', "c2SubFleet", "c2TaxiFleet", "c2EmptyDis", "c2TotalDis", "c2TravelTime"
                                      , "c3CPUTime", 'c3totalFleet', "c3SubFleet", "c3TaxiFleet", "c3EmptyDis", "c3TotalDis", "c3TravelTime"
                                      , "c4CPUTime", 'c4totalFleet', "c4SubFleet", "c4TaxiFleet", "c4EmptyDis", "c4TotalDis", "c4TravelTime"]
    CPU_time_fleetSchedule.to_csv('result/A_' + str(riderNum) + '_fleet_CPU_time_empty_' + str(emptyTime) + '.csv',
                                  encoding='utf-8')

