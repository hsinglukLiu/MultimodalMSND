from time import time

import numpy as np
import pandas as pd
import instance

if __name__ == '__main__':
    # linkSet = pd.read_csv('data_set/data2_link.csv', header=None).values
    linkSet = pd.read_csv('code0615/Instance/A_5000_2_linkSet.csv', header=None).values
    stationDict, riderList, dist, mode = instance.initInstance("instance1.xlsx")
    vehicleRouteList = instance.vehicleRouteSearch0(linkSet, mode)
    print(vehicleRouteList[1].shape[1])
    # for m in mode.ID:
    #     if vehicleRouteList[m] is None:
    #         continue
    np.savetxt('Instance/A_5000_2_mode'+str(1)+'Route.csv', vehicleRouteList[1], delimiter=',')