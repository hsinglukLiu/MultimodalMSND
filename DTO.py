import numpy as np


class Station:
    def __init__(self, stationId, pointX, pointY, subway):
        self.ID = stationId
        self.pointX = pointX
        self.pointY = pointY
        self.subway = subway


class Rider:
    def __init__(self, riderId, ed, la, tb, vr, ptype):
        self.ID = riderId
        self.earliestDepartTime = ed
        self.latestArriveTime = la
        self.maxTransferNum = vr
        self.origion = None
        self.destination = None

        self.totalDistance = 0

        self.maxPrice_taxi = 0
        self.maxPrice_subway = 0

        self.maxRideTime_taxi = 0
        self.maxRideTime_subway = 0

        self.availableSList = []

        self.routeList = {}  # key:[case, type]
        self.routePrice = {}

        self.linkSet = np.empty(shape=[0, 6])


class Link:
    def __init__(self, ti, si, tj, sj, m, p):
        self.headT = ti
        self.headS = si
        self.tailT = tj
        self.tailS = sj
        self.mode = m
        self.price = p


class Route:
    def __init__(self, t, n, n1, n2, m, c, c1, node):
        self.nodeList = []
        self.usedTime = t
        self.usedTransferNum = n
        self.taxiTransferNum = n1
        self.busTransferNum = n2
        self.price = c
        self.actualPrice = c1
        self.currentMode = m
        self.currentNode = node
        self.isVisited = []
        self.waited = False
        self.subRouteIndex = [0]
        self.subRouteMode = []

class Mode:
    def __init__(self):
        self.ID = []
        self.modeVelocity = []
        self.modePrice = []
        self.modeVehicleNum = []
        self.minRiderNum = []
        self.capacity = []
