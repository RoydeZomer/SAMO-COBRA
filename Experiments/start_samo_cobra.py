# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 13:48:24 2019

@author: r.dewinter
"""

from testFunctions.BNH import BNH
from testFunctions.CTP1 import CTP1
from testFunctions.OSY import OSY
from testFunctions.CEXP import CEXP
from testFunctions.C3DTLZ4 import C3DTLZ4
from testFunctions.TNK import TNK
from testFunctions.SRN import SRN


from testFunctions.TBTD import TBTD
from testFunctions.SRD import SRD
from testFunctions.WB import WB
from testFunctions.DBD import DBD
from testFunctions.NBP import NBP
from testFunctions.SPD import SPD
from testFunctions.CSI import CSI
from testFunctions.WP import WP

from testFunctions.BICOP1 import BICOP1
from testFunctions.BICOP2 import BICOP2
from testFunctions.TRICOP import TRICOP

from SAMO_COBRA_Init import SAMO_COBRA_Init
from SAMO_COBRA_PhaseII import SAMO_COBRA_PhaseII

import numpy as np
import copy
cobras = []

besthvosy = []
for i in range(10):
    np.random.seed(i)
    fn = OSY
    fName = 'OSY'
    lower = np.array([0,0,1,0,1,0])
    upper = np.array([10,10,5,6,5,10])
    ref = np.array([0,386])
    nConstraints = 6
    d=len(lower)
    xStart = lower+np.random.rand(len(upper))*upper
    feval = 40*d
    cobra = SAMO_COBRA_Init(xStart, fn, fName, lower, upper, nConstraints, ref=ref, feval=feval, initDesPoints=d+1, cobraSeed=i)
    cobra = SAMO_COBRA_PhaseII(cobra)
    cobras.append(copy.deepcopy(cobra))
    besthvosy.append(cobra['hypervolumeProgress'][-1])
    print('OSY', besthvosy, np.mean(besthvosy), np.std(besthvosy))

besthvnbp = []
for i in range(10):
    np.random.seed(i)
    fn = NBP
    fName = 'NBP'
    lower = np.array([20, 10])
    upper = np.array([250, 50])
    ref = np.array([11150, 12500])
    nConstraints = 5
    d=len(lower)
    xStart = lower+np.random.rand(len(upper))*upper
    feval = 40*d
    cobra = SAMO_COBRA_Init(xStart, fn, fName, lower, upper, nConstraints, ref=ref, feval=feval, initDesPoints=d+1, cobraSeed=i)
    cobra = SAMO_COBRA_PhaseII(cobra)
    cobras.append(copy.deepcopy(cobra))
    besthvnbp.append(cobra['hypervolumeProgress'][-1])
    print(fName, np.mean(besthvnbp), np.std(besthvnbp))

besthvbnh = []
for i in range(10):
    np.random.seed(i)
    fn = BNH
    fName = 'BNH'
    lower = np.array([0,0])
    upper = np.array([5,3])
    ref = np.array([140,50])
    nConstraints = 2
    d=len(lower)
    xStart = lower+np.random.rand(len(upper))*upper
    feval = 40*d
    cobra = SAMO_COBRA_Init(xStart, fn, fName, lower, upper, nConstraints, ref=ref, feval=feval, initDesPoints=d+1, cobraSeed=i)
    cobra = SAMO_COBRA_PhaseII(cobra)
    cobras.append(copy.deepcopy(cobra))
    besthvbnh.append(cobra['hypervolumeProgress'][-1])
    print('BNH', np.mean(besthvbnh), np.std(besthvbnh))

besthvcexp = []
for i in range(10):
    np.random.seed(i)
    fn = CEXP
    fName = 'CEXP'
    lower = np.array([0.1,0])
    upper = np.array([1,5])
    ref = np.array([1,9])
    nConstraints = 2
    d=len(lower)
    xStart = lower+np.random.rand(len(upper))*upper
    feval = 40*d
    cobra = SAMO_COBRA_Init(xStart, fn, fName, lower, upper, nConstraints, ref=ref, feval=feval, initDesPoints=d+1, cobraSeed=i)
    cobra = SAMO_COBRA_PhaseII(cobra)
    cobras.append(copy.deepcopy(cobra))
    besthvcexp.append(cobra['hypervolumeProgress'][-1])
    print('CEXP', np.mean(besthvcexp), np.std(besthvcexp))


besthvSRN = []
for i in range(10):
    np.random.seed(i)
    fn = SRN
    fName = 'SRN'
    lower = np.array([-20,-20])
    upper = np.array([20, 20])
    ref =  np.array([301,72])
    nConstraints = 2
    d=len(lower)
    xStart = lower+np.random.rand(len(upper))*upper
    feval = 40*d
    cobra = SAMO_COBRA_Init(xStart, fn, fName, lower, upper, nConstraints, ref=ref, feval=feval, initDesPoints=d+1, cobraSeed=i)
    cobra = SAMO_COBRA_PhaseII(cobra)
    cobras.append(copy.deepcopy(cobra))
    besthvSRN.append(cobra['hypervolumeProgress'][-1])
    print('SRN', np.mean(besthvSRN), np.std(besthvSRN))

besthvtnk = []
for i in range(10):
    np.random.seed(i)
    fn = TNK
    fName = 'TNK'
    lower = np.array([1e-5,1e-5])
    upper = np.array([np.pi, np.pi])
    ref =  np.array([3,3])
    nConstraints = 2
    d=len(lower)
    xStart = lower+np.random.rand(len(upper))*upper
    feval = 40*d
    cobra = SAMO_COBRA_Init(xStart, fn, fName, lower, upper, nConstraints, ref=ref, feval=feval, initDesPoints=d+1, cobraSeed=i)
    cobra = SAMO_COBRA_PhaseII(cobra)
    cobras.append(copy.deepcopy(cobra))
    besthvtnk.append(cobra['hypervolumeProgress'][-1])
    print('TNK', np.mean(besthvtnk), np.std(besthvtnk))

besthvctp1 = []
for i in range(10):
    np.random.seed(i)
    fn = CTP1
    fName = 'CTP1'
    lower = np.array([0,0])
    upper = np.array([1,1])
    ref = np.array([1,2])
    nConstraints = 2
    d=len(lower)
    xStart = lower+np.random.rand(len(upper))*upper
    feval = 40*d
    cobra = SAMO_COBRA_Init(xStart, fn, fName, lower, upper, nConstraints, ref=ref, feval=feval, initDesPoints=d+1, cobraSeed=i)
    cobra = SAMO_COBRA_PhaseII(cobra)
    cobras.append(copy.deepcopy(cobra))
    besthvctp1.append(cobra['hypervolumeProgress'][-1])
    print('CTP1', np.mean(besthvctp1), np.std(besthvctp1))

besthvwb = []
for i in range(10):
    np.random.seed(i)
    fn = WB
    fName = 'WB'
    lower = np.array([0.125, 0.1, 0.1, 0.125])
    upper = np.array([5, 10, 10, 5])
    d = len(lower)
    xStart = lower+np.random.rand(len(upper))*upper
    feval = 40*d
    nConstraints = 5
    ref = np.array([350,0.1])
    cobra = SAMO_COBRA_Init(xStart, fn, fName, lower, upper, nConstraints, ref=ref, feval=feval, initDesPoints=d+1, cobraSeed=i)
    cobra = SAMO_COBRA_PhaseII(cobra)
    cobras.append(copy.deepcopy(cobra))
    besthvwb.append(cobra['hypervolumeProgress'][-1])
    print('WB', np.mean(besthvwb), np.std(besthvwb))

bestTBTD = []
for i in range(10):
    np.random.seed(i)
    fn = TBTD
    fName = 'TBTD'
    lower = np.array([1,0.0005,0.0005])
    upper = np.array([3,0.05,0.05])
    xStart = lower+np.random.rand(len(upper))*upper
    d = len(lower)
    feval = 40*d
    nConstraints = 3
    ref = np.array([0.1,50000])
    cobra = SAMO_COBRA_Init(xStart, fn, fName, lower, upper, nConstraints, ref=ref, feval=feval, initDesPoints=d+1, cobraSeed=i)
    cobra = SAMO_COBRA_PhaseII(cobra)
    cobras.append(copy.deepcopy(cobra))
    bestTBTD.append(cobra['hypervolumeProgress'][-1])
    print('TBTD', np.mean(bestTBTD), np.std(bestTBTD))

bestDBD = []
for i in range(10):
    np.random.seed(i)
    fn = DBD
    fName = 'DBD'
    lower = np.array([55, 75, 500, 2])
    upper = np.array([80, 110, 3000, 20])
    d = len(lower)
    xStart = lower+np.random.rand(len(upper))*upper
    feval = 40*d
    nConstraints = 5
    ref = np.array([5,50])
    cobra = SAMO_COBRA_Init(xStart, fn, fName, lower, upper, nConstraints, ref=ref, feval=feval, initDesPoints=d+1, cobraSeed=i)
    cobra = SAMO_COBRA_PhaseII(cobra)
    cobras.append(copy.deepcopy(cobra))
    bestDBD.append(cobra['hypervolumeProgress'][-1])
    print('DBD', np.mean(bestDBD), np.std(bestDBD))

bestWP = []
for i in range(10): # lhs geeft betere resultaten dan optimalisatie
    np.random.seed(i)
    fn = WP
    fName = 'WP'
    lower = np.array([0.01,    0.01,  0.01])
    upper = np.array([0.45,    0.1,  0.1])
    d = len(lower)
    xStart = lower+np.random.rand(len(upper))*upper
    feval = 40*d
    nConstraints = 7
    ref = np.array([83000, 1350, 2.85, 15989825, 25000])
    cobra = SAMO_COBRA_Init(xStart, fn, fName, lower, upper, nConstraints, ref=ref, feval=feval, initDesPoints=d+1, cobraSeed=i)
    cobra = SAMO_COBRA_PhaseII(cobra)
    cobras.append(copy.deepcopy(cobra))
    bestWP.append(cobra['hypervolumeProgress'][-1])
    print('WP', np.mean(bestWP), np.std(bestWP))  

besthvc3dtlz4 = []
for i in range(10): 
    np.random.seed(i)
    fn = C3DTLZ4
    fName = 'C3DTLZ4'
    lower = np.array([0,0,0,0,0,0])
    upper = np.array([1,1,1,1,1,1])
    ref = np.array([3,3])
    nConstraints = 2
    d=len(lower)
    xStart = lower+np.random.rand(len(upper))*upper
    feval = 40*d
    cobra = SAMO_COBRA_Init(xStart, fn, fName, lower, upper, nConstraints, ref=ref, feval=feval, initDesPoints=d+1, cobraSeed=i)#, initDesign='BOUNDARIES')
    cobra = SAMO_COBRA_PhaseII(cobra)
    cobras.append(copy.deepcopy(cobra))
    besthvc3dtlz4.append(cobra['hypervolumeProgress'][-1])
    print('C3DTLZ4', np.mean(besthvc3dtlz4), np.std(besthvc3dtlz4))

bestSPD = []
for i in range(10):
    np.random.seed(i)
    fn = SPD
    fName = 'SPD'
    lower = np.array([150,    25,    12,   8,     14, 0.63])
    upper = np.array([274.32, 32.31, 22,   11.71, 18, 0.75])
    d = len(lower)
    xStart = lower+np.random.rand(len(upper))*upper
    feval = 40*d
    nConstraints=9
    ref = np.array([16,19000,-260000])
    cobra = SAMO_COBRA_Init(xStart, fn, fName, lower, upper, nConstraints, ref=ref, feval=feval, initDesPoints=d+1, cobraSeed=i)
    cobra = SAMO_COBRA_PhaseII(cobra)
    cobras.append(copy.deepcopy(cobra))
    bestSPD.append(cobra['hypervolumeProgress'][-1])
    print('SPD', np.mean(bestSPD), np.std(bestSPD))

bestCSI = []
for i in range(10):
    np.random.seed(i)
    fn = CSI
    fName = 'CSI'
    lower = np.array([0.5,    0.45,  0.5,  0.5,   0.875,     0.4,    0.4])
    upper = np.array([1.5,    1.35,  1.5,  1.5,   2.625,     1.2,    1.2])
    d = len(lower)
    xStart = lower+np.random.rand(len(upper))*upper
    feval = 40*d
    nConstraints = 10
    ref = np.array([42,4.5,13])
    cobra = SAMO_COBRA_Init(xStart, fn, fName, lower, upper, nConstraints, ref=ref, feval=feval, initDesPoints=d+1, cobraSeed=i)
    cobra = SAMO_COBRA_PhaseII(cobra)
    bestCSI.append(cobra['hypervolumeProgress'][-1])
    print('CSI', bestCSI, np.mean(bestCSI), np.std(bestCSI))  

bestSRD = []
for i in range(10):
    np.random.seed(i)
    fn = SRD
    fName = 'SRD'
    lower = np.array([2.6, 0.7, 17, 7.3, 7.3, 2.9, 5])
    upper = np.array([3.6,0.8,28,8.3,8.3,3.9,5.5])
    d = len(lower)
    feval = 40*d
    nConstraints = 11
    ref = np.array([7000,1700])
    xStart = lower+np.random.rand(len(upper))*upper
    cobra = SAMO_COBRA_Init(xStart, fn, fName, lower, upper, nConstraints, ref=ref, feval=feval, initDesPoints=d+1, cobraSeed=i)
    cobra = SAMO_COBRA_PhaseII(cobra)
    cobras.append(copy.deepcopy(cobra))
    bestSRD.append(cobra['hypervolumeProgress'][-1])
    print('SRD', np.mean(bestSRD), np.std(bestSRD))  

besthvTRICOP = []
for i in range(10):
    np.random.seed(i)
    fn = TRICOP
    fName = 'TRICOP'
    lower = np.array([-4,-4])
    upper = np.array([4,4])
    d = len(lower)
    xStart = lower+np.random.rand(len(upper))*upper
    feval = 40*d
    nConstraints = 3
    ref = np.array([34,-4,90])
    cobra = SAMO_COBRA_Init(xStart, fn, fName, lower, upper, nConstraints, ref=ref, feval=feval, initDesPoints=d+1, cobraSeed=i)
    cobra = SAMO_COBRA_PhaseII(cobra)
    cobras.append(copy.deepcopy(cobra))
    besthvTRICOP.append(cobra['hypervolumeProgress'][-1])
    print(np.mean(besthvTRICOP), np.std(besthvTRICOP))


besthvBICOP1 = []
for i in range(10):
    np.random.seed(i)
    fn = BICOP1
    fName = 'BICOP1'
    lower = np.zeros(10)
    upper = np.ones(10)
    d = len(lower)
    xStart = lower+np.random.rand(len(upper))*upper
    feval = 40*d
    nConstraints = 1
    ref = np.array([9,9])
    cobra = SAMO_COBRA_Init(xStart, fn, fName, lower, upper, nConstraints, ref=ref, feval=feval, initDesPoints=d+1, cobraSeed=i)
    cobra = SAMO_COBRA_PhaseII(cobra)
    cobras.append(copy.deepcopy(cobra))
    besthvBICOP1.append(cobra['hypervolumeProgress'][-1])
    print(np.mean(besthvBICOP1), np.std(besthvBICOP1))

besthvBICOP2 = []
for i in range(10):
    np.random.seed(i)
    fn = BICOP2
    fName = 'BICOP2'
    lower = np.zeros(10)
    upper = np.ones(10)
    d = len(lower)
    xStart = lower+np.random.rand(len(upper))*upper
    feval = 40*d
    nConstraints = 2
    ref = np.array([70,70])
    cobra = SAMO_COBRA_Init(xStart, fn, fName, lower, upper, nConstraints, ref=ref, feval=feval, initDesPoints=d+1, cobraSeed=i)
    cobra = SAMO_COBRA_PhaseII(cobra)
    cobras.append(copy.deepcopy(cobra))
    besthvBICOP2.append(cobra['hypervolumeProgress'][-1])
    print(besthvBICOP2, np.mean(besthvBICOP2), np.std(besthvBICOP2))
