# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 14:40:04 2017

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
from testFunctions.NBP import NBP
from testFunctions.DBD import DBD
from testFunctions.SPD import SPD
from testFunctions.CSI import CSI
from testFunctions.WP import WP

from testFunctions.BICOP1 import BICOP1
from testFunctions.BICOP2 import BICOP2
from testFunctions.TRICOP import TRICOP

from CONSTRAINED_SMSEGO import CONSTRAINED_SMSEGO # https://github.com/RoydeZomer/CEGO
import time

import numpy as np

for i in range(10):
    problemCall = OSY
    rngMin = np.array([0,0,1,0,1,0])
    rngMax = np.array([10,10,5,6,5,10])
    initEval = 7
    maxEval = 240
    smooth = 2
    runNo = i
    ref = np.array([0,386])
    nconstraints = 6
    
    epsilonInit=0.01
    epsilonMax=0.02
    s = time.time() 
    CONSTRAINED_SMSEGO(problemCall, rngMin, rngMax, ref, nconstraints, initEval, maxEval, smooth, runNo)
    print(time.time()-s)


for i in range(10):
    problemCall = SRD
    rngMin = np.array([2.6, 0.7, 17, 7.3, 7.3, 2.9, 5])
    rngMax =  np.array([3.6,0.8,28,8.3,8.3,3.9,5.5])
    initEval = 8
    maxEval = 280
    smooth = 2
    nVar = 7
    runNo = i
    ref = np.array([7000,1700])
    nconstraints = 11
    
    epsilonInit=0.01
    epsilonMax=0.02
    s = time.time() 
    CONSTRAINED_SMSEGO(problemCall, rngMin, rngMax, ref, nconstraints, initEval, maxEval, smooth, runNo)
    print(time.time()-s)

for i in range(10):
    problemCall = TRICOP
    rngMin = np.array([-4,-4])
    rngMax = np.array([4,4])
    initEval = 3
    maxEval = 80
    smooth = 2
    nVar = 7
    runNo = i
    ref = np.array([34,-4,90])
    nconstraints = 3
    
    epsilonInit=0.01
    epsilonMax=0.02
    s = time.time() 
    CONSTRAINED_SMSEGO(problemCall, rngMin, rngMax, ref, nconstraints, initEval, maxEval, smooth, runNo)
    print(time.time()-s)

for i in range(10):
    problemCall = WB
    rngMin = np.array([0.125, 0.1, 0.1, 0.125])
    rngMax = np.array([5, 10, 10, 5])
    initEval = 5
    maxEval = 160
    smooth = 2
    nVar = 4
    runNo = i
    ref = np.array([350,0.1])
    nconstraints = 5
    
    epsilonInit=0.01
    epsilonMax=0.02
    s = time.time() 
    CONSTRAINED_SMSEGO(problemCall, rngMin, rngMax, ref, nconstraints, initEval, maxEval, smooth, runNo)
    print(time.time()-s)

for i in range(10):
    problemCall = TBTD
    rngMin = np.array([1,0.0005,0.0005])
    rngMax = np.array([3,0.05,0.05])
    initEval = 4
    maxEval = 120
    smooth = 2
    runNo = i
    ref = np.array([0.1,50000])
    nconstraints = 3
    
    epsilonInit=0.01
    epsilonMax=0.02
    s = time.time() 
    CONSTRAINED_SMSEGO(problemCall, rngMin, rngMax, ref, nconstraints, initEval, maxEval, smooth, runNo)
    print(time.time()-s)

for i in range(10):
    problemCall = DBD
    rngMin = np.array([55, 75, 1000, 2])
    rngMax = np.array([80, 110, 3000, 20])
    initEval = 5
    maxEval = 160
    smooth = 2
    nVar = 4
    runNo = i
    ref = np.array([5,50])
    nconstraints = 5
    
    epsilonInit=0.01
    epsilonMax=0.02
    s = time.time() 
    CONSTRAINED_SMSEGO(problemCall, rngMin, rngMax, ref, nconstraints, initEval, maxEval, smooth, runNo)
    print(time.time()-s)

for i in range(10):
    problemCall = SPD
    rngMin = np.array([150,    25,    12,   8,     14, 0.63])
    rngMax = np.array([274.32, 32.31, 22,   11.71, 18, 0.75])
    initEval = 7
    maxEval = 240
    smooth = 2
    nVar = 6
    runNo = i
    ref = np.array([16,19000,-260000])
    nconstraints=9
    
    epsilonInit=0.01
    epsilonMax=0.02
    s = time.time() 
    CONSTRAINED_SMSEGO(problemCall, rngMin, rngMax, ref, nconstraints, initEval, maxEval, smooth, runNo)
    print(time.time()-s)



for i in range(10):
    problemCall = NBP
    rngMin = np.array([20, 10])
    rngMax = np.array([250, 50])
    initEval = 3
    maxEval = 80
    smooth = 2
    nVar = 2
    runNo = i
    ref = np.array([11150, 12500])
    nconstraints = 5
    
    s = time.time()
    CONSTRAINED_SMSEGO(problemCall, rngMin, rngMax, ref, nconstraints, initEval, maxEval, smooth, runNo)
    print(time.time()-s)


for i in range(10):
    problemCall = WP
    rngMin = np.array([0.01,    0.01,  0.01])
    rngMax = np.array([0.45,    0.1,  0.1])
    initEval = 4
    maxEval = 120
    smooth = 2
    nVar = 3
    runNo = i
    ref = np.array([83000, 1350, 2.85, 15989825, 25000])
    nconstraints = 7
    
    epsilonInit=0.01
    epsilonMax=0.02
    s = time.time() 
    CONSTRAINED_SMSEGO(problemCall, rngMin, rngMax, ref, nconstraints, initEval, maxEval, smooth, runNo)
    print(time.time()-s)

##########################theoreticala problems

for i in range(10):
    problemCall = BNH
    rngMin = np.array([0,0])
    rngMax = np.array([5,3])
    initEval = 3
    maxEval = 80
    smooth = 2
    runNo = i
    ref = np.array([140,50])
    nconstraints = 2
    
    epsilonInit=0.01
    epsilonMax=0.02
    s = time.time() 
    CONSTRAINED_SMSEGO(problemCall, rngMin, rngMax, ref, nconstraints, initEval, maxEval, smooth, runNo)
    print(time.time()-s)

for i in range(10):
    problemCall = SRN
    rngMin = np.array([-20,-20])
    rngMax = np.array([20, 20])
    initEval = 3
    maxEval = 80
    smooth = 2
    runNo = i
    ref = np.array([301,72])
    nconstraints = 2
    
    epsilonInit=0.01
    epsilonMax=0.02
    s = time.time() 
    CONSTRAINED_SMSEGO(problemCall, rngMin, rngMax, ref, nconstraints, initEval, maxEval, smooth, runNo)
    print(time.time()-s)

for i in range(10):
    problemCall = TNK
    rngMin = np.array([1e-5,1e-5])
    rngMax = np.array([np.pi, np.pi])
    initEval = 3
    maxEval = 80
    smooth = 2
    runNo = i
    ref = np.array([3,3])
    nconstraints = 2
    
    epsilonInit=0.01
    epsilonMax=0.02
    s = time.time() 
    CONSTRAINED_SMSEGO(problemCall, rngMin, rngMax, ref, nconstraints, initEval, maxEval, smooth, runNo)
    print(time.time()-s)

for i in range(10):
    problemCall = C3DTLZ4
    rngMin = np.array([0,0,0,0,0,0])
    rngMax = np.array([1,1,1,1,1,1])
    initEval = 7
    maxEval = 240
    smooth = 2
    runNo = i
    ref = np.array([3,3])
    nconstraints = 2
    
    epsilonInit=0.01
    epsilonMax=0.02
    s = time.time() 
    CONSTRAINED_SMSEGO(problemCall, rngMin, rngMax, ref, nconstraints, initEval, maxEval, smooth, runNo)
    print(time.time()-s)

for i in range(10):
    problemCall = CTP1
    rngMin = np.array([0,0])
    rngMax = np.array([1,1])
    initEval = 3
    maxEval = 80
    smooth = 2
    runNo = i
    ref = np.array([1,2])
    nconstraints = 2
    
    epsilonInit=0.01
    epsilonMax=0.02
    s = time.time() 
    CONSTRAINED_SMSEGO(problemCall, rngMin, rngMax, ref, nconstraints, initEval, maxEval, smooth, runNo)
    print(time.time()-s)

for i in range(10):
    problemCall = CEXP
    rngMin = np.array([0.1,0])
    rngMax = np.array([1,5])
    initEval = 3
    maxEval = 80
    smooth = 2
    runNo = i
    ref = np.array([1,9])
    nconstraints = 2
    
    epsilonInit=0.01
    epsilonMax=0.02
    s = time.time()
    CONSTRAINED_SMSEGO(problemCall, rngMin, rngMax, ref, nconstraints, initEval, maxEval, smooth, runNo)
    print(time.time()-s)

for i in range(10):
    problemCall = CSI
    rngMin = np.array([0.5,    0.45,  0.5,  0.5,   0.875,     0.4,    0.4])
    rngMax = np.array([1.5,    1.35,  1.5,  1.5,   2.625,     1.2,    1.2])
    initEval = 8
    maxEval = 280
    smooth = 2
    nVar = 7
    runNo = i
    ref = np.array([42,4.5,13])
    nconstraints = 10
    
    epsilonInit=0.01
    epsilonMax=0.02
    s = time.time() 
    CONSTRAINED_SMSEGO(problemCall, rngMin, rngMax, ref, nconstraints, initEval, maxEval, smooth, runNo)
    print(time.time()-s)

for i in range(10):
    problemCall = BICOP2
    rngMin = np.zeros(10)
    rngMax =  np.ones(10)
    initEval = 11
    maxEval = 400
    smooth = 2
    nVar = 7
    runNo = i
    ref = np.array([70,70])
    nconstraints = 2
    
    epsilonInit=0.01
    epsilonMax=0.02
    s = time.time() 
    CONSTRAINED_SMSEGO(problemCall, rngMin, rngMax, ref, nconstraints, initEval, maxEval, smooth, runNo)
    print(time.time()-s)


for i in range(10):
    problemCall = BICOP1
    rngMin = np.zeros(10)
    rngMax =  np.ones(10)
    initEval = 11
    maxEval = 400
    smooth = 2
    nVar = 7
    runNo = i
    ref = np.array([9,9])
    nconstraints = 1
    
    epsilonInit=0.01
    epsilonMax=0.02
    s = time.time() 
    CONSTRAINED_SMSEGO(problemCall, rngMin, rngMax, ref, nconstraints, initEval, maxEval, smooth, runNo)
    print(time.time()-s)