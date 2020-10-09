# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 17:57:13 2020

@author: r.dewinter
"""

from testFunctions.BNH import BNH
from testFunctions.OSY import OSY
from testFunctions.SRN import SRN
from testFunctions.TNK import TNK

from testFunctions.BICOP1 import BICOP1
from testFunctions.BICOP2 import BICOP2
from testFunctions.TRICOP import TRICOP

from SAMO_COBRA_Init import SAMO_COBRA_Init
from SAMO_COBRA_PhaseII import SAMO_COBRA_PhaseII

import numpy as np
from scipy.io import loadmat


import matplotlib.pyplot as plt

  

########################################### BNH experiment
np.random.seed(0)
fn = BNH
fName = 'BNH'
lower = np.array([0,0])
upper = np.array([5,3])
ref = loadmat('BNH-1000ev-nadirpoint.mat')['nadirpoint'][0]
nConstraints = 2
d=len(lower)
xStart = lower+np.random.rand(len(upper))*upper
feval = 150
cobra = SAMO_COBRA_Init(xStart, fn, fName, lower, upper, nConstraints, ref=ref, feval=feval, initDesPoints=d+1, cobraSeed=0, iterPlot=True)
cobra = SAMO_COBRA_PhaseII(cobra)
# reported HV after 200 evaluations: 5053.54 with std 4.56
# reported HV after 500 evaluations: 5082.90 with std 1.58
plt.plot(list(range(feval)),feval*[5053.54],'r',alpha=0.5,label='SMES-RBF HV after 200FE')
plt.plot(list(range(feval)),feval*[5082.90],'r',alpha=1.0,label='SMES-RBF HV after 500FE')
plt.plot(cobra['hypervolumeProgress'], label='HV SAMO-COBRA')
plt.title('SAMO-COBRA convergence plot of BNH')
plt.xlabel("Iterations")
plt.ylim(ymin=4500)
plt.ylabel("HV")
plt.legend()
plt.show()
plt.close() 

########################################### SRN experiment
np.random.seed(0)
fn = SRN
fName = 'SRN'
lower = np.array([-20,-20])
upper = np.array([20, 20])
ref = loadmat('SRN-1000ev-nadirpoint.mat')['nadirpoint'][0]
nConstraints = 2
d=len(lower)
xStart = lower+np.random.rand(len(upper))*upper
feval = 40
cobra = SAMO_COBRA_Init(xStart, fn, fName, lower, upper, nConstraints, ref=ref, feval=feval, initDesPoints=d+1, cobraSeed=0, iterPlot=True)
cobra = SAMO_COBRA_PhaseII(cobra)
# reported HV after 200 evaluations: 29268.88 with std 26.39
# reported HV after 500 evaluations: 29569.03 with std 14.48
plt.plot(list(range(feval)),feval*[29268.88],'r',alpha=0.5,label='SMES-RBF HV after 200FE')
plt.plot(list(range(feval)),feval*[29569.03],'r',alpha=1.0,label='SMES-RBF HV after 500FE')
plt.plot(cobra['hypervolumeProgress'], label='HV SAMO-COBRA')
plt.title('SAMO-COBRA convergence plot of SRN')
plt.xlabel("Iterations")
plt.ylim(ymin=20000)
plt.ylabel("HV")
plt.legend()
plt.show()
plt.close() 

########################################### TNK experiment
np.random.seed(0)
fn = TNK
fName = 'TNK'
lower = np.array([1e-5,1e-5])
upper = np.array([np.pi, np.pi])
ref = loadmat('TNK-1000ev-nadirpoint.mat')['nadirpoint'][0]
nConstraints = 2
d=len(lower)
xStart = lower+np.random.rand(len(upper))*upper
feval = 200
cobra = SAMO_COBRA_Init(xStart, fn, fName, lower, upper, nConstraints, ref=ref, feval=feval, initDesPoints=d+1, cobraSeed=0, iterPlot=True)
cobra = SAMO_COBRA_PhaseII(cobra)
# reported HV after 200 evaluations: 0.65 with std 0.01
# reported HV after 500 evaluations: 0.71 with std 0.00
plt.plot(list(range(feval)),feval*[0.65],'r',alpha=0.5,label='SMES-RBF HV after 200FE')
plt.plot(list(range(feval)),feval*[0.71],'r',alpha=1.0,label='SMES-RBF HV after 500FE')
plt.plot(cobra['hypervolumeProgress'], label='HV SAMO-COBRA')
plt.title('SAMO-COBRA convergence plot of TNK')
plt.xlabel("Iterations")
plt.ylim(ymin=0.5)
plt.ylabel("HV")
plt.legend()
plt.show()
plt.close() 


########################################### OSY experiment
np.random.seed(0)
fn = OSY
fName = 'OSY'
lower = np.array([0,0,1,0,1,0])
upper = np.array([10,10,5,6,5,10])
ref = loadmat('OSY-1000ev-nadirpoint.mat')['nadirpoint'][0]
nConstraints = 6
d=len(lower)
xStart = lower+np.random.rand(len(upper))*upper
feval = 30
cobra = SAMO_COBRA_Init(xStart, fn, fName, lower, upper, nConstraints, ref=ref, feval=feval, initDesPoints=d+1, cobraSeed=0, iterPlot=True)
cobra = SAMO_COBRA_PhaseII(cobra)
# reported HV after 500 evaluations: 24921.39 with std 449.53
# reported HV after 1000 evaluations: 26744.97 with std 460.10
# reported HV after 2000 evaluations: 27382.65 with std 470.54
plt.plot(list(range(feval)),feval*[24921.39],'r',alpha=0.5,label='SMES-RBF HV after 500FE')
plt.plot(list(range(feval)),feval*[26744.97],'r',alpha=0.75,label='SMES-RBF HV after 1000FE')
plt.plot(list(range(feval)),feval*[27382.65],'r',alpha=1.0,label='SMES-RBF HV after 2000FE')
plt.plot(cobra['hypervolumeProgress'], label='HV SAMO-COBRA')
plt.title('SAMO-COBRA convergence plot of OSY')
plt.xlabel("Iterations")
plt.ylabel("HV")
plt.legend()
plt.show()
plt.close() 

########################################### TRICOP experiment
np.random.seed(0)
fn = TRICOP
fName = 'TRICOP'
lower = np.array([-4,-4])
upper = np.array([4,4])
d = len(lower)
xStart = lower+np.random.rand(len(upper))*upper
feval = 30
nConstraints = 3
ref = loadmat('TRICOP-1000ev-nadirpoint.mat')['nadirpoint'][0]
cobra = SAMO_COBRA_Init(xStart, fn, fName, lower, upper, nConstraints, ref=ref, feval=feval, initDesPoints=d+1, cobraSeed=0, iterPlot=True)
cobra = SAMO_COBRA_PhaseII(cobra)
# reported HV after 200 evaluations: 54.42 with std 0.06
# reported HV after 500 evaluations: 55.13 with std 0.02
plt.plot(list(range(feval)),feval*[54.42],'r',alpha=0.75,label='SMES-RBF HV after 1000FE')
plt.plot(list(range(feval)),feval*[55.13],'r',alpha=1.0,label='SMES-RBF HV after 2000FE')
plt.plot(cobra['hypervolumeProgress'], label='HV SAMO-COBRA')
plt.title('SAMO-COBRA convergence plot of TRICOP')
plt.xlabel("Iterations")
# plt.ylim(ymin=20000)
plt.ylabel("HV")
plt.legend()
plt.show()
plt.close() 


########################################### BICOP1 experiment
np.random.seed(0)
fn = BICOP1
fName = 'BICOP1'
lower = np.zeros(10)
upper = np.ones(10)
d = len(lower)
xStart = lower+np.random.rand(len(upper))*upper
ref = loadmat('BICOP1-1000ev-nadirpoint.mat')['nadirpoint'][0]
feval = 80
nConstraints = 1
cobra = SAMO_COBRA_Init(xStart, fn, fName, lower, upper, nConstraints, ref=ref, feval=feval, initDesPoints=d+1, cobraSeed=0, iterPlot=True)
cobra = SAMO_COBRA_PhaseII(cobra)

# reported HV after 500 evaluations: 11.53 with std(0.09)
# reported HV after 1000 evaluations: 12.76 with std(0.06) which is inpossible with this reference point and this funciton
# reported HV after 2000 evaluations: 13.36 with std(0.0580) which is inpossible with this reference point and this funciton
# reported HV after 5000 evaluations: 14.6422 with std(0.0030) which is inpossible with this reference point and this funciton
plt.plot(list(range(feval)),feval*[11.53],'r',label='SMES-RBF HV after 500FE')
plt.plot(cobra['hypervolumeProgress'], label='HV SAMO-COBRA')
plt.title('SAMO-COBRA convergence plot of BICOP1')
plt.xlabel("Iterations")
plt.ylim(ymin=8.95,ymax=12.05)
plt.ylabel("HV")
plt.legend()
plt.show()
plt.close()

########################################### BICOP2 experiment
np.random.seed(0)
fn = BICOP2
fName = 'BICOP2'
lower = np.zeros(10)
upper = np.ones(10)
d = len(lower)
xStart = lower+np.random.rand(len(upper))*upper
ref = loadmat('BICOP2-1000ev-nadirpoint.mat')['nadirpoint'][0]
# reported HV after 5000 evaluations: 0.65 with std(0.05)
feval = 100
nConstraints = 2
cobra = SAMO_COBRA_Init(xStart, fn, fName, lower, upper, nConstraints, ref=ref, feval=feval, initDesPoints=d+1, cobraSeed=0, iterPlot=True)
cobra = SAMO_COBRA_PhaseII(cobra)

# reported HV after 500 evaluations: 0.35 with std(0.01)
# reported HV after 1000 evaluations: 0.38 with std(0.02)
# reported HV after 2000 evaluations: 0.40 with std(0.02) 
# reported HV after 5000 evaluations: 0.65 with std(0.05)
plt.plot(list(range(feval)),feval*[0.35],'r',alpha=0.25,label='SMES-RBF HV after 500FE')
plt.plot(list(range(feval)),feval*[0.38],'r',alpha=0.5,label='SMES-RBF HV after 1000FE')
plt.plot(list(range(feval)),feval*[0.40],'r',alpha=0.75,label='SMES-RBF HV after 2000FE')
plt.plot(list(range(feval)),feval*[0.65],'r',alpha=1.0,label='SMES-RBF HV after 5000FE')
plt.plot(cobra['hypervolumeProgress'], label='HV SAMO-COBRA')
plt.title('SAMO-COBRA convergence plot of BICOP2')
plt.xlabel("Iterations")
plt.ylabel("HV")
plt.legend()
plt.show()
plt.close() 
