# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 15:41:30 2021

@author: r.dewinter
"""

from testFunctions.BNH import BNH
from testFunctions.TRICOP import TRICOP

from SAMO_COBRA_Init import SAMO_COBRA_Init
from SAMO_COBRA_PhaseII import SAMO_COBRA_PhaseII

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

from platypus import NSGAII
from platypus import NSGAIII
from platypus import Problem
from platypus import Real
from hypervolume import hypervolume
import random


from pymoo.model.problem import Problem
from pymoo.optimize import minimize
import autograd.numpy as anp
from pycheapconstr.algorithms.sansga2 import SANSGA2
from pycheapconstr.algorithms.icsansga2 import ICSANSGA2



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
phv = SAMO_COBRA_Init(xStart, fn, fName, lower, upper, nConstraints, ref=ref, feval=feval, initDesPoints=d+1, cobraSeed=0, iterPlot=True)
phv = SAMO_COBRA_PhaseII(phv)



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
sms = SAMO_COBRA_Init(xStart, fn, fName, lower, upper, nConstraints, ref=ref, feval=feval, initDesPoints=d+1, cobraSeed=0, infillCriteria="SMS", iterPlot=True)
sms = SAMO_COBRA_PhaseII(sms)

fs = []
cs = []
problemCall = TRICOP
rngMin = np.array([-4,-4])
rngMax = np.array([4,4])
initEval = 3
maxEval = 30
smooth = 2
runNo = 0
ref = loadmat('TRICOP-1000ev-nadirpoint.mat')['nadirpoint'][0]
nconstraints = 3

epsilonInit=0.01
epsilonMax=0.02
CONSTRAINED_SMSEGO(problemCall, rngMin, rngMax, ref, nconstraints, initEval, maxEval, smooth, runNo)
cegores = []
cs = np.array(cs)
fs = np.array(fs)
feasible = np.sum(cs<=0,axis=1)==3
for i in range(1,len(fs)):
    fi = fs[:i][feasible[:i]]
    cegores.append(hypervolume(fi,ref))
    

fs = []
cs = []
random.seed(0)
problem = Problem(2,3,3)
problem.types[:] = [Real(-4,4),Real(-4,4)]
problem.constraints[:] = "<=0"
problem.function = TRICOP
algorithm = NSGAII(problem, 5)
algorithm.run(30)
nsgaiihv = []
cs = np.array(cs)
fs = np.array(fs)
feasible = np.sum(cs<=0,axis=1)==3
for i in range(1,len(fs)):
    fi = fs[:i][feasible[:i]]
    nsgaiihv.append(hypervolume(fi,ref))
nsgaiihv = nsgaiihv[:30]
print(nsgaiihv[-1])


fs = []
cs = []
random.seed(0)
problem = Problem(2,3,3)
problem.types[:] = [Real(-4,4),Real(-4,4)]
problem.constraints[:] = "<=0"
problem.function = TRICOP
algorithm = NSGAIII(problem, 1)
algorithm.run(30)
nsgaiiihv = []
cs = np.array(cs)
fs = np.array(fs)
feasible = np.sum(cs<=0,axis=1)==3
for i in range(1,len(fs)):
    fi = fs[:i][feasible[:i]]
    nsgaiiihv.append(hypervolume(fi,ref))
nsgaiiihv = nsgaiiihv[:30]
print(nsgaiiihv[-1])



class TRICOP_c(Problem):
    def __init__(self):
        super().__init__(n_var=2, n_obj=3, n_constr=3)
        self.xl = anp.array([-4.0,-4.0])
        self.xu = anp.array([4.0,4.0])

    def _evaluate(self, x, out, *args, **kwargs):
        if len(np.shape(x))>1:
            F = []
            G = []
            for i in range(len(x)):
                fi, gi = TRICOP(x[i])
                F.append(fi)
                G.append(gi)
            F = np.array(F)
            G = np.array(G)
            out["F"] = F 
            out["G"] = G
        else:
            F,G = TRICOP(x)
            out["F"] = F
            out["G"] = G

class TRICOP_c_withcheapconstraints(TRICOP_c):
    def _evaluate(self, x, out, *args, only_inexpensive_constraints=False, **kwargs):
        global fe
        global xs
        fe += 1
        if only_inexpensive_constraints:
            d = {}
            super()._evaluate(x, d, *args, **kwargs)
            out["G"] = d["G"]
        else:
            xs.append(x)
            super()._evaluate(x, out, *args, **kwargs)    

fs = []
cs = []
problem = TRICOP_c()
n_evals = 30
algorithm = SANSGA2()
res = minimize(problem, algorithm, ('n_evals', n_evals), seed=i, verbose=True)
sansgaiihv = []
cs = np.array(cs)
fs = np.array(fs)
feasible = np.sum(cs<=0,axis=1)==3
for i in range(1,len(fs)):
    fi = fs[:i][feasible[:i]]
    sansgaiihv.append(hypervolume(fi,ref))
sansgaiihv = sansgaiihv[:30]

fe = 0
fs = []
cs = []
xs = []
problem = TRICOP_c_withcheapconstraints()
n_evals = 30
algorithm = ICSANSGA2()
res = minimize(problem, algorithm, ('n_evals', n_evals), seed=1, verbose=True)
fs = []
cs = []
xs = xs[-3:]
for x in xs:
    for xi in x:    
        f, g = TRICOP(xi)

icsansgaiihv = []
cs = np.array(cs)
fs = np.array(fs)
feasible = np.sum(cs<=0,axis=1)==3
for i in range(1,len(fs)):
    fi = fs[:i][feasible[:i]]
    icsansgaiihv.append(hypervolume(fi,ref))
icsansgaiihv = icsansgaiihv[:30]

# reported HV after 200 evaluations: 54.42 with std 0.06
# reported HV after 500 evaluations: 55.13 with std 0.02
plt.plot(list(range(feval)),feval*[54.42],'r--',alpha=0.75,label='SMES-RBF HV after 1000FE')
plt.plot(list(range(feval)),feval*[55.13],'r--',alpha=1.0,label='SMES-RBF HV after 2000FE')
plt.plot(nsgaiihv, label='NSGA-II')
plt.plot(nsgaiiihv, label='NSGA-III')
plt.plot(sansgaiihv, label='SA-NSGA-II')
plt.plot(icsansgaiihv, label='IC-SA-NSGA-II')
plt.plot(cegores, label='CEGO')
plt.plot(phv['hypervolumeProgress'], label='PHV-SAMO-COBRA')
plt.plot(sms['hypervolumeProgress'], label='SMS-SAMO-COBRA')
plt.title('Convergence plot of TRICOP')
plt.xlabel("Iterations")
# plt.ylim(ymin=20000)
plt.ylabel("HV")
plt.legend()
plt.savefig('TRICOP Convergence.pdf')
plt.show()
plt.close()

######################################################
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
phv = SAMO_COBRA_Init(xStart, fn, fName, lower, upper, nConstraints, ref=ref, feval=feval, initDesPoints=d+1, cobraSeed=0, iterPlot=True)
phv = SAMO_COBRA_PhaseII(phv)

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
sms = SAMO_COBRA_Init(xStart, fn, fName, lower, upper, nConstraints, ref=ref, feval=feval, initDesPoints=d+1, cobraSeed=0, iterPlot=True, infillCriteria="SMS")
sms = SAMO_COBRA_PhaseII(sms)

fs = []
cs = []
problemCall = BNH
rngMin = np.array([0,0])
rngMax = np.array([5,3])
initEval = 3
maxEval = 150
smooth = 2
runNo = 0
ref = loadmat('BNH-1000ev-nadirpoint.mat')['nadirpoint'][0]
nconstraints = 2

epsilonInit=0.01
epsilonMax=0.02
CONSTRAINED_SMSEGO(problemCall, rngMin, rngMax, ref, nconstraints, initEval, maxEval, smooth, runNo)
cegores = []
cs = np.array(cs)
fs = np.array(fs)
feasible = np.sum(cs<=0,axis=1)==2
for i in range(1,len(fs)):
    fi = fs[:i][feasible[:i]]
    cegores.append(hypervolume(fi,ref))
cegores = cegores[:150]


fs = []
cs = []
random.seed(0)
problem = Problem(2,2,2)
problem.types[:] = [Real(0,5),Real(0,3)]
problem.constraints[:] = "<=0"
problem.function = BNH
algorithm = NSGAII(problem, 20*problem.nvars)
algorithm.run(150)
nsgaiihv = []
cs = np.array(cs)
fs = np.array(fs)
feasible = np.sum(cs<=0,axis=1)==2
for i in range(1,len(fs)):
    fi = fs[:i][feasible[:i]]
    nsgaiihv.append(hypervolume(fi,ref))
nsgaiihv = nsgaiihv[:150]

fs = []
cs = []
random.seed(0)
problem = Problem(2,2,2)
problem.types[:] = [Real(0,5),Real(0,3)]
problem.constraints[:] = "<=0"
problem.function = BNH
algorithm = NSGAIII(problem,8)
algorithm.run(150)
nsgaiiihv = []
cs = np.array(cs)
fs = np.array(fs)
feasible = np.sum(cs<=0,axis=1)==2
for i in range(1,len(fs)):
    fi = fs[:i][feasible[:i]]
    nsgaiiihv.append(hypervolume(fi,ref))
nsgaiiihv = nsgaiiihv[:150]


class BNH_c(Problem):
    def __init__(self):
        super().__init__(n_var=2, n_obj=2, n_constr=2)
        self.xl = anp.array([0.0,0.0])
        self.xu = anp.array([5.0,3.0])

    def _evaluate(self, x, out, *args, **kwargs):
        if len(np.shape(x))>1:
            F = []
            G = []
            for i in range(len(x)):
                fi, gi = BNH(x[i])
                F.append(fi)
                G.append(gi)
                
            F = np.array(F)
            G = np.array(G)
            out["F"] = F 
            out["G"] = G
        else:
            F,G = BNH(x)
            out["F"] = F
            out["G"] = G

fe = 0
class BNH_c_withcheapconstraints(BNH_c):
    def _evaluate(self, x, out, *args, only_inexpensive_constraints=False, **kwargs):
        global fe
        global xs
        fe += 1
        if only_inexpensive_constraints:
            d = {}
            super()._evaluate(x, d, *args, **kwargs)
            out["G"] = d["G"]
        else:
            xs.append(x)
            super()._evaluate(x, out, *args, **kwargs)    
fs = []
cs = []
problem = BNH_c()
n_evals = 150
algorithm = SANSGA2()
res = minimize(problem, algorithm, ('n_evals', n_evals), seed=1, verbose=True, save_history=True)
sansgaiihv = []
cs = np.array(cs)
fs = np.array(fs)
feasible = np.sum(cs<=0,axis=1)==2
for i in range(1,len(fs)):
    fi = fs[:i][feasible[:i]]
    sansgaiihv.append(hypervolume(fi,ref))
sansgaiihv = sansgaiihv[:150]

fe = 0
fs = []
cs = []
xs = []
problem = BNH_c_withcheapconstraints()
n_evals = 150
algorithm = ICSANSGA2()
res = minimize(problem, algorithm, ('n_evals', n_evals), seed=1, verbose=True)
fs = []
cs = []
xs = xs[-27:]
for x in xs:
    for xi in x:    
        f, g = BNH(xi)

icsansgaiihv = []
cs = np.array(cs)
fs = np.array(fs)
feasible = np.sum(cs<=0,axis=1)==2
for i in range(1,len(fs)):
    fi = fs[:i][feasible[:i]]
    icsansgaiihv.append(hypervolume(fi,ref))
icsansgaiihv = icsansgaiihv[:150]


# reported HV after 200 evaluations: 5053.54 with std 4.56
# reported HV after 500 evaluations: 5082.90 with std 1.58
plt.plot(list(range(feval)),feval*[5053.54],'r--',alpha=0.5,label='SMES-RBF HV after 200FE')
plt.plot(list(range(feval)),feval*[5082.90],'r--',alpha=1.0,label='SMES-RBF HV after 500FE')
plt.plot(nsgaiihv, label='NSGA-II')
plt.plot(nsgaiiihv, label='NSGA-III')
plt.plot(sansgaiihv, label='SA-NSGA-II')
plt.plot(icsansgaiihv, label='IC-SA-NSGA-II')
plt.plot(cegores, label='CEGO')
plt.plot(phv['hypervolumeProgress'], label='PHV-SAMO-COBRA')
plt.plot(sms['hypervolumeProgress'], label='SMS-SAMO-COBRA')
plt.title('Convergence plot of BNH')
plt.xlabel("Iterations")
plt.ylim(ymin=4500,ymax=5120)
plt.xlim(xmin=0,xmax=150)
plt.ylabel("HV")
plt.legend(prop={"size":8})
plt.savefig('BNH Convergence.pdf')
plt.show()
plt.close() 

###############################################################
 