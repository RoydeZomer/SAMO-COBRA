# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 09:37:08 2021

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

from hypervolume import hypervolume

import numpy as np
from pymoo.model.problem import Problem
from pymoo.optimize import minimize
import autograd.numpy as anp
from pycheapconstr.algorithms.sansga2 import SANSGA2
from pycheapconstr.algorithms.icsansga2 import ICSANSGA2

import json

sansga = {}
icsansga = {}

class OSY_c(Problem):
    def __init__(self):
        super().__init__(n_var=6, n_obj=2, n_constr=6)
        self.xl = anp.array([0.0,0.0,1.0,0.0,1.0,0.0])
        self.xu = anp.array([10.0,10.0,5.0,6.0,5.0,10.0])

    def _evaluate(self, x, out, *args, **kwargs):
        if len(np.shape(x))>1:
            F = []
            G = []
            for i in range(len(x)):
                fi, gi = OSY(x[i])
                F.append(fi)
                G.append(gi)
            F = np.array(F)
            G = np.array(G)
            out["F"] = F 
            out["G"] = G
        else:
            F,G = OSY(x)
            out["F"] = F
            out["G"] = G

class OSY_c_withcheapconstraints(OSY_c):
    def _evaluate(self, x, out, *args, only_inexpensive_constraints=False, **kwargs):

        if only_inexpensive_constraints:
            d = {}
            super()._evaluate(x, d, *args, **kwargs)
            out["G"] = d["G"]
        else:
            super()._evaluate(x, out, *args, **kwargs)    


OSYhv = []
ref = np.array([0,386])
for i in range(10):
    problem = OSY_c()
    n_evals = len(problem.xl)*40
    algorithm = SANSGA2(n_offsprings=10)
    res = minimize(problem, algorithm, ('n_evals', n_evals), seed=i, verbose=True)
    OSYhv.append(hypervolume(res.F, ref))
    print(i)
print(np.mean(OSYhv))
sansga['OSY'] = OSYhv

OSYhv2 = []
for i in range(10):
    problem = OSY_c_withcheapconstraints()
    algorithm = ICSANSGA2()
    res = minimize(problem, algorithm, ('n_evals', n_evals), seed=i, verbose=True)
    OSYhv2.append(hypervolume(res.F, ref))
    print(i)
print(np.mean(OSYhv2))
icsansga['OSYhv'] = OSYhv2



class NBP_c(Problem):
    def __init__(self):
        super().__init__(n_var=2, n_obj=2, n_constr=5)
        self.xl = anp.array([20.0, 10.0])
        self.xu = anp.array([250.0, 50.0])

    def _evaluate(self, x, out, *args, **kwargs):
        if len(np.shape(x))>1:
            F = []
            G = []
            for i in range(len(x)):
                fi, gi = NBP(x[i])
                F.append(fi)
                G.append(gi)
            F = np.array(F)
            G = np.array(G)
            out["F"] = F 
            out["G"] = G
        else:
            F,G = NBP(x)
            out["F"] = F
            out["G"] = G

class NBP_c_withcheapconstraints(NBP_c):
    def _evaluate(self, x, out, *args, only_inexpensive_constraints=False, **kwargs):

        if only_inexpensive_constraints:
            d = {}
            super()._evaluate(x, d, *args, **kwargs)
            out["G"] = d["G"]
        else:
            super()._evaluate(x, out, *args, **kwargs)    


NBPhv = []
ref = np.array([11150, 12500])
for i in range(10):
    problem = NBP_c()
    n_evals = len(problem.xl)*40
    algorithm = SANSGA2()
    res = minimize(problem, algorithm, ('n_evals', n_evals), seed=i, verbose=True)
    NBPhv.append(hypervolume(res.F, ref))
    print(i)
print(np.mean(NBPhv))
sansga['NBP'] = NBPhv

NBPhv2 = []
for i in range(10):
    problem = NBP_c_withcheapconstraints()
    algorithm = ICSANSGA2()
    res = minimize(problem, algorithm, ('n_evals', n_evals), seed=i, verbose=True)
    NBPhv2.append(hypervolume(res.F, ref))
    print(i)
print(np.mean(NBPhv2))
icsansga['NBPhv'] = NBPhv2



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

class BNH_c_withcheapconstraints(BNH_c):
    def _evaluate(self, x, out, *args, only_inexpensive_constraints=False, **kwargs):

        if only_inexpensive_constraints:
            d = {}
            super()._evaluate(x, d, *args, **kwargs)
            out["G"] = d["G"]
        else:
            super()._evaluate(x, out, *args, **kwargs)    


BNHhv = []
ref = np.array([140,50])
for i in range(10):
    problem = BNH_c()
    n_evals = len(problem.xl)*40
    algorithm = SANSGA2()
    res = minimize(problem, algorithm, ('n_evals', n_evals), seed=i, verbose=True)
    BNHhv.append(hypervolume(res.F, ref))
    print(i)
print(np.mean(BNHhv))
sansga['BNH'] = BNHhv

BNHhv2 = []
for i in range(10):
    problem = BNH_c_withcheapconstraints()
    algorithm = ICSANSGA2()
    res = minimize(problem, algorithm, ('n_evals', n_evals), seed=i, verbose=True)
    BNHhv2.append(hypervolume(res.F, ref))
    print(i)
print(np.mean(BNHhv2))
icsansga['BNHhv'] = BNHhv2



class CEXP_c(Problem):
    def __init__(self):
        super().__init__(n_var=2, n_obj=2, n_constr=2)
        self.xl = anp.array([0.1,0.0])
        self.xu = anp.array([1.0,5.0])

    def _evaluate(self, x, out, *args, **kwargs):
        if len(np.shape(x))>1:
            F = []
            G = []
            for i in range(len(x)):
                fi, gi = CEXP(x[i])
                F.append(fi)
                G.append(gi)
            F = np.array(F)
            G = np.array(G)
            out["F"] = F 
            out["G"] = G
        else:
            F,G = CEXP(x)
            out["F"] = F
            out["G"] = G

class CEXP_c_withcheapconstraints(CEXP_c):
    def _evaluate(self, x, out, *args, only_inexpensive_constraints=False, **kwargs):

        if only_inexpensive_constraints:
            d = {}
            super()._evaluate(x, d, *args, **kwargs)
            out["G"] = d["G"]
        else:
            super()._evaluate(x, out, *args, **kwargs)    


CEXPhv = []
ref = np.array([1,9])
for i in range(10):
    problem = CEXP_c()
    n_evals = len(problem.xl)*40
    algorithm = SANSGA2()
    res = minimize(problem, algorithm, ('n_evals', n_evals), seed=i, verbose=True)
    CEXPhv.append(hypervolume(res.F, ref))
    print(i)
print(np.mean(CEXPhv))
sansga['CEXP'] = CEXPhv

CEXPhv2 = []
for i in range(10):
    problem = CEXP_c_withcheapconstraints()
    algorithm = ICSANSGA2()
    res = minimize(problem, algorithm, ('n_evals', n_evals), seed=i, verbose=True)
    CEXPhv2.append(hypervolume(res.F, ref))
    print(i)
print(np.mean(CEXPhv2))
icsansga['CEXPhv'] = CEXPhv2

   


class SRN_c(Problem):
    def __init__(self):
        super().__init__(n_var=2, n_obj=2, n_constr=2)
        self.xl = anp.array([-20.0,-20.0])
        self.xu = anp.array([20.0, 20.0])

    def _evaluate(self, x, out, *args, **kwargs):
        if len(np.shape(x))>1:
            F = []
            G = []
            for i in range(len(x)):
                fi, gi = SRN(x[i])
                F.append(fi)
                G.append(gi)
            F = np.array(F)
            G = np.array(G)
            out["F"] = F 
            out["G"] = G
        else:
            F,G = SRN(x)
            out["F"] = F
            out["G"] = G

class SRN_c_withcheapconstraints(SRN_c):
    def _evaluate(self, x, out, *args, only_inexpensive_constraints=False, **kwargs):

        if only_inexpensive_constraints:
            d = {}
            super()._evaluate(x, d, *args, **kwargs)
            out["G"] = d["G"]
        else:
            super()._evaluate(x, out, *args, **kwargs)    


SRNhv = []
ref = np.array([301,72])
for i in range(10):
    problem = SRN_c()
    n_evals = len(problem.xl)*40
    algorithm = SANSGA2()
    res = minimize(problem, algorithm, ('n_evals', n_evals), seed=i, verbose=True)
    SRNhv.append(hypervolume(res.F, ref))
    print(i)
print(np.mean(SRNhv))
sansga['SRN'] = SRNhv

SRNhv2 = []
for i in range(10):
    problem = SRN_c_withcheapconstraints()
    algorithm = ICSANSGA2()
    res = minimize(problem, algorithm, ('n_evals', n_evals), seed=i, verbose=True)
    SRNhv2.append(hypervolume(res.F, ref))
    print(i)
print(np.mean(SRNhv2))
icsansga['SRNhv'] = SRNhv2



class TNK_c(Problem):
    def __init__(self):
        super().__init__(n_var=2, n_obj=2, n_constr=2)
        self.xl = anp.array([1e-5,1e-5])
        self.xu = anp.array([np.pi, np.pi])

    def _evaluate(self, x, out, *args, **kwargs):
        if len(np.shape(x))>1:
            F = []
            G = []
            for i in range(len(x)):
                fi, gi = TNK(x[i])
                F.append(fi)
                G.append(gi)
            F = np.array(F)
            G = np.array(G)
            out["F"] = F 
            out["G"] = G
        else:
            F,G = TNK(x)
            out["F"] = F
            out["G"] = G

class TNK_c_withcheapconstraints(TNK_c):
    def _evaluate(self, x, out, *args, only_inexpensive_constraints=False, **kwargs):

        if only_inexpensive_constraints:
            d = {}
            super()._evaluate(x, d, *args, **kwargs)
            out["G"] = d["G"]
        else:
            super()._evaluate(x, out, *args, **kwargs)    


TNKhv = []
ref = np.array([3,3])
for i in range(10):
    problem = TNK_c()
    n_evals = len(problem.xl)*40
    algorithm = SANSGA2()
    res = minimize(problem, algorithm, ('n_evals', n_evals), seed=i, verbose=True)
    TNKhv.append(hypervolume(res.F, ref))
    print(i)
print(np.mean(TNKhv))
sansga['TNK'] = TNKhv

TNKhv2 = []
for i in range(10):
    problem = TNK_c_withcheapconstraints()
    algorithm = ICSANSGA2()
    res = minimize(problem, algorithm, ('n_evals', n_evals), se2ed=i, verbose=True)
    TNKhv2.append(hypervolume(res.F, ref))
    print(i)
print(np.mean(TNKhv2))
icsansga['TNKhv'] = TNKhv2

class CTP1_c(Problem):
    def __init__(self):

        super().__init__(n_var=2, n_obj=2, n_constr=2)
        self.xl = anp.array([0.0,0.0])
        self.xu = anp.array([1.0,1.0])

    def _evaluate(self, x, out, *args, **kwargs):
        if len(np.shape(x))>1:
            F = []
            G = []
            for i in range(len(x)):
                fi, gi = CTP1(x[i])
                F.append(fi)
                G.append(gi)
            F = np.array(F)
            G = np.array(G)
            out["F"] = F 
            out["G"] = G
        else:
            F,G = CTP1(x)
            out["F"] = F
            out["G"] = G

class CTP1_c_withcheapconstraints(CTP1_c):
    def _evaluate(self, x, out, *args, only_inexpensive_constraints=False, **kwargs):

        if only_inexpensive_constraints:
            d = {}
            super()._evaluate(x, d, *args, **kwargs)
            out["G"] = d["G"]
        else:
            super()._evaluate(x, out, *args, **kwargs)    


CTP1hv = []
ref = np.array([1,2])
for i in range(10):
    problem = CTP1_c()
    n_evals = len(problem.xl)*40
    algorithm = SANSGA2()
    res = minimize(problem, algorithm, ('n_evals', n_evals), seed=i, verbose=True)
    CTP1hv.append(hypervolume(res.F, ref))
    print(i)
print(np.mean(CTP1hv))
sansga['CTP1'] = CTP1hv

CTP1hv2 = []
for i in range(10):
    problem = CTP1_c_withcheapconstraints()
    algorithm = ICSANSGA2()
    res = minimize(problem, algorithm, ('n_evals', n_evals), seed=i, verbose=True)
    CTP1hv2.append(hypervolume(res.F, ref))
    print(i)
print(np.mean(CTP1hv2))
icsansga['CTP1hv'] = CTP1hv2


class WB_c(Problem):
    def __init__(self):
        super().__init__(n_var=4, n_obj=2, n_constr=5)
        self.xl = anp.array([0.125, 0.1, 0.1, 0.125])
        self.xu = anp.array([5.0, 10.0, 10.0, 5.0])

    def _evaluate(self, x, out, *args, **kwargs):
        if len(np.shape(x))>1:
            F = []
            G = []
            for i in range(len(x)):
                fi, gi = WB(x[i])
                F.append(fi)
                G.append(gi)
            F = np.array(F)
            G = np.array(G)
            out["F"] = F 
            out["G"] = G
        else:
            F,G = WB(x)
            out["F"] = F
            out["G"] = G

fe = 0
class WB_c_withcheapconstraints(WB_c):
    def _evaluate(self, x, out, *args, only_inexpensive_constraints=False, **kwargs):
        global fe
        fe += 1
        if only_inexpensive_constraints:
            d = {}
            super()._evaluate(x, d, *args, **kwargs)
            out["G"] = d["G"]
        else:
            super()._evaluate(x, out, *args, **kwargs)    


WBhv = []
ref = np.array([350,0.1])
for i in range(10):
    problem = WB_c()
    n_evals = len(problem.xl)*40
    algorithm = SANSGA2()
    res = minimize(problem, algorithm, ('n_evals', n_evals), seed=i, verbose=True)
    WBhv.append(hypervolume(res.F, ref))
    print(i)
print(np.mean(WBhv))
sansga['WB'] = WBhv

WBhv2 = []
for i in range(10):
    problem = WB_c_withcheapconstraints()
    algorithm = ICSANSGA2()
    res = minimize(problem, algorithm, ('n_evals', n_evals), seed=i, verbose=True)
    WBhv2.append(hypervolume(res.F, ref))
    print(i)
print(np.mean(WBhv2))
icsansga['WBhv'] = WBhv2
print(fe/10)

class TBTD_c(Problem):
    def __init__(self):
        super().__init__(n_var=3, n_obj=2, n_constr=3)
        self.xl = anp.array([1.0,0.0005,0.0005])
        self.xu = anp.array([3.0,0.05,0.05])

    def _evaluate(self, x, out, *args, **kwargs):
        if len(np.shape(x))>1:
            F = []
            G = []
            for i in range(len(x)):
                fi, gi = TBTD(x[i])
                F.append(fi)
                G.append(gi)
            F = np.array(F)
            G = np.array(G)
            out["F"] = F 
            out["G"] = G
        else:
            F,G = TBTD(x)
            out["F"] = F
            out["G"] = G

class TBTD_c_withcheapconstraints(TBTD_c):
    def _evaluate(self, x, out, *args, only_inexpensive_constraints=False, **kwargs):

        if only_inexpensive_constraints:
            d = {}
            super()._evaluate(x, d, *args, **kwargs)
            out["G"] = d["G"]
        else:
            super()._evaluate(x, out, *args, **kwargs)    


TBTDhv = []
ref = np.array([0.1,50000])
for i in range(10):
    problem = TBTD_c()
    n_evals = len(problem.xl)*40
    algorithm = SANSGA2()
    res = minimize(problem, algorithm, ('n_evals', n_evals), seed=i, verbose=True)
    TBTDhv.append(hypervolume(res.F, ref))
    print(i)
print(np.mean(TBTDhv))
sansga['TBTD'] = TBTDhv

TBTDhv2 = []
for i in range(10):
    problem = TBTD_c_withcheapconstraints()
    algorithm = ICSANSGA2()
    res = minimize(problem, algorithm, ('n_evals', n_evals), seed=i, verbose=True)
    TBTDhv2.append(hypervolume(res.F, ref))
    print(i)
print(np.mean(TBTDhv2))
icsansga['TBTDhv'] = TBTDhv2


class DBD_c(Problem):
    def __init__(self):
        super().__init__(n_var=4, n_obj=2, n_constr=5)
        self.xl = anp.array([55.0, 75.0, 500.0, 2.0])
        self.xu = anp.array([80.0, 110.0, 3000.0, 20.0])

    def _evaluate(self, x, out, *args, **kwargs):
        if len(np.shape(x))>1:
            F = []
            G = []
            for i in range(len(x)):
                fi, gi = DBD(x[i])
                F.append(fi)
                G.append(gi)
            F = np.array(F)
            G = np.array(G)
            out["F"] = F 
            out["G"] = G
        else:
            F,G = DBD(x)
            out["F"] = F
            out["G"] = G

class DBD_c_withcheapconstraints(DBD_c):
    def _evaluate(self, x, out, *args, only_inexpensive_constraints=False, **kwargs):

        if only_inexpensive_constraints:
            d = {}
            super()._evaluate(x, d, *args, **kwargs)
            out["G"] = d["G"]
        else:
            super()._evaluate(x, out, *args, **kwargs)    


DBDhv = []
ref = np.array([5,50])
for i in range(10):
    problem = DBD_c()
    n_evals = len(problem.xl)*40
    algorithm = SANSGA2()
    res = minimize(problem, algorithm, ('n_evals', n_evals), seed=i, verbose=True)
    DBDhv.append(hypervolume(res.F, ref))
    print(i)
print(np.mean(DBDhv))
sansga['DBD'] = DBDhv

DBDhv2 = []
for i in range(10):
    problem = DBD_c_withcheapconstraints()
    algorithm = ICSANSGA2()
    res = minimize(problem, algorithm, ('n_evals', n_evals), seed=i, verbose=True)
    DBDhv2.append(hypervolume(res.F, ref))
    print(i)
print(np.mean(DBDhv2))
icsansga['DBDhv'] = DBDhv2


class WP_c(Problem):
    def __init__(self):
        super().__init__(n_var=3, n_obj=5, n_constr=7)
        self.xl = anp.array([0.01,    0.01,  0.01])
        self.xu = anp.array([0.45,    0.1,  0.1])

    def _evaluate(self, x, out, *args, **kwargs):
        if len(np.shape(x))>1:
            F = []
            G = []
            for i in range(len(x)):
                fi, gi = WP(x[i])
                F.append(fi)
                G.append(gi)
            F = np.array(F)
            G = np.array(G)
            out["F"] = F 
            out["G"] = G
        else:
            F,G = WP(x)
            out["F"] = F
            out["G"] = G

class WP_c_withcheapconstraints(WP_c):
    def _evaluate(self, x, out, *args, only_inexpensive_constraints=False, **kwargs):

        if only_inexpensive_constraints:
            d = {}
            super()._evaluate(x, d, *args, **kwargs)
            out["G"] = d["G"]
        else:
            super()._evaluate(x, out, *args, **kwargs)    


WPhv = []
ref = np.array([83000, 1350, 2.85, 15989825, 25000])
for i in range(10):
    problem = WP_c()
    n_evals = len(problem.xl)*40
    algorithm = SANSGA2(n_offsprings=10)
    res = minimize(problem, algorithm, ('n_evals', n_evals), seed=i, verbose=True)
    WPhv.append(hypervolume(res.F, ref))
    print(i)
print(np.mean(WPhv))
sansga['WP'] = WPhv

WPhv2 = []
for i in range(10):
    problem = WP_c_withcheapconstraints()
    algorithm = ICSANSGA2(n_offsprings=10)
    res = minimize(problem, algorithm, ('n_evals', n_evals), seed=i, verbose=True)
    WPhv2.append(hypervolume(res.F, ref))
    print(i)
print(np.mean(WPhv2))
icsansga['WPhv'] = WPhv2


class C3DTLZ4_c(Problem):
    def __init__(self):
        super().__init__(n_var=6, n_obj=2, n_constr=2)
        self.xl = anp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.xu = anp.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    def _evaluate(self, x, out, *args, **kwargs):
        if len(np.shape(x))>1:
            F = []
            G = []
            for i in range(len(x)):
                fi, gi = C3DTLZ4(x[i])
                F.append(fi)
                G.append(gi)
            F = np.array(F)
            G = np.array(G)
            out["F"] = F 
            out["G"] = G
        else:
            F,G = C3DTLZ4(x)
            out["F"] = F
            out["G"] = G

fe = 0
class C3DTLZ4_c_withcheapconstraints(C3DTLZ4_c):
    def _evaluate(self, x, out, *args, only_inexpensive_constraints=False, **kwargs):
        global fe
        fe += 1
        if only_inexpensive_constraints:
            d = {}
            super()._evaluate(x, d, *args, **kwargs)
            out["G"] = d["G"]
        else:
            super()._evaluate(x, out, *args, **kwargs)    


C3DTLZ4hv = []
ref = np.array([3,3])
for i in range(10):
    problem = C3DTLZ4_c()
    n_evals = len(problem.xl)*40
    algorithm = SANSGA2()
    res = minimize(problem, algorithm, ('n_evals', n_evals), seed=i, verbose=True)
    C3DTLZ4hv.append(hypervolume(res.F, ref))
    print(i)
print(np.mean(C3DTLZ4hv))
sansga['C3DTLZ4'] = C3DTLZ4hv

C3DTLZ4hv2 = []
for i in range(10):
    problem = C3DTLZ4_c_withcheapconstraints()
    algorithm = ICSANSGA2()
    res = minimize(problem, algorithm, ('n_evals', n_evals), seed=i, verbose=True)
    C3DTLZ4hv2.append(hypervolume(res.F, ref))
    print(i)
print(np.mean(C3DTLZ4hv2))
icsansga['C3DTLZ4hv'] = C3DTLZ4hv2
print(fe/10)


class SPD_c(Problem):
    def __init__(self):
        super().__init__(n_var=6, n_obj=3, n_constr=9)
        self.xl = anp.array([150.0,    25.0,    12.0,   8.0,     14.0, 0.63])
        self.xu = anp.array([274.32, 32.31, 22.0,   11.71, 18.0, 0.75])

    def _evaluate(self, x, out, *args, **kwargs):
        if len(np.shape(x))>1:
            F = []
            G = []
            for i in range(len(x)):
                fi, gi = SPD(x[i])
                F.append(fi)
                G.append(gi)
            F = np.array(F)
            G = np.array(G)
            out["F"] = F 
            out["G"] = G
        else:
            F,G = SPD(x)
            out["F"] = F
            out["G"] = G

class SPD_c_withcheapconstraints(SPD_c):
    def _evaluate(self, x, out, *args, only_inexpensive_constraints=False, **kwargs):

        if only_inexpensive_constraints:
            d = {}
            super()._evaluate(x, d, *args, **kwargs)
            out["G"] = d["G"]
        else:
            super()._evaluate(x, out, *args, **kwargs)    


SPDhv = []
ref = np.array([16,19000,-260000])
for i in range(10):
    problem = SPD_c()
    n_evals = len(problem.xl)*40
    algorithm = SANSGA2(n_offsprings=10)
    res = minimize(problem, algorithm, ('n_evals', n_evals), seed=i, verbose=True)
    SPDhv.append(hypervolume(res.F, ref))
    print(i)
print(np.mean(SPDhv))
sansga['SPD'] = SPDhv

SPDhv2 = []
for i in range(10):
    problem = SPD_c_withcheapconstraints()
    algorithm = ICSANSGA2(n_offsprings=10)
    res = minimize(problem, algorithm, ('n_evals', n_evals), seed=i, verbose=True)
    SPDhv2.append(hypervolume(res.F, ref))
    print(i)
print(np.mean(SPDhv2))
icsansga['SPDhv'] = SPDhv2


class CSI_c(Problem):
    def __init__(self):
        super().__init__(n_var=7, n_obj=3, n_constr=10)
        self.xl = anp.array([0.5,    0.45,  0.5,  0.5,   0.875,     0.4,    0.4])
        self.xu = anp.array([1.5,    1.35,  1.5,  1.5,   2.625,     1.2,    1.2])

    def _evaluate(self, x, out, *args, **kwargs):
        if len(np.shape(x))>1:
            F = []
            G = []
            for i in range(len(x)):
                fi, gi = CSI(x[i])
                F.append(fi)
                G.append(gi)
            F = np.array(F)
            G = np.array(G)
            out["F"] = F 
            out["G"] = G
        else:
            F,G = CSI(x)
            out["F"] = F
            out["G"] = G

class CSI_c_withcheapconstraints(CSI_c):
    def _evaluate(self, x, out, *args, only_inexpensive_constraints=False, **kwargs):

        if only_inexpensive_constraints:
            d = {}
            super()._evaluate(x, d, *args, **kwargs)
            out["G"] = d["G"]
        else:
            super()._evaluate(x, out, *args, **kwargs)    


CSIhv = []
ref = np.array([42,4.5,13])
for i in range(10):
    problem = CSI_c()
    n_evals = len(problem.xl)*40
    algorithm = SANSGA2(n_offsprings=20)
    res = minimize(problem, algorithm, ('n_evals', n_evals), seed=i, verbose=True)
    CSIhv.append(hypervolume(res.F, ref))
    print(i)
print(np.mean(CSIhv))
sansga['CSI'] = CSIhv

CSIhv2 = []
for i in range(10):
    problem = CSI_c_withcheapconstraints()
    algorithm = ICSANSGA2(n_offsprings=20)
    res = minimize(problem, algorithm, ('n_evals', n_evals), seed=i, verbose=True)
    CSIhv2.append(hypervolume(res.F, ref))
    print(i)
print(np.mean(CSIhv2))
icsansga['CSIhv'] = CSIhv2


class SRD_c(Problem):
    def __init__(self):
        super().__init__(n_var=7, n_obj=2, n_constr=11)
        self.xl = anp.array([2.6, 0.7, 17, 7.3, 7.3, 2.9, 5])
        self.xu = anp.array([3.6,0.8,28,8.3,8.3,3.9,5.5])

    def _evaluate(self, x, out, *args, **kwargs):
        if len(np.shape(x))>1:
            F = []
            G = []
            for i in range(len(x)):
                fi, gi = SRD(x[i])
                F.append(fi)
                G.append(gi)
            F = np.array(F)
            G = np.array(G)
            out["F"] = F 
            out["G"] = G
        else:
            F,G = SRD(x)
            out["F"] = F
            out["G"] = G

class SRD_c_withcheapconstraints(SRD_c):
    def _evaluate(self, x, out, *args, only_inexpensive_constraints=False, **kwargs):

        if only_inexpensive_constraints:
            d = {}
            super()._evaluate(x, d, *args, **kwargs)
            out["G"] = d["G"]
        else:
            super()._evaluate(x, out, *args, **kwargs)    


SRDhv = []
ref = np.array([7000,1700])
for i in range(10):
    problem = SRD_c()
    n_evals = len(problem.xl)*40
    algorithm = SANSGA2()
    res = minimize(problem, algorithm, ('n_evals', n_evals), seed=i, verbose=True)
    SRDhv.append(hypervolume(res.F, ref))
    print(i)
print(np.mean(SRDhv))
sansga['SRD'] = SRDhv

SRDhv2 = []
for i in range(10):
    problem = SRD_c_withcheapconstraints()
    algorithm = ICSANSGA2()
    res = minimize(problem, algorithm, ('n_evals', n_evals), seed=i, verbose=True)
    SRDhv2.append(hypervolume(res.F, ref))
    print(i)
print(np.mean(SRDhv2))
icsansga['SRDhv'] = SRDhv2



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

        if only_inexpensive_constraints:
            d = {}
            super()._evaluate(x, d, *args, **kwargs)
            out["G"] = d["G"]
        else:
            super()._evaluate(x, out, *args, **kwargs)    


TRICOPhv = []
ref = np.array([34,-4,90])
for i in range(10):
    problem = TRICOP_c()
    n_evals = len(problem.xl)*40
    algorithm = SANSGA2()
    res = minimize(problem, algorithm, ('n_evals', n_evals), seed=i, verbose=True)
    TRICOPhv.append(hypervolume(res.F, ref))
    print(i)
print(np.mean(TRICOPhv))
sansga['TRICOP'] = TRICOPhv

TRICOPhv2 = []
for i in range(10):
    problem = TRICOP_c_withcheapconstraints()
    algorithm = ICSANSGA2()
    res = minimize(problem, algorithm, ('n_evals', n_evals), seed=i, verbose=True)
    TRICOPhv2.append(hypervolume(res.F, ref))
    print(i)
print(np.mean(TRICOPhv2))
icsansga['TRICOPhv'] = TRICOPhv2



class BICOP1_c(Problem):
    def __init__(self):
        super().__init__(n_var=10, n_obj=2, n_constr=1)
        self.xl = anp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.xu = anp.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    def _evaluate(self, x, out, *args, **kwargs):
        if len(np.shape(x))>1:
            F = []
            G = []
            for i in range(len(x)):
                fi, gi = BICOP1(x[i])
                F.append(fi)
                G.append(gi)
            F = np.array(F)
            G = np.array(G)
            out["F"] = F 
            out["G"] = G
        else:
            F,G = BICOP1(x)
            out["F"] = F
            out["G"] = G

class BICOP1_c_withcheapconstraints(BICOP1_c):
    def _evaluate(self, x, out, *args, only_inexpensive_constraints=False, **kwargs):

        if only_inexpensive_constraints:
            d = {}
            super()._evaluate(x, d, *args, **kwargs)
            out["G"] = d["G"]
        else:
            super()._evaluate(x, out, *args, **kwargs)    


BICOP1hv = []
ref = np.array([9,9])
for i in range(10):
    problem = BICOP1_c()
    n_evals = len(problem.xl)*40
    algorithm = SANSGA2()
    res = minimize(problem, algorithm, ('n_evals', n_evals), seed=i, verbose=True)
    BICOP1hv.append(hypervolume(res.F, ref))
    print(i)
print(np.mean(BICOP1hv))
sansga['BICOP1'] = BICOP1hv

BICOP1hv2 = []
for i in range(10):
    problem = BICOP1_c_withcheapconstraints()
    algorithm = ICSANSGA2()
    res = minimize(problem, algorithm, ('n_evals', n_evals), seed=i, verbose=True)
    BICOP1hv2.append(hypervolume(res.F, ref))
    print(i)
print(np.mean(BICOP1hv2))
icsansga['BICOP1hv'] = BICOP1hv2



class BICOP2_c(Problem):
    def __init__(self):
        super().__init__(n_var=10, n_obj=2, n_constr=2)
        self.xl = anp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.xu = anp.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    def _evaluate(self, x, out, *args, **kwargs):
        if len(np.shape(x))>1:
            F = []
            G = []
            for i in range(len(x)):
                fi, gi = BICOP2(x[i])
                F.append(fi)
                G.append(gi)
            F = np.array(F)
            G = np.array(G)
            out["F"] = F 
            out["G"] = G
        else:
            F,G = BICOP2(x)
            out["F"] = F
            out["G"] = G

class BICOP2_c_withcheapconstraints(BICOP2_c):
    def _evaluate(self, x, out, *args, only_inexpensive_constraints=False, **kwargs):

        if only_inexpensive_constraints:
            d = {}
            super()._evaluate(x, d, *args, **kwargs)
            out["G"] = d["G"]
        else:
            super()._evaluate(x, out, *args, **kwargs)    

BICOP2hv = []
ref = np.array([70,70])
for i in range(10):
    problem = BICOP2_c()
    n_evals = len(problem.xl)*40
    algorithm = SANSGA2()
    res = minimize(problem, algorithm, ('n_evals', n_evals), seed=i, verbose=True)
    BICOP2hv.append(hypervolume(res.F, ref))
    print(i)
print(np.mean(BICOP2hv))
sansga['BICOP2'] = BICOP2hv

BICOP2hv2 = []
for i in range(10):
    problem = BICOP2_c_withcheapconstraints()
    algorithm = ICSANSGA2()
    res = minimize(problem, algorithm, ('n_evals', n_evals), seed=i, verbose=True)
    BICOP2hv2.append(hypervolume(res.F, ref))
    print(i)
print(np.mean(BICOP2hv2))
icsansga['BICOP2hv'] = BICOP2hv2

with open('sansgaii.json', 'w') as fp:
    json.dump(sansga, fp)

for key in sansga:
    print(key, np.mean(sansga[key]))

with open('icsansgaii.json', 'w') as fp:
    json.dump(icsansga, fp)

for key in icsansga:
    print(key, np.mean(icsansga[key]))