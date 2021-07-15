# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 14:20:41 2021

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



ref = np.array([0,386])
problem = OSY_c()
n_evals = len(problem.xl)*40
algorithm = SANSGA2()
res = minimize(problem, algorithm, ('n_evals', n_evals), seed=1, verbose=True, save_history=True)
F = []
for algorithm in res.history:
    evalss = algorithm.evaluator.n_eval
    opt = algorithm.opt
    feas = np.where(opt.get("feasible"))[0]
    _F = opt.get("F")[feas]
    F.append([evalss, hypervolume(_F, ref)])
print(F)


problem = OSY_c_withcheapconstraints()
algorithm = ICSANSGA2()
res = minimize(problem, algorithm, ('n_evals', n_evals), seed=0, verbose=True, save_history=True)
F = []
for algorithm in res.history:
    evalss = algorithm.evaluator.n_eval
    opt = algorithm.opt
    feas = np.where(opt.get("feasible"))[0]
    _F = opt.get("F")[feas]
    F.append([evalss, hypervolume(_F, ref)])
print(F)



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


ref = np.array([11150, 12500])
problem = NBP_c()
n_evals = len(problem.xl)*40
algorithm = SANSGA2()
res = minimize(problem, algorithm, ('n_evals', n_evals), seed=1, verbose=True, save_history=True)
F = []
for algorithm in res.history:
    evalss = algorithm.evaluator.n_eval
    opt = algorithm.opt
    feas = np.where(opt.get("feasible"))[0]
    _F = opt.get("F")[feas]
    F.append([evalss, hypervolume(_F, ref)])
print(F)

ref = np.array([11150, 12500])
problem = NBP_c_withcheapconstraints()
n_evals = len(problem.xl)*40
algorithm = ICSANSGA2()
res = minimize(problem, algorithm, ('n_evals', n_evals), seed=1, verbose=True, save_history=True)
F = []
for algorithm in res.history:
    evalss = algorithm.evaluator.n_eval
    opt = algorithm.opt
    feas = np.where(opt.get("feasible"))[0]
    _F = opt.get("F")[feas]
    F.append([evalss, hypervolume(_F, ref)])
print(F)



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


ref = np.array([140,50])
problem = BNH_c()
n_evals = len(problem.xl)*40
algorithm = SANSGA2()
res = minimize(problem, algorithm, ('n_evals', n_evals), seed=1, verbose=True, save_history=True)
F = []
for algorithm in res.history:
    evalss = algorithm.evaluator.n_eval
    opt = algorithm.opt
    feas = np.where(opt.get("feasible"))[0]
    _F = opt.get("F")[feas]
    F.append([evalss, hypervolume(_F, ref)])
print(F)

ref = np.array([140,50])
problem = BNH_c_withcheapconstraints()
n_evals = len(problem.xl)*40
algorithm = ICSANSGA2()
res = minimize(problem, algorithm, ('n_evals', n_evals), seed=1, verbose=True, save_history=True)
F = []
for algorithm in res.history:
    evalss = algorithm.evaluator.n_eval
    opt = algorithm.opt
    feas = np.where(opt.get("feasible"))[0]
    _F = opt.get("F")[feas]
    F.append([evalss, hypervolume(_F, ref)])
print(F)



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


ref = np.array([1,9])
problem = CEXP_c()
n_evals = len(problem.xl)*40
algorithm = SANSGA2()
res = minimize(problem, algorithm, ('n_evals', n_evals), seed=1, verbose=True, save_history=True)
F = []
for algorithm in res.history:
    evalss = algorithm.evaluator.n_eval
    opt = algorithm.opt
    feas = np.where(opt.get("feasible"))[0]
    _F = opt.get("F")[feas]
    F.append([evalss, hypervolume(_F, ref)])
print(F)

problem = CEXP_c_withcheapconstraints()
n_evals = len(problem.xl)*40
algorithm = ICSANSGA2()
res = minimize(problem, algorithm, ('n_evals', n_evals), seed=1, verbose=True, save_history=True)
F = []
for algorithm in res.history:
    evalss = algorithm.evaluator.n_eval
    opt = algorithm.opt
    feas = np.where(opt.get("feasible"))[0]
    _F = opt.get("F")[feas]
    F.append([evalss, hypervolume(_F, ref)])
print(F)

   


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

ref = np.array([301,72])
problem = SRN_c()
n_evals = len(problem.xl)*40
algorithm = SANSGA2()
res = minimize(problem, algorithm, ('n_evals', n_evals), seed=1, verbose=True, save_history=True)
F = []
for algorithm in res.history:
    evalss = algorithm.evaluator.n_eval
    opt = algorithm.opt
    feas = np.where(opt.get("feasible"))[0]
    _F = opt.get("F")[feas]
    F.append([evalss, hypervolume(_F, ref)])
print(F)

problem = SRN_c_withcheapconstraints()
n_evals = len(problem.xl)*40
algorithm = ICSANSGA2()
res = minimize(problem, algorithm, ('n_evals', n_evals), seed=1, verbose=True, save_history=True)
F = []
for algorithm in res.history:
    evalss = algorithm.evaluator.n_eval
    opt = algorithm.opt
    feas = np.where(opt.get("feasible"))[0]
    _F = opt.get("F")[feas]
    F.append([evalss, hypervolume(_F, ref)])
print(F)



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

ref = np.array([3,3])
problem = TNK_c()
n_evals = len(problem.xl)*40
algorithm = SANSGA2()
res = minimize(problem, algorithm, ('n_evals', n_evals), seed=1, verbose=True, save_history=True)
F = []
for algorithm in res.history:
    evalss = algorithm.evaluator.n_eval
    opt = algorithm.opt
    feas = np.where(opt.get("feasible"))[0]
    _F = opt.get("F")[feas]
    F.append([evalss, hypervolume(_F, ref)])
print(F)

problem = TNK_c_withcheapconstraints()
n_evals = len(problem.xl)*40
algorithm = ICSANSGA2()
res = minimize(problem, algorithm, ('n_evals', n_evals), seed=1, verbose=True, save_history=True)
F = []
for algorithm in res.history:
    evalss = algorithm.evaluator.n_eval
    opt = algorithm.opt
    feas = np.where(opt.get("feasible"))[0]
    _F = opt.get("F")[feas]
    F.append([evalss, hypervolume(_F, ref)])
print(F)


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

ref = np.array([1,2])
problem = CTP1_c()
n_evals = len(problem.xl)*40
algorithm = SANSGA2()
res = minimize(problem, algorithm, ('n_evals', n_evals), seed=1, verbose=True, save_history=True)
F = []
for algorithm in res.history:
    evalss = algorithm.evaluator.n_eval
    opt = algorithm.opt
    feas = np.where(opt.get("feasible"))[0]
    _F = opt.get("F")[feas]
    F.append([evalss, hypervolume(_F, ref)])
print(F)

problem = CTP1_c_withcheapconstraints()
n_evals = len(problem.xl)*40
algorithm = ICSANSGA2()
res = minimize(problem, algorithm, ('n_evals', n_evals), seed=1, verbose=True, save_history=True)
F = []
for algorithm in res.history:
    evalss = algorithm.evaluator.n_eval
    opt = algorithm.opt
    feas = np.where(opt.get("feasible"))[0]
    _F = opt.get("F")[feas]
    F.append([evalss, hypervolume(_F, ref)])
print(F)


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

class WB_c_withcheapconstraints(WB_c):
    def _evaluate(self, x, out, *args, only_inexpensive_constraints=False, **kwargs):

        if only_inexpensive_constraints:
            d = {}
            super()._evaluate(x, d, *args, **kwargs)
            out["G"] = d["G"]
        else:
            super()._evaluate(x, out, *args, **kwargs)    


ref = np.array([350,0.1])
problem = WB_c()
n_evals = len(problem.xl)*40
algorithm = SANSGA2()
res = minimize(problem, algorithm, ('n_evals', n_evals), seed=1, verbose=True, save_history=True)
F = []
for algorithm in res.history:
    evalss = algorithm.evaluator.n_eval
    opt = algorithm.opt
    feas = np.where(opt.get("feasible"))[0]
    _F = opt.get("F")[feas]
    F.append([evalss, hypervolume(_F, ref)])
print(F)

problem = WB_c_withcheapconstraints()
n_evals = len(problem.xl)*40
algorithm = ICSANSGA2()
res = minimize(problem, algorithm, ('n_evals', n_evals), seed=1, verbose=True, save_history=True)
F = []
for algorithm in res.history:
    evalss = algorithm.evaluator.n_eval
    opt = algorithm.opt
    feas = np.where(opt.get("feasible"))[0]
    _F = opt.get("F")[feas]
    F.append([evalss, hypervolume(_F, ref)])
print(F)


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

ref = np.array([0.1,50000])
problem = TBTD_c()
n_evals = len(problem.xl)*120
algorithm = SANSGA2()
res = minimize(problem, algorithm, ('n_evals', n_evals), seed=1, verbose=True, save_history=True)
F = []
for algorithm in res.history:
    evalss = algorithm.evaluator.n_eval
    opt = algorithm.opt
    feas = np.where(opt.get("feasible"))[0]
    _F = opt.get("F")[feas]
    F.append([evalss, hypervolume(_F, ref)])
print(F)

problem = TBTD_c_withcheapconstraints()
n_evals = len(problem.xl)*40
algorithm = ICSANSGA2()
res = minimize(problem, algorithm, ('n_evals', n_evals), seed=1, verbose=True, save_history=True)
F = []
for algorithm in res.history:
    evalss = algorithm.evaluator.n_eval
    opt = algorithm.opt
    feas = np.where(opt.get("feasible"))[0]
    _F = opt.get("F")[feas]
    F.append([evalss, hypervolume(_F, ref)])
print(F)



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

ref = np.array([5,50])
problem = DBD_c()
n_evals = len(problem.xl)*40
algorithm = SANSGA2()
res = minimize(problem, algorithm, ('n_evals', n_evals), seed=1, verbose=True, save_history=True)
F = []
for algorithm in res.history:
    evalss = algorithm.evaluator.n_eval
    opt = algorithm.opt
    feas = np.where(opt.get("feasible"))[0]
    _F = opt.get("F")[feas]
    F.append([evalss, hypervolume(_F, ref)])
print(F)

problem = DBD_c_withcheapconstraints()
n_evals = len(problem.xl)*40
algorithm = ICSANSGA2()
res = minimize(problem, algorithm, ('n_evals', n_evals), seed=1, verbose=True, save_history=True)
F = []
for algorithm in res.history:
    evalss = algorithm.evaluator.n_eval
    opt = algorithm.opt
    feas = np.where(opt.get("feasible"))[0]
    _F = opt.get("F")[feas]
    F.append([evalss, hypervolume(_F, ref)])
print(F)


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

ref = np.array([83000, 1350, 2.85, 15989825, 25000])
problem = WP_c()
n_evals = len(problem.xl)*100
algorithm = SANSGA2(n_offsprings=20)
res = minimize(problem, algorithm, ('n_evals', n_evals), seed=1, verbose=True, save_history=True)
F = []
for algorithm in res.history:
    evalss = algorithm.evaluator.n_eval
    opt = algorithm.opt
    feas = np.where(opt.get("feasible"))[0]
    _F = opt.get("F")[feas]
    F.append([evalss, hypervolume(_F, ref)])
print(F)

problem = WP_c_withcheapconstraints()
n_evals = len(problem.xl)*100
algorithm = ICSANSGA2(n_offsprings=20)
res = minimize(problem, algorithm, ('n_evals', n_evals), seed=1, verbose=True, save_history=True)
F = []
for algorithm in res.history:
    evalss = algorithm.evaluator.n_eval
    opt = algorithm.opt
    feas = np.where(opt.get("feasible"))[0]
    _F = opt.get("F")[feas]
    F.append([evalss, hypervolume(_F, ref)])
print(F)


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

class C3DTLZ4_c_withcheapconstraints(C3DTLZ4_c):
    def _evaluate(self, x, out, *args, only_inexpensive_constraints=False, **kwargs):

        if only_inexpensive_constraints:
            d = {}
            super()._evaluate(x, d, *args, **kwargs)
            out["G"] = d["G"]
        else:
            super()._evaluate(x, out, *args, **kwargs)    

ref = np.array([3,3])
problem = C3DTLZ4_c()
n_evals = len(problem.xl)*50
algorithm = SANSGA2(n_offsprings=10)
res = minimize(problem, algorithm, ('n_evals', n_evals), seed=1, verbose=True, save_history=True)
F = []
for algorithm in res.history:
    evalss = algorithm.evaluator.n_eval
    opt = algorithm.opt
    feas = np.where(opt.get("feasible"))[0]
    _F = opt.get("F")[feas]
    F.append([evalss, hypervolume(_F, ref)])
print(F)

problem = C3DTLZ4_c_withcheapconstraints()
n_evals = len(problem.xl)*40
algorithm = ICSANSGA2()
res = minimize(problem, algorithm, ('n_evals', n_evals), seed=1, verbose=True, save_history=True)
F = []
for algorithm in res.history:
    evalss = algorithm.evaluator.n_eval
    opt = algorithm.opt
    feas = np.where(opt.get("feasible"))[0]
    _F = opt.get("F")[feas]
    F.append([evalss, hypervolume(_F, ref)])
print(F)



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

ref = np.array([16,19000,-260000])
problem = SPD_c()
n_evals = len(problem.xl)*40
algorithm = SANSGA2(n_offsprings=10)
res = minimize(problem, algorithm, ('n_evals', n_evals), seed=1, verbose=True, save_history=True)
F = []
for algorithm in res.history:
    evalss = algorithm.evaluator.n_eval
    opt = algorithm.opt
    feas = np.where(opt.get("feasible"))[0]
    _F = opt.get("F")[feas]
    F.append([evalss, hypervolume(_F, ref)])
print(F)

problem = SPD_c_withcheapconstraints()
n_evals = len(problem.xl)*40
algorithm = ICSANSGA2(n_offsprings=10)
res = minimize(problem, algorithm, ('n_evals', n_evals), seed=1, verbose=True, save_history=True)
F = []
for algorithm in res.history:
    evalss = algorithm.evaluator.n_eval
    opt = algorithm.opt
    feas = np.where(opt.get("feasible"))[0]
    _F = opt.get("F")[feas]
    F.append([evalss, hypervolume(_F, ref)])
print(F)


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

ref = np.array([42,4.5,13])
problem = CSI_c()
n_evals = len(problem.xl)*100
algorithm = SANSGA2(n_offsprings=20)
res = minimize(problem, algorithm, ('n_evals', n_evals), seed=1, verbose=True, save_history=True)
F = []
for algorithm in res.history:
    evalss = algorithm.evaluator.n_eval
    opt = algorithm.opt
    feas = np.where(opt.get("feasible"))[0]
    _F = opt.get("F")[feas]
    F.append([evalss, hypervolume(_F, ref)])
print(F)

problem = CSI_c_withcheapconstraints()
n_evals = len(problem.xl)*100
algorithm = ICSANSGA2(n_offsprings=20)
res = minimize(problem, algorithm, ('n_evals', n_evals), seed=1, verbose=True, save_history=True)
F = []
for algorithm in res.history:
    evalss = algorithm.evaluator.n_eval
    opt = algorithm.opt
    feas = np.where(opt.get("feasible"))[0]
    _F = opt.get("F")[feas]
    F.append([evalss, hypervolume(_F, ref)])
print(F)


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

ref = np.array([7000,1700])
problem = SRD_c()
n_evals = len(problem.xl)*40
algorithm = SANSGA2()
res = minimize(problem, algorithm, ('n_evals', n_evals), seed=1, verbose=True, save_history=True)
F = []
for algorithm in res.history:
    evalss = algorithm.evaluator.n_eval
    opt = algorithm.opt
    feas = np.where(opt.get("feasible"))[0]
    _F = opt.get("F")[feas]
    F.append([evalss, hypervolume(_F, ref)])
print(F)

problem = SRD_c_withcheapconstraints()
n_evals = len(problem.xl)*40
algorithm = ICSANSGA2()
res = minimize(problem, algorithm, ('n_evals', n_evals), seed=1, verbose=True, save_history=True)
F = []
for algorithm in res.history:
    evalss = algorithm.evaluator.n_eval
    opt = algorithm.opt
    feas = np.where(opt.get("feasible"))[0]
    _F = opt.get("F")[feas]
    F.append([evalss, hypervolume(_F, ref)])
print(F)


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
xs = []
class TRICOP_c_withcheapconstraints(TRICOP_c):
    def _evaluate(self, x, out, *args, only_inexpensive_constraints=False, **kwargs):
        global xs
        if only_inexpensive_constraints:
            d = {}
            super()._evaluate(x, d, *args, **kwargs)
            out["G"] = d["G"]
        else:
            xs.append(x)
            super()._evaluate(x, out, *args, **kwargs)    

ref = np.array([34,-4,90])
problem = TRICOP_c()
n_evals = len(problem.xl)*40
algorithm = SANSGA2()
res = minimize(problem, algorithm, ('n_evals', n_evals), seed=1, verbose=True, save_history=True)
F = []
for algorithm in res.history:
    evalss = algorithm.evaluator.n_eval
    opt = algorithm.opt
    feas = np.where(opt.get("feasible"))[0]
    _F = opt.get("F")[feas]
    F.append([evalss, hypervolume(_F, ref)])
print(F)

problem = TRICOP_c_withcheapconstraints()
n_evals = len(problem.xl)*40
algorithm = ICSANSGA2()
res = minimize(problem, algorithm, ('n_evals', n_evals), seed=1, verbose=True, save_history=True)
F = []
for algorithm in res.history:
    evalss = algorithm.evaluator.n_eval
    opt = algorithm.opt
    feas = np.where(opt.get("feasible"))[0]
    _F = opt.get("F")[feas]
    F.append([evalss, hypervolume(_F, ref)])
print(F)

f=[]
g=[]
i=1
for x in xs[-1]:
    fi, gi = TRICOP(x)
    f.append(fi)
    g.append(gi)
    fhv = np.array(f)
    print(i, hypervolume(fhv, ref))
    i += 1
    
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

ref = np.array([9,9])
problem = BICOP1_c()
n_evals = len(problem.xl)*40
algorithm = SANSGA2()
res = minimize(problem, algorithm, ('n_evals', n_evals), seed=1, verbose=True, save_history=True)
F = []
for algorithm in res.history:
    evalss = algorithm.evaluator.n_eval
    opt = algorithm.opt
    feas = np.where(opt.get("feasible"))[0]
    _F = opt.get("F")[feas]
    F.append([evalss, hypervolume(_F, ref)])
print(F)

problem = BICOP1_c_withcheapconstraints()
n_evals = len(problem.xl)*40
algorithm = ICSANSGA2()
res = minimize(problem, algorithm, ('n_evals', n_evals), seed=1, verbose=True, save_history=True)
F = []
for algorithm in res.history:
    evalss = algorithm.evaluator.n_eval
    opt = algorithm.opt
    feas = np.where(opt.get("feasible"))[0]
    _F = opt.get("F")[feas]
    F.append([evalss, hypervolume(_F, ref)])
print(F)



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

ref = np.array([70,70])
problem = BICOP2_c()
n_evals = len(problem.xl)*40
algorithm = SANSGA2()
res = minimize(problem, algorithm, ('n_evals', n_evals), seed=1, verbose=True, save_history=True)
F = []
for algorithm in res.history:
    evalss = algorithm.evaluator.n_eval
    opt = algorithm.opt
    feas = np.where(opt.get("feasible"))[0]
    _F = opt.get("F")[feas]
    F.append([evalss, hypervolume(_F, ref)])
print(F)

problem = BICOP2_c_withcheapconstraints()
n_evals = len(problem.xl)*40
algorithm = ICSANSGA2()
res = minimize(problem, algorithm, ('n_evals', n_evals), seed=1, verbose=True, save_history=True)
F = []
for algorithm in res.history:
    evalss = algorithm.evaluator.n_eval
    opt = algorithm.opt
    feas = np.where(opt.get("feasible"))[0]
    _F = opt.get("F")[feas]
    F.append([evalss, hypervolume(_F, ref)])
print(F)
