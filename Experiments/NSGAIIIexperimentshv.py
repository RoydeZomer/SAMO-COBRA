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

import numpy as np
from platypus import NSGAIII
from platypus import Problem
from platypus import Real
from platypus import nondominated
from hypervolume import hypervolume
import ast
import random

def NSGAIII_Experiment():
    NSGAIII_results = {}
    hypNS3 = [0]
    random.seed(0)
    for d in range(1,10):
        hyp = []
        nfes = []
        for i in range(100):
            problem = Problem(7,2,11)
            problem.types[:] = [Real(2.6,3.6),Real(0.7,0.8),Real(17,28),Real(7.3,8.3),Real(7.3,8.3),Real(2.9,3.9),Real(5,5.5)]
            problem.constraints[:] = "<=0"
            problem.function = SRD
            algorithm = NSGAIII(problem,d)
            algorithm.run(40*problem.nvars)
            funcname = 'SRD'
            nondominated_solutions = nondominated(algorithm.result)
            ref = np.array([7000,1700])
            obj = []
            for s in nondominated_solutions:
                lijst = str(s.objectives)
                obj.append(ast.literal_eval(lijst))
            obj = np.array(obj)
            hyp.append(hypervolume(obj,ref))
            nfes.append(algorithm.nfe)
        if np.mean(nfes)<(40*problem.nvars*1.1):
            if np.mean(hypNS3) < np.mean(hyp):
                hypNS3 = hyp
    print(funcname, np.mean(hypNS3), '(', np.std(hypNS3), ')')
    NSGAIII_results[funcname] = hypNS3

    hypNS3 = [0]
    random.seed(0)
    for d in range(1,20):
        hyp = []
        nfes = []
        for i in range(100):
            problem = Problem(3,2,3)
            problem.types[:] = [Real(1,3),Real(0.0005,0.05),Real(0.0005,0.05)]
            problem.constraints[:] = "<=0"
            problem.function = TBTD
            algorithm = NSGAIII(problem,d)
            algorithm.run(40*problem.nvars)
            funcname = 'TBTD'
            nondominated_solutions = nondominated(algorithm.result)
            ref = np.array([0.1,50000])
            obj = []
            for s in nondominated_solutions:
                lijst = str(s.objectives)
                obj.append(ast.literal_eval(lijst))
            obj = np.array(obj)
            hyp.append(hypervolume(obj,ref))
            nfes.append(algorithm.nfe)
        if np.mean(nfes)<(40*problem.nvars*1.1):
            if np.mean(hypNS3) < np.mean(hyp):
                hypNS3 = hyp
    print(funcname, np.mean(hypNS3), '(', np.std(hypNS3), ')')
    NSGAIII_results[funcname] = hypNS3

    hypNS3 = [0]
    random.seed(0)
    for d in range(1,30):
        hyp = []
        nfes = []
        for i in range(100):
            problem = Problem(4,2,5)
            problem.types[:] = [Real(0.125,5),Real(0.1,10),Real(0.1,10),Real(0.125,5)]
            problem.constraints[:] = "<=0"
            problem.function = WB
            algorithm = NSGAIII(problem,d)
            algorithm.run(40*problem.nvars)
            funcname = 'WB'
            nondominated_solutions = nondominated(algorithm.result)
            ref = np.array([350,0.1])
            obj = []
            for s in nondominated_solutions:
                lijst = str(s.objectives)
                obj.append(ast.literal_eval(lijst))
            obj = np.array(obj)
            hyp.append(hypervolume(obj,ref))
            nfes.append(algorithm.nfe)
        if np.mean(nfes)<(40*problem.nvars*1.1):
            if np.mean(hypNS3) < np.mean(hyp):
                hypNS3 = hyp
    print(funcname, np.mean(hypNS3), '(', np.std(hypNS3), ')')
    NSGAIII_results[funcname] = hypNS3

    hypNS3 = [0]
    random.seed(0)
    for d in range(1,30):
        hyp = []
        nfes = []
        for i in range(100):
            problem = Problem(4,2,5)
            problem.types[:] = [Real(55,80),Real(75,110),Real(1000,3000),Real(2,20)]
            problem.constraints[:] = "<=0"
            problem.function = DBD
            algorithm = NSGAIII(problem,d)
            algorithm.run(40*problem.nvars)
            funcname = 'DBD'
            nondominated_solutions = nondominated(algorithm.result)
            ref = np.array([5,50])
            obj = []
            for s in nondominated_solutions:
                lijst = str(s.objectives)
                obj.append(ast.literal_eval(lijst))
            obj = np.array(obj)
            hyp.append(hypervolume(obj,ref))
            nfes.append(algorithm.nfe)
        if np.mean(nfes)<(40*problem.nvars*1.1):
            if np.mean(hypNS3) < np.mean(hyp):
                hypNS3 = hyp
    print(funcname, np.mean(hypNS3), '(', np.std(hypNS3), ')')
    NSGAIII_results[funcname] = hypNS3

    hypNS3 = [0]
    random.seed(0)
    for d in range(1,20):
        hyp = []
        nfes = []
        for i in range(100):
            problem = Problem(2,2,5)
            problem.types[:] = [Real(20,250),Real(10,50)]
            problem.constraints[:] = "<=0"
            problem.function = NBP
            algorithm = NSGAIII(problem,d)
            algorithm.run(40*problem.nvars)
            funcname = 'NBP'
            nondominated_solutions = nondominated(algorithm.result)
            ref =  np.array([11150, 12500])
            obj = []
            for s in nondominated_solutions:
                lijst = str(s.objectives)
                obj.append(ast.literal_eval(lijst))
            obj = np.array(obj)
            hyp.append(hypervolume(obj,ref))
            nfes.append(algorithm.nfe)
        if np.mean(nfes)<(40*problem.nvars*1.1):
            if np.mean(hypNS3) < np.mean(hyp):
                hypNS3 = hyp
    print(funcname, np.mean(hypNS3), '(', np.std(hypNS3), ')')
    NSGAIII_results[funcname] = hypNS3

    hypNS3 = [0]        
    random.seed(0)
    for d in range(1,10):
        hyp = []
        nfes = []
        for i in range(100):
            problem = Problem(6,3,9)
            problem.types[:] = [Real(150,274.32),Real(25,32.31),Real(12,22),Real(8,11.71),Real(14,18),Real(0.63,0.75)]
            problem.constraints[:] = "<=0"
            problem.function = SPD
            algorithm = NSGAIII(problem,d)
            algorithm.run(40*problem.nvars)
            funcname = 'SPD'
            nondominated_solutions = nondominated(algorithm.result)
            ref = np.array([16,19000,-260000])
            obj = []
            for s in nondominated_solutions:
                lijst = str(s.objectives)
                obj.append(ast.literal_eval(lijst))
            obj = np.array(obj)
            hyp.append(hypervolume(obj,ref))
            nfes.append(algorithm.nfe)
        if np.mean(nfes)<(40*problem.nvars*1.1):
            if np.mean(hypNS3) < np.mean(hyp):
                hypNS3 = hyp
    print(funcname, np.mean(hypNS3), '(', np.std(hypNS3), ')')
    NSGAIII_results[funcname] = hypNS3

    hypNS3 = [0]
    random.seed(0)
    for d in range(1,20):
        hyp = []
        nfes = []
        for i in range(100):
            problem = Problem(7,3,10)
            problem.types[:] = [Real(0.5,1.5),Real(0.45,1.35),Real(0.5,1.5),Real(0.5,1.5),Real(0.875,2.625),Real(0.4,1.2),Real(0.4,1.2)]
            problem.constraints[:] = "<=0"
            problem.function = CSI
            algorithm = NSGAIII(problem,d)
            algorithm.run(40*problem.nvars)
            funcname = 'CSI'
            nondominated_solutions = nondominated(algorithm.result)
            ref = np.array([42,4.5,13])
            obj = []
            for s in nondominated_solutions:
                lijst = str(s.objectives)
                obj.append(ast.literal_eval(lijst))
            obj = np.array(obj)
            hyp.append(hypervolume(obj,ref))
            nfes.append(algorithm.nfe)
        if np.mean(nfes)<(40*problem.nvars*1.1):
            if np.mean(hypNS3) < np.mean(hyp):
                hypNS3 = hyp
    print(funcname, np.mean(hypNS3), '(', np.std(hypNS3), ')')
    NSGAIII_results[funcname] = hypNS3

    hypNS3 = [0]        
    random.seed(0)
    for d in range(1,10):
        hyp = []
        nfes = []
        for i in range(100):
            problem = Problem(3,5,7)
            problem.types[:] = [Real(0.01,0.45),Real(0.01,0.1),Real(0.01,0.1)]
            problem.constraints[:] = "<=0"
            problem.function = WP
            algorithm = NSGAIII(problem,d)
            algorithm.run(40*problem.nvars)
            funcname = 'WP'
            nondominated_solutions = nondominated(algorithm.result)
            ref = np.array([83000, 1350, 2.85, 15989825, 25000])
            obj = []
            for s in nondominated_solutions:
                lijst = str(s.objectives)
                obj.append(ast.literal_eval(lijst))
            obj = np.array(obj)
            hyp.append(hypervolume(obj,ref))
            nfes.append(algorithm.nfe)
        if np.mean(nfes)<(40*problem.nvars*1.1):
            if np.mean(hypNS3) < np.mean(hyp):
                hypNS3 = hyp
    print(funcname, np.mean(hypNS3), '(', np.std(hypNS3), ')')
    NSGAIII_results[funcname] = hypNS3

    hypNS3 = [0]
    random.seed(0)
    for d in range(1,20):
        hyp = []
        nfes = []
        for i in range(1):
            problem = Problem(2,2,2)
            problem.types[:] = [Real(0,5),Real(0,3)]
            problem.constraints[:] = "<=0"
            problem.function = BNH
            algorithm = NSGAIII(problem,d)
            algorithm.run(40*problem.nvars)
            funcname = 'BNH'
            nondominated_solutions = nondominated(algorithm.result)
            ref = np.array([140,50])
            obj = []
            for s in nondominated_solutions:
                lijst = str(s.objectives)
                obj.append(ast.literal_eval(lijst))
            obj = np.array(obj)
            hyp.append(hypervolume(obj,ref))
            nfes.append(algorithm.nfe)
        if np.mean(nfes)<(40*problem.nvars*1.1):
            if np.mean(hypNS3) < np.mean(hyp):
                hypNS3 = hyp
                print(d)
    print(funcname, np.mean(hypNS3), '(', np.std(hypNS3), ')')
    NSGAIII_results[funcname] = hypNS3

    hypNS3 = [0]
    random.seed(0)
    for d in range(1,20):
        hyp = []
        nfes = []
        for i in range(100):
            problem = Problem(2,2,2)
            problem.types[:] = [Real(0.1,1),Real(0,5)]
            problem.constraints[:] = "<=0"
            problem.function = CEXP
            algorithm = NSGAIII(problem,d)
            algorithm.run(40*problem.nvars)
            funcname = 'CEXP'
            nondominated_solutions = nondominated(algorithm.result)
            ref = np.array([1,9])
            obj = []
            for s in nondominated_solutions:
                lijst = str(s.objectives)
                obj.append(ast.literal_eval(lijst))
            obj = np.array(obj)
            hyp.append(hypervolume(obj,ref))
            nfes.append(algorithm.nfe)
        if np.mean(nfes)<(40*problem.nvars*1.1):
            if np.mean(hypNS3) < np.mean(hyp):
                hypNS3 = hyp
    print(funcname, np.mean(hypNS3), '(', np.std(hypNS3), ')')
    NSGAIII_results[funcname] = hypNS3

    hypNS3 = [0]    
    random.seed(0)
    for d in range(1,100):
        hyp = []
        nfes = []
        for i in range(100):
            problem = Problem(6,2,2)
            problem.types[:] = [Real(0,1),Real(0,1),Real(0,1),Real(0,1),Real(0,1),Real(0,1)]
            problem.constraints[:] = "<=0"
            problem.function = C3DTLZ4
            algorithm = NSGAIII(problem,d)
            algorithm.run(40*problem.nvars)
            funcname = 'C3DTLZ4'
            nondominated_solutions = nondominated(algorithm.result)
            ref = np.array([3,3])
            obj = []
            for s in nondominated_solutions:
                lijst = str(s.objectives)
                obj.append(ast.literal_eval(lijst))
            obj = np.array(obj)
            hyp.append(hypervolume(obj,ref))
            nfes.append(algorithm.nfe)
        if np.mean(nfes)<(40*problem.nvars*1.1):
            if np.mean(hypNS3) < np.mean(hyp):
                hypNS3 = hyp
    print(funcname, np.mean(hypNS3), '(', np.std(hypNS3), ')')
    NSGAIII_results[funcname] = hypNS3

    hypNS3 = [0]
    random.seed(0)
    for d in range(1,40):
        hyp = []
        nfes = []
        for i in range(100):
            problem = Problem(2,2,2)
            problem.types[:] = [Real(-20,20),Real(-20,20)]
            problem.constraints[:] = "<=0"
            problem.function = SRN
            algorithm = NSGAIII(problem,d)
            algorithm.run(40*problem.nvars)
            funcname = 'SRN'
            nondominated_solutions = nondominated(algorithm.result)
            ref = np.array([301,72])
            obj = []
            for s in nondominated_solutions:
                lijst = str(s.objectives)
                obj.append(ast.literal_eval(lijst))
            obj = np.array(obj)
            hyp.append(hypervolume(obj,ref))
            nfes.append(algorithm.nfe)
        if np.mean(nfes)<(40*problem.nvars*1.1):
            if np.mean(hypNS3) < np.mean(hyp):
                hypNS3 = hyp
    print(funcname, np.mean(hypNS3), '(', np.std(hypNS3), ')')
    NSGAIII_results[funcname] = hypNS3

    hypNS3 = [0]
    random.seed(0)
    for d in range(1,40):
        hyp = []
        nfes = []
        for i in range(100):
            problem = Problem(2,2,2)
            problem.types[:] = [Real(1e-5,np.pi),Real(1e-5,np.pi)]
            problem.constraints[:] = "<=0"
            problem.function = TNK
            algorithm = NSGAIII(problem,d)
            algorithm.run(40*problem.nvars)
            funcname = 'TNK'
            nondominated_solutions = nondominated(algorithm.result)
            ref = np.array([3,3])
            obj = []
            for s in nondominated_solutions:
                lijst = str(s.objectives)
                obj.append(ast.literal_eval(lijst))
            obj = np.array(obj)
            hyp.append(hypervolume(obj,ref))
            nfes.append(algorithm.nfe)
        if np.mean(nfes)<(40*problem.nvars*1.1):
            if np.mean(hypNS3) < np.mean(hyp):
                hypNS3 = hyp
    print(funcname, np.mean(hypNS3), '(', np.std(hypNS3), ')')
    NSGAIII_results[funcname] = hypNS3

    hypNS3 = [0]
    random.seed(0)
    for d in range(1,10):
        hyp = []
        nfes = []
        for i in range(100):
            problem = Problem(6,2,6)
            problem.types[:] = [Real(0,10),Real(0,10),Real(1,5),Real(0,6),Real(1,5),Real(0,10)]
            problem.constraints[:] = "<=0"
            problem.function = OSY
            algorithm = NSGAIII(problem,d)
            algorithm.run(40*problem.nvars)
            funcname = 'OSY'
            nondominated_solutions = nondominated(algorithm.result)
            ref = np.array([0,386])
            obj = []
            for s in nondominated_solutions:
                lijst = str(s.objectives)
                obj.append(ast.literal_eval(lijst))
            obj = np.array(obj)
            hyp.append(hypervolume(obj,ref))
            nfes.append(algorithm.nfe)
        if np.mean(nfes)<(40*problem.nvars*1.1):
            if np.mean(hypNS3) < np.mean(hyp):
                hypNS3 = hyp
    print(funcname, np.mean(hypNS3), '(', np.std(hypNS3), ')')
    NSGAIII_results[funcname] = hypNS3

    hypNS3 = [0]
    random.seed(0)
    for d in range(1,40):
        hyp = []
        nfes = []
        for i in range(100):
            problem = Problem(2,2,2)
            problem.types[:] = [Real(0,1),Real(0,1)]
            problem.constraints[:] = "<=0"
            problem.function = CTP1
            algorithm = NSGAIII(problem,d)
            algorithm.run(40*problem.nvars)
            funcname = 'CTP1'
            nondominated_solutions = nondominated(algorithm.result)
            ref = np.array([1,2])
            obj = []
            for s in nondominated_solutions:
                lijst = str(s.objectives)
                obj.append(ast.literal_eval(lijst))
            obj = np.array(obj)
            hyp.append(hypervolume(obj,ref))
            nfes.append(algorithm.nfe)
        if np.mean(nfes)<(40*problem.nvars*1.1):
            if np.mean(hypNS3) < np.mean(hyp):
                hypNS3 = hyp
    print(funcname, np.mean(hypNS3), '(', np.std(hypNS3), ')')
    NSGAIII_results[funcname] = hypNS3

    hypNS3 = [0]
    random.seed(0)
    for d in range(1,20):
        hyp = []
        nfes = []
        for i in range(100):
            problem = Problem(10,2,1)
            problem.types[:] = [Real(0,1),Real(0,1),Real(0,1),Real(0,1),Real(0,1),Real(0,1),Real(0,1),Real(0,1),Real(0,1),Real(0,1)]
            problem.constraints[:] = "<=0"
            problem.function = BICOP1
            algorithm = NSGAIII(problem,d)
            algorithm.run(40*problem.nvars)
            funcname = 'BICOP1'
            nondominated_solutions = nondominated(algorithm.result)
            ref = np.array([9,9])
            obj = []
            for s in nondominated_solutions:
                lijst = str(s.objectives)
                obj.append(ast.literal_eval(lijst))
            obj = np.array(obj)
            hyp.append(hypervolume(obj,ref))
            nfes.append(algorithm.nfe)
        if np.mean(nfes)<(40*problem.nvars*1.1):
            if np.mean(hypNS3) < np.mean(hyp):
                hypNS3 = hyp
    print(funcname, np.mean(hypNS3), '(', np.std(hypNS3), ')')
    NSGAIII_results[funcname] = hypNS3

    hypNS3 = [0]        
    random.seed(0)
    for d in range(1,100):
        hyp = []
        nfes = []
        for i in range(100):
            problem = Problem(10,2,2)
            problem.types[:] = [Real(0,1),Real(0,1),Real(0,1),Real(0,1),Real(0,1),Real(0,1),Real(0,1),Real(0,1),Real(0,1),Real(0,1)]
            problem.constraints[:] = "<=0"
            problem.function = BICOP2
            algorithm = NSGAIII(problem,d)
            algorithm.run(40*problem.nvars)
            funcname = 'BICOP2'
            nondominated_solutions = nondominated(algorithm.result)
            ref = np.array([70,70])
            obj = []
            for s in nondominated_solutions:
                lijst = str(s.objectives)
                obj.append(ast.literal_eval(lijst))
            obj = np.array(obj)
            hyp.append(hypervolume(obj,ref))
            nfes.append(algorithm.nfe)
        if np.mean(nfes)<(40*problem.nvars*1.1):
            if np.mean(hypNS3) < np.mean(hyp):
                hypNS3 = hyp
    print(funcname, np.mean(hypNS3), '(', np.std(hypNS3), ')')
    NSGAIII_results[funcname] = hypNS3

    hypNS3 = [0]
    random.seed(0)
    for d in range(1,10):
        hyp = []
        nfes = []
        for i in range(100):
            problem = Problem(2,3,3)
            problem.types[:] = [Real(-4,4),Real(-4,4)]
            problem.constraints[:] = "<=0"
            problem.function = TRICOP
            algorithm = NSGAIII(problem,d)
            algorithm.run(40*problem.nvars)
            funcname = 'TRICOP'
            nondominated_solutions = nondominated(algorithm.result)
            ref = np.array([34,-4,90])
            obj = []
            for s in nondominated_solutions:
                lijst = str(s.objectives)
                obj.append(ast.literal_eval(lijst))
            obj = np.array(obj)
            hyp.append(hypervolume(obj,ref))
            nfes.append(algorithm.nfe)
        if np.mean(nfes)<(40*problem.nvars*1.1):
            if np.mean(hypNS3) < np.mean(hyp):
                hypNS3 = hyp
    print(funcname, np.mean(hypNS3), '(', np.std(hypNS3), ')')
    NSGAIII_results[funcname] = hypNS3

    return NSGAIII_results