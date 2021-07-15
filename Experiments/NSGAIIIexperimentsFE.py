# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 14:33:16 2021

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

import numpy as np
from platypus import NSGAIII
from platypus import Problem
from platypus import Real
from platypus import nondominated
from hypervolume import hypervolume
import ast
import random


random.seed(0)
for d in range(1,10):
    for g in range(1,10):
        hyp = []
        nfes = []
        for i in range(10):
            problem = Problem(2,2,2)
            problem.types[:] = [Real(0,5),Real(0,3)]
            problem.constraints[:] = "<=0"
            problem.function = BNH
            algorithm = NSGAIII(problem,d*problem.nvars)
            algorithm.run(d*g*problem.nvars)
            
            
            funcname = 'BNH'
            # if not os.path.exists(funcname):
            #     os.makedirs(funcname)
                
            nondominated_solutions = nondominated(algorithm.result)
            ref = np.array([140,50])
            obj = []
            for s in nondominated_solutions:
                lijst = str(s.objectives)
                obj.append(ast.literal_eval(lijst))
            obj = np.array(obj)
            # np.savetxt(str(funcname)+'/'+str(funcname)+'_pf_run_'+str(i)+'.csv', obj, delimiter=',')
            hyp.append(hypervolume(obj,ref))
            nfes.append(algorithm.nfe)
        print(np.mean(hyp))
        if np.mean(hyp) > 5005:
            print('BNH',np.mean(hyp), '(', np.std(hyp),')',g,d, np.mean(nfes))

print('BNH',np.mean(hyp), '(', np.std(hyp),')')
#print('BNH',np.max(hyp))
#print('BNH',np.std(hyp))
#print(funcname,algorithm.nfe)


random.seed(0)
for d in range(12,18,1):
    for g in range(10,15,1):
        nfes = []
        hyp = []
        for i in range(10):
            problem = Problem(2,2,2)
            problem.types[:] = [Real(0.1,1),Real(0,5)]
            problem.constraints[:] = "<=0"
            problem.function = CEXP
            algorithm = NSGAIII(problem,d*problem.nvars)
            algorithm.run(d*g*problem.nvars)
            
            
            funcname = 'CEXP'
            # if not os.path.exists(funcname):
            #     os.makedirs(funcname)
                
            nondominated_solutions = nondominated(algorithm.result)
            ref = np.array([1,9])
            obj = []
            for s in nondominated_solutions:
                lijst = str(s.objectives)
                obj.append(ast.literal_eval(lijst))
            obj = np.array(obj)
            # np.savetxt(str(funcname)+'/'+str(funcname)+'_pf_run_'+str(i)+'.csv', obj, delimiter=',')
            hyp.append(hypervolume(obj,ref))
            nfes.append(algorithm.nfe)
        print(np.mean(hyp))
        if np.mean(hyp) > 3.61811363037:
            print(funcname,np.mean(hyp), '(', np.std(hyp),')',g,d, np.mean(nfes))

print('CEXP',np.mean(hyp), '(', np.std(hyp),')')
#print('CEXP',np.max(hyp))
#print('CEXP',np.std(hyp))
#print(funcname,algorithm.nfe)


random.seed(0)
for d in range(5,15,1):
    for g in range(5,15,1):
        nfes = []
        hyp = []
        for i in range(10):
            problem = Problem(2,2,2)
            problem.types[:] = [Real(-20,20),Real(-20,20)]
            problem.constraints[:] = "<=0"
            problem.function = SRN
            algorithm = NSGAIII(problem,d*problem.nvars)
            algorithm.run(d*g*problem.nvars)
            
            funcname = 'SRN'
            # if not os.path.exists(funcname):
            #     os.makedirs(funcname)
                
            nondominated_solutions = nondominated(algorithm.result)
            ref = np.array([301,72])
            obj = []
            for s in nondominated_solutions:
                lijst = str(s.objectives)
                obj.append(ast.literal_eval(lijst))
            obj = np.array(obj)
            # np.savetxt(str(funcname)+'/'+str(funcname)+'_pf_run_'+str(i)+'.csv', obj, delimiter=',')
            hyp.append(hypervolume(obj,ref))
            nfes.append(algorithm.nfe)
        print(np.mean(hyp))
        if np.mean(hyp) > 59441.2892054:
            print(funcname,np.mean(hyp), '(', np.std(hyp),')',g,d, np.mean(nfes))

print('SRN',np.mean(hyp), '(', np.std(hyp),')')
#print('SRN',np.max(hyp))
#print('SRN',np.std(hyp))
#print(funcname,algorithm.nfe)


random.seed(0)
for d in range(14,18,1):
    for g in range(17,23,1):
        nfes = []
        hyp = []
        for i in range(10):
            problem = Problem(2,2,2)
            problem.types[:] = [Real(1e-5,np.pi),Real(1e-5,np.pi)]
            problem.constraints[:] = "<=0"
            problem.function = TNK
            algorithm = NSGAIII(problem,d*problem.nvars)
            algorithm.run(d*g*problem.nvars)
            
            funcname = 'TNK'
            # if not os.path.exists(funcname):
            #     os.makedirs(funcname)
                
            nondominated_solutions = nondominated(algorithm.result)
            ref = np.array([3,3])
            obj = []
            for s in nondominated_solutions:
                lijst = str(s.objectives)
                obj.append(ast.literal_eval(lijst))
            obj = np.array(obj)
            # np.savetxt(str(funcname)+'/'+str(funcname)+'_pf_run_'+str(i)+'.csv', obj, delimiter=',')
            hyp.append(hypervolume(obj,ref))
            nfes.append(algorithm.nfe)
        print(np.mean(hyp))
        if np.mean(hyp) > 7.65680404482:
            print(funcname,np.mean(hyp), '(', np.std(hyp),')',g,d, np.mean(nfes))

print('TNK',np.mean(hyp), '(', np.std(hyp),')')
#print('TNK',np.max(hyp))
#print('TNK',np.std(hyp))
#print(funcname,algorithm.nfe)


random.seed(0)
for d in range(7,12,1):
    for g in range(7,12,1):
        nfes = []
        hyp = []
        for i in range(10):
            problem = Problem(2,2,2)
            problem.types[:] = [Real(0,1),Real(0,1)]
            problem.constraints[:] = "<=0"
            problem.function = CTP1
            algorithm = NSGAIII(problem,d*problem.nvars)
            algorithm.run(d*g*problem.nvars)
            
            funcname = 'CTP1'
            # if not os.path.exists(funcname):
            #     os.makedirs(funcname)
                
            nondominated_solutions = nondominated(algorithm.result)
            ref = np.array([1,2])
            obj = []
            for s in nondominated_solutions:
                lijst = str(s.objectives)
                obj.append(ast.literal_eval(lijst))
            obj = np.array(obj)
            # np.savetxt(str(funcname)+'/'+str(funcname)+'_pf_run_'+str(i)+'.csv', obj, delimiter=',')
            hyp.append(hypervolume(obj,ref))
            nfes.append(algorithm.nfe)
        print(np.mean(hyp))
        if np.mean(hyp) > 1.23982829506:
            print(funcname,np.mean(hyp), '(', np.std(hyp),')',g,d, np.mean(nfes))


print('CTP1',np.mean(hyp), '(', np.std(hyp),')')
#print('CTP1',np.max(hyp))
#print('CTP1',np.std(hyp))
#print(funcname,algorithm.nfe)

random.seed(0)
for d in range(80,100,2):
    for g in range(40,55,2):
        nfes = []
        hyp = []
        for i in range(10):
            problem = Problem(6,2,2)
            problem.types[:] = [Real(0,1),Real(0,1),Real(0,1),Real(0,1),Real(0,1),Real(0,1)]
            problem.constraints[:] = "<=0"
            problem.function = C3DTLZ4
            algorithm = NSGAIII(problem,d*problem.nvars)
            algorithm.run(d*g*problem.nvars)
            
            funcname = 'C3DTLZ4'
            # if not os.path.exists(funcname):
            #     os.makedirs(funcname)
                
            nondominated_solutions = nondominated(algorithm.result)
            ref = np.array([3,3])
            obj = []
            for s in nondominated_solutions:
                lijst = str(s.objectives)
                obj.append(ast.literal_eval(lijst))
            obj = np.array(obj)
            # np.savetxt(str(funcname)+'/'+str(funcname)+'_pf_run_'+str(i)+'.csv', obj, delimiter=',')
            hyp.append(hypervolume(obj,ref))
            nfes.append(algorithm.nfe)
        print(np.mean(hyp))
        if np.mean(hyp) > 6.44303205286:
            print(funcname,np.mean(hyp), '(', np.std(hyp),')',g,d, np.mean(nfes))

print('C3DTLZ4',np.mean(hyp), '(', np.std(hyp),')')
#print('C3DTLZ4',np.max(hyp))
#print('C3DTLZ4',np.std(hyp))
#print(funcname,algorithm.nfe)


random.seed(0)
for d in range(65,72,1):
    for g in range(25,35,1):
        nfes = []
        hyp = []
        for i in range(10):
            problem = Problem(6,2,6)
            problem.types[:] = [Real(0,10),Real(0,10),Real(1,5),Real(0,6),Real(1,5),Real(0,10)]
            problem.constraints[:] = "<=0"
            problem.function = OSY
            algorithm = NSGAIII(problem,d*problem.nvars)
            algorithm.run(d*g*problem.nvars)
            
            funcname = 'OSY'
            # if not os.path.exists(funcname):
            #     os.makedirs(funcname)
                
            nondominated_solutions = nondominated(algorithm.result)
            ref = np.array([0,386])
            obj = []
            for s in nondominated_solutions:
                lijst = str(s.objectives)
                obj.append(ast.literal_eval(lijst))
            obj = np.array(obj)
            # np.savetxt(str(funcname)+'/'+str(funcname)+'_pf_run_'+str(i)+'.csv', obj, delimiter=',')
            hyp.append(hypervolume(obj,ref))
            nfes.append(algorithm.nfe)
        print(np.mean(hyp))
        if np.mean(hyp) > 95592.8275427:
            print(funcname,np.mean(hyp), '(', np.std(hyp),')',g,d, np.mean(nfes))

print('OSY',np.mean(hyp), '(', np.std(hyp),')')
#print('OSY',np.max(hyp))
#print('OSY',np.std(hyp))
#print(funcname,algorithm.nfe)


random.seed(0)
for d in range(5,15,1):
    for g in range(5,15,1):
        hyp = []
        nfes = []
        for i in range(10):
            problem = Problem(3,2,3)
            problem.types[:] = [Real(1,3),Real(0.0005,0.05),Real(0.0005,0.05)]
            problem.constraints[:] = "<=0"
            problem.function = TBTD
            algorithm = NSGAIII(problem,d*problem.nvars)
            algorithm.run(d*g*problem.nvars)
            
            funcname = 'TBTD'
            # if not os.path.exists(funcname):
            #     os.makedirs(funcname)
                
            nondominated_solutions = nondominated(algorithm.result)
            ref = np.array([0.1,50000])
            obj = []
            for s in nondominated_solutions:
                lijst = str(s.objectives)
                obj.append(ast.literal_eval(lijst))
            obj = np.array(obj)
            # np.savetxt(str(funcname)+'/'+str(funcname)+'_pf_run_'+str(i)+'.csv', obj, delimiter=',')
            hyp.append(hypervolume(obj,ref))
            nfes.append(algorithm.nfe)
        print(np.mean(hyp))
        if np.mean(hyp) > 3925:
            print(funcname,np.mean(hyp), '(', np.std(hyp),')',g,d, np.mean(nfes))

print('TBTD',np.mean(hyp), '(', np.std(hyp),')')
#print('TBTD',np.max(hyp))
#print('TBTD',np.std(hyp))
#print(funcname,algorithm.nfe)

random.seed(0)
for d in range(7,14,1):
    for g in range(7,14,1):
        hyp = []
        nfes = []
        for i in range(10):
            problem = Problem(2,2,5)
            problem.types[:] = [Real(20,250),Real(10,50)]
            problem.constraints[:] = "<=0"
            problem.function = NBP
            algorithm = NSGAIII(problem,d*problem.nvars)
            algorithm.run(d*g*problem.nvars)
            
            funcname = 'NBP'
            # if not os.path.exists(funcname):
            #     os.makedirs(funcname)
                
            nondominated_solutions = nondominated(algorithm.result)
            ref =  np.array([11150, 12500])
            obj = []
            for s in nondominated_solutions:
                lijst = str(s.objectives)
                obj.append(ast.literal_eval(lijst))
            obj = np.array(obj)
            # np.savetxt(str(funcname)+'/'+str(funcname)+'_pf_run_'+str(i)+'.csv', obj, delimiter=',')
            hyp.append(hypervolume(obj,ref))
            nfes.append(algorithm.nfe)
        print(np.mean(hyp))
        if np.mean(hyp) > 102407195:
            print(funcname,np.mean(hyp), '(', np.std(hyp),')',g,d, np.mean(nfes))

print('NBP',np.mean(hyp), '(', np.std(hyp),')')
#print('NBP',np.max(hyp))
#print('NBP',np.std(hyp))
#print(funcname,algorithm.nfe)

random.seed(0)
for d in range(3,10,1):
    for g in range(3,10,1):
        hyp = []
        nfes = []
        for i in range(10):
            problem = Problem(4,2,5)
            problem.types[:] = [Real(55,80),Real(75,110),Real(1000,3000),Real(2,20)]
            problem.constraints[:] = "<=0"
            problem.function = DBD
            algorithm = NSGAIII(problem,d*problem.nvars)
            algorithm.run(d*g*problem.nvars)
            
            funcname = 'DBD'
            # if not os.path.exists(funcname):
            #     os.makedirs(funcname)
                
            nondominated_solutions = nondominated(algorithm.result)
            ref = np.array([5,50])
            obj = []
            for s in nondominated_solutions:
                lijst = str(s.objectives)
                obj.append(ast.literal_eval(lijst))
            obj = np.array(obj)
            # np.savetxt(str(funcname)+'/'+str(funcname)+'_pf_run_'+str(i)+'.csv', obj, delimiter=',')
            hyp.append(hypervolume(obj,ref))
            nfes.append(algorithm.nfe)
        print(np.mean(hyp))
        if np.mean(hyp) > 217.30940:
            print(funcname,np.mean(hyp), '(', np.std(hyp),')',g,d, np.mean(nfes))

print('DBD',np.mean(hyp), '(', np.std(hyp),')')
#print('DBD',np.max(hyp))
#print('DBD',np.std(hyp))
#print(funcname,algorithm.nfe)

random.seed(0)
for d in range(250,300,10):
    for g in range(30,40,5):
        hyp = []
        nfes = []
        for i in range(10):
            problem = Problem(6,3,9)
            problem.types[:] = [Real(150,274.32),Real(25,32.31),Real(12,22),Real(8,11.71),Real(14,18),Real(0.63,0.75)]
            problem.constraints[:] = "<=0"
            problem.function = SPD
            algorithm = NSGAIII(problem,d*problem.nvars)
            algorithm.run(d*g*problem.nvars)
            
            funcname = 'SPD'
            # if not os.path.exists(funcname):
            #     os.makedirs(funcname)
                
            nondominated_solutions = nondominated(algorithm.result)
            ref = np.array([16,19000,-260000])
            obj = []
            for s in nondominated_solutions:
                lijst = str(s.objectives)
                obj.append(ast.literal_eval(lijst))
            obj = np.array(obj)
            # np.savetxt(str(funcname)+'/'+str(funcname)+'_pf_run_'+str(i)+'.csv', obj, delimiter=',')
            hyp.append(hypervolume(obj,ref))
            nfes.append(algorithm.nfe)
        print(np.mean(hyp))
        if np.mean(hyp) > 36886805013.7:
            print(funcname,np.mean(hyp), '(', np.std(hyp),')',g,d, np.mean(nfes))

print('SPD',np.mean(hyp), '(', np.std(hyp),')')
#print('SPD',np.max(hyp))
#print('SPD',np.std(hyp))
#print(funcname,algorithm.nfe)

random.seed(0)
for d in range(15,20,1):
    for g in range(150,151,5):
        hyp = []
        nfes = []
        for i in range(10):
            problem = Problem(7,3,10)
            problem.types[:] = [Real(0.5,1.5),Real(0.45,1.35),Real(0.5,1.5),Real(0.5,1.5),Real(0.875,2.625),Real(0.4,1.2),Real(0.4,1.2)]
            problem.constraints[:] = "<=0"
            problem.function = CSI
            algorithm = NSGAIII(problem,d*problem.nvars)
            algorithm.run(d*g*problem.nvars)
            
            funcname = 'CSI'
            # if not os.path.exists(funcname):
            #     os.makedirs(funcname)
                
            nondominated_solutions = nondominated(algorithm.result)
            ref = np.array([42,4.5,13])
            obj = []
            for s in nondominated_solutions:
                lijst = str(s.objectives)
                obj.append(ast.literal_eval(lijst))
            obj = np.array(obj)
            # np.savetxt(str(funcname)+'/'+str(funcname)+'_pf_run_'+str(i)+'.csv', obj, delimiter=',')
            hyp.append(hypervolume(obj,ref))
            nfes.append(algorithm.nfe)
        print(np.mean(hyp))
        if np.mean(hyp) > 25.7171858898:
            print(funcname,np.mean(hyp), '(', np.std(hyp),')',g,d, np.mean(nfes))
            
print('CSI',np.mean(hyp), '(', np.std(hyp),')')
#print('CSI',np.max(hyp))
#print('CSI',np.std(hyp))
#print(funcname,algorithm.nfe)

random.seed(0)
for d in range(10,15,1):
    for g in range(16,22,1):
        hyp = []
        nfes = []
        for i in range(10):
            problem = Problem(7,2,11)
            problem.types[:] = [Real(2.6,3.6),Real(0.7,0.8),Real(17,28),Real(7.3,8.3),Real(7.3,8.3),Real(2.9,3.9),Real(5,5.5)]
            problem.constraints[:] = "<=0"
            problem.function = SRD
            algorithm = NSGAIII(problem,d*problem.nvars)
            algorithm.run(d*g*problem.nvars)
            
            funcname = 'SRD'
            # if not os.path.exists(funcname):
            #     os.makedirs(funcname)
                
            nondominated_solutions = nondominated(algorithm.result)
            ref = np.array([7000,1700])
            obj = []
            for s in nondominated_solutions:
                lijst = str(s.objectives)
                obj.append(ast.literal_eval(lijst))
            obj = np.array(obj)
            # np.savetxt(str(funcname)+'/'+str(funcname)+'_pf_run_'+str(i)+'.csv', obj, delimiter=',')
            # print(hypervolume(obj,ref))
            hyp.append(hypervolume(obj,ref))
            nfes.append(algorithm.nfe)
        print(np.mean(hyp))
        if np.mean(hyp) > 3997308.6674:
            print(funcname,np.mean(hyp), '(', np.std(hyp),')',g,d, np.mean(nfes))

print('SRD',np.mean(hyp),'(',np.std(hyp),')')
#print('SRD',np.max(hyp))
#print('SRD',np.std(hyp))
#print(funcname,algorithm.nfe)

random.seed(0)
for d in range(1,10,1):
    for g in range(1,10,1):
        hyp = []
        nfes = []
        for i in range(10):
            problem = Problem(4,2,5)
            problem.types[:] = [Real(0.125,5),Real(0.1,10),Real(0.1,10),Real(0.125,5)]
            problem.constraints[:] = "<=0"
            problem.function = WB
            algorithm = NSGAIII(problem,d*problem.nvars)
            algorithm.run(d*g*problem.nvars)
            
            funcname = 'WB'
            # if not os.path.exists(funcname):
            #     os.makedirs(funcname)
                
            nondominated_solutions = nondominated(algorithm.result)
            ref = np.array([350,0.1])
            obj = []
            for s in nondominated_solutions:
                lijst = str(s.objectives)
                obj.append(ast.literal_eval(lijst))
            obj = np.array(obj)
            # np.savetxt(str(funcname)+'/'+str(funcname)+'_pf_run_'+str(i)+'.csv', obj, delimiter=',')
            hyp.append(hypervolume(obj,ref))
            nfes.append(algorithm.nfe)
        print(np.mean(hyp))
        if np.mean(hyp) > 32.9034195509:
            print(funcname,np.mean(hyp), '(', np.std(hyp),')',g,d, np.mean(nfes))

print('WB',np.mean(hyp), '(', np.std(hyp),')')
#print('WB',np.max(hyp))
#print('WB',np.std(hyp))
#print(funcname,algorithm.nfe)

random.seed(0)
for d in range(9,13,1):
    for g in range(17,20,1):
        hyp = []
        nfes = []
        for i in range(10):
            problem = Problem(10,2,1)
            problem.types[:] = [Real(0,1),Real(0,1),Real(0,1),Real(0,1),Real(0,1),Real(0,1),Real(0,1),Real(0,1),Real(0,1),Real(0,1)]
            problem.constraints[:] = "<=0"
            problem.function = BICOP1
            algorithm = NSGAIII(problem,d*problem.nvars)
            algorithm.run(d*g*problem.nvars)
            
            funcname = 'BICOP1'
            # if not os.path.exists(funcname):
            #     os.makedirs(funcname)
                
            nondominated_solutions = nondominated(algorithm.result)
            ref = np.array([9,9])
            obj = []
            for s in nondominated_solutions:
                lijst = str(s.objectives)
                obj.append(ast.literal_eval(lijst))
            obj = np.array(obj)
            # np.savetxt(str(funcname)+'/'+str(funcname)+'_pf_run_'+str(i)+'.csv', obj, delimiter=',')
            hyp.append(hypervolume(obj,ref))
            nfes.append(algorithm.nfe)
        print(np.mean(hyp))
        if np.mean(hyp) > 76.632825634:
            print(funcname,np.mean(hyp), '(', np.std(hyp),')',g,d, np.mean(nfes))
            
print('BICOP1',np.mean(hyp), '(', np.std(hyp),')')
#print('BICOP1',np.max(hyp))
#print('BICOP1',np.std(hyp))
#print(funcname,algorithm.nfe)

random.seed(0)
for d in range(1,2,1):
    for g in range(1,2,1):
        hyp = []
        nfes = []
        hyp = []
        for i in range(10):
            problem = Problem(10,2,2)
            problem.types[:] = [Real(0,1),Real(0,1),Real(0,1),Real(0,1),Real(0,1),Real(0,1),Real(0,1),Real(0,1),Real(0,1),Real(0,1)]
            problem.constraints[:] = "<=0"
            problem.function = BICOP2
            algorithm = NSGAIII(problem,d*problem.nvars)
            algorithm.run(d*g*problem.nvars)
            
            funcname = 'BICOP2'
            # if not os.path.exists(funcname):
            #     os.makedirs(funcname)
                
            nondominated_solutions = nondominated(algorithm.result)
            ref = np.array([70,70])
            obj = []
            for s in nondominated_solutions:
                lijst = str(s.objectives)
                obj.append(ast.literal_eval(lijst))
            obj = np.array(obj)
            # np.savetxt(str(funcname)+'/'+str(funcname)+'_pf_run_'+str(i)+'.csv', obj, delimiter=',')
            hyp.append(hypervolume(obj,ref))
            nfes.append(algorithm.nfe)
        print(np.mean(hyp))
        if np.mean(hyp) > 4606.57390886:
            print(funcname,np.mean(hyp), '(', np.std(hyp),')',g,d, np.mean(nfes))

print('BICOP2',np.mean(hyp), '(', np.std(hyp),')')
#print('BICOP2',np.max(hyp))
#print('BICOP2',np.std(hyp))
#print(funcname,algorithm.nfe)

random.seed(0)
for d in range(1,2,1):
    for g in range(1,2,1):
        hyp = []
        nfes = []
        hyp = []
        for i in range(10):
            problem = Problem(2,3,3)
            problem.types[:] = [Real(-4,4),Real(-4,4)]
            problem.constraints[:] = "<=0"
            problem.function = TRICOP
            algorithm = NSGAIII(problem,d*problem.nvars)
            algorithm.run(d*g*problem.nvars)
            
            funcname = 'TRICOP'
            # if not os.path.exists(funcname):
            #     os.makedirs(funcname)
                
            nondominated_solutions = nondominated(algorithm.result)
            ref = np.array([34,-4,90])
            obj = []
            for s in nondominated_solutions:
                lijst = str(s.objectives)
                obj.append(ast.literal_eval(lijst))
            obj = np.array(obj)
            # np.savetxt(str(funcname)+'/'+str(funcname)+'_pf_run_'+str(i)+'.csv', obj, delimiter=',')
            hyp.append(hypervolume(obj,ref))
            nfes.append(algorithm.nfe)
        print(np.mean(hyp))
        if np.mean(hyp) > 4606.57390886:
            print(funcname,np.mean(hyp), '(', np.std(hyp),')',g,d, np.mean(nfes))

print('TRICOP',np.mean(hyp), '(', np.std(hyp),')')
#print('TRICOP',np.max(hyp))
#print('TRICOP',np.std(hyp))
#print(funcname,algorithm.nfe)


random.seed(0)
for d in range(1,7):
    for g in range(1,7):
        hyp = []
        nfes = []
        for i in range(10):
            problem = Problem(3,5,7)
            problem.types[:] = [Real(0.01,0.45),Real(0.01,0.1),Real(0.01,0.1)]
            problem.constraints[:] = "<=0"
            problem.function = WP
            algorithm = NSGAIII(problem,d*problem.nvars)
            algorithm.run(d*g*problem.nvars)
            
            funcname = 'WP'
            # if not os.path.exists(funcname):
            #     os.makedirs(funcname)
                
            nondominated_solutions = nondominated(algorithm.result)
            ref = np.array([83000, 1350, 2.85, 15989825, 25000])
            obj = []
            for s in nondominated_solutions:
                lijst = str(s.objectives)
                obj.append(ast.literal_eval(lijst))
            obj = np.array(obj)
            # np.savetxt(str(funcname)+'/'+str(funcname)+'_pf_run_'+str(i)+'.csv', obj, delimiter=',')
            hyp.append(hypervolume(obj,ref))
            nfes.append(algorithm.nfe)
        print(np.mean(hyp))
        if np.mean(hyp) > 1.5147434E19:
            print(funcname,np.mean(hyp), '(', np.std(hyp),')',g,d, np.mean(nfes))
            
print('WP',np.mean(hyp), '(', np.std(hyp),')')
#print('WP',np.max(hyp))
#print('WP',np.std(hyp))
#print(funcname,algorithm.nfe)