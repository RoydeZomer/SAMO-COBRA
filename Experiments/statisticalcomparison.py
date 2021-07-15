# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 09:37:08 2021

@author: r.dewinter
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 14:29:39 2020

@author: r.dewinter
"""

from scipy.stats import ranksums
from scipy.stats import kruskal
from hypervolume import hypervolume
import numpy as np
import json

from NSGAIIexperimentshv import NSGAII_Experiment
from NSGAIIIexperimentshv import NSGAIII_Experiment


# NSGAII_results = NSGAII_Experiment()
# with open("nsgaii.json", 'w') as fout:
#     json_dumps_str = json.dumps(NSGAII_results, indent=4)
#     print(json_dumps_str, file=fout)
with open("C:/Users/r.dewinter/Downloads/SAMO-COBRA-main/Experiments//results/nsgaii.json") as f:
    NSGAII_results = json.load(f)
    
# NSGAIII_results=NSGAIII_Experiment()
# with open("nsgaiii.json", 'w') as fout:
#     json_dumps_str = json.dumps(NSGAIII_results, indent=4)
#     print(json_dumps_str, file=fout)

with open("C:/Users/r.dewinter/Downloads/SAMO-COBRA-main/Experiments/results/nsgaiii.json") as f:
    NSGAIII_results = json.load(f)
    
with open("C:/Users/r.dewinter/Downloads/SAMO-COBRA-main/Experiments//results/sansgaii.json") as f:
    SANSGA_results = json.load(f)

with open("C:/Users/r.dewinter/Downloads/SAMO-COBRA-main/Experiments//results/icsansgaii.json") as f:
    ICSANSGA_results = json.load(f)
    
CEGO_results = {}
PHV_results = {}
SMS_results = {}

funcNames = ["BNH","CEXP","SRN","TNK","C3DTLZ4","CTP1","OSY","TBTD","NBP","DBD","WP","SPD","SRD","WB","TRICOP","CSI","BICOP1","BICOP2"]
refs = [np.array([140,50]),np.array([1,9]),np.array([301,72]),np.array([3,3]),np.array([3,3]),np.array([1,2]),np.array([0,386]),np.array([0.1,50000]),np.array([11150,12500]),np.array([5,50]),np.array([83000,1350,2.85,15989825,25000]),np.array([16,19000,-260000]),np.array([7000,1700]),np.array([350,0.1]),np.array([34,-4,90]),np.array([42,4.5,13]),np.array([9,9]),np.array([70,70])]


SMSFolder = 'C:/Users/r.dewinter/Downloads/SAMO-COBRA-main/experiments/results/SMS/'
for i in range(len(funcNames)):
    hypSMS = []
    funcName = funcNames[i]
    ref = refs[i]
    
    for run in range(10):
        fname = str(funcName)+'/obj_run'+str(run)+'_finalPF.csv'
        file = SMSFolder+fname
        # file = cegoFolderFunction+'/obj_run'+str(run)+'.csv'
        objectives = np.genfromtxt(file, delimiter=',')
        hypSMS.append(hypervolume(objectives, ref))
    SMS_results[funcName] = hypSMS

PHVFolder = 'C:/Users/r.dewinter/Downloads/SAMO-COBRA-main/experiments/results/PHV/'
for i in range(len(funcNames)):
    hypPHV = []
    funcName = funcNames[i]
    ref = refs[i]
    
    for run in range(10):
        fname = '/'+str(funcName)+'/obj_run'+str(run)+'_finalPF.csv'
        file = PHVFolder+fname
        # file = cegoFolderFunction+'/obj_run'+str(run)+'.csv'
        objectives = np.genfromtxt(file, delimiter=',')
        hypPHV.append(hypervolume(objectives, ref))
    PHV_results[funcName] = hypPHV


CEGOFolder = 'C:/Users/r.dewinter/Downloads/SAMO-COBRA-main/experiments/results/CEGO/'
for i in range(len(funcNames)):
    hypCEGO = []
    funcName = funcNames[i]
    ref = refs[i]
    try:
        for run in range(10):
            fname = '/'+str(funcName)+'/obj_run'+str(run)+'_finalPF.csv'
            file = CEGOFolder+fname
            # file = cegoFolderFunction+'/obj_run'+str(run)+'.csv'
            objectives = np.genfromtxt(file, delimiter=',')
            hypCEGO.append(hypervolume(objectives, ref))
        CEGO_results[funcName] = hypCEGO
    except:
        print(funcName, run)
    
for funcName in funcNames:
    try:
        print(funcName)
        hypPHV = PHV_results[funcName]
        hypSMS = SMS_results[funcName]
        hypCEGO = CEGO_results[funcName]
        hypNSII = NSGAII_results[funcName]
        hypNSIII = NSGAIII_results[funcName]
        hypSANSII = SANSGA_results[funcName]
        hypICNSII = ICSANSGA_results[funcName]
        
        print(np.mean(hypPHV), '(',np.std(hypPHV),')')
        print(np.mean(hypSMS), '(',np.std(hypSMS),')')
        print(np.mean(hypCEGO), '(',np.std(hypCEGO),')')
        print(np.mean(hypNSII), '(',np.std(hypNSII),')')
        print(np.mean(hypNSIII), '(',np.std(hypNSIII),')')
        print(np.mean(hypSANSII), '(',np.std(hypSANSII),')')
        print(np.mean(hypICNSII), '(',np.std(hypICNSII),')')
        
        print('Kruskal test:', kruskal(hypPHV, hypSMS, hypCEGO, hypNSII, hypNSIII, hypSANSII, hypICNSII))
        
        #multiply with 6 for bonferroni correction. 
        if (np.mean(hypPHV) > np.mean(hypSMS) and np.mean(hypPHV) > np.mean(hypCEGO) and np.mean(hypPHV) > np.mean(hypNSII) and np.mean(hypPHV) > np.mean(hypNSIII) and np.mean(hypPHV) > np.mean(hypSANSII) and np.mean(hypPHV) > np.mean(hypICNSII)):
            print('PHV SMS', ranksums(hypPHV, hypSMS).pvalue*6)
            print('PHV CEGO', ranksums(hypPHV, hypCEGO).pvalue*6)
            print('PHV NSGAII', ranksums(hypPHV, hypNSII).pvalue*6)
            print('PHV NSGAIII', ranksums(hypPHV, hypNSIII).pvalue*6)
            print('PHV SANSGA', ranksums(hypPHV, hypSANSII).pvalue*6)
            print('PHV ICSANSGA', ranksums(hypPHV, hypICNSII).pvalue*6)
            
        if (np.mean(hypSMS) > np.mean(hypPHV) and np.mean(hypSMS) > np.mean(hypCEGO) and np.mean(hypSMS) > np.mean(hypNSII) and np.mean(hypSMS) > np.mean(hypNSIII) and np.mean(hypSMS) > np.mean(hypSANSII) and np.mean(hypSMS) > np.mean(hypICNSII)):
            print('SMS PHV', ranksums(hypSMS, hypPHV).pvalue*6)
            print('SMS CEGO', ranksums(hypSMS, hypCEGO).pvalue*6)
            print('SMS NSGAII', ranksums(hypSMS, hypNSII).pvalue*6)
            print('SMS NSGAIII', ranksums(hypSMS, hypNSIII).pvalue*6)
            print('SMS SANSGA', ranksums(hypSMS, hypSANSII).pvalue*6)
            print('SMS ICSANSGA', ranksums(hypSMS, hypICNSII).pvalue*6)
            
        if (np.mean(hypCEGO) > np.mean(hypPHV) and np.mean(hypCEGO) > np.mean(hypSMS) and np.mean(hypCEGO) > np.mean(hypNSII) and np.mean(hypCEGO) > np.mean(hypNSIII) and np.mean(hypCEGO) > np.mean(hypSANSII) and np.mean(hypCEGO) > np.mean(hypICNSII)):
            print('CEGO PHV', ranksums(hypCEGO, hypPHV).pvalue*6)
            print('CEGO SMS', ranksums(hypCEGO, hypSMS).pvalue*6)
            print('CEGO NSGAII', ranksums(hypCEGO, hypNSII).pvalue*6)
            print('CEGO NSGAIII', ranksums(hypCEGO, hypNSIII).pvalue*6)
            print('CEGO SANSGA', ranksums(hypCEGO, hypSANSII).pvalue*6)
            print('CEGO ICSANSGA', ranksums(hypCEGO, hypICNSII).pvalue*6)

        if (np.mean(hypNSII) > np.mean(hypPHV) and np.mean(hypNSII) > np.mean(hypSMS) and np.mean(hypNSII) > np.mean(hypCEGO) and np.mean(hypNSII) > np.mean(hypNSIII) and np.mean(hypNSII) > np.mean(hypSANSII) and np.mean(hypNSII) > np.mean(hypICNSII)):
            print('NSGAII PHV', ranksums(hypNSII, hypPHV).pvalue*6)
            print('NSGAII SMS', ranksums(hypNSII, hypSMS).pvalue*6)
            print('NSGAII CEGO', ranksums(hypNSII, hypCEGO).pvalue*6)
            print('NSGAII NSGAIII', ranksums(hypNSII, hypNSIII).pvalue*6)
            print('NSGAII SANSGA', ranksums(hypNSII, hypSANSII).pvalue*6)
            print('NSGAII ICSANSGA', ranksums(hypNSII, hypICNSII).pvalue*6)
            
        if (np.mean(hypNSIII) > np.mean(hypPHV) and np.mean(hypNSIII) > np.mean(hypSMS) and np.mean(hypNSIII) > np.mean(hypCEGO) and np.mean(hypNSIII) > np.mean(hypNSII) and np.mean(hypNSIII) > np.mean(hypSANSII) and np.mean(hypNSIII) > np.mean(hypICNSII)):
            print('NSGAIII PHV', ranksums(hypNSIII, hypPHV).pvalue*6)
            print('NSGAIII SMS', ranksums(hypNSIII, hypSMS).pvalue*6)
            print('NSGAIII CEGO', ranksums(hypNSIII, hypCEGO).pvalue*6)
            print('NSGAIII NSGAII', ranksums(hypNSIII, hypNSII).pvalue*6)
            print('NSGAIII SANSGA', ranksums(hypNSIII, hypSANSII).pvalue*6)
            print('NSGAIII ICSANSGA', ranksums(hypNSIII, hypICNSII).pvalue*6)

        if (np.mean(hypSANSII) > np.mean(hypPHV) and np.mean(hypSANSII) > np.mean(hypSMS) and np.mean(hypSANSII) > np.mean(hypCEGO) and np.mean(hypSANSII) > np.mean(hypNSII) and np.mean(hypSANSII) > np.mean(hypSANSII) and np.mean(hypSANSII) > np.mean(hypNSIII)):
            print('SANSGA PHV', ranksums(hypSANSII, hypPHV).pvalue*6)
            print('SANSGA SMS', ranksums(hypSANSII, hypSMS).pvalue*6)
            print('SANSGA CEGO', ranksums(hypSANSII, hypCEGO).pvalue*6)
            print('SANSGA NSGAII', ranksums(hypSANSII, hypNSII).pvalue*6)
            print('SANSGA NSGAIII', ranksums(hypSANSII, hypNSIII).pvalue*6)
            print('SANSGA ICSANSGA', ranksums(hypSANSII, hypICNSII).pvalue*6)

        if (np.mean(hypICNSII) > np.mean(hypPHV) and np.mean(hypICNSII) > np.mean(hypSMS) and np.mean(hypICNSII) > np.mean(hypCEGO) and np.mean(hypICNSII) > np.mean(hypNSII) and np.mean(hypICNSII) > np.mean(hypSANSII) and np.mean(hypICNSII) > np.mean(hypNSIII)):
            print('ICSANSGA PHV', ranksums(hypICNSII, hypPHV).pvalue*6)
            print('ICSANSGA SMS', ranksums(hypICNSII, hypSMS).pvalue*6)
            print('ICSANSGA CEGO', ranksums(hypICNSII, hypCEGO).pvalue*6)
            print('ICSANSGA NSGAII', ranksums(hypICNSII, hypNSII).pvalue*6)
            print('ICSANSGA NSGAIII', ranksums(hypICNSII, hypNSIII).pvalue*6)
            print('ICSANSGA SANSGA', ranksums(hypICNSII, hypSANSII).pvalue*6)

    except:
        print(funcName)
        hypPHV = PHV_results[funcName]
        hypSMS = SMS_results[funcName]
        hypNSII = NSGAII_results[funcName]
        hypNSIII = NSGAIII_results[funcName]
        hypSANSII = SANSGA_results[funcName]
        hypICNSII = ICSANSGA_results[funcName]
        
        print(np.mean(hypPHV), '(',np.std(hypPHV),')')
        print(np.mean(hypSMS), '(',np.std(hypSMS),')')
        print(np.mean(hypNSII), '(',np.std(hypNSII),')')
        print(np.mean(hypNSIII), '(',np.std(hypNSIII),')')
        print(np.mean(hypSANSII), '(',np.std(hypSANSII),')')
        print(np.mean(hypICNSII), '(',np.std(hypICNSII),')')
        
        print('Kruskal test:', kruskal(hypPHV, hypSMS, hypNSII, hypNSIII, hypSANSII, hypICNSII))
        
        #multiply with 6 for bonferroni correction. 
        if (np.mean(hypPHV) > np.mean(hypSMS) and np.mean(hypPHV) > np.mean(hypNSII) and np.mean(hypPHV) > np.mean(hypNSIII) and np.mean(hypPHV) > np.mean(hypSANSII) and np.mean(hypPHV) > np.mean(hypICNSII)):
            print('PHV SMS', ranksums(hypPHV, hypSMS).pvalue*6)
            print('PHV NSGAII', ranksums(hypPHV, hypNSII).pvalue*6)
            print('PHV NSGAIII', ranksums(hypPHV, hypNSIII).pvalue*6)
            print('PHV SANSGA', ranksums(hypPHV, hypSANSII).pvalue*6)
            print('PHV ICSANSGA', ranksums(hypPHV, hypICNSII).pvalue*6)
            
        if (np.mean(hypSMS) > np.mean(hypPHV) and np.mean(hypSMS) > np.mean(hypNSII) and np.mean(hypSMS) > np.mean(hypNSIII) and np.mean(hypSMS) > np.mean(hypSANSII) and np.mean(hypSMS) > np.mean(hypICNSII)):
            print('SMS PHV', ranksums(hypSMS, hypPHV).pvalue*6)
            print('SMS NSGAII', ranksums(hypSMS, hypNSII).pvalue*6)
            print('SMS NSGAIII', ranksums(hypSMS, hypNSIII).pvalue*6)
            print('SMS SANSGA', ranksums(hypSMS, hypSANSII).pvalue*6)
            print('SMS ICSANSGA', ranksums(hypSMS, hypICNSII).pvalue*6)

        if (np.mean(hypNSII) > np.mean(hypPHV) and np.mean(hypNSII) > np.mean(hypSMS) and np.mean(hypNSII) > np.mean(hypNSIII) and np.mean(hypNSII) > np.mean(hypSANSII) and np.mean(hypNSII) > np.mean(hypICNSII)):
            print('NSGAII PHV', ranksums(hypNSII, hypPHV).pvalue*6)
            print('NSGAII SMS', ranksums(hypNSII, hypSMS).pvalue*6)
            print('NSGAII NSGAIII', ranksums(hypNSII, hypNSIII).pvalue*6)
            print('NSGAII SANSGA', ranksums(hypNSII, hypSANSII).pvalue*6)
            print('NSGAII ICSANSGA', ranksums(hypNSII, hypICNSII).pvalue*6)
            
        if (np.mean(hypNSIII) > np.mean(hypPHV) and np.mean(hypNSIII) > np.mean(hypSMS) and np.mean(hypNSIII) > np.mean(hypNSII) and np.mean(hypNSIII) > np.mean(hypSANSII) and np.mean(hypNSIII) > np.mean(hypICNSII)):
            print('NSGAIII PHV', ranksums(hypNSIII, hypPHV).pvalue*6)
            print('NSGAIII SMS', ranksums(hypNSIII, hypSMS).pvalue*6)
            print('NSGAIII NSGAII', ranksums(hypNSIII, hypNSII).pvalue*6)
            print('NSGAIII SANSGA', ranksums(hypNSIII, hypSANSII).pvalue*6)
            print('NSGAIII ICSANSGA', ranksums(hypNSIII, hypICNSII).pvalue*6)

        if (np.mean(hypSANSII) > np.mean(hypPHV) and np.mean(hypSANSII) > np.mean(hypSMS) and np.mean(hypSANSII) > np.mean(hypNSII) and np.mean(hypSANSII) > np.mean(hypSANSII) and np.mean(hypSANSII) > np.mean(hypNSIII)):
            print('SANSGA PHV', ranksums(hypSANSII, hypPHV).pvalue*6)
            print('SANSGA SMS', ranksums(hypSANSII, hypSMS).pvalue*6)
            print('SANSGA NSGAII', ranksums(hypSANSII, hypNSII).pvalue*6)
            print('SANSGA NSGAIII', ranksums(hypSANSII, hypNSIII).pvalue*6)
            print('SANSGA ICSANSGA', ranksums(hypSANSII, hypICNSII).pvalue*6)

        if (np.mean(hypICNSII) > np.mean(hypPHV) and np.mean(hypICNSII) > np.mean(hypSMS) and np.mean(hypICNSII) > np.mean(hypNSII) and np.mean(hypICNSII) > np.mean(hypSANSII) and np.mean(hypICNSII) > np.mean(hypNSIII)):
            print('ICSANSGA PHV', ranksums(hypICNSII, hypPHV).pvalue*6)
            print('ICSANSGA SMS', ranksums(hypICNSII, hypSMS).pvalue*6)
            print('ICSANSGA NSGAII', ranksums(hypICNSII, hypNSII).pvalue*6)
            print('ICSANSGA NSGAIII', ranksums(hypICNSII, hypNSIII).pvalue*6)
            print('ICSANSGA SANSGA', ranksums(hypICNSII, hypSANSII).pvalue*6)
    print()
    print()
    
        



print('all', kruskal(hypCEGO, hypPHV, hypSMS))

print('phv sms', kruskal(hypPHV, hypSMS))
print('phv cego', kruskal(hypPHV, hypCEGO))

print('phv sms', ranksums(hypPHV, hypSMS))
print('phv cego', ranksums(hypPHV, hypCEGO))

