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
with open("/results/nsgaii.json") as f:
    NSGAII_results = json.load(f)
    
# NSGAIII_results=NSGAIII_Experiment()
# with open("nsgaiii.json", 'w') as fout:
#     json_dumps_str = json.dumps(NSGAIII_results, indent=4)
#     print(json_dumps_str, file=fout)
with open("/results/nsgaiii.json") as f:
    NSGAIII_results = json.load(f)
    
CEGO_results = {}
PHV_results = {}
SMS_results = {}

funcNames = ["BNH","CEXP","SRN","TNK","C3DTLZ4","CTP1","OSY","TBTD","NBP","DBD","WP","SPD","CSI","SRD","WB","BICOP1","TRICOP","BICOP2"]
refs = [np.array([140,50]),np.array([1,9]), np.array([301,72]),np.array([3,3]),np.array([3,3]),np.array([1,2]),np.array([0,386]),np.array([0.1,50000]),np.array([11150, 12500]),np.array([5,50]),np.array([83000, 1350, 2.85, 15989825, 25000]),np.array([16,19000,-260000]),np.array([42,4.5,13]),np.array([7000,1700]),np.array([350,0.1]),np.array([9,9]),np.array([34,-4,90]),np.array([70,70])]


SMSFolder = '/results/SMS/'
for i in range(len(funcNames)):
    hypSMS = []
    funcName = funcNames[i]
    ref = refs[i]
    
    for run in range(10):
        fname = '/'+str(funcName)+'/obj_run'+str(run)+'_finalPF.csv'
        file = SMSFolder+fname
        # file = cegoFolderFunction+'/obj_run'+str(run)+'.csv'
        objectives = np.genfromtxt(file, delimiter=',')
        hypSMS.append(hypervolume(objectives, ref))
    SMS_results[funcName] = hypSMS

PHVFolder = '/results/PHV/'
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

CEGOFolder = '/results/CEGO/'
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
    hypPHV = PHV_results[funcName]
    hypSMS = SMS_results[funcName]
    hypCEGO = CEGO_results[funcName]
    hypNSII = NSGAII_results[funcName]
    hypNSIII = NSGAIII_results[funcName]
    
    
    print(np.mean(hypPHV), '(',np.std(hypPHV),')')
    print(np.mean(hypSMS), '(',np.std(hypSMS),')')
    print(np.mean(hypCEGO), '(',np.std(hypCEGO),')')
    print(np.mean(hypNSII), '(',np.std(hypNSII),')')
    print(np.mean(hypNSIII), '(',np.std(hypNSIII),')')

    print('Kruskal test:', kruskal(hypPHV, hypSMS, hypCEGO, hypNSII, hypNSIII))
    
    if (np.mean(hypPHV) > np.mean(hypSMS) and np.mean(hypPHV) > np.mean(hypCEGO) and np.mean(hypPHV) > np.mean(hypNSII) and np.mean(hypPHV) > np.mean(hypNSIII)):
        #times 4 for bonferoni correction
        print('PHV SMS', ranksums(hypPHV, hypSMS)*4)
        print('PHV CEGO', ranksums(hypPHV, hypCEGO)*4)
        print('PHV NSGAII', ranksums(hypPHV, hypNSII)*4)
        print('PHV NSGAIII', ranksums(hypPHV, hypNSIII)*4)
        
    if (np.mean(hypSMS) > np.mean(hypPHV) and np.mean(hypSMS) > np.mean(hypCEGO) and np.mean(hypSMS) > np.mean(hypNSII) and np.mean(hypSMS) > np.mean(hypNSIII)):
        #times 4 for bonferoni correction
        print('SMS PHV', ranksums(hypSMS, hypPHV)*4)
        print('SMS CEGO', ranksums(hypSMS, hypCEGO)*4)
        print('SMS NSGAII', ranksums(hypSMS, hypNSII)*4)
        print('SMS NSGAIII', ranksums(hypSMS, hypNSIII)*4)
        
    if (np.mean(hypCEGO) > np.mean(hypPHV) and np.mean(hypCEGO) > np.mean(hypSMS) and np.mean(hypCEGO) > np.mean(hypNSII) and np.mean(hypCEGO) > np.mean(hypNSIII)):
        #times 4 for bonferoni correction
        print('CEGO PHV', ranksums(hypCEGO, hypPHV)*4)
        print('CEGO SMS', ranksums(hypCEGO, hypSMS)*4)
        print('CEGO NSGAII', ranksums(hypCEGO, hypNSII)*4)
        print('CEGO NSGAIII', ranksums(hypCEGO, hypNSIII)*4)
        
    if (np.mean(hypNSII) > np.mean(hypPHV) and np.mean(hypNSII) > np.mean(hypSMS) and np.mean(hypNSII) > np.mean(hypCEGO) and np.mean(hypNSII) > np.mean(hypNSIII)):
        #times 4 for bonferoni correction
        print('NSGAII PHV', ranksums(hypNSII, hypPHV)*4)
        print('NSGAII SMS', ranksums(hypNSII, hypSMS)*4)
        print('NSGAII CEGO', ranksums(hypNSII, hypCEGO)*4)
        print('NSGAII NSGAIII', ranksums(hypNSII, hypNSIII)*4)
        
    if (np.mean(hypNSIII) > np.mean(hypPHV) and np.mean(hypNSIII) > np.mean(hypSMS) and np.mean(hypNSIII) > np.mean(hypCEGO) and np.mean(hypNSIII) > np.mean(hypNSII)):
        #times 4 for bonferoni correction
        print('NSGAIII PHV', ranksums(hypNSIII, hypPHV)*4)
        print('NSGAIII SMS', ranksums(hypNSIII, hypSMS)*4)
        print('NSGAIII CEGO', ranksums(hypNSIII, hypCEGO)*4)
        print('NSGAIII NSGAIII', ranksums(hypNSIII, hypNSII)*4)
        



print('all', kruskal(hypCEGO, hypPHV, hypSMS))

print('phv sms', kruskal(hypPHV, hypSMS))
print('phv cego', kruskal(hypPHV, hypCEGO))

print('phv sms', ranksums(hypPHV, hypSMS))
print('phv cego', ranksums(hypPHV, hypCEGO))

