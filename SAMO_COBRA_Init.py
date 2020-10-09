# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 16:46:10 2017

@author: r.dewinter
"""
from SACOBRA import scaleRescale
from SACOBRA import rescaleWrapper
from SACOBRA import standardize_obj
from SACOBRA import rescale_constr
from SACOBRA import plog
from lhs import lhs 
from halton import halton
from transformLHS import transformLHS
from paretofrontFeasible import paretofrontFeasible
from hypervolume import hypervolume

import itertools
import numpy as np

def SAMO_COBRA_Init(xStart, fn, fName, lower, upper, nConstraints, ref, feval,
              infillCriteria="PHV",
              iterPlot=False,
              initDesign="HALTON",
              initDesPoints=None,
              seqTol=1e-6,
              epsilonInit=None,
              epsilonMax=None,
              newlower=-1,
              newupper=1,
              cobraSeed=1):
    
    print('start',fName,'with seed',cobraSeed)
    if initDesPoints is None:
        initDesPoints = len(xStart)+1

    nObj = len(ref)
    
    originalfn = fn
    originalL = lower
    originalU = upper
    phase = 'init'
    
    dimension = len(xStart) #number of parameters
    
    lower = np.array([newlower]*dimension)
    upper = np.array([newupper]*dimension)
    xStart = scaleRescale(xStart, originalL, originalU, newlower, newupper)
    fn = rescaleWrapper(fn,originalL,originalU,newlower,newupper)
    l = min(upper-lower)
    
    if epsilonInit is None:
        epsilonInit = [0.02*l]*nConstraints
    if epsilonMax is None:
        epsilonMax = [0.04*l]*nConstraints
        
    if initDesPoints>=feval:
        raise ValueError('feval should be larger then initial sample size')
    
    np.random.seed(cobraSeed)
    
    I = np.empty((1,1))
    I[:] = np.NaN
    Gres = np.empty((1,1))
    Gres[:] = np.NaN
    Fres = []
    
    np.random.seed(cobraSeed)
    if initDesign == 'RANDOM':
        I = np.random.uniform(low=lower,high=upper,size=(initDesPoints-1,dimension))
        I = np.vstack((xStart,I))
        I = np.clip(I,lower,upper)
           
    elif initDesign =='LHS':
        I = lhs(dimension, samples=initDesPoints-1, criterion="center", iterations=5)
        I = transformLHS(I, lower, upper)
        I = np.vstack((xStart,I))
        I = np.clip(I,lower,upper)

    elif initDesign == 'HALTON':
        I = halton(dimension, initDesPoints-1)
        I = scaleRescale(I, 0, 1, lower, upper)
        I = np.vstack((xStart,I))
        I = np.clip(I, lower, upper)
        
    elif initDesign == 'BOUNDARIES':
        inputdata = [[newlower,newupper]] * dimension
        result = np.array(list(itertools.product(*inputdata)))
        resultLength = len(result)
        indicator = [True]*(initDesPoints-1) + [False]*(resultLength - initDesPoints+1)
        np.random.shuffle(indicator)
        I = result[indicator]
        I = np.vstack((xStart,I))
        I = np.clip(I, lower, upper)
        
    else:
        raise ValueError('not yet implemented or invalid init design')
    Fres, Gres = randomResultsFactory(I,fn,nConstraints,nObj)
    
    hypervolumeProgress = np.empty((len(Fres),1))
    hypervolumeProgress[:] = np.NAN
    for i in range(len(Fres)):
        paretoOptimal = np.array([False]*(len(Fres)))
        paretoOptimal[:i+1] = paretofrontFeasible(Fres[:i+1,:],Gres[:i+1,:])
        paretoFront = Fres[paretoOptimal]
        hypervolumeProgress[i] = [hypervolume(paretoFront, ref)]
    
    Tfeas = 1 #np.floor(2*np.sqrt(dimension)) # The threshhold parameter for the number of consecutive iterations that yield feasible solution before the margin is reduced
    Tinfeas = 1 #np.floor(2*np.sqrt(dimension)) # The threshold parameter for the number of consecutive iterations that yield infeasible solutions before the margin is increased
    
    numViol = np.sum(Gres>0,axis=1)
    
    maxViol = np.max([np.zeros(len(Gres)),np.max(Gres, axis=1)],axis=0)
    
    A = I #contains all evaluated points
    
    pff = paretofrontFeasible(Fres,Gres)
    pf = Fres[pff]
    hv = hypervolumeProgress[-1].item()
    
    FresStandardized = np.full_like(Fres, 0)
    FresStandardizedMean = np.zeros(nObj)
    FresStandardizedStd = np.zeros(nObj)
    FresPlogStandardized = np.full_like(Fres, 0)
    FresPlogStandardizedMean = np.zeros(nObj)
    FresPlogStandardizedStd = np.zeros(nObj)
    for obji in range(nObj):
        res, mean, std = standardize_obj(Fres[:,obji])        
        FresStandardized[:,obji] = res
        FresStandardizedMean[obji] = mean 
        FresStandardizedStd[obji] = std
        
        plogFres = plog(Fres[:,obji])
        res, mean, std = standardize_obj(plogFres)        
        FresPlogStandardized[:,obji] = res
        FresPlogStandardizedMean[obji] = mean 
        FresPlogStandardizedStd[obji] = std
    
    GresRescaled = np.full_like(Gres, 0)
    GresRescaledDivider = np.zeros(nConstraints)
    GresPlogRescaled = np.full_like(Gres, 0)
    GresPlogRescaledDivider = np.zeros(nConstraints)
    for coni in range(nConstraints):
        GresRescaled[:,coni], GresRescaledDivider[coni] = rescale_constr(Gres[:,coni])
        plogGres = plog(Gres[:,coni])
        GresPlogRescaled[:,coni], GresPlogRescaledDivider[coni] = rescale_constr(plogGres)
        
    cobra = dict()
    cobra['ref'] = ref
    cobra['nObj'] = nObj
    cobra['currentHV'] = hv
    cobra['hypervolumeProgress'] = hypervolumeProgress
    cobra['paretoFrontier'] = pf
    cobra['paretoFrontierFeasible'] = pff
    cobra['fn'] = fn
    cobra['fName'] = fName
    cobra['dimension'] = dimension
    cobra['nConstraints'] = nConstraints
    cobra['lower'] = lower
    cobra['upper'] = upper
    cobra['originalL'] = originalL
    cobra['originalU'] = originalU
    cobra['originalfn'] = originalfn
    cobra['initDesPoints'] = initDesPoints
    cobra['feval'] = feval
    cobra['A'] = A
    cobra['Fres'] = Fres
    cobra['FresStandardized'] = FresStandardized
    cobra['FresStandardizedMean'] = FresStandardizedMean
    cobra['FresStandardizedStd'] = FresStandardizedStd
    cobra['FresPlogStandardized'] = FresPlogStandardized
    cobra['FresPlogStandardizedMean'] = FresPlogStandardizedMean 
    cobra['FresPlogStandardizedStd'] = FresPlogStandardizedStd
    cobra['Gres'] = Gres
    cobra['GresRescaled'] = GresRescaled
    cobra['GresRescaledDivider'] = GresRescaledDivider
    cobra['GresPlogRescaled'] = GresPlogRescaled
    cobra['GresPlogRescaledDivider'] = GresPlogRescaledDivider
    cobra['numViol'] = numViol
    cobra['maxViol'] = maxViol
    cobra['epsilonInit'] = epsilonInit
    cobra['epsilonMax'] = epsilonMax
    cobra['ptail'] = True
    cobra['seqFeval'] = (cobra['dimension']+cobra['nConstraints']+cobra['nObj'])*50
    cobra['computeStartPointsStrategy'] = 'multirandom'
    cobra['computeStartingPoints'] = (cobra['dimension']+cobra['nConstraints']+cobra['nObj'])*2
    cobra['seqTol'] = seqTol
    cobra['cobraSeed'] = cobraSeed
    cobra['RBFmodel'] = ['CUBIC','GAUSSIAN','MULTIQUADRIC','INVQUADRIC','INVMULTIQUADRIC','THINPLATESPLINE']#,'GAUSSIAN','MULTIQUADRIC','INVQUADRIC','INVMULTIQUADRIC','POLYHARMONIC1','POLYHARMONIC4','POLYHARMONIC5'] #['THINPLATESPLINE','GAUSSIAN','MULTIQUADRIC'] #
    cobra['bestPredictor'] = []
    cobra['phase'] = [phase]*initDesPoints
    cobra['plot'] = iterPlot
    cobra['optimizationTime'] = np.zeros(initDesPoints)
    
    if infillCriteria == "PHV" or infillCriteria == "SMS":
        cobra['infillCriteria'] = infillCriteria # "PHV" or "SMS"
    else:
        raise ValueError("This infill criteria is not implemented")
        
    surrogateErrors = {}
    for kernel in cobra['RBFmodel']:
        for obji in range(cobra['nObj']):
            surrogateErrors['OBJ'+str(obji)+kernel] = [0]*cobra['initDesPoints']
            surrogateErrors['OBJ'+str(obji)+'PLOG'+kernel] = [0]*cobra['initDesPoints']
        for coni in range(cobra['nConstraints']):
            surrogateErrors['CON'+str(coni)+kernel] = [0]*cobra['initDesPoints']
            surrogateErrors['CON'+str(coni)+'PLOG'+kernel] = [0]*cobra['initDesPoints']
    cobra['SurrogateErrors'] = surrogateErrors
    
    return(cobra)

def randomResultsFactory(I,fn,nConstraints,nObj):
    objs = np.empty((len(I),nObj))
    constr = np.empty((len(I),nConstraints))
    for i in range(len(I)):
        objs[i,:], constr[i,:] = fn(I[i])
    return [objs, constr]
        


            
