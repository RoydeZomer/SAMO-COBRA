[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4075509.svg)](https://doi.org/10.5281/zenodo.4075509)

# SAMO-COBRA: 
## A Fast Surrogate Assisted ConstrainedMulti-objective Optimization Algorithm

In this github repository, you can find a novel Self-Adaptive algorithm for Multi-Objective Constrained Optimization by using Radial Basis Function Approximations, SAMO-COBRA.
The algorithm models the constraints and objectives with Radial Basis Functions (RBFs), automatically chooses the best RBF-fit, uses the RBFs as a cheap surrogate to find new feasible Pareto-optimal solutions, and automatically tunes hyper-parameters of the local search strategy. 
In every iteration one solution is added and evaluated, resulting in a strategy requiring only a small amount of function evaluations for finding a set of feasible solutions on the Pareto frontier. 
The proposed algorithm is compared to several other algorithms (NSGA-II, NSGA-III, CEGO, SMES-RBF) on 18 constrained multi-objective problems. 
In the experiments we show that our algorithm outperforms the other algorithms in terms of achieved HyperVolume (HV) after a fixed number of function evaluations.

More information about this algorithm can be found in the original paper here: https://link.springer.com/chapter/10.1007%2F978-3-030-72062-9_22

## Usage

To use the optimization algorithm you need to define an objective function, the constraint function, and the search space before you can start the optimizer. Below is an examples that describe most of the functionality.
### Example - Optimizing CEXP problem

```python
import numpy as np

#imports from our package
from SAMO_COBRA_Init import SAMO_COBRA_Init
from SAMO_COBRA_PhaseII import SAMO_COBRA_PhaseII


# The "black-box" function
# returns objective array, and constraint array
# objective is to be minimized
# constraint should be transformed so that values samller then or equal to 0 are feasible
def CEXP(x):
    x1 = x[0]
    x2 = x[1]
    
    f1 = x1
    f2 = (1+x2)/x1
    
    g1 = x2 + 9*x1 - 6
    g2 = -1*x2 + 9*x1 - 1
    
    objectives = np.array([f1, f2])
    constraints = np.array([g1,g2])
    constraints = -1*constraints 
    return objectives, constraints


np.random.seed(0)
fn = CEXP
fName = 'CEXP'
# First we need to define the Search Space
# the search space consists of two continues variables in [0.1,1] and [0,5]
lower = np.array([0.1,0])
upper = np.array([1,5])
d = len(lower)
xStart = lower+np.random.rand(len(upper))*upper
#the objective space boundary is simply the largest values of the objective function we are interested in.
ref = np.array([1,9])
feval = 40*d
nConstraints = 2
cobra = SAMO_COBRA_Init(xStart, fn, fName, lower, upper, nConstraints, ref=ref, feval=feval, initDesPoints=d+1, cobraSeed=0, iterPlot=True)
cobra = SAMO_COBRA_PhaseII(cobra)

```


## Comparison with CEGO, NSGA-II, NSGA-III, SA-NSGA-II, IC-SA-NSGA-II
Mean HV after 40·d function evaluations for each algorithm on each test function. Note that for IC-SA-NSGA-II onlythe objective function evaluations are counted. PHV and SMS represents the SAMO-COBRA variants. The highest mean HV per test function are presented in bold. The Wilcoxon rank-sum test (with Bonferroni correction) significance is represented with a grayscale. Background colors represent:p≤0.001, p≤0.01, p≤0.05. Red shows that the algorithm took more than 24 hours. The experiments can be found in the Experiments folder. See the results here:

![alt text](https://github.com/RoydeZomer/SAMO-COBRA/blob/main/Experiments/samo-cobra-results-hv.PNG?raw=true)

Test Function, Threshold Hypervolume, and number of function evaluations required to achieve this threshold peroptimization algorithm. The results of the algorithm with the smallest number of function evaluations is reported in bold andaccompanied with a ↑. PHV and SMS represents the SAMO-COBRA variants. Experiments that required more then 5000 function evaluations or more then 24 hours are terminated.
![alt text](https://github.com/RoydeZomer/SAMO-COBRA/blob/main/Experiments/samo-cobra-results-fe.PNG?raw=true)

## Comparison with SMES-RBF
Also a comparison is made with SMES-RBF. Because the authors of SMES-RBF didn't provide an implementation, the results form their paper are compared to one run of SAMO-COBRA with the PHV infill criteria. See here two convergence and number of function evaluations to achieve the same results on the other problems:

![alt text](https://github.com/RoydeZomer/SAMO-COBRA/blob/main/Experiments/samo-cobra-vs-smes-rbf.PNG?raw=true))
![alt text](https://github.com/RoydeZomer/SAMO-COBRA/blob/main/Experiments/tricop-convergence.PNG?raw=true)
![alt text](https://github.com/RoydeZomer/SAMO-COBRA/blob/main/Experiments/BNH-convergence.PNG?raw=true)
