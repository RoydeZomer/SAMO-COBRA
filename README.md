# SAMO-COBRA: 
# A Fast Surrogate Assisted ConstrainedMulti-objective Optimization Algorithm

In this github repository, you can find a novel Self-Adaptive algorithm for Multi-Objective Constrained Optimization by using Radial Basis Function Approximations, SAMO-COBRA.
The algorithm models the constraints and objectives with Radial Basis Functions (RBFs), automatically chooses the best RBF-fit, uses the RBFs as a cheap surrogate to find new feasible Pareto-optimal solutions, and automatically tunes hyper-parameters of the local search strategy. 
In every iteration one solution is added and evaluated, resulting in a strategy requiring only a small amount of function evaluations for finding a set of feasible solutions on the Pareto frontier. 
The proposed algorithm is compared to several other algorithms (NSGA-II, NSGA-III, CEGO, SMES-RBF) on 18 constrained multi-objective problems. 
In the experiments we show that our algorithm outperforms the other algorithms in terms of achieved HyperVolume (HV) after a fixed number of function evaluations.

Two Different infill criteria are used for SAMO-COBRA. The Predicted HyperVolume (PHV) infill criteria, and the S-Metric Selection infill criteria (SMS). The variants are compared to CEGO, NSGAII, NSGAIII on 18 test problems. The experiments can be found in the Experiments folder. See the results here:
![alt text](https://github.com/RoydeZomer/SAMO-COBRA/blob/main/SAMO_COBRA_RESULTS.PNG?raw=true)

Also a comparison is made with SMES-RBF. Because the authors of SMES-RBF didn't provide an implementation, the results form their paper are compared to one run of SAMO-COBRA with the PHV infill criteria. See here the convergence plots on 7 test problems:
![alt text](https://github.com/RoydeZomer/SAMO-COBRA/blob/main/Experiments/SMES_ReferencePoints_rregis/Convergenceplot%20BNH%20SAMO-COBRA%20vs%20SAMO-COBRA.png?raw=true)
![alt text](https://github.com/RoydeZomer/SAMO-COBRA/blob/main/Experiments/SMES_ReferencePoints_rregis/Convergenceplot%20SRN%20SAMO-COBRA%20vs%20SAMO-COBRA.png?raw=true)
![alt text](https://github.com/RoydeZomer/SAMO-COBRA/blob/main/Experiments/SMES_ReferencePoints_rregis/Convergenceplot%20TNK%20SAMO-COBRA%20vs%20SAMO-COBRA.png?raw=true)
![alt text](https://github.com/RoydeZomer/SAMO-COBRA/blob/main/Experiments/SMES_ReferencePoints_rregis/Convergenceplot%20OSY%20SAMO-COBRA%20vs%20SAMO-COBRA.png?raw=true)
![alt text](https://github.com/RoydeZomer/SAMO-COBRA/blob/main/Experiments/SMES_ReferencePoints_rregis/Convergenceplot%20TRICOP%20SAMO-COBRA%20vs%20SAMO-COBRA.png?raw=true)
![alt text](https://github.com/RoydeZomer/SAMO-COBRA/blob/main/Experiments/SMES_ReferencePoints_rregis/Convergenceplot%20BICOP1%20SAMO-COBRA%20vs%20SAMO-COBRA.png?raw=true)
![alt text](https://github.com/RoydeZomer/SAMO-COBRA/blob/main/Experiments/SMES_ReferencePoints_rregis/Convergenceplot%20BICOP2%20SAMO-COBRA%20vs%20SAMO-COBRA.png?raw=true)
