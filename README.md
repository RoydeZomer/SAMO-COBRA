# SAMO-COBRA: 
# A Fast Surrogate Assisted ConstrainedMulti-objective Optimization Algorithm

In this github repository, you can find a novel Self-Adaptive algorithm for Multi-Objective Constrained Optimization by using Radial Basis Function Approximations, SAMO-COBRA.
The algorithm models the constraints and objectives with Radial Basis Functions (RBFs), automatically chooses the best RBF-fit, uses the RBFs as a cheap surrogate to find new feasible Pareto-optimal solutions, and automatically tunes hyper-parameters of the local search strategy. 
In every iteration one solution is added and evaluated, resulting in a strategy requiring only a small amount of function evaluations for finding a set of feasible solutions on the Pareto frontier. 
The proposed algorithm is compared to several other algorithms (NSGA-II, NSGA-III, CEGO, SMES-RBF) on 18 constrained multi-objective problems. 
In the experiments we show that our algorithm outperforms the other algorithms in terms of achieved HyperVolume (HV) after a fixed number of function evaluations.
