# Codes for "Experimental Design Networks: A Paradigm for Serving Heterogeneous  Learners under Networking Constraints"

> Y. Liu*, Y. Li*, L. Su, E. Yeh, and S. Ioannidis, "Experimental Design Networks: A Paradigm for Serving Heterogeneous  Learners under Networking Constraints", IEEE INFOCOM 2022

Please cite this paper if you intend to use this code for your research.

We implement both python version and matlab version of our algorithms. Here is the python version.

* [Dependencies](#dependencies)
* [Usage](#usage)
  

## Dependencies

The dependencies are specified in the [``requirements.txt``](requirements.txt) file. 

## Usage

[``ProbGenerate.py``](ProbGenerate.py) is to initialize the problems over different networks/topologies. Some execution examples is shown in [``run_Prob``](run_Prob).

After initialize the problem, we use [``FrankWolf.py``](FrankWolf.py), which implements FW as well as 3 baselines, e.g., MaxSum, MaxAlpha, and PGA, to solve the problems. This outputs both solution and objective of the problem. Some execution examples is shown in [``run_FW``](run_FW). To output objectives only, we process the results with [``Object.py``](Object.py), so that the resulting outputs could directly be fed into our plotter.

After solving the problem, we could use [``Beta.py``](Beta.py) to calculate the average norm of estimation error, where the estimation error is the difference between the true model and the MAP estimator. Some execution examples is shown in [``run_beta``](run_beta).

[``plot_topology.py``](plot_topology.py) is to plot bar figures in the paper.


