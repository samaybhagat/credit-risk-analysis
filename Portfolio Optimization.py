#!/usr/bin/env python
# coding: utf-8

# In[127]:


# initialization

import numpy as np
import matplotlib.pyplot as plt
import datetime 


# In[128]:


# importing qiskit tools

from qiskit import Aer
from qiskit.algorithms import VQE, NumPyMinimumEigensolver
from qiskit.algorithms.optimizers import COBYLA
from qiskit.circuit.library import TwoLocal
from qiskit.utils import QuantumInstance
from qiskit_finance.applications.optimization import PortfolioOptimization
from qiskit_finance.data_providers import RandomDataProvider
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.applications import OptimizationApplication
from qiskit_optimization.converters import QuadraticProgramToQubo


# In[129]:


# establishing the number of assets (same as # of cubits)

num_assets = 4
seed = 987

# select randomly generated data through RandomDataProvider

stocks = [("TICKER%s" % i) for i in range(num_assets)]
data = RandomDataProvider(
    tickers=stocks,
    start=datetime.datetime(2021, 1, 1),
    end=datetime.datetime(2021, 1, 30),
    seed=seed,
)


# In[130]:


# calculate expected return and covariance from generated data

data.run()
ER = data.get_period_return_mean_vector()
cov = data.get_period_return_covariance_matrix()


# In[131]:


# establishing parameters for optimizer
rf = 0.5  
budget = num_assets // 2    
portfolio = PortfolioOptimization(
    expected_returns=ER, covariances=cov, risk_factor=rf, budget=budget
)
qp = portfolio.to_quadratic_program()
qp


# In[136]:


# organizing print results

def print_result(result):
    selection = result.x
    value = result.fval
    print("Optimal: selection {}, value of obj func {:.4f}".format(selection, value))

    eigenstate = result.min_eigen_solver_result.eigenstate
    eigenvector = eigenstate if isinstance(eigenstate, np.ndarray) else eigenstate.to_matrix()
    probabilities = np.abs(eigenvector) ** 2
    i_sorted = reversed(np.argsort(probabilities))
    print("\n----------------- results ---------------------")
    print("selection\tvalue\t\tprobability")
    for i in i_sorted:
        x = index_to_selection(i, num_assets)
        value = QuadraticProgramToQubo().convert(qp).objective.evaluate(x)
        probability = probabilities[i]
        print("%10s\t%.4f\t\t%.4f" % (x, value, probability))


# In[137]:


# using vqe to find optimal asset choices

from qiskit.utils import algorithm_globals

algorithm_globals.random_seed = 9876
backend = Aer.get_backend("statevector_simulator")

cobyla = COBYLA()
cobyla.set_options(maxiter=500)
ry = TwoLocal(num_assets, "ry", "cz", reps=3, entanglement="full")
qi_VQE = QuantumInstance(backend=backend, seed_simulator=seed, seed_transpiler=seed)
vqe_mes = VQE(ry, optimizer=cobyla, quantum_instance=qi_VQE)
vqe = MinimumEigenOptimizer(vqe_mes)
result = vqe.solve(qp)


# In[138]:


# printing optimization results from VQE (quantum method)

print_result(result)


# In[139]:


# printing optimization results NumPyMinimumEigensolver (classical method)

mes = NumPyMinimumEigensolver()
eigensolver = MinimumEigenOptimizer(mes)

result = eigensolver.solve(qp)

print_result(result)


# In[ ]:




