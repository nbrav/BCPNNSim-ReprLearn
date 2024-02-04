# BCPNNSim

Bayesian Confidence Propagation Neural Network Simulator

This repository provides code for the experiments in the paper 

Ravichandran, N.B., Lansner, A., and Herman, P., 2024, Unsupervised Representation Learning with Hebbian Synaptic and Structural Plasticity in Brain-Like Feedforward Neural Networks

The code is implemented in C++, with MPI for message passing and HIP for GPU parallelization.

# Extract MNIST dataset
```
cd Data/mnist/
python3 extract.py
cd ../../
```

# Compile and Run
```
make reprlearn
mpirun -n 2 apps/reprlearn/reprlearnmain apps/reprlearn/reprlearn.par
```
The code was developed and tested on Dardel supercomputer (NAISS National Academic Infrastructure for Supercomputing in Sweden) equipped AMD EPYC Zen2 CPU with AMD Instinct MI250X GPUs. The code was compiled with rocm/5.0.2, hipBLAS, hipRAND, MPICH. 

# References

Ravichandran, N., Lansner, A. and Herman, P., 2023, Unsupervised Representation Learning with Hebbian Synaptic and Structural Plasticity in Brain-Like Feedforward Neural Networks. Available at SSRN 4613605.

Ravichandran, N.B., Lansner, A. and Herman, P., 2022, September. Brain-like combination of feedforward and recurrent network components achieves prototype extraction and robust pattern recognition. In International Conference on Machine Learning, Optimization, and Data Science (pp. 488-501). Cham: Springer Nature Switzerland.
