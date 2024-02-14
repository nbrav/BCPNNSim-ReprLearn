# BCPNNSim

Bayesian Confidence Propagation Neural Network Simulator

This repository provides code for the experiments in the paper 

Ravichandran, N.B., Lansner, A., and Herman, P., 2024, Unsupervised Representation Learning with Hebbian Synaptic and Structural Plasticity in Brain-Like Feedforward Neural Networks. Manuscript.

The code is implemented in C++, with MPI for message passing and HIP for GPU parallelization. It can also be converted easilily to CUDA and run on NVIDIA GPUs (https://rocm.docs.amd.com/projects/HIP/en/develop/user_guide/hip_porting_guide.html).

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

The code was developed and run with MPI/HIP on Dardel supercomputer with the following SLURM script:

```
#!/bin/bash -l
#SBATCH --account=<alloc-name>
#SBATCH --time=1:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=3
export MPICH_GPU_SUPPORT_ENABLED=1
ml rocm/5.0.2 craype-accel-amd-gfx90a
srun apps/reprlearn/reprlearnmain apps/reprlearn/reprlearn.par 
```

# References

Ravichandran, N., Lansner, A. and Herman, P., 2023, Unsupervised Representation Learning with Hebbian Synaptic and Structural Plasticity in Brain-Like Feedforward Neural Networks. Available at SSRN 4613605.

Ravichandran, N.B., Lansner, A. and Herman, P., 2022, September. Brain-like combination of feedforward and recurrent network components achieves prototype extraction and robust pattern recognition. In International Conference on Machine Learning, Optimization, and Data Science (pp. 488-501). Cham: Springer Nature Switzerland.
