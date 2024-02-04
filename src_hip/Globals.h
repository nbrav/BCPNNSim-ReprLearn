/*

MIT License

Copyright (c) 2024 Anders Lansner, Naresh Ravichandran

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/

#ifndef __Globals_included
#define __Globals_included

#include <vector>
#include <string>
#include <random>

/* HIP specific */
#include <hip/hip_runtime.h>
#include "hipblas.h"
#include "hiprand.h"
#include <hiprand_kernel.h>

/* CUDA specific */
// #include <cuda.h>
// #include <cuda_runtime.h>
// #include "cublas_v2.h"
// #include <curand.h>
// #include <curand_kernel.h>

#include <mpi.h>

namespace Globals {

    extern int simstep;
    extern float timestep,simtime;

    void error(std::string errloc,std::string errstr,int errcode = -1);
    void warning(std::string warnloc,std::string warnstr);
    void reset();
    void advance();
    void gsetpoissonmean(double mean) ;
    void gsetseed(long seed) ;
    int gnextint() ;
    float gnextfloat() ;
    int gnextpoisson() ;
    int argmax(float* vec,int i1,int n);
    int argmax(std::vector<float> vec,int i1,int n);
    int argmin(std::vector<float> vec,int i1,int n);
    float vlen(std::vector<float> vec) ;
    float vdiff(std::vector<float> vec1,std::vector<float> vec2) ;
    float vl1(std::vector<float> vec1,std::vector<float> vec2) ;
    void tofile(std::vector<float> vec,FILE *outfp);
    void tofile(std::vector<float> vec,std::string filename);
    void tofile(std::vector<std::vector<float> > mat,FILE *outfp);
    void tofile(std::vector<std::vector<float> > mat,std::string filename) ;
    void tofile(std::vector<int> vec,FILE *outfp);
    void tofile(std::vector<int> vec,std::string filename);
    void tofile(std::vector<std::vector<int> > mat,FILE *outfp);
    void tofile(std::vector<std::vector<int> > mat,std::string filename) ;

    // MPI stuff
    extern int mpi_world_rank, mpi_world_size;
    void initialize_comm();    
    bool isroot() ;
    void finalize_comm();

    // GPU stuff
    extern int gpu_local_rank, gpu_local_size;
    void initialize_gpu();
    int get_gpu_local_rank();
    int get_gpu_local_size();
    extern int blockDim1d;
    extern int blockDim2d;

} ;

class RndGen {

public:
    
    long seed ;
    std::mt19937_64 generator ;
    std::uniform_real_distribution<float> uniformfloatdistr ;
    std::uniform_int_distribution<int> uniformintdistr ;
    std::poisson_distribution<int> poissondistr ;
    
    static RndGen *grndgen;
    RndGen(long seedoffs = -1) ;
    void setseed(long seed,int hcuid = -1) ;
    void setpoissonmean(float mean) ;
    long getseed() ;
    int nextint() ;
    float nextfloat() ;
    int nextpoisson() ;

} ;

/* CUDA specific */

/*
#define CHECK_CUDA_ERROR(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t status, const char *file, int line, bool abort=true) {
    
    if (status == cudaSuccess) return;
       
    fprintf(stderr,"\nCUDA Assert: %s %s %d\n", cudaGetErrorString(status), file, line);
    if (abort) exit(status);

}

#define CHECK_CUBLAS_ERROR(ans) { cublasAssert((ans), __FILE__, __LINE__); }
inline void cublasAssert(cublasStatus_t status, const char *file, int line, bool abort=true) {
    
    if (status == CUBLAS_STATUS_SUCCESS) return;

    std::string errormsg;
    
    switch(status) {
        
    case CUBLAS_STATUS_SUCCESS: errormsg = "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED: errormsg = "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED: errormsg = "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE: errormsg = "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH: errormsg = "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR: errormsg = "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED: errormsg = "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR: errormsg = "CUBLAS_STATUS_INTERNAL_ERROR";
    default: errormsg = "UNKNOWN CUBLAS ERROR";
        
    }

    fprintf(stderr,"\nCUBLAS Assert: %s %s %d\n", errormsg.c_str(), file, line);
    if (abort) exit(status);

}
*/

/* HIP specific */

#define CHECK_HIP_ERROR(error)                       \
    if(error != hipSuccess)                          \
        {                                            \
            fprintf(stderr,                          \
                    "Hip error: '%s'(%d) at %s:%d\n",   \
                    hipGetErrorString(error),           \
                    error,                              \
                    __FILE__,                           \
                    __LINE__);                          \
            exit(EXIT_FAILURE);                         \
        }

#define CHECK_HIPBLAS_ERROR(error)                                 \
    if(error != HIPBLAS_STATUS_SUCCESS)                            \
        {                                                          \
            fprintf(stderr, "rocBLAS error: ");                    \
            if(error == HIPBLAS_STATUS_NOT_INITIALIZED)            \
                fprintf(stderr, "HIPBLAS_STATUS_NOT_INITIALIZED"); \
            if(error == HIPBLAS_STATUS_ALLOC_FAILED)               \
                fprintf(stderr, "HIPBLAS_STATUS_ALLOC_FAILED");    \
            if(error == HIPBLAS_STATUS_INVALID_VALUE)              \
                fprintf(stderr, "HIPBLAS_STATUS_INVALID_VALUE");   \
            if(error == HIPBLAS_STATUS_MAPPING_ERROR)              \
                fprintf(stderr, "HIPBLAS_STATUS_MAPPING_ERROR");   \
            if(error == HIPBLAS_STATUS_EXECUTION_FAILED)           \
                fprintf(stderr, "HIPBLAS_STATUS_EXECUTION_FAILED");     \
            if(error == HIPBLAS_STATUS_INTERNAL_ERROR)                  \
                fprintf(stderr, "HIPBLAS_STATUS_INTERNAL_ERROR");       \
            if(error == HIPBLAS_STATUS_NOT_SUPPORTED)                   \
                fprintf(stderr, "HIPBLAS_STATUS_NOT_SUPPORTED");        \
            if(error == HIPBLAS_STATUS_INVALID_ENUM)                    \
                fprintf(stderr, "HIPBLAS_STATUS_INVALID_ENUM");         \
            if(error == HIPBLAS_STATUS_UNKNOWN)                         \
                fprintf(stderr, "HIPBLAS_STATUS_UNKNOWN");              \
            fprintf(stderr, "\n");                                      \
            exit(EXIT_FAILURE);                                         \
        }

#endif // __Globals_included
