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

#include <vector>
#include <string>
#include <random>
#include <iostream>

#include "Globals.h"
#include "Pop.h"

using namespace std;
using namespace Globals;

int Pop::npop = 0;

Pop::Pop(int H, int M, std::string actfn, float again, int rank) {

    id = npop++;
    this->H = H;
    this->M = M;
    this->N = H * M;    
    this->actfn = actfn;
    this->again = again;
    this->rank = rank;
    eps = 1e-7;

    if (onthisrank()) {
        printf("\nNew Pop id = %-5d rank=%d mpi = %d/%d gpu = %d/%d H = %-5d M = %-5d N = %-5d again=%-.3f", id, rank, Globals::mpi_world_rank, Globals::mpi_world_size, Globals::gpu_local_rank, Globals::gpu_local_size, H, M, N, this->again);
        fflush(stdout);
    }
    
}

Pop::~Pop() {

    if (not onthisrank()) return;

    free(lgi);
    free(sup);
    free(act);
    
}

bool Pop::onthisrank() {

    if (rank<0 or rank>=Globals::mpi_world_size) { printf("Accessing rank = %d", rank); error("Pop::onthisrank", "Illegal rank!"); }

    return Globals::mpi_world_rank == rank;

}

__global__
void setup_noise_kernel(hiprandState *state, int N) {
    
    int n = threadIdx.x + blockIdx.x * blockDim.x;

    if (n>=N) return;
  
    hiprand_init(1234, n, 0, &state[n]);
  
}

void Pop::allocate_memory() {

    if (not onthisrank()) return;    

    lgi = (float*)malloc(N*sizeof(float));
    sup = (float*)malloc(N*sizeof(float));
    supinf = (float*)malloc(N*sizeof(float));
    act = (float*)malloc(N*sizeof(float));
    spkidx = (int*)malloc(H*sizeof(int));
    cumsum = (float*)malloc(N*sizeof(float));
    
    for (int i=0; i<N; i++) lgi[i] = 0; // logf(eps);
    for (int i=0; i<N; i++) sup[i] = logf(eps);
    for (int i=0; i<N; i++) supinf[i] = logf(eps);
    for (int i=0; i<N; i++) act[i] = eps;
    for (int h=0; h<H; h++) spkidx[h] = 0;
    for (int i=0; i<N; i++) cumsum[i] = 0;

    // HIP stuff

    CHECK_HIP_ERROR(hipMalloc(&d_lgi, N*sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&d_sup, N*sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&d_supinf, N*sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&d_act, N*sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&d_spkidx, H*sizeof(int)));
    CHECK_HIP_ERROR(hipMalloc(&d_cumsum, N*sizeof(float)));

    CHECK_HIP_ERROR(hipMemcpy(d_lgi, lgi, N*sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_sup, sup, N*sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_supinf, supinf, N*sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_act, act, N*sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_spkidx, spkidx, H*sizeof(int), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_cumsum, cumsum, N*sizeof(float), hipMemcpyHostToDevice));

    // MPI stuff

    sendRequests = new MPI_Request[axons.size()];
    sendStats = new MPI_Status[axons.size()];
    recvRequests = new MPI_Request[dends.size()];
    recvStats = new MPI_Status[dends.size()];

    CHECK_HIP_ERROR(hipMalloc(&d_sendBuf, N*sizeof(float)));
    for (int did=0; did<dends.size(); did++) {
        float *tmpptr;
        CHECK_HIP_ERROR(hipMalloc(&tmpptr, dends[did]->Ni*sizeof(float)));
        d_recvBuf.push_back(tmpptr);
    }

    // hipStreams stuff
    
    // hipStreamCreate(&comm_stream);
    // hipStreamCreate(&compute_stream);
    // hipStreamCreateWithFlags(&compute_stream, hipStreamNonBlocking); // wrong results?!

    // hipblas stuff
    
    hipblasCreate(&handle);
    // hipblasSetStream(handle, compute_stream);

    // hiprand stuff

    hipMalloc((void**)&devStates, N*sizeof(hiprandState));
    setup_noise_kernel<<<dim3((N+Globals::blockDim1d-1)/Globals::blockDim1d, 1, 1), dim3(Globals::blockDim1d, 1, 1), 0, 0>>>(devStates, N);
 
    CHECK_HIP_ERROR(hipPeekAtLastError());
    
    // Sync up

    sync_device();

}

void Pop::print_comm_summary() {

    if (not onthisrank()) return;

    double avgSendDur = 0, avgRecvDur = 0;
    
    if (sendNum!=0) avgSendDur = 1000000. * sendDur / sendNum;
    if (recvNum!=0) avgRecvDur = 1000000. * recvDur / recvNum;

    printf("\nPop rank = %d recvs num = %10d dur = %10.3f avgDur = %10.3fus sends num = %10d dur = %10.3fs avgDur = %10.3fus \n",
           rank, recvNum, recvDur, avgRecvDur, sendNum, sendDur, avgSendDur);
    fflush(stdout);

}

void Pop::store(std::string field, FILE* f) {

    if (not onthisrank()) return;

    if (field == "act") {        
        CHECK_HIP_ERROR(hipMemcpy(act, d_act, N*sizeof(float), hipMemcpyDeviceToHost));
        fwrite(act, sizeof(float), N, f);        
    } else if (field == "lgi") {
        CHECK_HIP_ERROR(hipMemcpy(lgi, d_lgi, N*sizeof(float), hipMemcpyDeviceToHost));
        fwrite(lgi, sizeof(float), N, f);
    } else if (field == "sup") {
        CHECK_HIP_ERROR(hipMemcpy(sup, d_sup, N*sizeof(float), hipMemcpyDeviceToHost));
        fwrite(sup, sizeof(float), N, f);
    } else if (field == "supinf") {
        CHECK_HIP_ERROR(hipMemcpy(supinf, d_supinf, N*sizeof(float), hipMemcpyDeviceToHost));
        fwrite(supinf, sizeof(float), N, f);   
    } else {
        printf("\nPop::store Invalid field!");
    }
            
}

float* Pop::getact() {

    if (not onthisrank()) return nullptr;
    
    return d_act;

}

void Pop::resetsup() {

    if (not onthisrank()) return;

    CHECK_HIP_ERROR(hipMemsetAsync(d_supinf, 0, N*sizeof(float)));
    CHECK_HIP_ERROR(hipMemsetAsync(d_act, 0, N*sizeof(float)));

}

__global__
void setinput_to_zero_kernel(float* lgi, float eps, int N) {

    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    lgi[n] = 0; // logf(eps);
  
}

__global__
void setinput_kernel(float* lgi, float* inp, float eps, int N) {

    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    lgi[n] = logf(inp[n] + eps);
  
}

void Pop::setinput(float *d_inp = nullptr) {

    if (not onthisrank()) return;

    if (d_inp==nullptr) {
        setinput_to_zero_kernel<<<dim3((N+Globals::blockDim1d-1)/Globals::blockDim1d, 1, 1), dim3(Globals::blockDim1d, 1, 1), 0, 0>>>(d_lgi, eps, N);
        CHECK_HIP_ERROR(hipPeekAtLastError());
    } else {
        setinput_kernel<<<dim3((N+Globals::blockDim1d-1)/Globals::blockDim1d, 1, 1), dim3(Globals::blockDim1d, 1, 1), 0, 0>>>(d_lgi, d_inp, eps, N);
        CHECK_HIP_ERROR(hipPeekAtLastError());
    }
         
}

void Pop::sync_device() {

    if (not onthisrank()) return;

    CHECK_HIP_ERROR(hipDeviceSynchronize());
    
}

void Pop::start_send() {

    if (not onthisrank()) return;

    if (axons.size()==0) return;    

    for (int aid=0; aid<axons.size(); aid++) {

        Prj *axon = axons[aid];

        if (this->rank == axon->pop_j->rank) {
            CHECK_HIP_ERROR(hipMemcpy(axon->d_Xi, d_act, N*sizeof(float), hipMemcpyDeviceToDevice));
            sendRequests[aid] = MPI_REQUEST_NULL;
        } else {
            MPI_Isend(d_act, N, MPI_FLOAT, axon->pop_j->rank, 0, MPI_COMM_WORLD, &sendRequests[aid]);
        }

    }

}

void Pop::wait_and_end_send() {

    if (not onthisrank()) return;

    if (axons.size()==0) return;

    double t1 = MPI_Wtime();

    MPI_Waitall(axons.size(), sendRequests, sendStats); // wait till previous activity is fully sent
    
    double t2 = MPI_Wtime();

    sendDur += t2 - t1;
    sendNum += 1;

}

void Pop::start_recv() {

    if (not onthisrank()) return;

    if (dends.size()==0) return;

    for (int did=0; did<dends.size(); did++) {

        Prj *dend = dends[did];

        if (this->rank == dend->pop_i->rank) {
            recvRequests[did] = MPI_REQUEST_NULL;
        } else {
            MPI_Irecv(dend->d_Xi, dend->Ni, MPI_FLOAT, dend->pop_i->rank, 0, MPI_COMM_WORLD, &recvRequests[did]);
        }

    }

}

void Pop::wait_and_end_recv() {

    if (not onthisrank()) return;

    if (dends.size()==0) return;

    double t1 = MPI_Wtime(); 
 
    MPI_Waitall(dends.size(), recvRequests, recvStats); // wait till activity is fully received

    double t2 = MPI_Wtime();

    recvDur += t2 - t1;
    recvNum += 1;
    
}

__global__
void addtosupinf_kernel(float *supinf, float *inc, int N) {

    int n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n>=N) return;

    supinf[n] += inc[n];
        
}

__global__
void calcsup_kernel(float *sup, float *supinf, int N) {

    int n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n>=N) return;

    sup[n] += (supinf[n] - sup[n]);

}

void Pop::integrate() {

    /* integrate all dendrite prj's support into pop's support */

    if (not onthisrank()) return;

    addtosupinf_kernel<<<dim3((N+Globals::blockDim1d-1)/Globals::blockDim1d, 1, 1), dim3(Globals::blockDim1d, 1, 1), 0, 0>>>(d_supinf, d_lgi, N);
    
    for (int did=0; did<dends.size(); did++) {

        Prj *dend = dends[did];

        if (dend->bwgain<eps) continue;
        
        addtosupinf_kernel<<<dim3((N+Globals::blockDim1d-1)/Globals::blockDim1d, 1, 1), dim3(Globals::blockDim1d, 1, 1), 0, 0>>>(d_supinf, dend->d_bwsup, N);
        CHECK_HIP_ERROR(hipPeekAtLastError());
        
    }

    calcsup_kernel<<<dim3((N+Globals::blockDim1d-1)/Globals::blockDim1d, 1, 1), dim3(Globals::blockDim1d, 1, 1), 0, 0>>>(d_sup, d_supinf, N);
    CHECK_HIP_ERROR(hipPeekAtLastError());
        
}

__global__
void inject_noise_kernel(hiprandState *state, float *sup, float nampl, int N) {

    /* inject noise to supoprt from a uniform distribution U(0,1) scaled by nampl */
    
    int n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n>=N) return;

    sup[n] += nampl * hiprand_uniform(&state[n]);
    
}

void Pop::inject_noise(float nampl) {

    if (not onthisrank()) return;

    inject_noise_kernel<<<dim3((N+Globals::blockDim1d-1)/Globals::blockDim1d, 1, 1), dim3(Globals::blockDim1d, 1, 1), 0, 0>>>(devStates, d_sup, nampl, N);
    CHECK_HIP_ERROR(hipPeekAtLastError());
    
}

__global__
void softmax_kernel(float* sup, float* act, float again, int H, int M) {

    /* soft winner-takes-all activation */

    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= H) return;
    
    float maxsup = sup[M*h], sumexpsup = 0;  
    for (int m=0; m<M; m++) maxsup = max(maxsup, sup[M*h+m]); 
    for (int m=0; m<M; m++) act[M*h+m] = expf(sup[M*h+m] - maxsup);
    for (int m=0; m<M; m++) sumexpsup += act[M*h+m];  
    for (int m=0; m<M; m++) act[M*h+m] /= sumexpsup;
    
}

void Pop::updact() {

    if (not onthisrank()) return;

    if (actfn=="softmax")
        softmax_kernel<<<dim3((H+Globals::blockDim1d-1)/Globals::blockDim1d, 1, 1), dim3(Globals::blockDim1d, 1, 1), 0, 0>>>(d_sup, d_act, again, H, M);
    else 
        error("Pop::updact", "Unknown actfn");
    CHECK_HIP_ERROR(hipPeekAtLastError());
    
}

void Pop::resetncorrect() {

    if (not onthisrank()) return;
    
    ncorrect = 0;

}

void Pop::compute_accuracy(float *d_target) {

    if (not onthisrank()) return;

    float *d_pred = this->d_act;

    float *pred = new float[N];
    float *target = new float[N];
    
    CHECK_HIP_ERROR(hipMemcpy(target, d_target, N*sizeof(float), hipMemcpyDeviceToHost)); 
    CHECK_HIP_ERROR(hipMemcpy(pred, d_pred, N*sizeof(float), hipMemcpyDeviceToHost));
    
    ncorrect += argmax(target, 0, N) == argmax(pred, 0, N);
    
}

void Pop::settracc(int ntrpat) {

    if (not onthisrank()) return;

    tracc = float(ncorrect) / ntrpat * 100.;

}

void Pop::setteacc(int ntepat) {

    if (not onthisrank()) return;

    teacc = float(ncorrect) / ntepat * 100.;

}
