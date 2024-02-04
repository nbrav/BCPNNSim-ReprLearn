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
#include "Prj.h"

using namespace std;
using namespace Globals;

int Prj::nprj = 0;

Prj::Prj(Pop *pop_i, Pop *pop_j, string lrule) {

    id = nprj++;
    this->pop_i = pop_i;
    this->pop_j = pop_j;
    this->Hi = pop_i->H;
    this->Mi = pop_i->M;
    this->Ni = pop_i->N;
    this->Hj = pop_j->H;
    this->Mj = pop_j->M;
    this->Nj = pop_j->N;
    this->Hij = Hj * Hi;
    this->Nij = Nj * Ni;
    this->lrule = lrule;
    ncorrect = 0; // if this is a classifier

    if (pop_i->onthisrank() or pop_j->onthisrank()) {
        pop_i->axons.push_back(this);                
        pop_j->dends.push_back(this);
    }

    if (pop_j->onthisrank()) {
        printf("\nNew Prj id = %-5d rank=%d mpi = %d/%d gpu = %d/%d Pop[%d]->Pop[%d] lrule=%s Nji = %-10ld", pop_j->id, Globals::mpi_world_rank, Globals::mpi_world_rank, Globals::mpi_world_size, Globals::gpu_local_rank, Globals::gpu_local_size, pop_i->id, pop_j->id, lrule.c_str(), Nij);
        fflush(stdout);
    }

}

Prj::~Prj() {

}

void Prj::set_eps(float paramval) {

    eps = paramval;

}

void Prj::set_bgain(float paramval) {

    bgain = paramval;

}

void Prj::set_wgain(float paramval) {

    wgain = paramval;

}

void Prj::allocate_memory() {

    if (not pop_j->onthisrank()) return;

    if (true) { // rewire = true

        updconn_nswap = (int*)malloc(Hj*sizeof(int));
        mutual_info = (float*)malloc(Hij*sizeof(float));
        score = (float*)malloc(Hij*sizeof(float));
        Connij = (int*)malloc(Hij*sizeof(int));
        WConnij = (int*)malloc(Nij*sizeof(int));

        for (int hji=0; hji<Hij; hji++) 
            mutual_info[hji] = 0;
        for (int hji=0; hji<Hij; hji++) 
            score[hji] = 0;
        for (int hji=0; hji<Hij; hji++) 
            Connij[hji] = 1;
        for (int ji=0; ji<Nij; ji++) 
            WConnij[ji] = 1;

        CHECK_HIP_ERROR(hipMalloc(&d_updconn_nswap, Hj*sizeof(int)));
        CHECK_HIP_ERROR(hipMalloc(&d_mutual_info, Hij*sizeof(float)));
        CHECK_HIP_ERROR(hipMalloc(&d_score, Hij*sizeof(float)));
        CHECK_HIP_ERROR(hipMalloc(&d_Connij, Hij*sizeof(int)));
        CHECK_HIP_ERROR(hipMalloc(&d_WConnij, Nij*sizeof(int)));
        CHECK_HIP_ERROR(hipMalloc(&d_fanout, Hi*sizeof(int)));

        CHECK_HIP_ERROR(hipMemcpyAsync(d_mutual_info, mutual_info, Hij*sizeof(float), hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpyAsync(d_score, score, Hij*sizeof(float), hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpyAsync(d_Connij, Connij, Hij*sizeof(int), hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpyAsync(d_WConnij, WConnij, Nij*sizeof(int), hipMemcpyHostToDevice));

    }

    CHECK_HIP_ERROR(hipDeviceSynchronize());

}

void Prj::store(std::string field, FILE* f) {

    if (not pop_j->onthisrank()) return;

    if (field == "bwsup") {
        CHECK_HIP_ERROR(hipMemcpy(bwsup, d_bwsup, Nj*sizeof(float), hipMemcpyDeviceToHost));        
        fwrite(bwsup, sizeof(float), Nj, f);    
    } else if (field == "wij") {
        CHECK_HIP_ERROR(hipMemcpy(Wij, d_Wij, Nij*sizeof(float), hipMemcpyDeviceToHost));
        fwrite(Wij, sizeof(float), Nij, f);    
    } else if (field == "bj") {
        CHECK_HIP_ERROR(hipMemcpy(Bj, d_Bj, Nj*sizeof(float), hipMemcpyDeviceToHost));        
        fwrite(Bj, sizeof(float), Nj, f);    
    } else if (field == "conn") {
        CHECK_HIP_ERROR(hipMemcpy(Connij, d_Connij, Hij*sizeof(int), hipMemcpyDeviceToHost));       
        fwrite(Connij, sizeof(int), Hij, f);
    } else if (field == "wconn") {
        CHECK_HIP_ERROR(hipMemcpy(WConnij, d_WConnij, Nij*sizeof(int), hipMemcpyDeviceToHost));       
        fwrite(WConnij, sizeof(int), Nij, f);   
    } else {
        printf("\nPrj::store Invalid field!");
    }

}

__global__
void add_bias(float *bwsup, float *b, float alpha, int N) {

    int n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n>=N) return;

    bwsup[n] += b[n] * alpha;

}

void Prj::depolarize() {

    /* multiply pre-synaptic activity by weights and calculate dendrite's support to post-synaptic neuron */

    if (not pop_j->onthisrank()) return;

    if (bwgain<eps) return;

    /* just an empty shell */
    
    if (lrule=="BCP") {
        printf("\nWARNING! Prj::depolarize trying to implement BCP function");
    } else if (lrule=="LSGD") {
        printf("\nWARNING! Prj::depolarize trying to implement LSGD function");
    } else {
        printf("\nWARNING! Prj::updtraces lrule not valid!");
    }

}

void Prj::updtraces() {

    if (not pop_j->onthisrank()) return;

    /* just an empty shell */

    if (lrule=="BCP") {
        printf("\nWARNING! Prj::updtraces trying to implement BCP function!");
    } else if (lrule=="LSGD") {
        printf("\nWARNING! Prj::updtraces trying to implement LSGD function!");
    } else {
        printf("\nWARNING! Prj::updtraces lrule not valid!");
    }

}

void Prj::updbw() {

    if (not pop_j->onthisrank()) return;

    if (printnow<eps) return;

    /* just an empty shell */

    if (lrule=="BCP") {
        printf("\nWARNING! Prj::updbw trying to implement BCP function!");
    } else if (lrule=="LSGD") {
        printf("\nWARNING! Prj::updbw trying to implement LSGD function!");
    } else {
        printf("\nWARNING! Prj::updbw lrule not valid!");   
    } 

}

__global__
void updwconn_kernel(int *WConnij, int *Connij, int Hi, int Mi, int Hj, int Mj) {

    int hi = blockIdx.x * blockDim.x + threadIdx.x;
    int hj = blockIdx.y * blockDim.y + threadIdx.y;
    if ((hi >= Hi) || (hj >= Hj)) return;

    int Ni = Hi * Mi;
    
    for (int mj=0; mj<Mj; mj++) {

        for (int ms=0; ms<Mi; ms++) {

            int r = hj * Mj + mj;
            int s = hi * Mi + ms;
            WConnij[r*Ni + s] = Connij[hj*Hi+hi]==1;
            
        }
    }
    
}

void Prj::initconn_rand(int nconn) {

    /* initialize random connections as active */

    if (not pop_j->onthisrank()) return;
    
    this->nconn = nconn;

    // Create index vector later to be shuffled
    std::vector<int> shuffled(Hi);
    for (size_t hi=0; hi<Hi; hi++)
        shuffled[hi] = hi;

    for (size_t hj=0; hj<Hj; hj++) {

        // Shuffle index for selecting connections
        shuffle(begin(shuffled), end(shuffled), RndGen::grndgen->generator);

        for (size_t id=0; id<Hi; id++) {
            int hi = shuffled[id];
            if (id<this->nconn) {
                Connij[hj*Hi+hi] = 1;              
            } else {
                Connij[hj*Hi+hi] = 0;
            }
        }
    }

    CHECK_HIP_ERROR(hipMemcpy(d_Connij, Connij, Hij*sizeof(int), hipMemcpyHostToDevice));

    // update wconns

    updwconn_kernel<<<dim3((Hi+Globals::blockDim2d-1)/Globals::blockDim2d, (Hj+Globals::blockDim2d-1)/Globals::blockDim2d), dim3(Globals::blockDim2d, Globals::blockDim2d), 0, 0>>>(d_WConnij, d_Connij, Hi, Mi, Hj, Mj);
    CHECK_HIP_ERROR(hipPeekAtLastError());   

}

void Prj::initconn_sqr(int nconn) {

    /* initialize square receptive fields connections as active */

    if (not pop_j->onthisrank()) return;

    this->nconn = nconn;

    int Wj = int(sqrt(Hj)); // width & height of output layer
    int Wi = int(sqrt(Hi)); // weidth & heright of input layer
    int Wf = int(sqrt(nconn)); // weidth & height of filter
    int stride = int(Wi/Wj); 

    if (not Wi * Wi == Hi) printf("\nWarning! Hi not square but fitting as 2D square");
    if (not Wj * Wj == Hj) printf("\nWarning! Hj not square but fitting as 2D square");
    if (not Wf * Wf == nconn) printf("\nWarning! nconn not square but fitting as 2D square");
    if (not stride * Wj == Wi) printf("\nWarning! input size should be multiple of output");
    
    for (int hji=0; hji<Hij; hji++) 
        Connij[hji] = 0;

    for (int wj=0; wj<Wj; wj++) {
        for (int lj=0; lj<Wj; lj++) {
            int hj = wj + lj * Wj;
            for (int wi=wj*stride; wi<=wj*stride+Wf; wi++) {
                for (int li=lj*stride; li<=lj*stride+Wf; li++) {
                    int wi_mod = (wi > 0 ? wi : 0) < Wi ? wi : Wi-1;
                    int li_mod = (li > 0 ? li : 0) < Wi ? li : Wi-1;
                    int hi = wi_mod + li_mod * Wi;
                    Connij[hj*Hi+hi] = 1;
                }
            }
        }
    }

    CHECK_HIP_ERROR(hipMemcpy(d_Connij, Connij, Hij*sizeof(int), hipMemcpyHostToDevice));

    // update wconns

    updwconn_kernel<<<dim3((Hi+Globals::blockDim2d-1)/Globals::blockDim2d, (Hj+Globals::blockDim2d-1)/Globals::blockDim2d), dim3(Globals::blockDim2d, Globals::blockDim2d), 0, 0>>>(d_WConnij, d_Connij, Hi, Mi, Hj, Mj);
    CHECK_HIP_ERROR(hipPeekAtLastError());

}

void Prj::updconn() {

    if (not pop_j->onthisrank()) return;

    if (not REWIRE) return;

    /* just an empty shell */

    if (lrule=="BCP") {
        printf("\nWARNING! Prj::updconn trying to implement BCP function!");
    } else if (lrule=="LSGD") {
        printf("\nWARNING! Prj::updconn trying to implement LSGD function!");
    } else {
        printf("\nWARNING! Prj::updconn lrule not valid!");   
    } 

}

/*-------------------------------------  BCP  ----------------------------------*/

BCP::BCP(Pop* pop_i, Pop* pop_j, std::string lrule) : Prj(pop_i, pop_j, lrule) {

}

void BCP::set_taup(float paramval) {

    taup = paramval;
    taupdt = Globals::timestep / paramval;

}

void BCP::allocate_memory() {

    if (not pop_j->onthisrank()) return;

    Prj::allocate_memory();

    Xi = (float*)malloc(Ni*sizeof(float));
    Bj = (float*)malloc(Nj*sizeof(float));
    Wij = (float*)malloc(Nij*sizeof(float));
    bwsup = (float*)malloc(Nj*sizeof(float));
    P = eps;    
    Pi = (float*)malloc(Ni*sizeof(float));
    Pj = (float*)malloc(Nj*sizeof(float));
    Pij = (float*)malloc(Nij*sizeof(float));

    for (int s=0; s<Ni; s++) 
        Xi[s] = 1./Mi;
    for (int r=0; r<Nj; r++) 
        Bj[r] = eps;
    for (int ji=0; ji<Nij; ji++) 
        Wij[ji] = 0; // 0.1 * (gnextfloat() - 0.1); // eps*eps;
    for (int r=0; r<Nj; r++) 
        bwsup[r] = 0;
    for (int s=0; s<Ni; s++) 
        Pi[s] = eps;
    for (int r=0; r<Nj; r++) 
        Pj[r] = eps;
    for (int ji=0; ji<Nij; ji++) 
        Pij[ji] = eps*eps;

    P =  Mi * Mj * 10;
    Globals::gsetpoissonmean(P / Mi);
    for (int i = 0; i < Ni; i++) {
        Pi[i] = 1 + Globals::gnextpoisson();
    }
    Globals::gsetpoissonmean(P / Mj);
    for (int j = 0; j < Nj; j++) {
        Pj[j] = 1 + Globals::gnextpoisson();
    }
    Globals::gsetpoissonmean(P / (Mi * Mj));
    for (int k = 0; k < Ni * Nj; k++) {
        Pij[k] = 1 + Globals::gnextpoisson();
    }
    for (int hi = 0; hi < Hi; hi++) {
        float hsum = 0;
        for (int i = hi * Mi; i < (hi + 1) * Mi; i++)
            hsum += Pi[i];
        if (hsum > 0) {
            for (int i = hi * Mi; i < (hi + 1) * Mi; i++) {
                Pi[i] = Pi[i] / hsum * taupdt;
            }
        }
    }
    for (int hj = 0; hj < Hj; hj++) {
        float hsum = 0;
        for (int j = hj * Mj; j < (hj + 1) * Mj; j++)
            hsum += Pj[j];
        if (hsum > 0) {
            for (int j = hj * Mj; j < (hj + 1) * Mj; j++) {
                Pj[j] = Pj[j] / hsum * taupdt;
            }
        }
    }
    for (int hi = 0; hi < Hi; hi++) {
        for (int hj = 0; hj < Hj; hj++) {
            float hijsum = 0;
            for (int i = hi * Mi; i < (hi + 1) * Mi; i++) {
                for (int j = hj * Mj; j < (hj + 1) * Mj; j++) {
                    hijsum += Pij[i * Nj + j];
                }
            }
            for (int i = hi * Mi; i < (hi + 1) * Mi; i++) {
                for (int j = hj * Mj; j < (hj + 1) * Mj; j++) {
                    Pij[i * Nj + j] = Pij[i * Nj + j] / hijsum * taupdt;
                }
            }
        }
    }
    P = taupdt;

    CHECK_HIP_ERROR(hipMalloc(&d_Xi, Ni*sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&d_Bj, Nj*sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&d_Wij, Nij*sizeof(float)));    
    CHECK_HIP_ERROR(hipMalloc(&d_bwsup, Nj*sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&d_Pi, Ni*sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&d_Pj, Nj*sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&d_Pij, Nij*sizeof(float)));

    CHECK_HIP_ERROR(hipMemcpyAsync(d_Xi, Xi, Ni*sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpyAsync(d_Bj, Bj, Nj*sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpyAsync(d_Wij, Wij, Nij*sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpyAsync(d_bwsup, bwsup, Nj*sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpyAsync(d_Pi, Pi, Ni*sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpyAsync(d_Pj, Pj, Nj*sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpyAsync(d_Pij, Pij, Nij*sizeof(float), hipMemcpyHostToDevice));

}

void BCP::store(std::string field, FILE* f) {

    if (not pop_j->onthisrank()) return;

    if (field == "pij") {
        CHECK_HIP_ERROR(hipMemcpy(Pij, d_Pij, Nij*sizeof(float), hipMemcpyDeviceToHost));       
        fwrite(Pij, sizeof(float), Nij, f);
    } else if (field == "pi") {
        CHECK_HIP_ERROR(hipMemcpy(Pi, d_Pi, Ni*sizeof(float), hipMemcpyDeviceToHost));
        fwrite(Pi, sizeof(float), Ni, f);
    } else if (field == "pj") {
        CHECK_HIP_ERROR(hipMemcpy(Pj, d_Pj, Nj*sizeof(float), hipMemcpyDeviceToHost));
        fwrite(Pj, sizeof(float), Nj, f);
    } else if (field == "mi") {        
        CHECK_HIP_ERROR(hipMemcpy(mutual_info, d_mutual_info, Hij*sizeof(float), hipMemcpyDeviceToHost));        
        fwrite(mutual_info, sizeof(float), Hij, f);  
    } else if (field == "nmi") {
        CHECK_HIP_ERROR(hipMemcpy(score, d_score, Hij*sizeof(float), hipMemcpyDeviceToHost)); 
        fwrite(score, sizeof(float), Hij, f);   
    } else if (field == "updconn_nswap") {
        CHECK_HIP_ERROR(hipMemcpy(updconn_nswap, d_updconn_nswap, Hj*sizeof(int), hipMemcpyDeviceToHost));  
        fwrite(updconn_nswap, sizeof(int), Hj, f);   
    } else {
        Prj::store(field, f);
    }

}

void BCP::depolarize() {

    /* multiply pre-synaptic activity by weights and calculate dendrite's support to post-synaptic neuron */

    if (not pop_j->onthisrank()) return;

    if (bwgain<eps) return;
    
    /*  
        SGEMV matrix-vector dot product
        performs y := alpha * A * x + beta * y
        params (handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
        more info: https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-gemv
    */

    float alpha = 1; // multiplier for synaptic current
    float beta = 0; // multiplier for previous support term
    float biasmul = 1; // multiplier for bias term
    CHECK_HIPBLAS_ERROR(hipblasSgemv(pop_j->handle, HIPBLAS_OP_T, Ni, Nj, &alpha, d_Wij, Ni, d_Xi, 1, &beta, d_bwsup, 1));
    add_bias<<<dim3((Nj+Globals::blockDim1d-1)/Globals::blockDim1d, 1, 1), dim3(Globals::blockDim1d, 1, 1), 0, 0>>>(d_bwsup, d_Bj, biasmul, Nj);
    CHECK_HIP_ERROR(hipPeekAtLastError());

}

__global__
void updpi_kernel(float *xi, float *pi, float taupdt, int Ni, float PRN) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= Ni) return;
    pi[i] += (xi[i] - pi[i]) * taupdt * PRN;
}

__global__
void updpj_kernel(float *xj, float *pj, float taupdt, int Nj, float PRN) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= Nj) return;
    pj[j] += (xj[j] - pj[j]) * taupdt * PRN;
}

__global__
void updpij_kernel(float *xi, float *xj, float *pij, float taupdt, float eps, int Ni, int Nj, float PRN) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if ((i >= Ni) || (j >= Nj)) return;
    int ij = j * Ni + i;
    pij[ij] += (xi[i] * xj[j] - pij[ij]) * taupdt * PRN;
}

void BCP::updtraces() {

    if (not pop_j->onthisrank()) return;

    float *d_Xj = pop_j->d_act; // this should always be in the same rank and set by Pop

    if (printnow<eps) return;

    P += (1 - P) * taupdt * printnow;

    updpi_kernel<<<dim3((Ni+Globals::blockDim1d-1)/Globals::blockDim1d, 1, 1), dim3(Globals::blockDim1d, 1, 1), 0, 0>>>(d_Xi, d_Pi, taupdt, Ni, printnow);
    CHECK_HIP_ERROR(hipPeekAtLastError());

    updpj_kernel<<<dim3((Nj+Globals::blockDim1d-1)/Globals::blockDim1d, 1, 1), dim3(Globals::blockDim1d, 1, 1), 0, 0>>>(d_Xj, d_Pj, taupdt, Nj, printnow);
    CHECK_HIP_ERROR(hipPeekAtLastError());
    
    updpij_kernel<<<dim3((Ni+Globals::blockDim2d-1)/Globals::blockDim2d, (Nj+Globals::blockDim2d-1)/Globals::blockDim2d, 1), dim3(Globals::blockDim2d, Globals::blockDim2d, 1), 0, 0>>>(d_Xi, d_Xj, d_Pij, taupdt, eps, Ni, Nj, printnow);
    CHECK_HIP_ERROR(hipPeekAtLastError());
    
}

__global__
void updbw_kernel(float p, float* pi, float* pj, float* pij, float* bj, float* wij, int* wconnij, float bgain, float wgain, int Ni, int Nj, float eps) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if ((i >= Ni) || (j >= Nj)) return;
    int ij = j * Ni + i;

    if (i==0) {
        bj[j] = bgain * logf(pj[j]/p);
    }

    wij[ij] = wgain * logf(pij[ij]*p/(pi[i]*pj[j])) * wconnij[ij];  

}

void BCP::updbw() {

    if (not pop_j->onthisrank()) return;

    if (printnow<eps) return;

    updbw_kernel<<<dim3((Ni+Globals::blockDim2d-1)/Globals::blockDim2d, (Nj+Globals::blockDim2d-1)/Globals::blockDim2d, 1), dim3(Globals::blockDim2d, Globals::blockDim2d, 1), 0, 0>>>(P, d_Pi, d_Pj, d_Pij, d_Bj, d_Wij, d_WConnij, bgain, wgain, Ni, Nj, eps);
    CHECK_HIP_ERROR(hipPeekAtLastError());

}

/*  Calculate mutual-information between two populations
 *  Naive implementation with serial loops per hypercolumn-pair
 */
__global__
void calc_mutualinfo_kernel(float *mutual_info, float *Pi, float *Pj, float *Pij, float P, float eps, int Hi, int Mi, int Hj, int Mj) {

    int hi = blockIdx.x * blockDim.x + threadIdx.x;
    int hj = blockIdx.y * blockDim.y + threadIdx.y;
    if ((hi >= Hi) || (hj >= Hj)) return;
    int hji = hj * Hi + hi;

    int Ni = Hi * Mi;

    float tmpsum = 0;

    for (int mj=0; mj<Mj; mj++) {

        for (int mi=0; mi<Mi; mi++) {

            int j = hj * Mj + mj;
            int i = hi * Mi + mi;
            int ji = j * Ni + i;
            float Qi = fmax(Pi[i]/P, eps);                
            float Qj = fmax(Pj[j]/P, eps);
            float Qij = fmax(Pij[ji]/P, eps*eps);            
            tmpsum += Qij * log(Qij/(Qi*Qj));
            
        }
    }
    
    mutual_info[hji] = tmpsum;
}

__global__
void compute_fanout_kernel(int *fanout, int *Connij, int Hi, int Hj) {

    int hi = blockIdx.x * blockDim.x + threadIdx.x;
    if (hi >= Hi) return;

    int tmpsum = 0;

    for (int hj=0; hj<Hj; hj++) tmpsum += Connij[hj*Hi+hi]==1;

        fanout[hi] = tmpsum;
    
}

__global__
void recompute_score_kernel(float *score, float *mutual_info, int *fanout, int Hi, int Hj) {

    int hi = blockIdx.x * blockDim.x + threadIdx.x;
    int hj = blockIdx.y * blockDim.y + threadIdx.y;
    if ((hi >= Hi) || (hj >= Hj)) return;
    int hji = hj * Hi + hi;

    score[hji] = mutual_info[hji] / (fanout[hi] + 1); 
    
}

/* Swap connections (one prune + one grow) x nswap times
 * Extremeley inefficient, basically serial code running on single GPU thread
 */
__global__
void swap_kernel(int *Connij, float *score, int updconn_nswapmax, int* updconn_nswap, float updconn_threshold, int Hi, int hj) {

    bool converged = false;

    updconn_nswap[hj] = updconn_nswapmax;

    for (int swapid=0; swapid<updconn_nswapmax; swapid++) {

        if (converged) {
            updconn_nswap[hj] = swapid;
            break;
        }

        int active_id, silent_id;
        float active_minscore = FLT_MAX, silent_maxscore = -FLT_MAX;
        
        for (int hi=0; hi<Hi; hi++) {
            if (Connij[hj*Hi+hi]==1 and score[hj*Hi+hi] < active_minscore) {
                active_id = hi;
                active_minscore = score[hj*Hi+hi];
            }
            if (Connij[hj*Hi+hi]==0 and score[hj*Hi+hi] > silent_maxscore) {
                silent_id = hi;
                silent_maxscore = score[hj*Hi+hi];
            }
        }
        
        if (silent_maxscore > updconn_threshold * active_minscore) {
            Connij[hj*Hi + active_id] = 0;                
            Connij[hj*Hi + silent_id] = 1;
        } else {
            converged = true;
        }

    }    
    
}

void BCP::updconn() {

    if (not pop_j->onthisrank()) return;
    
    if (not REWIRE) return;
    
    // compute mutual-info
    
    calc_mutualinfo_kernel<<<dim3((Hi+Globals::blockDim2d-1)/Globals::blockDim2d, (Hj+Globals::blockDim2d-1)/Globals::blockDim2d, 1), dim3(Globals::blockDim2d, Globals::blockDim2d, 1), 0, 0>>>(d_mutual_info, d_Pi, d_Pj, d_Pij, P, eps, Hi, Mi, Hj, Mj);
    CHECK_HIP_ERROR(hipPeekAtLastError());
    
    for (int hj=0; hj<Hj; hj++) {

        // (re)compute score from mutual info

        compute_fanout_kernel<<<dim3((Hi+Globals::blockDim1d-1)/Globals::blockDim1d), dim3(Globals::blockDim1d), 0, 0>>>(d_fanout, d_Connij, Hi, Hj);
        CHECK_HIP_ERROR(hipPeekAtLastError());
        
        recompute_score_kernel<<<dim3((Hi+Globals::blockDim2d-1)/Globals::blockDim2d, (Hj+Globals::blockDim2d-1)/Globals::blockDim2d, 1), dim3(Globals::blockDim2d, Globals::blockDim2d, 1), 0, 0>>>(d_score, d_mutual_info, d_fanout, Hi, Hj);
        CHECK_HIP_ERROR(hipPeekAtLastError());
        
        // update connections (slower than serial!)
        
        swap_kernel<<<dim3(1), dim3(1), 0, 0>>>(d_Connij, d_score, updconn_nswapmax, d_updconn_nswap, updconn_threshold, Hi, hj);
        CHECK_HIP_ERROR(hipPeekAtLastError());
        
    }

    // (re)compute score from mutual info

    compute_fanout_kernel<<<dim3((Hi+Globals::blockDim1d-1)/Globals::blockDim1d), dim3(Globals::blockDim1d), 0, 0>>>(d_fanout, d_Connij, Hi, Hj);
    CHECK_HIP_ERROR(hipPeekAtLastError());
        
    recompute_score_kernel<<<dim3((Hi+Globals::blockDim2d-1)/Globals::blockDim2d, (Hj+Globals::blockDim2d-1)/Globals::blockDim2d, 1), dim3(Globals::blockDim2d, Globals::blockDim2d, 1), 0, 0>>>(d_score, d_mutual_info, d_fanout, Hi, Hj);
    CHECK_HIP_ERROR(hipPeekAtLastError());
            
    // update wconns
    
    updwconn_kernel<<<dim3((Hi+Globals::blockDim2d-1)/Globals::blockDim2d, (Hj+Globals::blockDim2d-1)/Globals::blockDim2d), dim3(Globals::blockDim2d, Globals::blockDim2d), 0, 0>>>(d_WConnij, d_Connij, Hi, Mi, Hj, Mj);
    CHECK_HIP_ERROR(hipPeekAtLastError());    
    
}

/*-------------------------------------  LSGD ----------------------------------*/

LSGD::LSGD(Pop* pop_i, Pop* pop_j, std::string lrule) : Prj(pop_i, pop_j, lrule) {

    d_target = nullptr;

    batch_size = 100;
    t = 0;

    // From Kingma & Ba, 2014
    beta1 = 0.9;
    beta2 = 0.999;
    epsilon = 1e-7f;
    alpha = 0.001;

}

void LSGD::allocate_memory() {

    if (not pop_j->onthisrank()) return;
    
    Prj::allocate_memory();
    
    Xi = (float*)malloc(Ni*sizeof(float));
    Bj = (float*)malloc(Nj*sizeof(float));
    Wij = (float*)malloc(Nij*sizeof(float));
    bwsup = (float*)malloc(Nj*sizeof(float));
    db = (float*)malloc(Nj*sizeof(float));
    m_db = (float*)malloc(Nj*sizeof(float));
    v_db = (float*)malloc(Nj*sizeof(float));
    m_db_corr = (float*)malloc(Nj*sizeof(float));
    v_db_corr = (float*)malloc(Nj*sizeof(float));
    dw = (float*)malloc(Nj*Ni*sizeof(float));
    m_dw = (float*)malloc(Nj*Ni*sizeof(float));
    v_dw = (float*)malloc(Nj*Ni*sizeof(float));
    m_dw_corr = (float*)malloc(Nj*Ni*sizeof(float));
    v_dw_corr = (float*)malloc(Nj*Ni*sizeof(float));
    
    for (int s=0; s<Ni; s++) 
        Xi[s] = 0;
    for (int r=0; r<Nj; r++) 
        Bj[r] = 0;
    for (int ji=0; ji<Nij; ji++) 
        Wij[ji] = 0;
    for (int r=0; r<Nj; r++) 
        bwsup[r] = 0;
    memset(db, 0, Nj*sizeof(float));
    memset(m_db, 0, Nj*sizeof(float));
    memset(v_db, 0, Nj*sizeof(float));
    memset(m_db_corr, 0, Nj*sizeof(float));
    memset(v_db_corr, 0, Nj*sizeof(float));
    memset(dw, 0, Nj*Ni*sizeof(float));
    memset(m_dw, 0, Nj*Ni*sizeof(float));
    memset(v_dw, 0, Nj*Ni*sizeof(float));
    memset(m_dw_corr, 0, Nj*Ni*sizeof(float));
    memset(v_dw_corr, 0, Nj*Ni*sizeof(float));

    for (int ji=0; ji<Nij; ji++) 
        Wij[ji] = gnextfloat() * 0.05;

    CHECK_HIP_ERROR(hipMalloc(&d_Xi, Ni*sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&d_Bj, Nj*sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&d_Wij, Nij*sizeof(float)));    
    CHECK_HIP_ERROR(hipMalloc(&d_bwsup, Nj*sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&d_db, Nj*sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&d_m_db, Nj*sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&d_v_db, Nj*sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&d_m_db_corr, Nj*sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&d_v_db_corr, Nj*sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&d_dw, Ni*Nj*sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&d_m_dw, Ni*Nj*sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&d_v_dw, Ni*Nj*sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&d_m_dw_corr, Ni*Nj*sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&d_v_dw_corr, Ni*Nj*sizeof(float)));   

    CHECK_HIP_ERROR(hipMemcpyAsync(d_Xi, Xi, Ni*sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpyAsync(d_Bj, Bj, Nj*sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpyAsync(d_Wij, Wij, Nij*sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpyAsync(d_bwsup, bwsup, Nj*sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemset(d_db, 0, Nj*sizeof(float)));
    CHECK_HIP_ERROR(hipMemset(d_m_db, 0, Nj*sizeof(float)));
    CHECK_HIP_ERROR(hipMemset(d_v_db, 0, Nj*sizeof(float)));
    CHECK_HIP_ERROR(hipMemset(d_m_db_corr, 0, Nj*sizeof(float)));
    CHECK_HIP_ERROR(hipMemset(d_v_db_corr, 0, Nj*sizeof(float)));
    CHECK_HIP_ERROR(hipMemset(d_dw, 0, Ni*Nj*sizeof(float)));
    CHECK_HIP_ERROR(hipMemset(d_m_dw, 0, Ni*Nj*sizeof(float)));
    CHECK_HIP_ERROR(hipMemset(d_v_dw, 0, Ni*Nj*sizeof(float)));
    CHECK_HIP_ERROR(hipMemset(d_m_dw_corr, 0, Ni*Nj*sizeof(float)));
    CHECK_HIP_ERROR(hipMemset(d_v_dw_corr, 0, Ni*Nj*sizeof(float)));    

}

void LSGD::store(std::string field, FILE* f) {

    // Prj::store(std::string field, FILE* f);

}

void LSGD::settarget(float* d_target = nullptr) {

    if (not pop_j->onthisrank()) return;

    if (d_target!=nullptr) {
        this->d_target = d_target;
    }

}

void LSGD::depolarize() {

    /* multiply pre-synaptic activity by weights and calculate dendrite's support to post-synaptic neuron */

    if (not pop_j->onthisrank()) return;

    if (bwgain<eps) return;
    
    /*  
        SGEMV matrix-vector dot product
        performs y := alpha * A * x + beta * y
        params (handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
        more info: https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-gemv
    */

    float alpha = 1 ; // * pop_i->again // multiplier for synaptic current
    float beta = 0; // multiplier for previous support term 
    float biasmul = 1; // multiplier for bias term
    CHECK_HIPBLAS_ERROR(hipblasSgemv(pop_j->handle, HIPBLAS_OP_T, Ni, Nj, &alpha, d_Wij, Ni, d_Xi, 1, &beta, d_bwsup, 1));
    add_bias<<<dim3((Nj+Globals::blockDim1d-1)/Globals::blockDim1d, 1, 1), dim3(Globals::blockDim1d, 1, 1), 0, 0>>>(d_bwsup, d_Bj, biasmul, Nj);
    CHECK_HIP_ERROR(hipPeekAtLastError());

}

__global__
void upd_traces_lsgd(float *db, float *dw, float *src, float *target, float *pred, int Ni, int Nj) {

    int s = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    if ((s >= Ni) || (r >= Nj)) return;
    int rs = r * Ni + s;

    if (s==0) {
        db[r] += (target[r] - pred[r]);
    }

    dw[rs] += src[s] * (target[r] - pred[r]);

}

void LSGD::updtraces() {

    if (not pop_j->onthisrank()) return;
    
    if (d_target==nullptr) return;

    if (printnow<eps) return;

    float *srcact = d_Xi;
    float *d_Xj = pop_j->d_act; // this should always be in the same rank

    upd_traces_lsgd<<<dim3((Ni+Globals::blockDim2d-1)/Globals::blockDim2d, (Nj+Globals::blockDim2d-1)/Globals::blockDim2d, 1), dim3(Globals::blockDim2d, Globals::blockDim2d, 1), 0, 0>>>(d_db, d_dw, srcact, d_target, d_Xj, Ni, Nj);
    CHECK_HIP_ERROR(hipPeekAtLastError());
    
}

__global__
void updbw_lsgd_kernel(float *b, float *db, float *m_db, float *v_db, float *m_db_corr, float *v_db_corr,
    float *w, float *dw, float *m_dw, float *v_dw, float *m_dw_corr, float *v_dw_corr,
    float alpha, float beta1, float beta2, float epsilon, float t, int batch_size, 
    int Ni, int Nj) {

    int s = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    if ((s >= Ni) || (r >= Nj)) return;
    int rs = r * Ni + s;
    
    m_dw[rs] = beta1 * m_dw[rs] + (1-beta1) * dw[rs] / (float) batch_size; // Update biased first moment estimate
    v_dw[rs] = beta2 * v_dw[rs] + (1-beta2) * dw[rs] * dw[rs] / (float) batch_size ; // Update biased second raw moment estimate
    m_dw_corr[rs] = m_dw[rs] / (1-pow(beta1,t)) ; //  Compute bias-corrected first moment estimate
    v_dw_corr[rs] = v_dw[rs] / (1-pow(beta2,t)) ; // Compute bias-corrected second raw moment estimate
    w[rs] = w[rs] + alpha * (m_dw_corr[rs] / (sqrt(v_dw_corr[rs])+epsilon)) ; //  Update parameters
    
    if (s==0) {

        m_db[r] = beta1 * m_db[r] + (1-beta1) * db[r] / (float) batch_size ; // Update biased first moment estimate
        v_db[r] = beta1 * v_db[r] + (1-beta2) * db[r] * db[r] / (float) batch_size ; // Update biased second raw moment estimate
        m_db_corr[r] = m_db[r] / (1-pow(beta1,t)) ; //  Compute bias-corrected first moment estimate
        v_db_corr[r] = v_db[r] / (1-pow(beta2,t)) ; // Compute bias-corrected second raw moment estimate  
        b[r] = b[r] + alpha * (m_db_corr[r] / (sqrt(v_db_corr[r])+epsilon)) ; //  Update parameters
        
    }
    
}

void LSGD::updbw() {

    if (not pop_j->onthisrank()) return;
    
    if (printnow<eps) return;
    
    /* @brief update w, b from dw, db using Adam (Adaptive Moment Estimation) optimizer      
     * Original : Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980. 
     * Help : https://towardsdatascience.com/how-to-implement-an-adam-optimizer-from-scratch-76e7b217f1cc  
     */
    
    t++;
    
    updbw_lsgd_kernel<<<dim3((Ni+Globals::blockDim2d-1)/Globals::blockDim2d, (Nj+Globals::blockDim2d-1)/Globals::blockDim2d, 1), dim3(Globals::blockDim2d, Globals::blockDim2d, 1), 0, 0>>>(d_Bj, d_db, d_m_db, d_v_db, d_m_db_corr, d_v_db_corr, d_Wij, d_dw, d_m_dw, d_v_dw, d_m_dw_corr, d_v_dw_corr, alpha, beta1, beta2, epsilon, t, batch_size, Ni, Nj);
    CHECK_HIP_ERROR(hipPeekAtLastError());
    
    CHECK_HIP_ERROR(hipMemset(d_db, 0, Nj*sizeof(float)));
    CHECK_HIP_ERROR(hipMemset(d_dw, 0, Nij*sizeof(float)));   

}
