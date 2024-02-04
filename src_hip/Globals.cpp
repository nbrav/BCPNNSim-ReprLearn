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

#include <stdio.h>

#include "Globals.h"

using namespace std;
using namespace Globals;

int Globals::simstep = 0;

float Globals::timestep = 0.001, Globals::simtime = simstep * timestep;

void Globals::error(string errloc,string errstr,int errcode) {

    fprintf(stderr, "\nERROR in %s: %s\n",errloc.c_str(),errstr.c_str());

    exit(errcode);

}


void Globals::warning(string warnloc,string warnstr) {

    fprintf(stderr, "\nWARNING in %s: %s\n",warnloc.c_str(),warnstr.c_str());

}

  
void Globals::reset() {

    simstep = 0;

    simtime = 0;

}


void Globals::gsetpoissonmean(double mean) { RndGen::grndgen->setpoissonmean(mean); }


void Globals::gsetseed(long seed) { RndGen::grndgen->setseed(seed); }


int Globals::gnextint()  { return RndGen::grndgen->nextint(); }


float Globals::gnextfloat() { return RndGen::grndgen->nextfloat(); }


int Globals::gnextpoisson() { return RndGen::grndgen->nextpoisson(); }


void Globals::advance() {

    simstep += 1;

    simtime = simstep * timestep;

}


int Globals::argmax(float *vec, int i1, int n) {

    int maxi = i1;
    float maxv = vec[maxi];

    for (int i=i1+1; i<i1+n; i++) if (vec[i]>maxv) { maxi = i; maxv = vec[i]; }

    return maxi;

}


int Globals::argmax(vector<float> vec,int i1,int n) {

    int maxi = i1;
    float maxv = vec[maxi];

    for (int i=i1+1; i<i1+n; i++) if (vec[i]>maxv) { maxi = i; maxv = vec[i]; }

    return maxi;

}


int Globals::argmin(vector<float> vec,int i1,int n) {

    int mini = i1;
    float minv = vec[mini];

    for (int i=i1+1; i<i1+n; i++) if (vec[i]<minv) { mini = i; minv = vec[i]; }

    return mini;

}


float Globals::vlen(vector<float> vec) {

    float vlen = 0;

    for (size_t i=0; i<vec.size(); i++) vlen += vec[i]*vec[i];

    return sqrt(vlen);

}


float Globals::vdiff(vector<float> vec1,vector<float> vec2) {

    if (vec1.size()!=vec2.size()) error("Globals::vdiff","vec1 -- vec2 length mismatch");

    float vdiff = 0;

    for (size_t i=0; i<vec1.size(); i++) vdiff += (vec2[i]-vec1[i])*(vec2[i]-vec1[i]);

    return sqrt(vdiff);

}


float Globals::vl1(vector<float> vec1,vector<float> vec2) {

    if (vec1.size()!=vec2.size()) error("Globals::vl1","vec1 -- vec2 length mismatch");

    float vl1 = 0;

    for (size_t i=0; i<vec1.size(); i++) vl1 += abs(vec2[i]-vec1[i]);

    return vl1;

}


void Globals::tofile(vector<float> vec,FILE *outfp) {

    fwrite (vec.data(),sizeof(float),vec.size(),outfp);    

}


void Globals::tofile(vector<float> vec,string filename) {

    FILE *outfp = fopen(filename.c_str(),"wb");

    tofile(vec,outfp);

    fclose(outfp);

}


void Globals::tofile(vector<vector<float> > mat,FILE *outfp) {

    for (size_t r=0; r<mat.size(); r++) tofile(mat[r],outfp);

}


void Globals::tofile(vector<vector<float> > mat,string filename) {

    FILE *outfp = fopen(filename.c_str(),"wb");

    tofile(mat,outfp);

    fclose(outfp);

}

void Globals::tofile(vector<int> vec,FILE *outfp) {

    fwrite (vec.data(),sizeof(int),vec.size(),outfp);    

}


void Globals::tofile(vector<int> vec,string filename) {

    FILE *outfp = fopen(filename.c_str(),"wb");

    tofile(vec,outfp);

    fclose(outfp);

}

void Globals::tofile(vector<vector<int> > mat,FILE *outfp) {

    for (size_t r=0; r<mat.size(); r++) tofile(mat[r],outfp);

}

void Globals::tofile(vector<vector<int> > mat,string filename) {

    FILE *outfp = fopen(filename.c_str(),"wb");

    tofile(mat,outfp);

    fclose(outfp);

}

RndGen *RndGen::grndgen = new RndGen();

RndGen::RndGen(long seedoffs) {

    uniformfloatdistr = uniform_real_distribution<float> (0.0,1.0);
    uniformintdistr = uniform_int_distribution<int>();
    poissondistr = poisson_distribution<int>(1); 
    setseed(4711174,seedoffs);

}

void RndGen::setseed(long seed,int hcuid) {

    // seed==0 gives random seed
    if (seed==0) seed = random_device{}();
    this->seed = seed + 17 * (hcuid + 1);
    generator.seed(this->seed);

}

void RndGen::setpoissonmean(float mean) {
    poissondistr = poisson_distribution<int>(mean);
}

long RndGen::getseed() { return seed; }

int RndGen::nextint() { return uniformintdistr(generator); }

float RndGen::nextfloat() { return uniformfloatdistr(generator); }

int RndGen::nextpoisson() { return poissondistr(generator); }

// MPI Stuff

int Globals::mpi_world_size, Globals::mpi_world_rank;

void Globals::initialize_comm() {

    MPI_Init(NULL,NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_world_rank);
    
}

bool Globals::isroot() {

    return mpi_world_rank==0;

}

void Globals::finalize_comm() {

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
}

// GPU stuff

int Globals::gpu_local_rank = 0, Globals::gpu_local_size = 0;
int Globals::blockDim1d = 128, Globals::blockDim2d = 16;

void Globals::initialize_gpu() {

    if (hipGetDeviceCount(&gpu_local_size) != hipSuccess) error("\nGlobals::inititalize_gpu", "No GPU devices");

    gpu_local_rank = mpi_world_rank; // Assign each GPU per MPI rank

    if (gpu_local_rank >= gpu_local_size) { printf("gpu=%d ngpu=%d ", gpu_local_rank, gpu_local_size); error("Globals::initialize_gpu", "Assigning to more GPU than available!"); }

    if (gpu_local_rank < 0) error("Globals::initialize_gpu", "Illegal GPU assignment!");

    if (hipSetDevice(Globals::gpu_local_rank) != hipSuccess) {

           error("Globals::initialize_gpu", "Error initializing device");

    }

}

int Globals::get_gpu_local_size() { return gpu_local_size; }

int Globals::get_gpu_local_rank() { return gpu_local_rank; }

