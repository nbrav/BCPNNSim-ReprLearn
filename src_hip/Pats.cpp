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

#include "Globals.h"
#include "Pats.h"

using namespace std;
using namespace Globals;


Pats::Pats(int H, int M, std::string dir, std::string filename, bool binarize, std::string patype) {

    this->H = H;
    this->M = M;
    this->N = H * M;
    this->dir = dir;
    this->filename = filename;
    this->binarize = binarize;
    this->patype = patype;

    pats = NULL;
    d_pats = NULL;
    dispats = NULL;
    d_dispats = NULL;

}

void Pats::loadpats(int qbegin, int qend) {

    /* load binary patterns in range of query begin to end */

    if (this->pbegin == qbegin and this->pend == qend) return; // we already got this data; do nothing

    clearpats();

    pbegin = qbegin;
    pend = qend;
    npat = pend - pbegin;

    pats = (float *)malloc(npat * N * sizeof(float));
    FILE *fileptr = fopen((dir+filename).c_str(), "rb");

    if (binarize) {

        if (M!=2) fprintf(stderr, "Warning! Layer not binary HCUs but binarizing data.");
        
        fseek(fileptr, pbegin * H * sizeof(float), SEEK_SET);
        float* tmppats = (float *)calloc(npat * H, sizeof(float));
        int nitem = fread(tmppats, sizeof(float), npat * H, fileptr);

        for (int p=0; p<npat; p++)
            for (int h=0; h<H; h++) {
                pats[p*N + h*M + 0] = tmppats[p*H + h];
                pats[p*N + h*M + 1] = 1 - tmppats[p*H + h];
            }

        delete [] tmppats;

        // printf("\nLoaded %10s [begin=%d, end=%d, npat=%d] N=%d nitem=%d ", filename.c_str(), pbegin, pend, npat, N, nitem/N);

    } else {

        fseek(fileptr, pbegin * N * sizeof(float), SEEK_SET);
        int nitem = fread(pats, sizeof(float), npat * N, fileptr);

        // printf("\nLoaded %10s [begin=%d, end=%d, npat=%d] N=%d nitem=%d ", filename.c_str(), pbegin, pend, npat, N, nitem/N);
        
    }
    
    fclose(fileptr);
    
    // HIP stuff

    hipMalloc(&d_pats, N*npat*sizeof(float));
    hipMemcpy(d_pats, pats, N*npat*sizeof(float), hipMemcpyHostToDevice);  

}

void Pats::mkbinpats(int npat) {

    pats = new float[npat*N]; memset(pats,0,npat*N*sizeof(float));

    this->npat = npat;

    for (int p=0; p<npat; p++) 

	if (patype=="ortho") {

	    for (int h=0; h<H; h++) pats[p*N + h*M + p%M] = 1;

	} else if (patype=="rand") {

	    for (int h=0; h<H; h++) pats[p*N + h*M + gnextint()%M] = 1;

	} else error("Pats::mkbinpats","No such patype: " + patype);

    // HIP stuff

    if (d_pats==NULL) hipMalloc(&d_pats, N*npat*sizeof(float));
    hipMemcpy(d_pats, pats, N*npat*sizeof(float), hipMemcpyHostToDevice);  

}


void Pats::distortpats(string distype,int disarg) {

    if (disarg>H) error("Pats::distortpats","Illegal disarg>H: H = " + to_string(H) + " disarg = " +
			to_string(disarg));

    dispats = new float[npat*N]; memset(dispats,0,npat*N*sizeof(float));

    for (int p=0; p<npat; p++) 

        for (int i=0; i<N; i++)

            dispats[p*N + i] = pats[p*N + i];

    bool *tmph = new bool[H]; memset(tmph,0,H*sizeof(bool));

    int h;

    for (size_t p=0; p<npat; p++) {

	memset(tmph,0,H*sizeof(bool));
		
	if (distype=="hblank") {
			
            for (int d=0; d<disarg; d++) {
				
                h = gnextint()%H;
				
                while (tmph[h]) h = gnextint()%H;
		
                for (int m=0; m<M; m++) dispats[p*N + h*M+m] = 0;
				
                tmph[h] = true;
				
            }
			
	} else if (distype=="nflip") {
			
            for (int d=0; d<disarg; d++) {
				
                h = gnextint()%H;

                while (tmph[h]) h = gnextint()%H;
				
                for (int m=0; m<M; m++) dispats[p*N + h*M+m] = 0;
				
                dispats[p*N + h*M + gnextint()%M] = 1;
				
                tmph[h] = true;
				
            }
			
	} else error("Pats::distortpats","No such distype: " + distype);
		
    }

    if (d_dispats==NULL) hipMalloc(&d_dispats, N*npat*sizeof(float));
    hipMemcpy(d_dispats, dispats, N*npat*sizeof(float), hipMemcpyHostToDevice);  

}


void Pats::clearpats() {

    if (npat<=0) return;
        
    delete [] pats;
        
}

float* Pats::getpat(int p) {

    if (filename!="" and (pbegin>p or p>=pend)) { fprintf(stderr, "Warning! Data %d range not loaded!",p); }

    return &(d_pats[(p-pbegin) * N]);
    
}


float* Pats::getdispat(int p) {

    if (filename!="" and (pbegin>p or p>=pend)) { fprintf(stderr, "Warning! Data %d range not loaded!",p); }

    return &(d_dispats[(p-pbegin) * N]);
    
}


