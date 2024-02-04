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

#ifndef __Pats_included
#define __Pats_included

#include <vector>
#include <string>
#include <random>

class Pats {

 public:

    std::string patype, dir, filename;
    int H, M, N;
    int pbegin=0, pend=-1, npat=-1;
    bool binarize;
    float *pats,*dispats;
    float *d_pats,*d_dispats;

 public:

    Pats(int H, int M, std::string dir = "", std::string filename = "", bool binarize = false, std::string patype = "rand");
    void mkbinpats(int npat);
    void distortpats(std::string distype = "nflip",int disarg = 1);
    void loadpats(int, int);
    void clearpats();
    float* getpat(int);
    float* getdispat(int);

} ;

#endif // __Pats_included
