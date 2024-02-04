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

#ifndef __Parseparam_INCLUDED__
#define __Parseparam_INCLUDED__

#include <vector>
#include <string>

// #include "Globals.h"

enum Value_t { Int = 0, Long, Float, Boole, String } ;

class Parseparam {

 public:

    Parseparam(std::string paramfile) ;

    void error(std::string errloc,std::string errstr) ;

    void postparam(std::string paramstring,void *paramvalue,Value_t paramtype) ;

    int findparam(std::string paramstring) ;

    void doparse(std::string paramlogfile = "") ;

    int getintparam(std::string paramstring,int v = 0) ;

    float getfltparam(std::string paramstring,int v = 0) ;

    long getlongparam(std::string paramstring,int v = 0) ;

    bool getbooleparam(std::string paramstring,int v = 0) ;

    std::string getstringparam(std::string paramstring,int v = 0) ;

    bool haschanged() ;

    char *timestamp() ;

    char *dolog(bool usetimestamp) ;

 protected:

    int nparampost;
    std::string _paramlogfile,_paramfile ;
    std::vector<std::string> _paramstring ;
    std::vector<void *> _paramvalue,*_vparamvalue ;
    std::vector<std::vector<int> > _vintvalue ;
    std::vector<std::vector<long> > _vlongvalue ;
    std::vector<std::vector<float> > _vfltvalue ;
    std::vector<std::vector<bool> > _vboolevalue;
    std::vector<std::vector<std::string> > _vstringvalue;
    std::vector<Value_t> _paramtype;
    time_t _oldmtime;
    char *_timestamp;

} ;

#endif // __Parseparam_INCLUDED__
