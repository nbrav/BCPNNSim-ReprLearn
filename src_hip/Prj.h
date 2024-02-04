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

#ifndef __Prj_included
#define __Prj_included

#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <cfloat>
#include <list>

#include "Globals.h"
#include "Pop.h"

#include <mpi.h>

class Pop;

class Prj {

public:
  
  static int nprj;
  
  int id;
  
  bool REWIRE=false;
  float bwgain=1, printnow=1, eps;
  
  Pop *pop_i, *pop_j;
  long int Hi, Mi, Ni, Hj, Mj, Nj, Nij, Hij;
  bool selfconn;
  
  // Synaptic plasticity parameters
  std::string lrule;
  float wgain=1, bgain=1;
  float *bwsup, *Xi, *Wij, *Bj;
  float *d_bwsup, *d_Xi, *d_Wij, *d_Bj;
  int ncorrect;
  float invmu;

  // Structural plasticity parameters
  int nconn;
  int updconn_nswapmax;
  float updconn_threshold;
  int *WConnij, *Connij;
  float *mutual_info, *score;
  int *d_WConnij, *d_Connij;
  float *d_mutual_info, *d_score;
  int *d_fanout, *updconn_nswap, *d_updconn_nswap;
  
  Prj(Pop *pop_i, Pop *pop_j, std::string lrule = "BCPNN");
  ~Prj();
  void set_eps(float);
  void set_bgain(float);
  void set_wgain(float);
  virtual void allocate_memory();
  virtual void store(std::string field, FILE*);
  void set_learningrate(float); 
  virtual void depolarize();
  virtual void updtraces();
  virtual void updbw();
  void initconn_rand(int);
  void initconn_sqr(int);
  virtual void updconn();

} ;

class BCP : public Prj {
  
public:
  
  float P;
  float taup, taupdt;
  float *Pi, *Pj, *Pij;
  float *d_Pi, *d_Pj, *d_Pij;

  BCP(Pop*, Pop*, std::string);
  void set_taup(float);
  void allocate_memory();
  void store(std::string field, FILE*);
  void depolarize();
  void updtraces();
  void updbw();
  void updconn();
  
} ;

class LSGD : public Prj {

public:
  
  int batch_size;
  float t, beta1, beta2, epsilon, alpha;
  float *db, *m_db, *v_db, *m_db_corr, *v_db_corr;
  float *dw, *m_dw, *v_dw, *m_dw_corr, *v_dw_corr;
  int *d_ncorrect, *d_argpred, *d_argtarget;
  float *d_db, *d_m_db, *d_v_db, *d_m_db_corr, *d_v_db_corr;
  float *d_dw, *d_m_dw, *d_v_dw, *d_m_dw_corr, *d_v_dw_corr;

  float *d_target;

  LSGD(Pop*, Pop*, std::string);
  void allocate_memory();
  void store(std::string field, FILE*);
  void settarget(float*);
  void depolarize();
  void updtraces();
  void updbw();
 
} ;

#endif // __Prj_included
