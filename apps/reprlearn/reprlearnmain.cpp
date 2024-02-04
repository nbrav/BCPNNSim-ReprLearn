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
#include <iostream>
#include <string>
#include <chrono>
#include <time.h>
#include <sys/time.h>

#include "Globals.h"
#include "Parseparam.h"
#include "Pats.h"
#include "Pop.h"
#include "Prj.h"
#include "Logger.h"

using namespace std;
using namespace Globals;

// Global parameters
long seed ;
float nampl, eps;
int verbosity = 1;
struct timeval start_time;

// Pop parameters
std::vector<Pop*> pops;
std::string actfn;
int Hin, Min, Hhid, Mhid, Hout, Mout;
float again;

// Prj parameters
std::vector<Prj*> prjs;
int nconnih, updconn_interval, updconn_nswapmax;
float updconn_threshold, taup, bgain, wgain;

// Training parameters
int nstep_per_pat, nusupepoch, nsupepoch;

// Classifier objects
std::vector<Pop*> lsgdclf_pops;
std::vector<LSGD*> lsgdclf_prjs;

// Dataset parameters
int ntrpat, ntepat, binarize_input;
std::string datadir, trimgfile, teimgfile, trlblfile, telblfile, parfile;
Pats *trimg = nullptr, *trlbl = nullptr, *teimg = nullptr, *telbl = nullptr;

// Storage parameters
bool storeacts=1, storeweights=1;

float getDiffTime(struct timeval start_time) {
    /* time difference in milli-seconds */
    struct timeval t_time;
    gettimeofday(&t_time, 0);
    return (1000.0 * (t_time.tv_sec - start_time.tv_sec) + (0.001 * (t_time.tv_usec - start_time.tv_usec)));
}

void parseparam() {

    /* Parse parameters from parameter file */

    Parseparam *parseparam = new Parseparam(parfile);

    parseparam->postparam("seed", &seed, Long);
    parseparam->postparam("Hin", &Hin, Int);
    parseparam->postparam("Min", &Min, Int);
    parseparam->postparam("Hhid", &Hhid, Int);
    parseparam->postparam("Mhid", &Mhid, Int);
    parseparam->postparam("Hout", &Hout, Int);
    parseparam->postparam("Mout", &Mout, Int);
    parseparam->postparam("binarize_input", &binarize_input, Int);
    parseparam->postparam("nconnih", &nconnih, Int);
    parseparam->postparam("taup", &taup, Float);
    parseparam->postparam("actfn", &actfn, String);
    parseparam->postparam("again", &again, Float);
    parseparam->postparam("updconn_interval", &updconn_interval, Int);
    parseparam->postparam("updconn_nswapmax", &updconn_nswapmax, Int);
    parseparam->postparam("updconn_threshold", &updconn_threshold, Int);
    parseparam->postparam("bgain", &bgain, Float);
    parseparam->postparam("wgain", &wgain, Float);
    parseparam->postparam("nstep_per_pat", &nstep_per_pat, Int);
    parseparam->postparam("nampl", &nampl, Float);
    parseparam->postparam("eps", &eps, Float);
    parseparam->postparam("nusupepoch", &nusupepoch, Int);
    parseparam->postparam("nsupepoch", &nsupepoch, Int);
    parseparam->postparam("ntrpat", &ntrpat, Int);
    parseparam->postparam("ntepat", &ntepat, Int);
    parseparam->postparam("datadir", &datadir, String);
    parseparam->postparam("trimgfile", &trimgfile, String);
    parseparam->postparam("trlblfile", &trlblfile, String);
    parseparam->postparam("teimgfile", &teimgfile, String);
    parseparam->postparam("telblfile", &telblfile, String);

    parseparam->doparse();
    
    if (taup==0)
        taup = timestep * ntrpat;
    
}

void initpats() {

    /* Initialize pattern data from files */

    bool binarize = true; // TODO add to par file
    
    trimg = new Pats(Hin, Min, datadir, trimgfile, binarize);
    trlbl = new Pats(Hout, Mout, datadir, trlblfile);
    teimg = new Pats(Hin, Min, datadir, teimgfile, binarize);
    telbl = new Pats(Hout, Mout, datadir, telblfile);
    
    trimg->loadpats(0, ntrpat);
    trlbl->loadpats(0, ntrpat);
    teimg->loadpats(0, ntepat);
    telbl->loadpats(0, ntepat);    
    
}

void buildnet() {

    /* Build network */

    if (isroot() and verbosity) { printf("\nBuilding net"); fflush(stdout); }
    
    gettimeofday(&start_time,0);

    // adding pops
    int rank_iter = 0;
    pops.push_back(new Pop(Hin, Min, actfn, again, rank_iter++));
    pops.push_back(new Pop(Hhid, Mhid, actfn, again, rank_iter++));
    
    // adding prjs
    BCP* ih_prj = new BCP(pops[0], pops[1], "BCP");
    ih_prj->set_taup(taup);
    ih_prj->updconn_nswapmax = updconn_nswapmax;
    ih_prj->updconn_threshold = updconn_threshold;
    prjs.push_back(ih_prj);

    // lsgd classifiers
    Pop *o_pop = new Pop(Hout, Mout, actfn, pops[1]->rank);
    LSGD *ho_prj = new LSGD(pops[1], o_pop, "LSGD");
    pops.push_back(o_pop);
    prjs.push_back(ho_prj);
    lsgdclf_pops.push_back(o_pop);
    lsgdclf_prjs.push_back(ho_prj);
    
    for (auto prj: prjs) {
        prj->set_eps(eps);
        prj->set_bgain(bgain);
        prj->set_wgain(wgain);
    }
    
    for (auto pop: pops)
        pop->allocate_memory();
    for (auto prj: prjs)
        prj->allocate_memory();

    prjs[0]->initconn_rand(nconnih);

    MPI_Barrier(MPI_COMM_WORLD);

    if (verbosity and isroot()) {
        printf("\n%-50s %10d steps %10.2f ms", "Building net done.", simstep, getDiffTime(start_time));
        fflush(stdout);
    }

}

void run() {
    
    /* Run one full step of network */

    for (auto pop: pops) pop->sync_device(); // waiting on each host for prev kernels to finish before sending/receving    
    for (auto pop: pops) pop->start_send();
    for (auto pop: pops) pop->start_recv();
    for (auto pop: pops) pop->wait_and_end_send();
    for (auto pop: pops) pop->wait_and_end_recv();
    for (auto pop: pops) pop->resetsup();
    for (auto prj: prjs) prj->depolarize();
    for (auto pop: pops) pop->integrate();
    for (auto pop: pops) pop->inject_noise(nampl);
    for (auto pop: pops) pop->updact();
    for (auto prj: prjs) prj->updtraces();
    for (auto prj: prjs) prj->updbw();    
    for (auto prj: prjs) prj->updconn();
    Logger::dologall();
    advance();
    
}

void learn() {
    /* Learn representations unsupervised */
    if (isroot() and verbosity) { printf("\nUnsupervised learning "); fflush(stdout); }
    gettimeofday(&start_time, 0);
    vector<Logger*> learn_loggers;
    if (storeweights) {
        for (auto prj: prjs) {
            if (prj==nullptr) continue;
            if (not prj->pop_j->onthisrank()) continue;
            learn_loggers.push_back( new Logger(prj, "conn", "learn.cij.l"+to_string(prj->pop_j->id)+".bin") );
        }
    }
    vector<int> randids(ntrpat);
    for (int q=0; q<ntrpat; q++) randids[q] = q;
    for (int epoch=0; epoch<nusupepoch; epoch++ ) {
        shuffle(begin(randids), end(randids), RndGen::grndgen->generator); // Only used by pops[0]
        for (int q=0, p; q<ntrpat; q++) {
            p = randids[q];
            for (int t=0; t<nstep_per_pat; t++) {
                int its = q + epoch * ntrpat;
                vector<float> probes = {0, 1e0, 2e0, 5e0, 1e1, 2e1, 5e1, 1e2, 2e2, 5e2, 1e3, 2e3, 5e3, 1e4, 2e4, 5e4, 1e5, 2e5, 5e5, 1e6, 2e6, 5e6, 1e7, 2e7, 5e7, 1e8, 2e8, 5e8, 1e9};
                if (std::find(probes.begin(), probes.end(), its) != probes.end() and t==nstep_per_pat-1) {
                    for (auto logger: learn_loggers) logger->on();
                } else { 
                    for (auto logger: learn_loggers) logger->off();
                }
                float PRN = t==nstep_per_pat-1;
                for (auto prj: prjs) {
                    prj->bwgain = 1;
                    prj->printnow = PRN; 
                    prj->REWIRE = q%updconn_interval==0 and PRN;
                }
                for (auto prj : lsgdclf_prjs) {
                    prj->bwgain = 0;
                    prj->printnow = 0;
                    prj->REWIRE = false;
                }     
                pops[0]->setinput(trimg->getpat(p));
                for (auto pop : lsgdclf_pops) pop->setinput(NULL);        
                run();
            }
        }   
        if (isroot() and verbosity) { printf("\nEpoch %d ",epoch+1); fflush(stdout); }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    for (auto logger: learn_loggers) logger->off();
    if (verbosity and isroot()) { printf("\n%-50s %10d steps %10.2f ms", "Unsupervised learning done.", simstep, getDiffTime(start_time)); fflush(stdout); }
}

void learn_clfs() {
    /* Learn linear classifier from representations */
    if (isroot() and verbosity) { printf("\nClassifier learning "); fflush(stdout); }
    gettimeofday(&start_time, 0);
    for (int epoch=0; epoch<nsupepoch; epoch++ ) {
        for (int p=0; p<ntrpat; p++) {
            for (int t=0; t<nstep_per_pat; t++) {
                float PRN = t==nstep_per_pat-1;
                for (auto prj: prjs) {
                    prj->bwgain = 1;
                    prj->printnow = 0;
                    prj->REWIRE = false;
                }
                for (auto prj : lsgdclf_prjs) {
                    prj->bwgain = 1;
                    prj->printnow = PRN;
                    prj->REWIRE = false;
                }
                pops[0]->setinput(trimg->getpat(p));
                for (auto prj : lsgdclf_prjs)
                    prj->settarget(trlbl->getpat(p)); // clamp targets for lsgd learning
                run();
            }
        }
        if (isroot() and verbosity) { printf("\nEpoch %d ",epoch+1); fflush(stdout); }            
    }
    MPI_Barrier(MPI_COMM_WORLD);   
    if (verbosity and isroot()) { printf("\n%-50s %10d steps %10.2f ms", "Classifier learning done.", simstep, getDiffTime(start_time)); fflush(stdout); }
}

void predict_train() {
    /* Predict labels on train dataset */
    if (verbosity and isroot()) { printf("\nPredicting from train samples "); fflush(stdout); }
    gettimeofday(&start_time, 0);
    for (auto pop : lsgdclf_pops) pop->resetncorrect();
        vector<Logger*> everystep_loggers;
    if (storeacts) {
        for (auto pop: pops) {
            everystep_loggers.push_back(new Logger(pop, "act", "predict.tract.l"+to_string(pop->id)+".log"));
        }
    }
    if (storeacts) for (auto logger: everystep_loggers) logger->on();
    for (int p=0; p<ntrpat; p++) {
        for (int t=0; t<nstep_per_pat; t++) {
            for (auto prj: prjs) {
                prj->bwgain = 1;
                prj->printnow = 0;
                prj->REWIRE = false;
            }
            for (auto prj : lsgdclf_prjs) {
                prj->bwgain = 1;
                prj->printnow = 0;
                prj->REWIRE = false;
            }
            pops[0]->setinput(trimg->getpat(p));
            for (auto pop : lsgdclf_pops) pop->setinput(nullptr);
            run();
        }
        for (auto pop : lsgdclf_pops) pop->compute_accuracy(trlbl->getpat(p));
    }
    if (storeacts) for (auto logger: everystep_loggers) logger->off();
    for (auto pop : lsgdclf_pops) pop->settracc(ntrpat);
    MPI_Barrier(MPI_COMM_WORLD);
    if (verbosity and isroot()) { printf("\n%-50s %10d steps %10.2f ms", "Predict train done.", simstep, getDiffTime(start_time)); fflush(stdout); }
}

void predict_test() {
    /* Predict labels on test dataset */
    if (verbosity and isroot()) { printf("\nPredicting from test samples "); fflush(stdout); }
    gettimeofday(&start_time, 0);
    for (auto pop : lsgdclf_pops) pop->resetncorrect();
    vector<Logger*> everystep_loggers;
    if (storeacts) {
        for (auto pop: pops) {
            everystep_loggers.push_back(new Logger(pop, "act", "predict.teact.l"+to_string(pop->id)+".log"));
        }
    }
    if (storeacts) for (auto logger: everystep_loggers) logger->on();
    for (int p=0; p<ntepat; p++) {
        for (int t=0; t<nstep_per_pat; t++) {
            for (auto prj: prjs) {
                prj->bwgain = 1;
                prj->printnow = 0;
                prj->REWIRE = false;
            }
            for (auto prj : lsgdclf_prjs) {
                prj->bwgain = 1; 
                prj->printnow = 0;
                prj->REWIRE = false;
            }           
            pops[0]->setinput(teimg->getpat(p));
            for (auto pop : lsgdclf_pops) pop->setinput(nullptr);                    
            run();
        }
        for (auto pop : lsgdclf_pops) pop->compute_accuracy(telbl->getpat(p));
    }
    if (storeacts) for (auto logger: everystep_loggers) logger->off();
    for (auto pop : lsgdclf_pops) pop->setteacc(ntepat);
    MPI_Barrier(MPI_COMM_WORLD);
    if (verbosity and isroot()) { printf("\n%-50s %10d steps %10.2f ms", "Predict test done.", simstep, getDiffTime(start_time)); fflush(stdout); }
}

void summary() {
    /* Summarize by printing accuracy */
    for (auto pop : lsgdclf_pops) {
        if (pop->onthisrank()) {
            int layer = pop->id;
            printf("\nLSGD Layer %-4d Accuracy (train) = %.2f %% Accuracy (test) = %.2f %%", layer, pop->tracc, pop->teacc);
            fflush(stdout);
        }
    }
}

int main(int argc, char **args) {

    for (int i=0; i<argc; i++) parfile = args[i];
    initialize_comm();
    initialize_gpu();
    parseparam();
    gsetseed(seed);
    initpats();
    buildnet();
    learn();
    learn_clfs();
    predict_train();
    predict_test();
    summary();
    finalize_comm();
    if (isroot()) { printf("\nFin.\n"); fflush(stdout); }

}
