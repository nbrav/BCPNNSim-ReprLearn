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
#include "Pop.h"
#include "Prj.h"

#include "Logger.h"

using namespace std;
using namespace Globals;

vector<Logger *> Logger::loggers;

Logger::Logger(Pop *pop, string field, string filename, int logstep, int logstepoffs) {

    this->pop = pop;
    this->prj = NULL;
    this->field = field;
    this->filename = filename;
    this->logstep = logstep;
    this->logstepoffs = logstepoffs;
    ison = true;
    idx = loggers.size();
    loggers.push_back(this);
    if (pop->onthisrank()) {
        remove(filename.c_str());
        if (filename!="") logfp = fopen(filename.c_str(),"a"); else logfp = NULL;
    }
    
}

Logger::Logger(Prj *prj,string field,string filename,int logstep,int logstepoffs) {

    this->pop = NULL;
    this->prj = prj;
    this->field = field;
    this->filename = filename;
    this->logstep = logstep;
    this->logstepoffs = logstepoffs;
    ison = true;
    idx = loggers.size();
    loggers.push_back(this);
    if (prj->pop_j->onthisrank()) {
        remove(filename.c_str());
        if (filename!="") logfp = fopen(filename.c_str(),"a"); else logfp = NULL;
    }
    
}

void Logger::reset() { ison = true; }

void Logger::on() { ison = true; }

void Logger::off() { ison = false; }

void Logger::dolog() {

    if (not ison) return;

    if (simstep<logstepoffs) return;

    if ((simstep-logstepoffs)%logstep!=0) return;

    if (pop!=NULL and logfp!=NULL)
        pop->store(field, logfp);
    else if (prj!=NULL and logfp!=NULL)
        prj->store(field,logfp);

}


void Logger::close() {

    if (logfp!=NULL) { fclose(logfp); logfp = NULL; }

    // loggers.erase(loggers.begin() + idx);

}


void Logger::clear() { loggers.clear(); }


void Logger::dologall() { for (size_t l=0; l<loggers.size(); l++) loggers[l]->dolog(); }


void Logger::onall() { for (size_t l=0; l<loggers.size(); l++) loggers[l]->on(); }


void Logger::offall() { for (size_t l=0; l<loggers.size(); l++) loggers[l]->off(); }


void Logger::closeall() {

    for (size_t l=0; l<loggers.size(); l++) loggers[l]->close();

    clear();

}
