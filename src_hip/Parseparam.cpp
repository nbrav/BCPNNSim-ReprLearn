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

#include <stdlib.h>
#include <cstring>
#include <math.h>
#include <fstream>
#include <sys/types.h>
#include <sys/stat.h>

#include "Parseparam.h"

using namespace std;
// using namespace Globals;

void Parseparam::error(string errloc,string errstr) {

    fprintf(stderr,"ERROR in Parseparam::%s: %s\n",errloc.c_str(),errstr.c_str());

    exit(0);

}

Parseparam::Parseparam(string paramfile) {
    _paramfile = paramfile;
    time(&_oldmtime);
    nparampost = 0;
}

void Parseparam::postparam(string paramstring,void *paramvalue,Value_t paramtype) {
    _paramstring.push_back(paramstring);
    _paramvalue.push_back(paramvalue);
    _paramtype.push_back(paramtype);
    nparampost++;
}

int Parseparam::findparam(string paramstring) {
    /* Find index in _paramstring */
    for (int k=0; k<(int)_paramstring.size(); k++)
	if (strcmp(paramstring.c_str(),_paramstring[k].c_str())==0) return k;
    return -1;
}

void Parseparam::doparse(string paramlogfile) {
    char *str1,*str2;
    int k;
    string linestr,str;
	
    std::ifstream paramf(_paramfile.c_str());
    if (!paramf) {
		fprintf(stderr,"paramfile \"%s\" not found!\n",_paramfile.c_str());
		error("Parseparam::doparse","Could not open paramfile: " + _paramfile);
	}

    _vintvalue.resize(nparampost);
    _vlongvalue.resize(nparampost);
    _vfltvalue.resize(nparampost);
    _vboolevalue.resize(nparampost);
    _vstringvalue.resize(nparampost);

    bool comment;
    while (getline(paramf,linestr)) {
	vector<string> valstrs;

	valstrs.clear();

	comment = false;
	for (int i=0; i<(int)linestr.size(); i++) {
	    if (linestr[i]=='#') comment = true;
	    if (comment) linestr[i] = (char)0;
	}
	if (strlen(linestr.c_str())==0) continue;

	str1 = strtok((char *)linestr.c_str()," \t");
	str2 = strtok(NULL," \t");
	valstrs.push_back(string(str2));

	char *str3 = str2;

	while (str2!=NULL) {
	    str2 = strtok(NULL," \t");
	    if (str2!=NULL) { valstrs.push_back(string(str2)); }
	}

	str2 = str3;

	if (str1==NULL || valstrs.size()==0) error("Parseparam::doparse","Illegal str1 or str2");

	k = findparam(string(str1));
	if (k<0) continue;

	switch (_paramtype[k]) {
	case Int:
	    int intval;
	    *(int *)_paramvalue[k] = atoi(str2);
	    for (size_t s=0; s<valstrs.size(); s++) {
	      	intval = atoi(valstrs[s].c_str());
	     	_vintvalue[k].push_back(intval);
	    }
	    break;
	case Long:
	    long longval;
	    *(long *)_paramvalue[k] = atol(str2);
	    for (size_t s=0; s<valstrs.size(); s++) {
	      	longval = atol(valstrs[s].c_str());
	     	_vlongvalue[k].push_back(longval);
	    }
	    break;
	case Float:
	    *(float *)_paramvalue[k] = atof(str2);
	    float fltval;
	    for (size_t s=0; s<valstrs.size(); s++) {
	     	fltval = atof(valstrs[s].c_str());
	    	_vfltvalue[k].push_back(fltval);
	    }
	    break;
	case Boole:
	    if (strcmp(str2,"true")==0)
		*(bool *)_paramvalue[k] = true;
	    else if (strcmp(str2,"false")==0)
		*(bool *)_paramvalue[k] = false;
	    else
		error("Parseparam::doparse","Illegal boolean value");
	    for (size_t s=0; s<valstrs.size(); s++) {
		if (strcmp(valstrs[s].c_str(),"true")==0)
		    _vboolevalue[k].push_back(true);
	    	else if (strcmp(valstrs[s].c_str(),"false")==0)
	    	    _vboolevalue[k].push_back(false);
	    	else
	    	    error("Parseparam::doparse","Illegal boolean value");
	    }
	    break;
	case String:
	    *(string *)_paramvalue[k] = string(str2);
	    if (_paramstring[k].compare("paramlogfile")==0) _paramlogfile = string(str2);
	    for (size_t s=0; s<valstrs.size(); s++)
	    	_vstringvalue[k].push_back(valstrs[s]);
	    break;
	default:
	    error("Parseparam::doparse","Illegal paramtype");
	}

    }
    paramf.close();

}

int Parseparam::getintparam(string paramstring,int v) {

    int k = findparam(paramstring);
    if (v<0 or _vintvalue[k].size()<=v)
	error("Parseparam::getintparam","Paramstring '" + paramstring + "' not found.");
    return _vintvalue[k][v];

}

long Parseparam::getlongparam(std::string paramstring,int v) {

    int k = findparam(paramstring);
    if (v<0 or _vlongvalue[k].size()<=v)
	error("Parseparam::getlongparam","Paramstring '" + paramstring + "' not found.");
    return _vlongvalue[k][v];

}

float Parseparam::getfltparam(string paramstring,int v) {

    int k = findparam(paramstring);
    if (v<0 or _vfltvalue[k].size()<=v)
	error("Parseparam::getfltparam","Paramstring '" + paramstring + "' not found.");
    return _vfltvalue[k][v];

}

bool Parseparam::getbooleparam(std::string paramstring,int v) {

    int k = findparam(paramstring);
    if (v<0 or _vboolevalue[k].size()<=v)
	error("Parseparam::getbooleparam","Paramstring '" + paramstring + "' not found.");
    return _vboolevalue[k][v];

}

string Parseparam::getstringparam(std::string paramstring,int v) {

    int k = findparam(paramstring);
    if (v<0 or _vstringvalue[k].size()<=v)
	error("Parseparam::getstringparam","Paramstring '" + paramstring + "' not found.");
    return _vstringvalue[k][v];

}


bool Parseparam::haschanged() {
    struct stat file_stat; bool haschd = false;
    int err = stat(_paramfile.c_str(),&file_stat);
    if (err != 0) {
        perror("Parseparam::haschanged");
        exit(9);
    }
    if (file_stat.st_mtime > _oldmtime) {
	haschd = true;
	time(&_oldmtime);
    }
    return haschd;
}

char *Parseparam::timestamp() {
    // current date/time based on current system
    time_t now = time(0);
    // convert now to tm struct for UTC
    tm *ltm = localtime(&now);
    char *ldot = (char *)malloc(15);
    error("Parseparam::timestamp","Timestamp removed");
    return ldot;
}

char *Parseparam::dolog(bool usetimestamp) {
    _timestamp = timestamp();
    if (usetimestamp) _paramlogfile += "_" + string(_timestamp);
    _paramlogfile += ".sav";
    FILE *logf = fopen(_paramlogfile.c_str(),"w");
    if (usetimestamp)
	fprintf(logf,"%-30s%s\n","TIMESTAMP",_timestamp);
    for (size_t p=0; p<_paramstring.size(); p++) {
	fprintf(logf,"%-30s",_paramstring[p].c_str());
	switch (_paramtype[p]) {
	case Int:
	    fprintf(logf,"%d",*(int *)_paramvalue[p]);
	    break;
	case Long:
	    fprintf(logf,"%ld",*(long *)_paramvalue[p]);
	    break;
	case Float:
	    fprintf(logf,"%f",*(float *)_paramvalue[p]);
	    break;
	case Boole:
	    if (*(bool *)_paramvalue[p])
		fprintf(logf,"true");
	    else 
		fprintf(logf,"false");
	    break;
	case String:
	    fprintf(logf,"%s",(*(string *)_paramvalue[p]).c_str());
	    break;
	default:
	    error("Parseparam::doparse","Illegal paramtype");
	}
	fprintf(logf,"\n");
    }
    fclose(logf);
    return _timestamp;
}
