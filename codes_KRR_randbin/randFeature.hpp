#ifndef _RANDBIN_
#define _RANDBIN_

#include <stdio.h>
#include <stdlib.h>
#include <cfloat>
#include <iostream>
#include <sstream>
#include <fstream>
#include <utility>
#include <vector>
#include <cmath>
#include <omp.h>
#include <random>
#include <set>
#include <map>
#include <vector>

using namespace std;

class VectComp{ //assume vector of the length
	public:
	bool operator()(const vector<int>* a, const vector<int>* b){
		
		for(int i=0; i<a->size(); i++){
			
			int v1 = a->at(i);
			int v2 = b->at(i);
			if( v1 < v2 )
				return 1;
			else if( v1 > v2 )
				return 0;
		}
		
		return 0; //equal means not <
	}
};

typedef map<vector<int>*, int, VectComp> CodeIndexMap;
typedef map<vector<int>*, int, VectComp>::iterator CodeIndexMapIter;

double randn(double mu=0.0, double sigma=1.0) {

	static bool deviateAvailable=false;		//flag
	static float storedDeviate;		//		deviate from previous calculation
	double dist, angle;

	if (!deviateAvailable) {

		dist=sqrt( -2.0 * log(double(rand()) / double(RAND_MAX)) );
		angle=2.0 * M_PI * (double(rand()) / double(RAND_MAX));

		storedDeviate=dist*cos(angle);
		deviateAvailable=true;

		return dist * sin(angle) * sigma + mu;
	}
	else {
		deviateAvailable=false;
		return storedDeviate*sigma + mu;
	}
}

class Gamma{
	
	public:
	default_random_engine gen;
	
	double gamma;
	gamma_distribution<double>* dist;

	Gamma(double g){
		gamma=g;
		dist = new gamma_distribution<double>(2.0, 1.0/gamma);
	}

	double generate(){
		return (*dist)(gen);
	}
};

const double NONE_LABEL = -19191.0;

vector<int>* compute_bin_num( vector<pair<int,double> >* ins, double* delta, double* u, int d ){

	vector<int>* code = new vector<int>();
	code->resize(d);

	int j=0;
	for(vector<pair<int,double> >::iterator it=ins->begin(); it!=ins->end(); it++){
		while( j < it->first ){
			(*code)[j] = floor((0.0-u[j])/delta[j]) ;
			j++;
		}

		(*code)[j] = floor( (it->second-u[j])/delta[j] ) ;
		j++;
	}
	while( j < d ){
		(*code)[j] = floor((0.0-u[j])/delta[j]) ;
		j++;
	}

	return code;
}

void random_binning_feature(int d, int D, vector< vector< pair<int,double> > >& ins_old, vector< vector< pair<int,double> > >& ins_new, double gamma ){

	double** delta = new double*[D];
	double** u = new double*[D];

	Gamma gamma_dist(gamma);

	for(int i = 0; i < D; i++){
		delta[i] = new double[d];
		u[i] = new double[d];
	}
	for(int i = 0; i < D; i++){
		for(int j = 0; j < d; j++){
			delta[i][j] = gamma_dist.generate();
			u[i][j] = ((double)rand()/RAND_MAX)*delta[i][j];
		}
	}

	int N = ins_old.size();
	vector< vector<pair<int,double> > > features;
	features.resize(D);
	for(int j=0;j<D;j++)
		features[j].resize(N);
	ins_new.resize(N);
	for(int i=0;i<N;i++)
		ins_new[i].resize(D);
	int* offset = new int[D];
	offset[0] = 0;

	int j_offset = 0;
	double sqrt_D = sqrt(D);
#pragma omp parallel for
	for(int j=0; j<D; j++){

		double* delta_j = delta[j];
		double* u_j = u[j];
		vector<pair<int,double> >* fea = &(features[j]);

		CodeIndexMap code_ind_map;
		CodeIndexMapIter it;
		vector<int>* code;
		for(int i=0;i<N;i++){
			if( ins_old[i].size() == 0 )
				continue;

			code = compute_bin_num( &(ins_old[i]), delta_j,  u_j, d );
			//cerr << "code size=" << code->size() << endl;
			int ind;
			if( (it=code_ind_map.find(code)) == code_ind_map.end() ){
				ind = code_ind_map.size();
				code_ind_map.insert(make_pair(code, ind));
				//cerr << "ind=" << ind << endl;
			}else{
				ind = it->second;
				delete code;
			}

			//(*fea)[i] = make_pair(ind + 1, 1.0/sqrt_D) ;
			(*fea)[i] = make_pair(ind + 1, 1.0) ;
		}
    //delete vector keys in the map
    for(CodeIndexMap::iterator it=code_ind_map.begin(); it!=code_ind_map.end(); it++){
      delete (it->first);
    }
		
    //cerr << "code size=" << code_ind_map.size() << endl;
		if(j<D-1)
			offset[j+1] = code_ind_map.size();
	}

	//compute offset
	for(int j=1;j<D;j++){
		offset[j] = offset[j] + offset[j-1];
	}

	for(int j=1;j<D;j++){
		vector<pair<int,double> >* fea = &(features[j]);
		int offset_j = offset[j];
		for(vector<pair<int,double> >::iterator it=fea->begin(); it!=fea->end(); it++)
			it->first += offset_j;
	}

	//convert to data_new
	for(int j=0;j<D;j++){

		vector<pair<int,double> >* fea = &(features[j]);
		int i=0;
		for(vector<pair<int,double> >::iterator it=fea->begin(); it!=fea->end(); it++, i++){
			ins_new[i][j] = make_pair(it->first, sqrt(1.0/D)*it->second);
		}
	}

	for(int i = 0; i < D; i++){
		delete[] delta[i];
		delete[] u[i];
	}
	delete[] offset;
	delete[] delta;
	delete[] u;
}

void random_fourier_feature(int d, int D, vector< vector< pair<int,double> > >& ins_old, vector< vector< pair<int,double> > >& ins_new, double gamma ){
	
	double* w[d];
	double b[D];

	for(int i = 0; i < d; i++){
		w[i] = new double[D];
	}
	
	for(int i = 0; i < d; i++){
		for(int j = 0; j < D; j++){
			w[i][j] = randn(0.0, sqrt(gamma));
		}
	}
	
	for(int i = 0; i < D; i++){
		b[i] = 2.0 * M_PI * (double) rand() / (double) RAND_MAX;
	}
	
	ins_new.resize(ins_old.size());

#pragma omp parallel for
	for(int i = 0; i < ins_old.size(); i++){
		
		if( ins_old[i].size() == 0 ){
			continue;
		}

		double wx[D];
		
		//ins_new.push_back(vector< pair<int,double> >());
		
		for(int j = 0; j < D; j++){
			wx[j] = 0.0;
		}
		
		for(int j = 0; j < ins_old[i].size(); j++){
			int index = ins_old[i][j].first;
			double feature = ins_old[i][j].second;
			
			for(int k = 0; k < D; k++){
				wx[k] += w[index-1][k] * feature;
			}
		}

		ins_new[i].clear();

		for(int k = 0; k < D; k++){
			ins_new[i].push_back(pair<int,double>(k + 1, sqrt(2.0/D) * cos(wx[k] + b[k])));
			//ins_new[i].push_back(pair<int,double>(k + 1, cos(wx[k] + b[k])));
		}
	}
	
	for(int i = 0; i < d; i++){
		delete [] w[i];
	}
}

void random_stump_feature(int d, int D, vector< vector< pair<int,double> > >& ins_old, vector< vector< pair<int,double> > >& ins_new){
	
	double s[D];
	
	for(int i = 0; i < D; i++){
		s[i] = (double) rand() / (double) RAND_MAX > 0.5 ? 1.0 : -1.0;
	}
	
	double cmax[d], cmin[d], cpd[d];
	
	for(int i = 0; i < d; i++){
		cmax[i] = -DBL_MAX;
		cmin[i] = DBL_MAX;
	}

	for(int i = 0; i < ins_old.size(); i++){
		
		for(int j = 0; j < ins_old[i].size(); j++){
			int index = ins_old[i][j].first;
			double feature = ins_old[i][j].second;
			
			if(cmax[index-1] < feature){
				cmax[index-1] = feature;
			}
			
			if(cmin[index-1] > feature){
				cmin[index-1] = feature;
			}
		}
	}

	cpd[0] = cmax[0] - cmin[0];
	for(int i = 1; i < d; i++){
		cpd[i] = cpd[i-1] + (cmax[i] - cmin[i]);
	}

	int r[D];
	double p[D];
	
	for(int i = 0; i < D; i++){
		r[i] = d;
		p[i] = cpd[d-1] * (double) rand() / (double) RAND_MAX;

		for(int j = 0; j < d; j++){
			if(p[i] < cpd[j]){
				r[i] = j;
				if(j == 0){
					p[i] = p[i] / cpd[j];
				}else{
					p[i] = (p[i] - cpd[j-1]) / (cpd[j] - cpd[j-1]);
				}
				break;
			}
		}

	}
	
	for(int i = 0; i < D; i++){
		cout << r[i] << " " << p[i] << endl;
	}
	
	for(int i = 0; i < d; i++){
		cout << d << " "<< cpd[i] << " " << cmax[i] << " " << cmin[i] << endl;
	}


	for(int i = 0; i < ins_old.size(); i++){

		//ins_new.push_back(vector< pair<int,double> >());

		for(int j = 0; j < ins_old[i].size(); j++){
			int index = ins_old[i][j].first;
			double feature = ins_old[i][j].second;
		
			for(int k = 0; k < D; k++){
				if(index-1 == r[k]){
					ins_new[i].push_back(pair<int,double>(k + 1, s[k] * feature * p[k]));
				}
			}
		}
	}
}

#endif

