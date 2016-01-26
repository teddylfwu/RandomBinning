#include <stdio.h>
#include <stdlib.h>
#include <cfloat>

#include "Gaussian.h"
#include "Gamma.h"
#include "util.h"

#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <utility>
#include <vector>
#include <cmath>

#include <omp.h>

using namespace std;

const double NONE_LABEL = -19191.0;

void readSVMfile(const char* f, int& d, vector< vector< pair<int,double> > >& ins, vector<double>& labs){
	
	d = 0;

	fstream fs;
	fs.open(f, fstream::in);

	if(fs.fail()){
		cerr << "error reading file." << endl;
		return;
	}

	double label;
	int index;
	char colon;
	double feature;
	string line, token;

	while(getline(fs, line)){
		
		if( line.length() < 1 ){

			labs.push_back(NONE_LABEL);
			ins.push_back(vector< pair<int,double> >());
			continue;
		}
		
		ins.push_back(vector< pair<int,double> >());
		
		stringstream ss;
		ss << line;
		ss >> label;
		
		labs.push_back(label);
		
		while(getline(ss, token, ' ')){
			
			if(token.size() == 0){
				continue;
			}
			
			stringstream ss2;

			ss2 << token;
			ss2 >> index >> colon >> feature;

			ins.back().push_back(pair<int,double>(index, feature));
			if(index > d){
				d = index;
			}
		}
	}

	fs.close();
}

void writeSVMfile(const char* f, vector< vector< pair<int,double> > >& ins, vector<double>& labs){
	
	fstream fs;
	fs.open(f, fstream::out);

	if(fs.fail()){
		cerr << "error writing file." << endl;
		return;
	}

	for(int i = 0; i < ins.size(); i++){
		
		if( labs[i] == NONE_LABEL ){
			fs << endl;
			continue;
		}

		fs << (labs[i]);
		for(int j = 0; j < ins[i].size(); j++){
			fs << " " << ins[i][j].first << ":" << ins[i][j].second;
		}
		fs << endl;
	}

	fs.close();
}

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
	ins_new.resize(N);
	for(int i=0;i<N;i++)
		ins_new[i].resize(D);
	
	int j_offset = 0;
	double sqrt_D = sqrt(D);
	for(int j=0; j<D; j++){
		
		double* delta_j = delta[j];
		double* u_j = u[j];
		
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
			
			ins_new[i][j] = make_pair(j_offset + ind + 1, 1.0/sqrt_D) ;
			//ins_new[i][j] = make_pair(j_offset + ind + 1, 1.0) ;
		}
        //delete vector keys in the map
        for(CodeIndexMap::iterator it=code_ind_map.begin(); it!=code_ind_map.end(); it++){
        delete (it->first);
        }
		//cerr << "code size=" << code_ind_map.size() << endl;
		j_offset += code_ind_map.size();
        if( j % 100== 0 ){
            cerr << "j=" << j << endl;
        }
	}
	
	for(int i = 0; i < D; i++){
		delete[] delta[i];
		delete[] u[i];
	}
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

int main(int argc, char* argv[]){

    if(argc < 1+6){
        cerr << "Usage: " << argv[0] << " [inTrain] [inTest] [outTrain] [outTest] [D] [gamma]" << endl;
        exit(0);
    }

    char* trainFile = argv[1];
    char* testFile = argv[2];
    char* trainOut = argv[3];
    char* testOut = argv[4];
    int D = atoi(argv[5]);
    double gamma = atof(argv[6]);

    int dimension;
    vector<double> label_train, label_test;
    vector< vector< pair<int,double> > > train_old, test_old, train_new, test_new;

    readSVMfile(trainFile, dimension, train_old, label_train);
    readSVMfile(testFile, dimension, test_old, label_test);
    dimension += 1;

    cerr << "#train_sample=" << train_old.size() << endl;
    cerr << "#test_sample=" << test_old.size() << endl;
    cerr << "dim=" << dimension << endl;
    cerr << "D=" << D << endl;

    vector< vector<pair<int,double> > >  data_old, data_new;
    for(int i=0;i<train_old.size();i++)
        data_old.push_back(train_old[i]);
    for(int i=0;i<test_old.size();i++)
        data_old.push_back(test_old[i]);

    random_binning_feature(dimension, D, data_old, data_new, gamma);
    //random_fourier_feature(dimension, D, data_old, data_new, gamma);
    //find max index
    int max_ind = -1;
    for(vector<vector<pair<int,double> > >::iterator it=data_new.begin(); it!=data_new.end(); it++){
        vector<pair<int,double> >* ins = &(*it);
        int ind = ins->at(ins->size()-1).first;
        if( max_ind < ind  )
            max_ind = ind;
    }
    cerr << "max-rf-index=" << max_ind << endl;

    for(int i=0;i<train_old.size();i++)
        train_new.push_back(data_new[i]);
    int Ntr = train_old.size();
    for(int i=Ntr;i<Ntr+test_old.size();i++)
        test_new.push_back(data_new[i]);

    writeSVMfile(trainOut, train_new, label_train);
    writeSVMfile(testOut, test_new, label_test);

    return 0;
}
