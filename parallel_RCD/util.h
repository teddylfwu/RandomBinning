#ifndef UTIL
#define UTIL
#include <cmath>
#include <vector>
#include <string>
#include <stdlib.h>
#include <fstream>
#include <iostream>
using namespace std;

typedef vector<pair<int,double> > Instance;
typedef vector<pair<int,double> > SparseVec;
class Feature{
	public:
	int id;
	SparseVec values;
};

const int LINE_LEN = 100000000;
const int FNAME_LEN = 10000;

bool nnz_comp(const SparseVec* a, const SparseVec* b){
    return (a->size() > b->size());
}

bool pair_comp(const pair<int,double> a, const pair<int,double> b){
    return (a.second > b.second);
}

double inner_product(SparseVec* a, double* b){
    double sum = 0.0;
    for (SparseVec::iterator it = a->begin(); it != a->end(); ++it){
        sum += it->second * b[it->first];
    }
    return sum;
}

double inner_product(SparseVec* a1, SparseVec* a2){
    double sum = 0.0;
    /*SparseVec::iterator it2 = a2->begin();
    for (SparseVec::iterator it1 = a1->begin(); it1 != a1->end(); ++it1){
        while (it2 != a2->end() && it2->first < it1->first){
            it2++;
        }
        if (it2 == a2->end())
            break;
        if (it2->first == it1->first)
            sum += it1->second * it2->second;
    }*/
    SparseVec::iterator it1 = a1->begin();
    SparseVec::iterator it2 = a2->begin();
    while( it1 != a1->end() && it2 != a2->end() ){
	if( it1->first == it2->first ){
		sum += it1->second * it2->second;
		it1++;
		it2++;
	}else if( it1->first < it2->first ){
		it1++;
	}else{
		it2++;
	}
    }
    //cerr << "prod=" << sum << endl;
    return sum;
}

void vAdd(SparseVec* v1, SparseVec* v2){
	
	SparseVec fea;
	
	SparseVec::iterator it = v1->begin();
	SparseVec::iterator it2 = v2->begin();
	while( it != v1->end() && it2 != v2->end() ){
		
		if(it->first==it2->first){
			fea.push_back(make_pair(it->first, it->second+it2->second));
			it++;
			it2++;
		}else if(it->first < it2->first){
			fea.push_back(make_pair(it->first, it->second));
			it++;
		}else{
			fea.push_back(make_pair(it2->first, it2->second));
			it2++;
		}
	}

	SparseVec::iterator it_r;
	SparseVec* v_r;
	if( it==v1->end() ){
		it_r = it2;
		v_r = v2;
	}else{
		it_r = it;
		v_r = v1;
	}
	
	for(; it_r != v_r->end(); it_r++){
		fea.push_back(make_pair(it_r->first, it_r->second));
	}
	
	*(v1) = fea;
}

void vTimes(SparseVec* v1, double a){
	
	SparseVec::iterator it;
	for(it=v1->begin();it!=v1->end();it++){
		it->second *= a;
	}
}

vector<string> split(string str, string pattern){

	vector<string> str_split;
	size_t i=0;
	size_t index=0;
	while( index != string::npos ){

		index = str.find(pattern,i);
		str_split.push_back(str.substr(i,index-i));

		i = index+1;
	}

	if( str_split.back()=="" )
		str_split.pop_back();

	return str_split;
}

void readData(char* fname, vector<Instance*>& data, vector<double>& labels, int& d){
	
	ifstream fin(fname);
	char* line = new char[LINE_LEN];
	
	d = -1;
	while( !fin.eof() ){
		
		fin.getline(line, LINE_LEN);
		string line_str(line);

		if( line_str.length() < 2 && fin.eof() )
			break;

		vector<string> tokens = split(line_str, " ");
		double label = atof(tokens[0].c_str());
		Instance* ins = new Instance();
		for(int i=1;i<tokens.size();i++){
			vector<string> kv = split(tokens[i],":");
			int ind = atoi(kv[0].c_str());
			double val = atof(kv[1].c_str());
			ins->push_back(make_pair(ind,val));

			if( ind > d ){
				d = ind;
			}
		}
		
		data.push_back(ins);
		labels.push_back(label);
	}
	fin.close();

	d = d+1; //adding bias

	delete[] line;
}

void writeData( char* outputName, vector<Instance*>& data, vector<double>& labels){
	
	ofstream fout(outputName);
	for(int i=0;i<data.size();i++){
		fout << labels[i] << " ";
		Instance* ins = data[i];
		for(Instance::iterator it=ins->begin();
			it!=ins->end();it++){
			fout << it->first << ":" << it->second << " ";
		}
		fout << endl;
	}
	fout.close();
}

void dataToFeatures(vector<Instance*>& data, int d, vector<Feature*>& features){
	
	int n = data.size();
	
	features.clear();
	features.resize(d);
	for(int j=0;j<d;j++){
		Feature* fea = new Feature();
		fea->id = j;
		features[j] = fea;
	}
	
	for(int i=0;i<n;i++){

		Instance* ins = data[i];
		for(Instance::iterator it=ins->begin();
			it!=ins->end(); it++){
			
			features[it->first]->values.push_back(make_pair(i,it->second));
		}
	}
}

void featuresToData(vector<SparseVec*>& features, int n, vector<Instance*>& data){
	
	int d = features.size();
	
	for(int i=0;i<data.size();i++){
		delete data[i];
	}
	data.clear();
	data.resize(n);
	for(int i=0;i<n;i++){
		data[i] = new Instance();
	}
	
	for(int j=0;j<d;j++){

		SparseVec* fea = features[j];
		for(SparseVec::iterator it=fea->begin();
			it!=fea->end(); it++){
			
			data[it->first]->push_back(make_pair(j,it->second));
		}
	}
}

double softThd(const double &x,const double  &thd){
	if (x>thd)
		return x-thd;
	else if (x < -thd)
		return x+thd;
	else
		return 0;
}


double abs_maximum(double* values, int size,int &posi){

	double ret = -1e300;
	for(int i=0;i<size;i++){
		if( fabs(values[i]) > ret ){
			ret = fabs(values[i]);
			posi = i;
		}
	}
	return ret;
}

double prod(double* q, SparseVec* x){
	
	double sum = 0.0;
	for(SparseVec::iterator it=x->begin();
		it!=x->end(); it++){
		
		sum += q[it->first] * it->second;
	}
	return sum;
}

#endif
