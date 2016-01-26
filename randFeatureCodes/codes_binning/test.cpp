#include <cmath>
#include <iostream>
#include "util.h"
using namespace std;


int main(){
	
	CodeIndexMap s;
	
	int k = floor(-2.3);
	cerr << "k=" << k << endl;

	int a1[3] = {2,3,1};
	int a2[3] = {3,2,1};
	int a3[3] = {k,2,3};
	int a4[3] = {1,0,1};
	vector<int> v1(a1,a1+3);
	vector<int> v2(a2,a2+3);
	vector<int> v3(a3,a3+3);
	vector<int> v4(a4,a4+3);
	
	s.insert( make_pair(&v1,1) );
	s.insert( make_pair(&v2,2) );
	s.insert( make_pair(&v3,3) );
	s.insert( make_pair(&v4,4) );
	
	for(CodeIndexMapIter it = s.begin(); it!=s.end(); it++){
		vector<int>* v = it->first;
		for(int i=0;i<v->size();i++){
			cout << v->at(i) << " ";
		}
		cout << " : " << it->second << endl;
	}
}
