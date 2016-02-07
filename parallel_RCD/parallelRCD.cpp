#include <iostream>
#include <omp.h>
#include "util.h"
#include "loss.h"
#include "rcd.h"
using namespace std;

double RMSE( vector<Instance*>& data, vector<double>& labels, double* w, int d ){
	
	double y_sq_sum = 0.0;
	double sqerr = 0.0;
	for(int i=0;i<data.size();i++){
		Instance* ins = data[i];
		double y = labels[i];
		double prod = 0.0;
		for(Instance::iterator it=ins->begin(); it!=ins->end(); it++)
			if( it->first < d )
				prod += w[it->first] * it->second;
		sqerr += (y-prod)*(y-prod);
		y_sq_sum += y*y;
	}
	sqerr /= y_sq_sum;
	return sqrt(sqerr);
}

double accuracy( vector<Instance*>& data, vector<double>& labels, double* w, int d ){
	
	double err = 0.0;
	for(int i=0;i<data.size();i++){
		Instance* ins = data[i];
		double y = labels[i];
		double prod = 0.0;
		for(Instance::iterator it=ins->begin(); it!=ins->end(); it++)
			if( it->first < d )
				prod += w[it->first] * it->second;
		if( prod*y < 0.0 )
			err += 1.0;
	}
	err /= data.size();
	return 1.0-err;
}

int main(int argc, char** argv){

	if( argc < 1+6 ){
		cerr << "./parallelRCD [data] [testdata] [L1_lambda] [loss(0:square,1:L2-hinge,2:logistic)] [num_threads] [num_iter] (stop_obj)" << endl;
		exit(0);
	}

	char* dataFname = argv[1];
	char* testFname = argv[2];
	double lambda = atof(argv[3]);
	int loss_to_use = atoi(argv[4]);
	int nThreads =atoi(argv[5]);
	int nIter = atoi(argv[6]);
	
	double stop_obj = -1e300;
	if( argc >= 1+7 ){
		stop_obj = atof(argv[7]);
	}
	
	omp_set_num_threads(nThreads);
	
	vector<Instance*> data;
	vector<Instance*> testdata;
	vector<Feature*> features;
	vector<double> labels;
	vector<double> testlabels;
	int d,n;
	
	srand(time(NULL));
	readData(dataFname, data, labels, d);
	n = data.size();
	dataToFeatures( data, d, features);
	cout << "iterations\ttime(s)\tobjective\tnnz"<< endl;
	cout << "#samples n="  << data.size() <<"; #features d=" << features.size() << endl;
	
	LossFunc* loss;
       	if( loss_to_use==0 )
		loss = new SquareLoss();
	else if( loss_to_use==1 )
		loss = new L2hingeLoss();
	else
		loss = new LogisticLoss();

	double* w = new double[d];
	rcd(features,labels,n,loss,lambda,w,nThreads, nIter, stop_obj);
	
	int d2;
	readData(testFname, testdata, testlabels, d2);
	if( loss_to_use==0 ){
		double rmse = RMSE( testdata, testlabels, w, d );
		cout << "test rmse = " << rmse << endl;
	}else{
		double acc = accuracy( testdata, testlabels, w, d );
		cout << "test acc = " << acc << endl;
	}
	
	return 0;
}
