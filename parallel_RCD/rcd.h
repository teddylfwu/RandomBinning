#include <algorithm>
#include <iomanip>
#include<cmath>
#include "loss.h"
#include "util.h"
#include "omp.h"
void rcd(vector<Feature*>& features, vector<double>& labels, int n, LossFunc* loss, double lambda,  double* w_ret, int nr_threads){
	double* factors = new double[n];
	
	int d = features.size();
	double k1 = loss->sec_deriv_ubound();//second derivative upper bound of the Loss
	double* H_diag = new double[d];
	for(int r=0;r<features.size();r++){
		Feature* fea = features[r];
		H_diag[r] = 0.0;
		for(SparseVec::iterator it=fea->values.begin(); it!=fea->values.end(); it++){
			H_diag[r] += it->second*it->second;
		}
	}
	
	double* w = new double[d];
	for (int i=0;i<d; i++)
		w[i] = 0.0;
	double funval = 0.0;
	for (int i=0;i<n;i++){
		factors[i] = 0.0;
	}
	int max_iter = 300;
	
	int ins_id;
	//double w_old = 0;
	double fval_new;
	//int* pi = new int[d];
	//for (int i=0;i<d;i++)
	//	pi[i] = rand()%d;
	
	int chunk = 10000;
	double start = omp_get_wtime();
	double minus_time=0.0;
	for (int iter=0;iter<max_iter; iter++){
		
		#pragma omp parallel shared(chunk)
		{
			#pragma omp for schedule(dynamic,chunk) nowait
			
			//#pragma omp parallel for 
			for (int inner_iter = 0; inner_iter < d; inner_iter++){
				
				int j=inner_iter;
				Feature* fea = features[j];
				double Qii = H_diag[j]*k1;
				//update w
				double gradient =  0.0;
				for (SparseVec::iterator ii = fea->values.begin(); ii != fea->values.end(); ++ii){
					gradient += loss->deriv(factors[ii->first],labels[ii->first]) * ii->second;
				}
				double eta = softThd(w[j] - gradient/(Qii),lambda/(Qii)) - w[j];
				if( fabs(eta)>1e-3 ){
					w[j] += eta;
					for (SparseVec::iterator ii = fea->values.begin(); ii != fea->values.end(); ++ii){
						#pragma omp atomic
						factors[ii->first] += eta * ii->second;
					}
				}
			}
		}
		
		if (iter % 10 == 0){

			minus_time -= omp_get_wtime();
			funval = 0.0;
			for (int i=0;i<n;i++){
				funval +=  loss->fval(factors[i],labels[i]);
			}
			int nnz = 0;
			for (int i=0; i<d;i++){
				funval += lambda*fabs(w[i]);
				if (fabs(w[i])>1e-8)
					nnz++;
			}
			
			for(int i=0;i<d;i++){
				int j = i+rand()%(d-i);
				swap(features[i],features[j]);
				swap(w[i],w[j]);
				swap(H_diag[i],H_diag[j]);
			}
			
			minus_time += omp_get_wtime();

			double end = omp_get_wtime();
			double time_used = end-start-minus_time;
			cerr  << setprecision(15)<<setw(20) << iter << setw(20) << time_used << setw(20) << funval << setw(10)<<nnz<< endl;
		}
	}
	
	for(int i=0;i<d;i++){
		int j = features[i]->id;
		w_ret[j] = w[i];
	}

	delete [] H_diag;
	delete [] factors;
	delete [] w;
}
