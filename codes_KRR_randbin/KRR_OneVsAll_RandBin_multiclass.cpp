// This program performs multiclass classification based on the
// one-vs-all scheme. In each binary classification problem, the
// classifer is a kernel ridge regression. The kernel ridge
// regresssion is approximated by using the Random Binning feature
// method. This program is suitable for the case of a large number of
// classes.
//
// This program implements the following kernels (lambda is the
// regularization):
//
// 1. IsotropicGaussian: k(x,y) = exp(-r^2/sigma^2/2) + lambda,
//     where r = sum_i (x_i-y_i)^2.
//
// 2. IsotropicLaplace: k(x,y) = exp(-r/sigma) + lambda,
//     where r = sqrt[ sum_i (x_i-y_i)^2 ].
//
// 3. ProdLaplace: k(x,y) = prod_i exp(-r_i/sigma) + lambda,
//     where r_i = abs(x_i-y_i).
//
// This program uses the LibSVM data format. The attribute indices
// must start from 1. The class labels must be consecutive integers
// starting from 0.
//
// This program can be run with more than one thread. Most parts of
// the program have been threaded except for data I/O and random
// number generation. The program accepts the following compile-time
// macros: -DUSE_OPENBLAS -DUSE_OPENMP .
//
// Usage:
//
//   KRR_OneVsAll.ex NumThreads FileTrain FileTest NumClasses d r Seed
//   lambda Num_sigma List_sigma
//
//   NumThreads:  Number of threads
//   FileTrain:   File name (including path) of train data
//   FileTest:    File name (including path) of test data
//   NumClasses:  Number of classes
//   d:           Data dimension
//   r:           Rank (number of random features)
//   Seed:        Seed for RNG. If < 0, use current time
//   Kernel:      One of IsotropicGaussian, IsotropicLaplace, ProdLaplace
//   lambda:      Regularization
//   Num_lambda:  Number of lambda's for parameter tuning
//   List_lambda: List of lambda's
//   Num_sigma:   Number of sigma's for parameter tuning
//   List_sigma:  List of sigma's


#include <math.h>
#include <time.h>
#include <float.h>
#include <stdio.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#ifdef USE_OPENBLAS
#include <openblas/cblas.h>
#elif USE_ESSL
//#define _ESV_COMPLEX99_
#include <essl.h>
#include <omp.h>
#elif USE_OPENMP
#include <omp.h>
#endif

#include "randFeature.hpp"
#include "LibCMatrix.hpp"

//--------------------------------------------------------------------------
typedef enum { IsotropicGaussian, IsotropicLaplace, ProdLaplace } KERNEL;

#define TWO_PI 6.28318530717958647692
/*
#define PREPARE_CLOCK(is_timing)                 \
    struct timespec time_start, time_end;        \
    double ELAPSED_TIME = 0.0;                   \
    int timing = is_timing;

#define START_CLOCK                                 \
    if (timing) {                                   \
      clock_gettime(CLOCK_MONOTONIC, &time_start);  \
    }

#define END_CLOCK                                                       \
    if (timing) {                                                       \
      clock_gettime(CLOCK_MONOTONIC, &time_end);                        \
      ELAPSED_TIME = time_end.tv_sec - time_start.tv_sec;               \
      ELAPSED_TIME += (time_end.tv_nsec - time_start.tv_nsec) / 1e9;    \
    }
*/

//--------------------------------------------------------------------------
//int ReadData(const char *filename, double **X, int **y, int d, int *n);
//int VerifyFileFormat(const char *filename, int d, int *n);
void UniformRandom01(double *a, int n);
void StandardNormal(double *a, int n);
void StudentT1(double *a, int n);
void MultivariateStudentT1(double *a, int n);

// For multiclass classification, need to convert a single vector
// ytest to a matrix Ytest.
void ConvertYtrain(const DVector &ytrain, DMatrix &Ytrain, int NumClasses);

// If NumClasses = 1 (regression), return relative error. If
// NumClasses = 2 (binary classification), return accuracy% (between 0
// and 100).
double Performance(const DVector &ytest_truth, const DVector &ytest_predict,
                   int NumClasses);

// If Numclasses > 2 (multiclass classification), must call this
// routine. The input Ytest_predict consists of NumClasses
// columns. Return accuracy% (between 0 and 100).
double Performance(const DVector &ytest_truth, const DMatrix &Ytest_predict,
                   int NumClasses);

//--------------------------------------------------------------------------
int main(int argc, char **argv) {

  // Temporary variables
  long int i, j, k, ii, idx = 1;
  double ElapsedTime;

  // Arguments
  int NumThreads = atoi(argv[idx++]);
  char *FileTrain = argv[idx++];
  char *FileTest = argv[idx++];
  int NumClasses = atoi(argv[idx++]);
  int d = atoi(argv[idx++]);
  int r = atoi(argv[idx++]);
  int Num_lambda = atoi(argv[idx++]);
  double *List_lambda = (double *)malloc(Num_lambda*sizeof(double));
  for (ii = 0; ii < Num_lambda; ii++) {
    List_lambda[ii] = atof(argv[idx++]);
  }
  int Num_sigma = atoi(argv[idx++]);
  double *List_sigma = (double *)malloc(Num_sigma*sizeof(double));
  for (i = 0; i < Num_sigma; i++) {
    List_sigma[i] = atof(argv[idx++]);
  }
  int MAXIT = atoi(argv[idx++]);
  double TOL = atof(argv[idx++]);
  bool verbose = atoi(argv[idx++]);

  // Threading
#ifdef USE_OPENBLAS
  openblas_set_num_threads(NumThreads);
#elif USE_ESSL

#elif USE_OPENMP
  omp_set_num_threads(NumThreads);
#else
  NumThreads = 1; // To avoid compiler warining of unused variable
#endif

  PREPARE_CLOCK(1);
  START_CLOCK;

  // Read in X = Xtrain (n*d), y = ytrain (n*1),
  //     and X0 = Xtest (m*d), y0 = ytest (m*1)
  DPointArray Xtrain;        // read all data points from train
  DPointArray Xtest;        // read all data points from test
  DVector ytrain;           // Training labels
  DVector ytest;            // Testing labels (ground truth)
  DVector ytest_predict;    // Predictions

  if (ReadData(FileTrain, Xtrain, ytrain, d) == 0) {
    return -1;
  }
  if (ReadData(FileTest, Xtest, ytest, d) == 0) {
    return -1;
  }

  END_CLOCK;
  ElapsedTime = ELAPSED_TIME;
  printf("OneVsAll: time loading data = %g seconds\n", ElapsedTime); fflush(stdout);

  // For multiclass classification, need to convert a single vector
  // ytrain to a matrix Ytrain. The "predictions" are stored in the
  // corresponding matrix Ytest_predict. The vector ytest_predict is
  DMatrix Ytrain;
  ConvertYtrain(ytrain, Ytrain, NumClasses);

  int Seed = 0; // initialize seed as zero
  // Loop over List_lambda
  for (ii = 0; ii < Num_lambda; ii++) {
    double lambda = List_lambda[ii];
    // Loop over List_sigma
    for (k = 0; k < Num_sigma; k++) {
    double sigma = List_sigma[k];
    // Seed the RNG
    srandom(Seed);

    START_CLOCK;
    // Generate feature matrix Xdata_randbin given Xdata
    vector< vector< pair<int,double> > > instances_old, instances_new;
    long Xtrain_N = Xtrain.GetN();
    for(i=0;i<Xtrain_N;i++){
      instances_old.push_back(vector<pair<int,double> >());
      for(j=0;j<d;j++){
        int index = j+1;
        double *myXtrain = Xtrain.GetPointer();
        double  myXtrain_feature = myXtrain[j*Xtrain_N+i];
        if (myXtrain_feature != 0)
          instances_old.back().push_back(pair<int,double>(index, myXtrain_feature));
      }
    }
    long Xtest_N = Xtest.GetN();
    for(i=0;i<Xtest_N;i++){
      instances_old.push_back(vector<pair<int,double> >());
      for(j=0;j<d;j++){
        int index = j+1;
        double *myXtest = Xtest.GetPointer();
        double  myXtest_feature = myXtest[j*Xtest_N+i];
        if (myXtest_feature != 0)
          instances_old.back().push_back(pair<int,double>(index, myXtest_feature));
      }
    }
    END_CLOCK;
    printf("Train. RandBin: Time (in seconds) for converting data format: %g\n", ELAPSED_TIME);fflush(stdout);
     
     // add 0 feature for Enxu's code
    START_CLOCK;
    random_binning_feature(d+1, r, instances_old, instances_new, sigma);
    END_CLOCK;
    printf("Train. RandBin: Time (in seconds) for generating random binning features: %g\n", ELAPSED_TIME);fflush(stdout);

    START_CLOCK;
    SPointArray Xdata_randbin;  // Generate random binning features
    long int nnz = r*(Xtrain_N + Xtest_N);
    long int dd = 0;
    for(i = 0; i < instances_new.size(); i++){
      if(dd < instances_new[i][r-1].first)
        dd = instances_new[i][r-1].first;
    }
    Xdata_randbin.Init(Xtrain_N+Xtest_N, dd, nnz);
    long int ind = 0;
    long int *mystart = Xdata_randbin.GetPointerStart();
    int *myidx = Xdata_randbin.GetPointerIdx();
    double *myX = Xdata_randbin.GetPointerX();
    for(i = 0; i < instances_new.size(); i++){
      if (i == 0)
        mystart[i] = 0;
      else
        mystart[i] = mystart[i-1] + r;
      for(j = 0; j < instances_new[i].size(); j++){
        myidx[ind] = instances_new[i][j].first-1;
        myX[ind] = instances_new[i][j].second;
        ind++;
      }
    }
    mystart[i] = nnz; // mystart has a length N+1
    // generate random binning features for Xtrain and Xtest
    SPointArray Xtrain;         // Training points
    SPointArray Xtest;          // Testing points
    long Row_start = 0;
    Xdata_randbin.GetSubset(Row_start, Xtrain_N,Xtrain);
    Xdata_randbin.GetSubset(Xtrain_N,Xtest_N,Xtest);
    Xdata_randbin.ReleaseAllMemory();
    END_CLOCK;
    printf("Train. RandBin: Time (in seconds) for converting data format back: %g\n", ELAPSED_TIME);fflush(stdout);
    printf("OneVsAll: n train = %ld, m test = %ld, r = %d, D = %ld, Gamma = %f, num threads = %d\n", Xtrain_N, Xtest_N, r, dd, sigma, NumThreads); fflush(stdout);

    // solve (Z'Z + lambdaI)w = Z'y, note that we never explicitly form
    // Z'Z since Z is a large sparse matrix N*dd
    START_CLOCK;
    int m = Ytrain.GetN(); // number of classes
    long N = Xtrain.GetN(); // number of training points
    long NN = Xtest.GetN(); // number of training points
    long M = Xtrain.GetD(); // dimension of randome binning features
    DMatrix Ytest_predict(NN,m);
    DMatrix W(M,m);
    SPointArray EYE;
    EYE.Init(M,M,M);
    mystart = EYE.GetPointerStart();
    myidx = EYE.GetPointerIdx();
    myX = EYE.GetPointerX();
    for(i=0;i<M;i++){
      mystart[i] = i;
      myidx[i] = i;
      myX[i] = 1;
    }
    mystart[i] = M+1; // mystart has a length N+1
    for (i = 0; i < m; i++) {
      DVector w;
      w.Init(M);
      DVector ytrain, yy;
      Ytrain.GetColumn(i, ytrain);
      Xtrain.MatVec(ytrain, yy, TRANSPOSE);
      double NormRHS = yy.Norm2();
      PCG pcg_solver;
      pcg_solver.Solve<SPointArray, SPointArray>(Xtrain, yy, w, EYE, MAXIT, TOL, 1);
      if (verbose) {
        int Iter = 0;
        const double *ResHistory = pcg_solver.GetResHistory(Iter);
        printf("RLCM::Train, PCG. iteration = %d, Relative residual = %g\n",
          Iter, ResHistory[Iter-1]/NormRHS);fflush(stdout);
      }

      pcg_solver.GetSolution(w);
      W.SetColumn(i, w);
    }
    END_CLOCK;
    printf("Train. RandBin: Time (in seconds) for solving linear system solution: %g\n", ELAPSED_TIME);fflush(stdout);

    // y = Xtest*W = z(x)'*w
    START_CLOCK;
    Xtest.MatMat(W,Ytest_predict,NORMAL,NORMAL);
    double accuracy = Performance(ytest, Ytest_predict, NumClasses);
    END_CLOCK;
    ElapsedTime = ELAPSED_TIME;
    printf("Test. RandBin: param = %g %g, perf = %g, time = %g\n", sigma, lambda, accuracy, ElapsedTime); fflush(stdout);

  }// End loop over List_sigma
  }// End loop over List_lambda

  // Clean up
  free(List_sigma);
  free(List_lambda);

  return 0;
}

/*
//--------------------------------------------------------------------------
int ReadData(const char *filename, double **X, int **y, int d, int *n) {

  if (VerifyFileFormat(filename, d, n) == -1) {
    return -1;
  }

  FILE *fp = NULL;
  fp = fopen(filename, "r");
  
  // Allocate memory for X and y
  (*X) = (double *)malloc((*n)*d*sizeof(double));
  (*y) = (int *)malloc((*n)*sizeof(int));

  // Read the file again and populate X and y
  long i = 0;
  int numi = 0;
  double numf = 0.0;
  char *line = NULL, *str = NULL, *saveptr = NULL, *subtoken = NULL;
  ssize_t read = 0;
  size_t len = 0;
  while ((read = getline(&line, &len, fp)) != -1) { // Read in a line
    int cnt = 0;
    for (str = line; ; str = NULL) {
      subtoken = strtok_r(str, ": ", &saveptr); // Tokenize the line
      if (subtoken == NULL) {
        break;
      }
      else {
        cnt++;
      }
      if (cnt == 1) {
        (*y)[i] = atoi(subtoken);
      }
      else if (cnt%2 == 0) {
        numi = atoi(subtoken);
      }
      else {
        numf = atof(subtoken);
        (*X)[i+(*n)*(numi-1)] = numf; // LibSVM index starts from 1
      }
    }
    i++;
  }
  if (line) {
    free(line);
  }

  fclose(fp);
  return 0;

}


//--------------------------------------------------------------------------
int VerifyFileFormat(const char *filename, int d, int *n) {

  FILE *fp = NULL;
  fp = fopen(filename, "r");
  if (fp == NULL) {
    printf("VerifyFileFormat. Error: Cannot open file %s. Function call takes no effect.\n", filename);
    return -1;
  }

  (*n) = 0;
  char *line = NULL, *str = NULL, *saveptr = NULL, *subtoken = NULL;
  ssize_t read = 0;
  size_t len = 0;
  while ((read = getline(&line, &len, fp)) != -1) { // Read in a line
    (*n)++;
    int cnt = 0, maxdim = 0;
    for (str = line; ; str = NULL) {
      subtoken = strtok_r(str, ": ", &saveptr); // Tokenize the line
      if (subtoken == NULL) {
        break;
      }
      else {
        cnt++;
      }
      if (cnt%2 == 0) { // Verify format
        int num = atoi(subtoken);
        if (maxdim > num) {
          printf("VerifyFileFormat. Error: Indices in line %d are not in the ascending order. Stop reading data. Function call takes no effect.\n", (*n));
          fclose(fp);
          return -1;
        }
        else {
          maxdim = num;
        }
      }
    }
    if (cnt%2 != 1) { // Verify format
      printf("VerifyFileFormat. Error: Line %d does not conform with a LibSVM format. Stop reading data. Function call takes no effect.\n", (*n));
      fclose(fp);
      return -1;
    }
    else if (d < maxdim) {
      printf("VerifyFileFormat. Error: Line %d indicates a point of dimension larger than d. Stop reading data. Function call takes no effect.\n", (*n));
      fclose(fp);
      return -1;
    }
  }
  if (line) {
    free(line);
  }

  fclose(fp);
  if ((*n) == 0) {
    printf("VerifyFileFormat. Error: Empty file!\n");
    return -1;
  }
  return 0;

}
*/

//--------------------------------------------------------------------------
void UniformRandom01(double *a, int n) {
  int i;
  for (i = 0; i < n; i++) {
    a[i] = (double)random()/RAND_MAX;
  }
}


//--------------------------------------------------------------------------
// Box-Muller: U, V from uniform[0,1]
// X = sqrt(-2.0*log(U))*cos(2.0*M_PI*V)
// Y = sqrt(-2.0*log(U))*sin(2.0*M_PI*V)
void StandardNormal(double *a, int n) {
  int i;
  for (i = 0; i < n/2; i++) {
    double U = (double)random()/RAND_MAX;
    double V = (double)random()/RAND_MAX;
    double common1 = sqrt(-2.0*log(U));
    double common2 = TWO_PI*V;
    a[2*i] = common1 * cos(common2);
    a[2*i+1] = common1 * sin(common2);
  }
  if (n%2 == 1) {
    double U = (double)random()/RAND_MAX;
    double V = (double)random()/RAND_MAX;
    double common1 = sqrt(-2.0*log(U));
    double common2 = TWO_PI*V;
    a[n-1] = common1 * cos(common2);
  }
}


//--------------------------------------------------------------------------
// Let X and Y be independent standard normal. Then X/fabs(Y) follows
// student t with 1 degree of freedom.
void StudentT1(double *a, int n) {
  int i;
  for (i = 0; i < n; i++) {
    double V = (double)random()/RAND_MAX;
    a[i] = tan(2.0*M_PI*V); // Why is cot not in math.h??? :-(
    if (V > 0.5) {
      a[i] = -a[i];
    }
  }
}


//--------------------------------------------------------------------------
// Let X and Y be independent standard normal. Then X/fabs(Y) follows
// student t with 1 degree of freedom.
void MultivariateStudentT1(double *a, int n) {
  StandardNormal(a, n);
  double b = 0.0;
  StandardNormal(&b, 1);
  b = fabs(b);
  int i;
  for (i = 0; i < n; i++) {
    a[i] /= b;
  }
}


//--------------------------------------------------------------------------
void ConvertYtrain(const DVector &ytrain, DMatrix &Ytrain, int NumClasses) {
  Ytrain.Init(ytrain.GetN(), NumClasses);
  // Lingfei: the y label is supposed to start from 0.
  for (int i = 0; i < NumClasses; i++) {
    DVector y = ytrain;
    double *my = y.GetPointer();
    for (long j = 0; j < y.GetN(); j++) {
      my[j] = ((int)my[j])==i ? 1.0 : -1.0;
    }
    Ytrain.SetColumn(i, y);
  }
}


//--------------------------------------------------------------------------
double Performance(const DVector &ytest_truth, const DVector &ytest_predict,
                   int NumClasses) {

  long n = ytest_truth.GetN();
  if (n != ytest_predict.GetN()) {
    printf("Performance. Error: Vector lengths mismatch. Return NAN");
    return NAN;
  }
  if (NumClasses != 1 && NumClasses != 2) {
    printf("Performance. Error: Neither regression nor binary classification. Return NAN");
    return NAN;
  }
  double perf = 0.0;

  if (NumClasses == 1) { // Relative error
    DVector diff;
    ytest_truth.Subtract(ytest_predict, diff);
    perf = diff.Norm2()/ytest_truth.Norm2();
  }
  else if (NumClasses == 2) { // Accuracy
    double *y1 = ytest_truth.GetPointer();
    double *y2 = ytest_predict.GetPointer();
    for (long i = 0; i < n; i++) {
      perf += y1[i]*y2[i]>0 ? 1.0:0.0;
    }
    perf = perf/n * 100.0;
  }

  return perf;
}


//--------------------------------------------------------------------------
double Performance(const DVector &ytest_truth, const DMatrix &Ytest_predict,
                   int NumClasses) {

  long n = ytest_truth.GetN();
  if (n != Ytest_predict.GetM() || Ytest_predict.GetN() != NumClasses ) {
    printf("Performance. Error: Size mismatch. Return NAN");
    return NAN;
  }
  if (NumClasses <= 2) {
    printf("Performance. Error: Not multiclass classification. Return NAN");
    return NAN;
  }
  double perf = 0.0;

  // Compute ytest_predict
  DVector ytest_predict(n);
  double *y = ytest_predict.GetPointer();
  for (long i = 0; i < n; i++) {
    DVector row(NumClasses);
    Ytest_predict.GetRow(i, row);
    long idx = -1;
    row.Max(idx);
    y[i] = (double)idx;
  }

  // Accuracy
  double *y1 = ytest_truth.GetPointer();
  double *y2 = ytest_predict.GetPointer();
  for (long i = 0; i < n; i++) {
    perf += ((int)y1[i])==((int)y2[i]) ? 1.0 : 0.0;
  }
  perf = perf/n * 100.0;

  return perf;
}


