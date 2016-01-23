#ifndef _RLCM_TPP_
#define _RLCM_TPP_

#define INITVAL_ymean DBL_MAX


//--------------------------------------------------------------------------
template<class Kernel, class Point, class PointArray>
RLCM<Kernel, Point, PointArray>::
RLCM() {
  Init();
}


//--------------------------------------------------------------------------
template<class Kernel, class Point, class PointArray>
void RLCM<Kernel, Point, PointArray>::
Init(void) {
  ReleaseAllMemory();
  ymean = INITVAL_ymean;
}


//--------------------------------------------------------------------------
template<class Kernel, class Point, class PointArray>
void RLCM<Kernel, Point, PointArray>::
ReleaseAllMemory(void) {
  K.ReleaseAllMemory();
  c.ReleaseAllMemory();
  invK.ReleaseAllMemory();
  ymean = INITVAL_ymean;
}


//--------------------------------------------------------------------------
template<class Kernel, class Point, class PointArray>
RLCM<Kernel, Point, PointArray>::
RLCM(const RLCM &G) {
  Init();
  DeepCopy(G);
}


//--------------------------------------------------------------------------
template<class Kernel, class Point, class PointArray>
RLCM<Kernel, Point, PointArray>& RLCM<Kernel, Point, PointArray>::
operator= (const RLCM &G) {
  if (this != &G) {
    DeepCopy(G);
  }
  return *this;
}


//--------------------------------------------------------------------------
template<class Kernel, class Point, class PointArray>
void RLCM<Kernel, Point, PointArray>::
DeepCopy(const RLCM &G) {
  ReleaseAllMemory();
  K = G.K;
  c = G.c;
  ymean = G.ymean;
  invK = G.invK;
}


//--------------------------------------------------------------------------
template<class Kernel, class Point, class PointArray>
RLCM<Kernel, Point, PointArray>::
~RLCM() {
  ReleaseAllMemory();
}


//--------------------------------------------------------------------------
template<class Kernel, class Point, class PointArray>
bool RLCM<Kernel, Point, PointArray>::
Train(const Kernel &mKernel, double lambda,
      PointArray &Xtrain,
      DVector &ytrain,
      long *Perm, long *iPerm,
      long r, unsigned Seed, bool verbose, double &mem_est) {

  // Seed the RNG
  srandom(Seed);

  // N
  long N = Xtrain.GetN();

  // N0
  long N0 = r;

  PREPARE_CLOCK(verbose);

  // K = phi(X,X) + lambda*I
  START_CLOCK;
  bool success = K.BuildKernelMatrix<Kernel, Point, PointArray>
    (Xtrain, Perm, iPerm, mKernel, lambda, r, N0);
  END_CLOCK;
  if (verbose) {
    printf("RLCM::Train. Time (in seconds) for construting K: %g\n", ELAPSED_TIME);fflush(stdout);
  }
  if (!success) {
    mem_est = NAN;
    return false;
  }

  // y
  ytrain.Permute(Perm, N);

  // ymean = mean(y)
  ymean = ytrain.Mean();

  // invK
  START_CLOCK;
  K.Invert(invK);
  END_CLOCK;
  if (verbose) {
    printf("RLCM::Train. Time (in seconds) for inverting K: %g\n", ELAPSED_TIME);fflush(stdout);
  }

  // c = K\(y-ymean)
  START_CLOCK;
  DVector yy;
  ytrain.Subtract(ymean, yy); // yy = y-ymean
  invK.MatVec(yy, c, NORMAL); // c = K\yy
  END_CLOCK;
  if (verbose) {
    printf("RLCM::Train. Time (in seconds) for multiplying invK: %g\n", ELAPSED_TIME);fflush(stdout);
  }

  // If accuracy is not good enough, do an iterative solve
  START_CLOCK;
  DVector d;
  K.MatVec(c, d, NORMAL);
  DVector rr;
  d.Subtract(yy, rr);
  double NormRHS = yy.Norm2();
  double Res = rr.Norm2()/NormRHS;
  if (verbose) {
    printf("RLCM::Train, Invert. Relative residual = %g\n", Res);fflush(stdout);
  }
  if (Res > RLCM_PCG_RTOL) {
    PCG pcg_solver;
    pcg_solver.Solve<CMatrix, CMatrix>(K, yy, c, invK, RLCM_PCG_MAXIT,
                                       RLCM_PCG_RTOL, 0, 0);
    if (verbose) {
      int Iter = 0;
      const double *ResHistory = pcg_solver.GetResHistory(Iter);
      printf("RLCM::Train, PCG. iteration = %d, Relative residual = %g\n",
             Iter, ResHistory[Iter-1]/NormRHS);fflush(stdout);
    }
    pcg_solver.GetSolution(c);
  }
  END_CLOCK;
  if (verbose) {
    printf("RLCM::Train. Time (in seconds) for refining linear system solution: %g\n", ELAPSED_TIME);fflush(stdout);
  }

  // lingfei: add performance measure for computing training error
  if (verbose){
    K.MatVec(c, d, NORMAL);
    d.Add(ymean);
    double Perf = 0.0L;
    Perf = TrainPerf(ytrain,d,2);
    printf("RLCM::Training error = %g\n", 100-Perf);fflush(stdout);
  }

  // Release memory
  invK.ReleaseAllMemory();

  // Memory estimation
  mem_est = (double)K.GetMemEst() / N;

  return true;

}


//--------------------------------------------------------------------------
template<class Kernel, class Point, class PointArray>
void RLCM<Kernel, Point, PointArray>::
Test(const Kernel &mKernel, double lambda,
     const PointArray &Xtrain,
     const PointArray &Xtest,
     DVector &ytest_predict) {

  // K0 = phi(Xtest,Xtrain)
  // y = K0*c
  K.EvalKernelMatAndDoMatVec<Kernel, Point, PointArray>
    (Xtrain, Xtest, mKernel, lambda, c, ytest_predict);

  // y = K0*c + ymean
  ytest_predict.Add(ymean);

}


//--------------------------------------------------------------------------
template<class Kernel, class Point, class PointArray>
bool RLCM<Kernel, Point, PointArray>::
TrainAlt(const Kernel &mKernel, double lambda,
         PointArray &Xtrain,
         long *Perm, long *iPerm,
         long r, unsigned Seed, bool verbose, double &mem_est) {

  // Seed the RNG
  srandom(Seed);

  // N
  long N = Xtrain.GetN();

  // N0
  long N0 = r;

  PREPARE_CLOCK(verbose);

  // K = phi(X,X) + lambda*I
  START_CLOCK;
  bool success = K.BuildKernelMatrix<Kernel, Point, PointArray>
    (Xtrain, Perm, iPerm, mKernel, lambda, r, N0);
  END_CLOCK;
  if (verbose) {
    printf("RLCM::TrainAlt. Time (in seconds) for construting K: %g\n", ELAPSED_TIME);fflush(stdout);
  }
  if (!success) {
    mem_est = NAN;
    return false;
  }

  // invK
  START_CLOCK;
  K.Invert(invK);
  END_CLOCK;
  if (verbose) {
    printf("RLCM::TrainAlt. Time (in seconds) for inverting K: %g\n", ELAPSED_TIME);fflush(stdout);
  }

  // Memory estimation
  mem_est = (double)K.GetMemEst() / N;

  return true;

}


//--------------------------------------------------------------------------
template<class Kernel, class Point, class PointArray>
void RLCM<Kernel, Point, PointArray>::
TestAlt(const Kernel &mKernel, double lambda,
        const PointArray &Xtrain,
        const PointArray &Xtest,
        DMatrix &Ytrain,
        DVector &ytrain_truth,
        const long *Perm, const long *iPerm,
        bool PermuteYtrain,
        bool verbose,
        DMatrix &Ytest_predict) {

  int m = Ytrain.GetN();
  double *ymean2 = NULL;
  New_1D_Array<double, int>(&ymean2, m);
  long N = Xtrain.GetN();
  DMatrix C(N, m);

  PREPARE_CLOCK(verbose);

  START_CLOCK;
  // c = K\(y-ymean)
  for (int i = 0; i < m; i++) {

    DVector ytrain, yy;
    Ytrain.GetColumn(i, ytrain);
    if (PermuteYtrain) {
      ytrain.Permute(Perm, N);
    }
    ymean2[i] = ytrain.Mean(); // ymean = mean(y)
    ytrain.Subtract(ymean2[i], yy); // yy = y-ymean
    invK.MatVec(yy, c, NORMAL); // c = K\yy
    C.SetColumn(i, c);
    if (PermuteYtrain) {
      Ytrain.SetColumn(i, ytrain);
    }

    // If accuracy is not good enough, do an iterative solve
    DVector d;
    K.MatVec(c, d, NORMAL);
    DVector rr;
    d.Subtract(yy, rr);
    double NormRHS = yy.Norm2();
    double Res = rr.Norm2()/NormRHS;
//    if (verbose) {
//      printf("RLCM::Test, Invert. Relative residual for class %d = %g\n", i,Res);fflush(stdout);
//    }
    if (Res > RLCM_PCG_RTOL) {
      PCG pcg_solver;
      pcg_solver.Solve<CMatrix, CMatrix>(K, yy, c, invK, RLCM_PCG_MAXIT,
                                       RLCM_PCG_RTOL, 0 , 0);
      pcg_solver.GetSolution(c);
      C.SetColumn(i, c);
    }
    END_CLOCK;
  }
  // lingfei: add performance measure for computing training error
  if (verbose){
    printf("RLCM::TestAlt. Time (in seconds) for multiplying invK: %g\n", ELAPSED_TIME);fflush(stdout);
    DMatrix D(N, m);
    for (int i=0;i<m;i++){
      C.GetColumn(i, c);
      DVector d;
      K.MatVec(c, d, NORMAL);
      d.Add(ymean2[i]);
      D.SetColumn(i, d);
    }
    double Perf = 0.0L;
    if (PermuteYtrain) {
      ytrain_truth.Permute(Perm, N);
    }
    Perf = TrainPerf(ytrain_truth,D, m);
    printf("RLCM::TestAlt. Training error for %d classes = %g\n", m, 100-Perf);fflush(stdout);
  }

  // K0 = phi(Xtest,Xtrain)
  // y = K0*c
  K.EvalKernelMatAndDoMatMat<Kernel, Point, PointArray>
    (Xtrain, Xtest, mKernel, lambda, C, Ytest_predict);

  // y = K0*c + ymean
  for (int i = 0; i < m; i++) {
    DVector ytest_predict;
    Ytest_predict.GetColumn(i, ytest_predict);
    ytest_predict.Add(ymean2[i]);
    Ytest_predict.SetColumn(i, ytest_predict);

  }

  // Clean up
  Delete_1D_Array<double>(&ymean2);

}

// lingfei: for computing training error
//--------------------------------------------------------------------------
template<class Kernel, class Point, class PointArray>
double RLCM<Kernel, Point, PointArray>::
TrainPerf(const DVector &ytrain_truth, const DVector &ytrain_predict,
      const int NumClasses) {

  long n = ytrain_truth.GetN();
  if (n != ytrain_predict.GetN()) {
    printf("Performance. Error: Vector lengths mismatch. Return NAN");
    return NAN;
  }
  double perf = 0.0;

  if (NumClasses == 1) { // Relative error
    DVector diff;
    ytrain_truth.Subtract(ytrain_predict, diff);
    perf = diff.Norm2()/ytrain_truth.Norm2();
  }
  else if (NumClasses == 2) { // Accuracy
    double *y1 = ytrain_truth.GetPointer();
    double *y2 = ytrain_predict.GetPointer();
    for (long i = 0; i < n; i++) {
      perf += y1[i]*y2[i]>0 ? 1.0:0.0;
    }
    perf = perf/n * 100.0;
  }

  return perf;
}

// lingfei: for computing training error
//--------------------------------------------------------------------------
template<class Kernel, class Point, class PointArray>
double RLCM<Kernel, Point, PointArray>::
TrainPerf(const DVector &ytrain_truth, const DMatrix &Ytrain_predict, 
      const int NumClasses) {

  long n = ytrain_truth.GetN();
  if (n != Ytrain_predict.GetM()) {
    printf("Performance. Error: Size mismatch. Return NAN");
    return NAN;
  }
  double perf = 0.0;

  // Compute ytrain_predict
  DVector ytrain_predict(n);
  double *y = ytrain_predict.GetPointer();
  for (long i = 0; i < n; i++) {
    DVector row(NumClasses);
    Ytrain_predict.GetRow(i, row);
    long idx = -1;
    row.Max(idx);
    y[i] = (double)idx;
  }

  // Accuracy
  double *y1 = ytrain_truth.GetPointer();
  double *y2 = ytrain_predict.GetPointer();
  for (long i = 0; i < n; i++) {
    perf += ((int)y1[i])==((int)y2[i]) ? 1.0 : 0.0;
  }
  perf = perf/n * 100.0;

  return perf;
}


#endif
