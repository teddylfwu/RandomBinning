#ifndef _BLOCKDIAG_TPP_
#define _BLOCKDIAG_TPP_

#define INITVAL_ymean DBL_MAX


//--------------------------------------------------------------------------
template<class Kernel, class Point, class PointArray>
BlockDiag<Kernel, Point, PointArray>::
BlockDiag() {
  Init();
}


//--------------------------------------------------------------------------
template<class Kernel, class Point, class PointArray>
void BlockDiag<Kernel, Point, PointArray>::
Init(void) {
  ReleaseAllMemory();
  ymean = INITVAL_ymean;
}


//--------------------------------------------------------------------------
template<class Kernel, class Point, class PointArray>
void BlockDiag<Kernel, Point, PointArray>::
ReleaseAllMemory(void) {
  K.ReleaseAllMemory();
  c.ReleaseAllMemory();
  invK.ReleaseAllMemory();
  ymean = INITVAL_ymean;
}


//--------------------------------------------------------------------------
template<class Kernel, class Point, class PointArray>
BlockDiag<Kernel, Point, PointArray>::
BlockDiag(const BlockDiag &G) {
  Init();
  DeepCopy(G);
}


//--------------------------------------------------------------------------
template<class Kernel, class Point, class PointArray>
BlockDiag<Kernel, Point, PointArray>& BlockDiag<Kernel, Point, PointArray>::
operator= (const BlockDiag &G) {
  if (this != &G) {
    DeepCopy(G);
  }
  return *this;
}


//--------------------------------------------------------------------------
template<class Kernel, class Point, class PointArray>
void BlockDiag<Kernel, Point, PointArray>::
DeepCopy(const BlockDiag &G) {
  ReleaseAllMemory();
  K = G.K;
  c = G.c;
  ymean = G.ymean;
  invK = G.invK;
}


//--------------------------------------------------------------------------
template<class Kernel, class Point, class PointArray>
BlockDiag<Kernel, Point, PointArray>::
~BlockDiag() {
  ReleaseAllMemory();
}


//--------------------------------------------------------------------------
template<class Kernel, class Point, class PointArray>
bool BlockDiag<Kernel, Point, PointArray>::
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
    printf("BlockDiag::Train. Time (in seconds) for construting K: %g\n", ELAPSED_TIME);fflush(stdout);
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
    printf("BlockDiag::Train. Time (in seconds) for inverting K: %g\n", ELAPSED_TIME);fflush(stdout);
  }

  // c = K\(y-ymean)
  START_CLOCK;
  DVector yy;
  ytrain.Subtract(ymean, yy); // yy = y-ymean
  invK.MatVec(yy, c, NORMAL, FACTORED); // c = K\yy
  END_CLOCK;
  if (verbose) {
    printf("BlockDiag::Train. Time (in seconds) for multiplying invK: %g\n", ELAPSED_TIME);fflush(stdout);
  }

  // Memory estimation
  mem_est = (double)K.GetMemEst() / N;

  // Release memory
  invK.ReleaseAllMemory();

  return true;

}


//--------------------------------------------------------------------------
template<class Kernel, class Point, class PointArray>
void BlockDiag<Kernel, Point, PointArray>::
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
bool BlockDiag<Kernel, Point, PointArray>::
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
    printf("BlockDiag::TrainAlt. Time (in seconds) for construting K: %g\n", ELAPSED_TIME);fflush(stdout);
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
    printf("BlockDiag::TrainAlt. Time (in seconds) for inverting K: %g\n", ELAPSED_TIME);fflush(stdout);
  }

  // Memory estimation
  mem_est = (double)K.GetMemEst() / N;

  return true;

}


//--------------------------------------------------------------------------
template<class Kernel, class Point, class PointArray>
void BlockDiag<Kernel, Point, PointArray>::
TestAlt(const Kernel &mKernel, double lambda,
        const PointArray &Xtrain,
        const PointArray &Xtest,
        DMatrix &Ytrain,
        const long *Perm, const long *iPerm,
        bool PermuteYtrain,
        DMatrix &Ytest_predict) {

  // c = K\(y-ymean)
  int m = Ytrain.GetN();
  double *ymean2 = NULL;
  New_1D_Array<double, int>(&ymean2, m);
  long N = Xtrain.GetN();
  DMatrix C(N, m);
  for (int i = 0; i < m; i++) {
    DVector ytrain, yy;
    Ytrain.GetColumn(i, ytrain);
    if (PermuteYtrain) {
      ytrain.Permute(Perm, N);
    }
    ymean2[i] = ytrain.Mean(); // ymean = mean(y)
    ytrain.Subtract(ymean2[i], yy); // yy = y-ymean
    invK.MatVec(yy, c, NORMAL, FACTORED); // c = K\yy
    C.SetColumn(i, c);
    if (PermuteYtrain) {
      Ytrain.SetColumn(i, ytrain);
    }
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


#endif
