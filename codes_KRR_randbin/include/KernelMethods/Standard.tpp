#ifndef _STANDARD_TPP_
#define _STANDARD_TPP_

#define INITVAL_ymean DBL_MAX


//--------------------------------------------------------------------------
template<class Kernel, class Point, class PointArray>
Standard<Kernel, Point, PointArray>::
Standard() {
  Init();
}


//--------------------------------------------------------------------------
template<class Kernel, class Point, class PointArray>
void Standard<Kernel, Point, PointArray>::
Init(void) {
  ReleaseAllMemory();
  ymean = INITVAL_ymean;
}


//--------------------------------------------------------------------------
template<class Kernel, class Point, class PointArray>
void Standard<Kernel, Point, PointArray>::
ReleaseAllMemory(void) {
  c.ReleaseAllMemory();
  LU.ReleaseAllMemory();
  ymean = INITVAL_ymean;
}


//--------------------------------------------------------------------------
template<class Kernel, class Point, class PointArray>
Standard<Kernel, Point, PointArray>::
Standard(const Standard &G) {
  Init();
  DeepCopy(G);
}


//--------------------------------------------------------------------------
template<class Kernel, class Point, class PointArray>
Standard<Kernel, Point, PointArray>& Standard<Kernel, Point, PointArray>::
operator= (const Standard &G) {
  if (this != &G) {
    DeepCopy(G);
  }
  return *this;
}


//--------------------------------------------------------------------------
template<class Kernel, class Point, class PointArray>
void Standard<Kernel, Point, PointArray>::
DeepCopy(const Standard &G) {
  ReleaseAllMemory();
  c = G.c;
  ymean = G.ymean;
  LU = G.LU;
}


//--------------------------------------------------------------------------
template<class Kernel, class Point, class PointArray>
Standard<Kernel, Point, PointArray>::
~Standard() {
  ReleaseAllMemory();
}


//--------------------------------------------------------------------------
template<class Kernel, class Point, class PointArray>
void Standard<Kernel, Point, PointArray>::
Train(const Kernel &mKernel, double lambda,
      const PointArray &Xtrain,
      const DVector &ytrain,
      double &mem_est) {

  // K = phi(X,X)
  DMatrix K;
  K.BuildKernelMatrix<Kernel, Point, PointArray>(mKernel, Xtrain);

  // ymean = mean(y)
  ymean = ytrain.Mean();

  // c = (K+lambda*I)\(y-ymean)
  K.AddDiagonal(lambda);
  DVector yy;
  ytrain.Subtract(ymean, yy); // yy = y-ymean
  K.Mldivide(yy, c, NORMAL); // c = K\yy

  // Memory estimation
  mem_est = (double)Xtrain.GetN();

}


//--------------------------------------------------------------------------
template<class Kernel, class Point, class PointArray>
void Standard<Kernel, Point, PointArray>::
Test(const Kernel &mKernel,
     const PointArray &Xtrain,
     const PointArray &Xtest,
     DVector &ytest_predict,
     long Budget) const {

  long batch_size = (long)ceil(1.0*Budget/Xtrain.GetN());
  long remain_size = Xtest.GetN();
  long this_batch_size;
  long this_batch_start = 0;
  ytest_predict.Init(Xtest.GetN());
  while (remain_size > 0) {
    this_batch_size = batch_size<remain_size ? batch_size : remain_size;
    PointArray Xtest_batch;
    Xtest.GetSubset(this_batch_start, this_batch_size, Xtest_batch);
    DVector ytest_predict_batch;

    // K0 = phi(Xtest,Xtrain)
    DMatrix K0;
    K0.BuildKernelMatrix<Kernel, Point, PointArray>
      (mKernel, Xtest_batch, Xtrain);

    // y = K0*c + ymean
    K0.MatVec(c, ytest_predict_batch, NORMAL);
    ytest_predict_batch.Add(ymean);
    ytest_predict.SetBlock(this_batch_start, this_batch_size,
                           ytest_predict_batch);

    remain_size -= this_batch_size;
    this_batch_start += this_batch_size;
  }

}


//--------------------------------------------------------------------------
template<class Kernel, class Point, class PointArray>
void Standard<Kernel, Point, PointArray>::
TrainAlt(const Kernel &mKernel, double lambda,
         const PointArray &Xtrain,
         double &mem_est) {

  // K = phi(X,X)
  LU.BuildKernelMatrix<Kernel, Point, PointArray>(mKernel, Xtrain);

  // Add lambda to diagonal
  LU.AddDiagonal(lambda);

  // LU factorization of K
  LU.DGETRF();

  // Memory estimation
  mem_est = (double)Xtrain.GetN();

}


//--------------------------------------------------------------------------
template<class Kernel, class Point, class PointArray>
void Standard<Kernel, Point, PointArray>::
TestAlt(const Kernel &mKernel,
        const PointArray &Xtrain,
        const PointArray &Xtest,
        const DMatrix &Ytrain,
        DMatrix &Ytest_predict,
        long Budget) {

  // yy = y-ymean
  int m = Ytrain.GetN();
  double *ymean2 = NULL;
  New_1D_Array<double, int>(&ymean2, m);
  DMatrix YY(Ytrain.GetM(), m);
  for (int i = 0; i < m; i++) {
    DVector ytrain, yy;
    Ytrain.GetColumn(i, ytrain);
    ymean2[i] = ytrain.Mean();
    ytrain.Subtract(ymean2[i], yy);
    YY.SetColumn(i, yy);
  }

  // c = (K+lambda*I)\(y-ymean)
  DMatrix C;
  LU.DGETRS(YY, C, NORMAL); // c = K\yy

  long batch_size = (long)ceil(1.0*Budget/Xtrain.GetN());
  long remain_size = Xtest.GetN();
  long this_batch_size;
  long this_batch_start = 0;
  Ytest_predict.Init(Xtest.GetN(), m);
  while (remain_size > 0) {
    this_batch_size = batch_size<remain_size ? batch_size : remain_size;
    PointArray Xtest_batch;
    Xtest.GetSubset(this_batch_start, this_batch_size, Xtest_batch);
    DMatrix Ytest_predict_batch;

    // K0 = phi(Xtest,Xtrain)
    DMatrix K0;
    K0.BuildKernelMatrix<Kernel, Point, PointArray>
      (mKernel, Xtest_batch, Xtrain);

    // y = K0*c + ymean
    K0.MatMat(C, Ytest_predict_batch, NORMAL, NORMAL);
    DVector ytest_predict_batch;
    for (int i = 0; i < m; i++) {
      Ytest_predict_batch.GetColumn(i, ytest_predict_batch);
      ytest_predict_batch.Add(ymean2[i]);
      Ytest_predict_batch.SetColumn(i, ytest_predict_batch);
    }
    Ytest_predict.SetBlock(this_batch_start, this_batch_size, 0, m,
                           Ytest_predict_batch);

    remain_size -= this_batch_size;
    this_batch_start += this_batch_size;
  }

  // Clean up
  Delete_1D_Array<double>(&ymean2);

}


#endif
