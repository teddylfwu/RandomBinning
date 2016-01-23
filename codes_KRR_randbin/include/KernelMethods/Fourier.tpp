#ifndef _FOURIER_TPP_
#define _FOURIER_TPP_

#define INITVAL_ymean DBL_MAX


//--------------------------------------------------------------------------
template<class Kernel, class Point, class PointArray>
Fourier<Kernel, Point, PointArray>::
Fourier() {
  Init();
}


//--------------------------------------------------------------------------
template<class Kernel, class Point, class PointArray>
void Fourier<Kernel, Point, PointArray>::
Init(void) {
  ReleaseAllMemory();
  ymean = INITVAL_ymean;
}


//--------------------------------------------------------------------------
template<class Kernel, class Point, class PointArray>
void Fourier<Kernel, Point, PointArray>::
ReleaseAllMemory(void) {
  w.ReleaseAllMemory();
  b.ReleaseAllMemory();
  c.ReleaseAllMemory();
  Z.ReleaseAllMemory();
  ZZ.ReleaseAllMemory();
  ymean = INITVAL_ymean;
}


//--------------------------------------------------------------------------
template<class Kernel, class Point, class PointArray>
Fourier<Kernel, Point, PointArray>::
Fourier(const Fourier &G) {
  Init();
  DeepCopy(G);
}


//--------------------------------------------------------------------------
template<class Kernel, class Point, class PointArray>
Fourier<Kernel, Point, PointArray>& Fourier<Kernel, Point, PointArray>::
operator= (const Fourier &G) {
  if (this != &G) {
    DeepCopy(G);
  }
  return *this;
}


//--------------------------------------------------------------------------
template<class Kernel, class Point, class PointArray>
void Fourier<Kernel, Point, PointArray>::
DeepCopy(const Fourier &G) {
  ReleaseAllMemory();
  w = G.w;
  b = G.b;
  c = G.c;
  ymean = G.ymean;
  Z = G.Z;
  ZZ = G.ZZ;
}


//--------------------------------------------------------------------------
template<class Kernel, class Point, class PointArray>
Fourier<Kernel, Point, PointArray>::
~Fourier() {
  ReleaseAllMemory();
}


//--------------------------------------------------------------------------
template<class Kernel, class Point, class PointArray>
void Fourier<Kernel, Point, PointArray>::
Train(const Kernel &mKernel, double lambda,
      const PointArray &Xtrain,
      const DVector &ytrain,
      long r, unsigned Seed, double &mem_est) {

  // Seed the RNG
  srandom(Seed);

  // Dimensions
  long N = Xtrain.GetN();
  int d = Xtrain.GetD();

  // Generate w. Currently support only the following kernels:
  //   IsotropicGaussian, IsotropicLaplace, ProductLaplace
  w.Init(d, r);
  if (mKernel.GetKernelName() == "IsotropicGaussian") {

    // Standard normal scaled by 1/sigma
    w.SetStandardNormal();
    w.Divide(mKernel.GetSigma());

  }
  else if (mKernel.GetKernelName() == "IsotropicLaplace") {

    // Multivariate student-t of degree 1, scaled by 1/sigma
    w.SetMultivariateStudentT1();
    w.Divide(mKernel.GetSigma());

  }
  else if (mKernel.GetKernelName() == "ProductLaplace") {

    // Student-t of degree 1, scaled by 1/sigma
    w.SetStudentT1();
    w.Divide(mKernel.GetSigma());

  }
  else {

    printf("Fourier::Train. Error: The supplied kernel is not supported by the Fourier method. Function call takes no effect.\n");
    return;

  }

  // b = rand(1,r)*2*pi
  b.Init(r);
  b.SetUniformRandom01();
  b.Multiply(2.0*M_PI);

  // Z = cos(X*w+repmat(b,n,1)) * sqrt(2/r) * sqrt(s).
  DMatrix Xw;
  Xtrain.MatMat(w, Xw, NORMAL, NORMAL);
  DVector ones(N);
  ones.SetConstVal(1.0);
  DMatrix oneb;
  oneb.OuterProduct(ones, b);
  Z = Xw;
  Z.Add(oneb);
  Z.Cos();
  Z.Multiply(sqrt(2.0/r*mKernel.GetS()));

  // ymean = mean(y)
  ymean = ytrain.Mean();

  // c = (Z'*Z+lambda*I)\(Z'*(y-ymean))
  Z.MatMat(Z, ZZ, TRANSPOSE, NORMAL); // ZZ = Z'*Z
  ZZ.AddDiagonal(lambda); // ZZ = Z'*Z + lambda*I
  DVector yy;
  ytrain.Subtract(ymean, yy); // yy = y-ymean
  DVector Z2;
  Z.MatVec(yy, Z2, TRANSPOSE); // Z2 = Z'*(y-ymean)
  ZZ.Mldivide(Z2, c, NORMAL); // c = ZZ\Z2

  // Z and ZZ are no longer needed
  Z.ReleaseAllMemory();
  ZZ.ReleaseAllMemory();

  // Memory estimation
  mem_est = (double)r;

}


//--------------------------------------------------------------------------
template<class Kernel, class Point, class PointArray>
void Fourier<Kernel, Point, PointArray>::
Test(const Kernel &mKernel,
     const PointArray &Xtest,
     DVector &ytest_predict,
     long r) const {

  // Number of points
  long n = Xtest.GetN();
  ytest_predict.Init(n);

  // Z0 = cos(X*w+repmat(b,n,1)) * sqrt(2/r) * sqrt(s)
  DMatrix Xw;
  Xtest.MatMat(w, Xw, NORMAL, NORMAL);
  DVector ones(n);
  ones.SetConstVal(1.0);
  DMatrix oneb;
  oneb.OuterProduct(ones, b);
  DMatrix Z0 = Xw;
  Z0.Add(oneb);
  Z0.Cos();
  Z0.Multiply(sqrt(2.0/r*mKernel.GetS()));

  // y = Z0*c + ymean
  Z0.MatVec(c, ytest_predict, NORMAL);
  ytest_predict.Add(ymean);

}


//--------------------------------------------------------------------------
template<class Kernel, class Point, class PointArray>
void Fourier<Kernel, Point, PointArray>::
TrainAlt(const Kernel &mKernel, double lambda,
         const PointArray &Xtrain,
         long r, unsigned Seed, double &mem_est) {

  // Seed the RNG
  srandom(Seed);

  // Dimensions
  long N = Xtrain.GetN();
  int d = Xtrain.GetD();

  // Generate w. Currently support only the following kernels:
  //   IsotropicGaussian, IsotropicLaplace, ProductLaplace
  w.Init(d, r);
  if (mKernel.GetKernelName() == "IsotropicGaussian") {

    // Standard normal scaled by 1/sigma
    w.SetStandardNormal();
    w.Divide(mKernel.GetSigma());

  }
  else if (mKernel.GetKernelName() == "IsotropicLaplace") {

    // Multivariate student-t of degree 1, scaled by 1/sigma
    w.SetMultivariateStudentT1();
    w.Divide(mKernel.GetSigma());

  }
  else if (mKernel.GetKernelName() == "ProductLaplace") {

    // Student-t of degree 1, scaled by 1/sigma
    w.SetStudentT1();
    w.Divide(mKernel.GetSigma());

  }
  else {

    printf("Fourier::TrainAlt. Error: The supplied kernel is not supported by the Fourier method. Function call takes no effect.\n");
    return;

  }

  // b = rand(1,r)*2*pi
  b.Init(r);
  b.SetUniformRandom01();
  b.Multiply(2.0*M_PI);

  // Z = cos(X*w+repmat(b,n,1)) * sqrt(2/r) * sqrt(s).
  DMatrix Xw;
  Xtrain.MatMat(w, Xw, NORMAL, NORMAL);
  DVector ones(N);
  ones.SetConstVal(1.0);
  DMatrix oneb;
  oneb.OuterProduct(ones, b);
  Z = Xw;
  Z.Add(oneb);
  Z.Cos();
  Z.Multiply(sqrt(2.0/r*mKernel.GetS()));

  // c = (Z'*Z+lambda*I)\(Z'*(y-ymean))
  Z.MatMat(Z, ZZ, TRANSPOSE, NORMAL); // ZZ = Z'*Z
  ZZ.AddDiagonal(lambda); // ZZ = Z'*Z + lambda*I
  ZZ.DGETRF(); // Factorize ZZ

  // Memory estimation
  mem_est = (double)r;

}


//--------------------------------------------------------------------------
template<class Kernel, class Point, class PointArray>
void Fourier<Kernel, Point, PointArray>::
TestAlt(const Kernel &mKernel,
        const PointArray &Xtest,
        const DMatrix &Ytrain,
        DMatrix &Ytest_predict,
        long r) {

  // Number of points
  long n = Xtest.GetN();

  // Z0 = cos(X*w+repmat(b,n,1)) * sqrt(2/r) * sqrt(s)
  DMatrix Xw;
  Xtest.MatMat(w, Xw, NORMAL, NORMAL);
  DVector ones(n);
  ones.SetConstVal(1.0);
  DMatrix oneb;
  oneb.OuterProduct(ones, b);
  DMatrix Z0 = Xw;
  Z0.Add(oneb);
  Z0.Cos();
  Z0.Multiply(sqrt(2.0/r*mKernel.GetS()));

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

  // c = (Z'*Z+lambda*I)\(Z'*(y-ymean))
  DMatrix Z2;
  Z.MatMat(YY, Z2, TRANSPOSE, NORMAL); // Z2 = Z'*(y-ymean)
  DMatrix C;
  ZZ.DGETRS(Z2, C, NORMAL); // c = ZZ\Z2

  // y = Z0*c + ymean
  Z0.MatMat(C, Ytest_predict, NORMAL, NORMAL);
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
