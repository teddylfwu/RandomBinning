#ifndef _NYSTROM_TPP_
#define _NYSTROM_TPP_

#define INITVAL_ymean DBL_MAX


//--------------------------------------------------------------------------
template<class Kernel, class Point, class PointArray>
Nystrom<Kernel, Point, PointArray>::
Nystrom() {
  Init();
}


//--------------------------------------------------------------------------
template<class Kernel, class Point, class PointArray>
void Nystrom<Kernel, Point, PointArray>::
Init(void) {
  ReleaseAllMemory();
  ymean = INITVAL_ymean;
}


//--------------------------------------------------------------------------
template<class Kernel, class Point, class PointArray>
void Nystrom<Kernel, Point, PointArray>::
ReleaseAllMemory(void) {
  Y.ReleaseAllMemory();
  K3.ReleaseAllMemory();
  c.ReleaseAllMemory();
  Z.ReleaseAllMemory();
  ZZ.ReleaseAllMemory();
  ymean = INITVAL_ymean;
}


//--------------------------------------------------------------------------
template<class Kernel, class Point, class PointArray>
Nystrom<Kernel, Point, PointArray>::
Nystrom(const Nystrom &G) {
  Init();
  DeepCopy(G);
}


//--------------------------------------------------------------------------
template<class Kernel, class Point, class PointArray>
Nystrom<Kernel, Point, PointArray>& Nystrom<Kernel, Point, PointArray>::
operator= (const Nystrom &G) {
  if (this != &G) {
    DeepCopy(G);
  }
  return *this;
}


//--------------------------------------------------------------------------
template<class Kernel, class Point, class PointArray>
void Nystrom<Kernel, Point, PointArray>::
DeepCopy(const Nystrom &G) {
  ReleaseAllMemory();
  Y = G.Y;
  K3 = G.K3;
  c = G.c;
  ymean = G.ymean;
  Z = G.Z;
  ZZ = G.ZZ;
}


//--------------------------------------------------------------------------
template<class Kernel, class Point, class PointArray>
Nystrom<Kernel, Point, PointArray>::
~Nystrom() {
  ReleaseAllMemory();
}


//--------------------------------------------------------------------------
template<class Kernel, class Point, class PointArray>
void Nystrom<Kernel, Point, PointArray>::
Train(const Kernel &mKernel, double lambda,
      const PointArray &Xtrain,
      const DVector &ytrain,
      long r, unsigned Seed, double &mem_est) {

  // Seed the RNG
  srandom(Seed);

  // Centers Y: random subset of X
  long *idx = NULL;
  New_1D_Array<long, long>(&idx, r);
  RandPerm(Xtrain.GetN(), r, idx);
  Xtrain.GetSubset(idx, r, Y);

  // K1 = phi(X,Y)
  DMatrix K1;
  K1.BuildKernelMatrix<Kernel, Point, PointArray>(mKernel, Xtrain, Y);

  // K2 = phi(Y,Y)
  DMatrix K2;
  K2.BuildKernelMatrix<Kernel, Point, PointArray>(mKernel, Y);

  // K3 = some form of sqrt of pinv(K2)
  DVector D;
  DMatrix V;
  K2.SymEig(D, V); // K2 = V*diag(D)*V'
  long *idx2 = NULL;
  New_1D_Array<long, long>(&idx2, r);
  long num = D.FindLargerThan(D.Max()*EPS, idx2);
  DMatrix V2;
  V.GetColumns(idx2, num, V2); // V2 = V(:,idx2)
  DVector D2;
  D.GetBlock(idx2, num, D2);
  D2.Sqrt();
  D2.Inv(); // D2 = 1/sqrt(D(idx2))
  DMatrix D3;
  D3.MakeDiag(D2); // D3 = diag(D2)
  V2.MatMat(D3, K3, NORMAL, NORMAL); // K3 = V2*D3

  // Z = K1*K3
  K1.MatMat(K3, Z, NORMAL, NORMAL);

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

  // Cleanup
  Delete_1D_Array<long>(&idx);
  Delete_1D_Array<long>(&idx2);
  Z.ReleaseAllMemory();
  ZZ.ReleaseAllMemory();

  // Memory estimation
  mem_est = (double)r;

}


//--------------------------------------------------------------------------
template<class Kernel, class Point, class PointArray>
void Nystrom<Kernel, Point, PointArray>::
Test(const Kernel &mKernel,
     const PointArray &Xtest,
     DVector &ytest_predict) const {

  // K1 = phi(X,Y)
  DMatrix K1;
  K1.BuildKernelMatrix<Kernel, Point, PointArray>(mKernel, Xtest, Y);

  // Z0 = K1*K3
  DMatrix Z0;
  K1.MatMat(K3, Z0, NORMAL, NORMAL);

  // y = Z0*c + ymean
  Z0.MatVec(c, ytest_predict, NORMAL);
  ytest_predict.Add(ymean);

}


//--------------------------------------------------------------------------
template<class Kernel, class Point, class PointArray>
void Nystrom<Kernel, Point, PointArray>::
TrainAlt(const Kernel &mKernel, double lambda,
         const PointArray &Xtrain,
         long r, unsigned Seed, double &mem_est) {

  // Seed the RNG
  srandom(Seed);

  // Centers Y: random subset of X
  long *idx = NULL;
  New_1D_Array<long, long>(&idx, r);
  RandPerm(Xtrain.GetN(), r, idx);
  Xtrain.GetSubset(idx, r, Y);

  // K1 = phi(X,Y)
  DMatrix K1;
  K1.BuildKernelMatrix<Kernel, Point, PointArray>(mKernel, Xtrain, Y);

  // K2 = phi(Y,Y)
  DMatrix K2;
  K2.BuildKernelMatrix<Kernel, Point, PointArray>(mKernel, Y);

  // K3 = some form of sqrt of pinv(K2)
  DVector D;
  DMatrix V;
  K2.SymEig(D, V); // K2 = V*diag(D)*V'
  long *idx2 = NULL;
  New_1D_Array<long, long>(&idx2, r);
  long num = D.FindLargerThan(D.Max()*EPS, idx2);
  DMatrix V2;
  V.GetColumns(idx2, num, V2); // V2 = V(:,idx2)
  DVector D2;
  D.GetBlock(idx2, num, D2);
  D2.Sqrt();
  D2.Inv(); // D2 = 1/sqrt(D(idx2))
  DMatrix D3;
  D3.MakeDiag(D2); // D3 = diag(D2)
  V2.MatMat(D3, K3, NORMAL, NORMAL); // K3 = V2*D3

  // Z = K1*K3
  K1.MatMat(K3, Z, NORMAL, NORMAL);

  // c = (Z'*Z+lambda*I)\(Z'*(y-ymean))
  Z.MatMat(Z, ZZ, TRANSPOSE, NORMAL); // ZZ = Z'*Z
  ZZ.AddDiagonal(lambda); // ZZ = Z'*Z + lambda*I
  ZZ.DGETRF(); // Factorize ZZ

  // Cleanup
  Delete_1D_Array<long>(&idx);
  Delete_1D_Array<long>(&idx2);

  // Memory estimation
  mem_est = (double)r;

}


//--------------------------------------------------------------------------
template<class Kernel, class Point, class PointArray>
void Nystrom<Kernel, Point, PointArray>::
TestAlt(const Kernel &mKernel,
        const PointArray &Xtest,
        const DMatrix &Ytrain,
        DMatrix &Ytest_predict) {

  // K1 = phi(X,Y)
  DMatrix K1;
  K1.BuildKernelMatrix<Kernel, Point, PointArray>(mKernel, Xtest, Y);

  // Z0 = K1*K3
  DMatrix Z0;
  K1.MatMat(K3, Z0, NORMAL, NORMAL);

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
