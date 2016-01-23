#ifndef _PCG_TPP_
#define _PCG_TPP_


//--------------------------------------------------------------------------
template<class MatrixA, class MatrixM>
void PCG::
Solve(MatrixA &A,  // Martix A
      DVector &b,  // Right-hand side b
      DVector &x0, // Initial guess x0
      MatrixM &M,  // Preconditioner M approx inv(A)
      int MaxIt,   // Maximum # of iterations
      double RTol,  // Relative residual tolerance
      bool ATA,     // Enable different matrix type such as A = C'C
      double lambda // Enable sparse matrix with regulaizer A = C'C + lambda*I
      ) {

  NormB = b.Norm2();
  double Tol = RTol * NormB;
  Delete_1D_Array<double>(&mRes);
  New_1D_Array<double, int>(&mRes, MaxIt);

  mIter = 0;

  // r = b - Ax
  x = x0;
  DVector Ax;
  if (ATA){
    DVector Ax_temp;
    A.MatVec(x,Ax_temp,NORMAL);
    A.MatVec(Ax_temp,Ax,TRANSPOSE);
    x.Multiply(lambda,Ax_temp);
    Ax.Add(Ax_temp);
  }
  else{
    A.MatVec(x, Ax, NORMAL);
  }
  DVector r;
  b.Subtract(Ax, r);
  mRes[mIter++] = r.Norm2();

  if (mRes[0] < Tol) {
    return;
  }

  // z = M(r)
  DVector z;
  if (ATA)
    z = r;
  else
    M.MatVec(r, z, NORMAL);

  // p = z
  DVector p = z;

  // rz = r'*z
  double rz = r.InProd(z);

  while (mIter < MaxIt) {

    // Ap = A(p)
    DVector Ap;
    if (ATA){
      DVector Ap_temp;
      A.MatVec(p,Ap_temp,NORMAL);
      A.MatVec(Ap_temp,Ap,TRANSPOSE);
      p.Multiply(lambda, Ap_temp);
      Ap.Add(Ap_temp);
    }
    else{
      A.MatVec(p, Ap, NORMAL);
    }

    // alpha = rz / (Ap'*p)
    double App = Ap.InProd(p);
    double alpha = rz / App;

    // x = x + alpha*p
    DVector v;
    p.Multiply(alpha, v);
    x.Add(v);

    // r = r - alpha*Ap
    Ap.Multiply(alpha, v);
    r.Subtract(v);

    mRes[mIter] = r.Norm2();
    if (mRes[mIter] < Tol) {
      mIter++;
      break;
    }

    // z = M(r)
    if (ATA)
      z = r;
    else
      M.MatVec(r, z, NORMAL);

    // rz_new = r'*z
    double rz_new = r.InProd(z);

    // beta = rz_new / rz
    double beta = rz_new / rz;

    // rz = rz_new
    rz = rz_new;

    // p = z + beta*p
    p.Multiply(beta, v);
    z.Add(v, p);

    mIter++;

  }

}


#endif
