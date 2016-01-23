// The DMatrix class implements a dense matrix in double precision. A
// substantial portion of the methods in this class are wrappers of
// BLAS2, BLAS3, and LAPACK routines. The implementation tends to be
// coherent with Matlab constructs. The matrix data is stored by using
// column major order.
//
// In addition, this class provides methods for building the kernel
// matrix from point sets.

#ifndef _DMATRIX_
#define _DMATRIX_

#include "DVector.hpp"

class DMatrix {

public:


  DMatrix();
  DMatrix(long N_);
  DMatrix(long M_, long N_);
  DMatrix(const DMatrix &G);
  DMatrix& operator= (const DMatrix &G);
  ~DMatrix();

  void Init(void);
  void Init(long N_);          // Initialized as a zero matrix
  void Init(long M_, long N_); // Initialized as a zero matrix
  void ReleaseAllMemory(void);
  void DeepCopy(const DMatrix &G);

  //-------------------- Utilities --------------------

  // Get dimension
  long GetM(void) const;
  long GetN(void) const;

  // Get A(i,j)
  double GetEntry(long i, long j) const;
  // b = A(:,i)
  void GetColumn(long i, DVector &b) const;
  // B = A(:,idx)
  void GetColumns(long *idx, long n, DMatrix &B) const;
  // b = A(i,:)
  void GetRow(long i, DVector &b) const;
  // B = A(RowStart:RowStart+nRow-1, ColStart:ColStart+nCol-1)
  void GetBlock(long RowStart, long nRow, long ColStart, long nCol,
                DMatrix &B) const;
  // Get the double* pointer
  double* GetPointer(void) const;

  // Set A(i,j) = b
  void SetEntry(long i, long j, double b);
  // A(:,i) = b
  void SetColumn(long i, const DVector &b);
  // A(i,:) = b
  void SetRow(long i, const DVector &b);
  // A(RowStart:RowStart+nRow-1, ColStart:ColStart+nCol-1) = B
  void SetBlock(long RowStart, long nRow, long ColStart, long nCol,
                const DMatrix &B);
  void SetBlock(long RowStart, long nRow, long ColStart, long nCol,
                const double *B);

  // A = I
  void SetIdentity(void);
  // A = c
  void SetConstVal(double c);
  // A = rand()
  void SetUniformRandom01(void);
  // A = randn()
  void SetStandardNormal(void);
  // A = rtnd(1); each element is a student-t of degree 1
  void SetStudentT1(void);
  // Each column of A is a multivariate student-t of degree 1
  void SetMultivariateStudentT1(void);
  // A = diag(b)
  void MakeDiag(const DVector &b);

  // Build a compressed kernel matrix from point set X (and Y). The
  // class Kernel must have the following methods:
  //
  //   template<class Point>
  //   double Eval(const Point &x, const Point &y) const;
  //   bool IsSymmetric(void) const;
  //
  // The class PointArray must have the following methods:
  //
  //   long GetN(void) const;
  //   void GetPoint(long i, DPoint &x) const;
  template<class Kernel, class Point, class PointArray>
  void BuildKernelMatrix(const Kernel &mKernel, const PointArray &X,
                         double lambda = 0.0);
  template<class Kernel, class Point, class PointArray>
  void BuildKernelMatrix(const Kernel &mKernel, const PointArray &X,
                         const PointArray &Y, double lambda = 0.0);

  // Print the matrix in the Matlab form
  void PrintMatrixMatlabForm(const char *name) const;

  //-------------------- Matrix computations --------------------

  // A = A'  or  B = A'
  void Transpose(void);
  void Transpose(DMatrix &B) const;

  // A = (A+A')/2  or  B = (A+A')/2
  void Symmetrize(void);
  void Symmetrize(DMatrix &B) const;

  // A = -A  or  B = -A
  void Negate(void);
  void Negate(DMatrix &B) const;

  // A = A + B  or  C = A + B
  void Add(const DMatrix &B);
  void Add(const DMatrix &B, DMatrix &C) const;

  // A = A - B  or  C = A - B
  void Subtract(const DMatrix &B);
  void Subtract(const DMatrix &B, DMatrix &C) const;

  // A = A * b  or  C = A * b  (b is scalar)
  void Multiply(double b);
  void Multiply(double b, DMatrix &C) const;

  // A = A / b  or  C = A / b  (b is scalar)
  void Divide(double b);
  void Divide(double b, DMatrix &C) const;

  // A = A + sI  or  B = A + sI
  void AddDiagonal(double s);
  void AddDiagonal(double s, DMatrix &B) const;

  // A = A - sI  or  B = A - sI
  void SubtractDiagonal(double s);
  void SubtractDiagonal(double s, DMatrix &B) const;

  // A = cos(A)  or  B = cos(A)
  void Cos(void);
  void Cos(DMatrix &B) const;

  // A = b * c'
  void OuterProduct(const DVector &b, const DVector &c);

  // y = mode(A) * b
  void MatVec(const DVector &b, DVector &y, MatrixMode ModeA) const;

  // y = alpha * mode(A) * b + beta * y
  void DGEMV(const DVector &b, DVector &y,
             double alpha, double beta, MatrixMode ModeA) const;

  // C = mode(A) * mode(B)
  void MatMat(const DMatrix &B, DMatrix &C,
              MatrixMode ModeA, MatrixMode ModeB) const;

  // C = alpha * mode(A) * mode(B) + beta * C
  void DGEMM(const DMatrix &B, DMatrix &C, double alpha, double beta,
             MatrixMode ModeA, MatrixMode ModeB) const;

  // Linear system solve: X = mode(A)\B
  void Mldivide(const DVector &b, DVector &x, MatrixMode ModeA) const;
  void Mldivide(const DMatrix &B, DMatrix &X, MatrixMode ModeA) const;
  // Least squares solve: X = A\B
  // Res must be either NULL or an allocated array with length size(A,1).
  void Mldivide(const DVector &b, DVector &x, double *Res = NULL) const;
  void Mldivide(const DMatrix &B, DMatrix &X, double *Res = NULL) const;

  // The following three routines are related to linear system
  // solves. They can be used separately from Mldivide in places where
  // efficiency is fine grained.
  //
  // Factorize the matrix. The data array A is destroyed to hold the
  // LU factors.
  void DGETRF(void);
  //
  // Solve linear systems Ax = b by using the factorization.
  void DGETRS(const DVector &b, DVector &x, MatrixMode ModeA);
  void DGETRS(const DMatrix &B, DMatrix &X, MatrixMode ModeA);
  //
  // Estimate the reciprocal condition number of the original matrix A.
  double DGECON(void);

  // A = GG'
  void Chol(DMatrix &G) const;

  // lambda = eig(A), A symmetric
  void SymEig(DVector &lambda) const;

  // [V,lambda] = eig(A), A symmetric
  void SymEig(DVector &lambda, DMatrix &V) const;

  // lambda = eig(A,B), A,B symmetric, but B is not definite. In such
  // a case, only nonsymmetric algorithm can be used. Note that
  // eigenvalues must be real.
  void SymEigBIndef(const DMatrix &B, DVector &lambda) const;

  // [V,lambda] = eig(A,B), A,B symmetric, but B is not definite. In
  //such a case, only nonsymmetric algorithm can be used. Note that
  //eigenvalues must be real.
  void SymEigBIndef(const DMatrix &B, DVector &lambda, DMatrix &V) const;

  // [U,T] = schur(A, 'real')
  void RealSchur(DMatrix &T, DMatrix &U, SelectType type) const;

  // B = orth(A). A must be square
  void Orth(DMatrix &B) const;

  // Perform QR with column pivoting, return the first k pivots. That
  // is, pivots[0] is the index of the original matrix column that is
  // pivoted to the first column under the QR factorization
  void QRpivots(long k, long *pivots) const;

  // rank(A)
  long Rank(void) const;

  // cond(A,2)
  double Cond2(void) const;

  // norm(A,2)
  double Norm2(void) const;

  // norm(A,'fro')
  double NormF(void) const;

  // b = diag(A)
  void Diag(DVector &b) const;

  // trace(A) = sum(diag(A))
  double Tr(void) const;

protected:

private:

  long M, N; // Matrix dimension
  double *A; // Matrix data in column major order
  int *IPIV; // Needed by DGETRF, DGETRS, and DGECON. Length = min(M,N)

  // Called by Mldivide()
  void Mldivide_LinearSystemSolve(const DMatrix &B, DMatrix &X,
                                  MatrixMode ModeA) const;
  void Mldivide_LeastSquaresSolve(const DMatrix &B, DMatrix &X,
                                  double *Res = NULL) const;

  // Called by DGETRS()
  void DGETRS_sub(const DMatrix &B, DMatrix &X, MatrixMode ModeA);

  // Called by SymEig()
  void DSYEV(char JOBZ, DVector &lambda, DMatrix &V) const;

  // Called by SymEigBIndef()
  void DGGEV(char JOBVR, const DMatrix &B,
             DVector &lambda, DMatrix &V) const;

  // Called by RealSchur()
  void DGEES(DMatrix &T, DMatrix &U,
             LOGICAL (*Select)(double *WR, double *WI)) const;

  // Called by Rank(), Cond2() and Norm2()
  void DGESVD(double *S) const;

};

#include "DMatrix.tpp"

#endif
