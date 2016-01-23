// The PCG class implements the preconditioned conjugate gradient
// method for solving a linear system of equations
//
//   Ax = b
//
// by using an initial guess x0 and a preconditioner M (that
// approximates the inverse of A).

#ifndef _PCG_
#define _PCG_

#include "../Matrices/DVector.hpp"

class PCG {

public:

  PCG();
  PCG(const PCG &G);
  PCG& operator= (const PCG &G);
  ~PCG();

  void Init(void);
  void ReleaseAllMemory(void);
  void DeepCopy(const PCG &G);

  // Solve.
  // MatrixA and Matrix M must have the following method:
  //   void MatVec(const DVector &b, DVector &y, MatrixMode ModeA);
  // Only ModeA = NORMAL is required.
  template<class MatrixA, class MatrixM>
  void Solve(MatrixA &A,  // Martix A
             DVector &b,  // Right-hand side b
             DVector &x0, // Initial guess x0
             MatrixM &M,  // Preconditioner M approx inv(A)
             int MaxIt,   // Maximum # of iterations
             double RTol,  // Relative residual tolerance
             bool ATA, // Enable different matrix type such as A = C'C
             double lambda // Enable sparse matrix with regulaizer A = C'C + lambda*I
             );

  // Get the norm of the right hand side b.
  double GetNormRHS(void) const;

  // Get the solution vector x
  void GetSolution(DVector &Sol) const;

  // Get the pointer to the array of residual history (NOTE: not
  // relative residuals). Iter is the number of iterations. That is,
  // the residuals are stored in [0 .. Iter-1].
  const double* GetResHistory(int &Iter) const;

protected:

private:

  double NormB;
  DVector x;
  int mIter;
  double *mRes;

};

#include "PCG.tpp"

#endif
