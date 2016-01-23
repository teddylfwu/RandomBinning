// The BMatrix class implements a block diagonal matrix in double
// precision.

#ifndef _BMATRIX_
#define _BMATRIX_

#include "Node.hpp"

// When multiplying a block diagonal matrix A with a vector, one needs
// to distinguish two cases: (ORIGINAL) the usual case; (FACTORED) the
// case when A is the inverse of another block diagonal matrix B. In
// the ORIGINAL case, we do the straightforward multiplication. In the
// FACTORED case, each diagonal block is represented by the LU factor
// of the corresponding block of B. Then, the matrix-vector
// multiplication is done through a back solve followed by a forward
// solve.
enum MultMode { ORIGINAL, FACTORED };

class BMatrix {

public:

  BMatrix();
  BMatrix(const BMatrix &G);
  BMatrix& operator= (const BMatrix &G);
  ~BMatrix();

  void Init(void);
  void ReleaseAllMemory(void);
  void DeepCopy(const BMatrix &G);

  //-------------------- Utilities --------------------

  // Get matrix size
  long GetN(void);

  // Get root
  Node* GetRoot(void) const;

  // Get number of nonzero elements in the matrix
  long GetMemEst(void) const;

  //-------------------- Build the matrix --------------------

  // Build a block diagonal matrix from point array X. The tree is a
  // binary tree.
  //
  // The template classes Kernel, Point and PointArray are subject to
  // the same requirements as those for
  // DMatrix::BuildKernelMatrix. Additionally, the template class
  // PointArray must have the following method:
  //
  //   long RandomBipartition(long start, long n, long N0, long *perm,
  //                          DPoint &normal, double &offset);
  //
  // The points X will be permuted. The arrays Perm and iPerm are used
  // to record the permutation. Perm can be either NULL or
  // preallocated with sufficient memory. If Perm is NULL, iPerm is
  // not referenced. If Perm is preallocated, iPerm can be either NULL
  // or preallocated with sufficient memory.
  //
  // The routine returns true if the matrix can be successfully built;
  // otherwise false.
  template<class Kernel, class Point, class PointArray>
  bool BuildKernelMatrix(PointArray &X, // Points
                         long *Perm,    // Xnew[i] = Xold[Perm[i]]
                         long *iPerm,   // Xnew[iPerm[i]] = Xold[i]
                         const Kernel &mKernel, // Kernel
                         double lambda, // Regularization
                         int r_,        // Rank
                         int N0         // Maximum # of points per leaf node
                         );

  // Given a new set of points Y, let B be the kernel matrix between X
  // and Y; that is B = phi(X,Y). This routine computes z = B'*w for
  // any input vector w, without explicitly forming the matrix B.
  template<class Kernel, class Point, class PointArray>
  void EvalKernelMatAndDoMatVec(const PointArray &X,   // Points (after permut.)
                                const PointArray &Y,   // New points
                                const Kernel &mKernel, // Kernel
                                double lambda,         // Regularization
                                const DVector &w,      // The vector to multiply
                                DVector &z             // z = B'*w
                                );

  // Augmented for MatMat Z = B'*W
  template<class Kernel, class Point, class PointArray>
  void EvalKernelMatAndDoMatMat(const PointArray &X,   // Points (after permut.)
                                const PointArray &Y,   // New points
                                const Kernel &mKernel, // Kernel
                                double lambda,         // Regularization
                                const DMatrix &W,      // The matrix to multiply
                                DMatrix &Z             // Z = B'*W
                                );

  //-------------------- Inspect tree structure --------------------

  void PrintTree(void) const;

  //-------------------- Convert to full matrix --------------------

  // This routine requires O(N^2) memory. Use it wisely.
  void ConvertToDMatrix(DMatrix &B);

  //-------------------- Matrix computations --------------------

  // y = A*b
  void MatVec(const DVector &b, DVector &y, MatrixMode ModeA, MultMode mtMode);

  // tA = inv(A)
  void Invert(BMatrix &tA);

protected:

private:

  Node *Root;      // Tree root
  int MaxNumChild; // Maximum number of children for each node

  long N; // Matrix dimension
  int r;  // Rank

  // Called by BuildKernelMatrix()
  template<class Kernel, class Point, class PointArray>
  void BuildKernelMatrixDownward1(Node *mNode, PointArray &X,
                                  long *Perm, int N0);
  template<class Kernel, class Point, class PointArray>
  void BuildKernelMatrixDownward2(Node *mNode, const PointArray &X,
                                  const Kernel &mKernel, double lambda);

  // Called by EvalKernelMatAndDoMatVec()
  template<class Kernel, class Point, class PointArray>
  void EvalKernelMatAndDoMatVecDownward(Node *mNode, const PointArray &X,
                                        const PointArray &y,
                                        const Kernel &mKernel,
                                        double lambda,
                                        const DVector &w, double &z0);

  // Called by EvalKernelMatAndDoMatMat()
  template<class Kernel, class Point, class PointArray>
  void EvalKernelMatAndDoMatMatDownward(Node *mNode, const PointArray &X,
                                        const PointArray &y,
                                        const Kernel &mKernel,
                                        double lambda,
                                        const DMatrix &W, DVector &z0);

  // Called by ConvertToDMatrix()
  void ConvertToDMatrixUpward(DMatrix &B, const Node *mNode);

  // Called by MatVec()
  void MatVecUpward(const DVector &b, DVector &y,
                    Node *mNode, MatrixMode ModeA, MultMode mtMode);

  // Called by Invert()
  void InvertUpward(Node *mNodeTA);

};

#include "BMatrix.tpp"

#endif
