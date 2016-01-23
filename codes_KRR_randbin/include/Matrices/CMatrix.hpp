// The CMatrix class implements a recursively low-rank compressed
// matrix in double precision.

#ifndef _CMATRIX_
#define _CMATRIX_

#include "Node.hpp"

class CMatrix {

public:

  CMatrix();
  CMatrix(const CMatrix &G);
  CMatrix& operator= (const CMatrix &G);
  ~CMatrix();

  void Init(void);
  void ReleaseAllMemory(void);
  void DeepCopy(const CMatrix &G);

  //-------------------- Utilities --------------------

  // Get matrix size
  long GetN(void);

  // Get root
  Node* GetRoot(void) const;

  // Get an estimation of the memory consumption of the matrix. The
  // estimation is based on a highly simplified model, where the sizes
  // of A, Sigma, U, V, W, and Z in all nodes are summed.
  long GetMemEst(void) const;

  //-------------------- Build the matrix --------------------

  // Build a compressed kernel matrix from point array X. The tree is
  // a binary tree.
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

  // Given a new set of points Y, let K be the kernel matrix between X
  // and Y; that is K = phi(X,Y). This routine computes z = K'*w for
  // any input vector w, without explicitly forming the matrix K.
  template<class Kernel, class Point, class PointArray>
  void EvalKernelMatAndDoMatVec(const PointArray &X,   // Points (after permut.)
                                const PointArray &Y,   // New points
                                const Kernel &mKernel, // Kernel
                                double lambda,         // Regularization
                                const DVector &w,      // The vector to multiply
                                DVector &z             // z = K'*w
                                );

  // Augmented for MatMat Z = K'*W
  template<class Kernel, class Point, class PointArray>
  void EvalKernelMatAndDoMatMat(const PointArray &X,   // Points (after permut.)
                                const PointArray &Y,   // New points
                                const Kernel &mKernel, // Kernel
                                double lambda,         // Regularization
                                const DMatrix &W,      // The matrix to multiply
                                DMatrix &Z             // Z = K'*W
                                );

  //-------------------- Inspect tree structure --------------------

  void PrintTree(void) const;

  //-------------------- Convert to full matrix --------------------

  // This routine requires O(N^2) memory. Use it wisely.
  void ConvertToDMatrix(DMatrix &B);

  //-------------------- Matrix computations --------------------

  // y = A*b
  void MatVec(const DVector &b, DVector &y, MatrixMode ModeA);

  // tA = inv(A)
  void Invert(CMatrix &tA);

protected:

private:

  Node *Root;      // Tree root
  int MaxNumChild; // Maximum number of children for each node

  long N; // Matrix dimension
  int r;  // Rank

  // Augmented for ConvertToDMatrixUpward()
  DMatrix Ut;

  // Called by BuildKernelMatrix()
  template<class Kernel, class Point, class PointArray>
  void BuildKernelMatrixDownward1(Node *mNode, PointArray &X,
                                  long *Perm, int N0);
  template<class Kernel, class Point, class PointArray>
  void BuildKernelMatrixDownward2(Node *mNode, const PointArray &X,
                                  const Kernel &mKernel, double lambda);

  // Called by EvalKernelMatAndDoMatVec()
  template<class Kernel, class Point, class PointArray>
  void EvalKernelMatAndDoMatVecInitAugmentedData(Node *mNode);
  template<class Kernel, class Point, class PointArray>
  void EvalKernelMatAndDoMatVecReleaseAugmentedData(Node *mNode);
  template<class Kernel, class Point, class PointArray>
  void EvalKernelMatAndDoMatVecUpward1(Node *mNode, const DVector &w);
  template<class Kernel, class Point, class PointArray>
  void EvalKernelMatAndDoMatVecUpward2(Node *mNode, const PointArray &X,
                                       const PointArray &y,
                                       const Kernel &mKernel,
                                       double lambda,
                                       const DVector &W, double &z0);

  // Called by EvalKernelMatAndDoMatMat()
  template<class Kernel, class Point, class PointArray>
  void EvalKernelMatAndDoMatMatInitAugmentedData(Node *mNode, const DMatrix &W);
  template<class Kernel, class Point, class PointArray>
  void EvalKernelMatAndDoMatMatReleaseAugmentedData(Node *mNode);
  template<class Kernel, class Point, class PointArray>
  void EvalKernelMatAndDoMatMatUpward1(Node *mNode, const DMatrix &W);
  template<class Kernel, class Point, class PointArray>
  void EvalKernelMatAndDoMatMatUpward2(Node *mNode, const PointArray &X,
                                       const PointArray &y,
                                       const Kernel &mKernel,
                                       double lambda,
                                       const DMatrix &W, DVector &z0);

  // Called by ConvertToDMatrix()
  void ConvertToDMatrixUpward(DMatrix &B, const Node *mNode);

  // Called by MatVec()
  void MatVecInitAugmentedData(Node *mNode);
  void MatVecReleaseAugmentedData(Node *mNode);
  void MatVecUpward(const DVector &b, DVector &y,
                    Node *mNode, MatrixMode ModeA);
  void MatVecDownward(const DVector &b, DVector &y,
                      Node *mNode, MatrixMode ModeA);

  // Called by Invert()
  void InvertInitAugmentedData(Node *mNodeTA);
  void InvertReleaseAugmentedData(Node *mNodeTA);
  void InvertUpward(Node *mNodeA, Node *mNodeTA);
  void InvertDownward(Node *mNodeTA);

};

#include "CMatrix.tpp"

#endif
