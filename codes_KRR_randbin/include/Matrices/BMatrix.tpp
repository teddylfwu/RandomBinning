#ifndef _BMATRIX_TPP_
#define _BMATRIX_TPP_


//--------------------------------------------------------------------------
template<class Kernel, class Point, class PointArray>
bool BMatrix::
BuildKernelMatrix(PointArray &X, // Points
                  long *Perm,    // Xnew[i] = Xold[Perm[i]]
                  long *iPerm,   // Xnew[iPerm[i]] = Xold[i]
                  const Kernel &mKernel, // Kernel
                  double lambda, // Regularization
                  int r_,        // Rank
                  int N0         // Maximum # of points per leaf node
                  ) {

  if (N0 < r_) {
    printf("BMatrix::BuildKernelMatrix. Error: N0 must >= r. Function call takes no effect.\n");
    return false;
  }

  // Destroy the old matrix
  Init();
  MaxNumChild = 2;
  N = X.GetN();
  r = r_;

  // Handle permutation
  if (Perm != NULL) {
    for (long i = 0; i < N; i++) {
      Perm[i] = i;
    }
  }

  // Create root
  Root = new Node;
  Root->r = r;
  Root->n = N;
  Root->start = 0;

  // Build a barebone tree
  BuildKernelMatrixDownward1<Kernel, Point, PointArray>
    (Root, X, Perm, N0);

  if (Root->NumChild == 0) {
    printf("BMatrix::BuildKernelMatrix. Error: Fail to partition the data set and thus cannot build the matrix.\n");
    Init();
    return false;
  }

  // Instantiate matrix components
  BuildKernelMatrixDownward2<Kernel, Point, PointArray>
    (Root, X, mKernel, lambda);

  // Compute the inverse permutation
  if (Perm != NULL && iPerm != NULL) {
    for (long i = 0; i < N; i++) {
      iPerm[Perm[i]] = i;
    }
  }

  return true;
}


//--------------------------------------------------------------------------
template<class Kernel, class Point, class PointArray>
void BMatrix::
BuildKernelMatrixDownward1(Node *mNode, PointArray &X,
                           long *Perm, int N0) {

  // Partition the current point set
  //long m1 = X.TwoMeansPartition(mNode->start, mNode->n, N0, Perm,
  //                              mNode->normal, mNode->offset);
  long m1 = X.RandomBipartition(mNode->start, mNode->n, N0, Perm,
                                mNode->normal, mNode->offset);

  // If partitioning yields clusters of size less than N0, quit.
  if (m1 == 0) {
    return;
  }

  // Otherwise, create children
  mNode->NumChild = 2;

  Node *NewNodeL = new Node;
  NewNodeL->Parent = mNode;
  mNode->LeftChild = NewNodeL;
  NewNodeL->r = r;
  NewNodeL->start = mNode->start;
  NewNodeL->n = m1;

  Node *NewNodeR = new Node;
  NewNodeR->Parent = mNode;
  NewNodeL->RightSibling = NewNodeR;
  NewNodeR->r = r;
  NewNodeR->start = mNode->start + m1;
  NewNodeR->n = mNode->n - m1;

  // Recurse on children
  BuildKernelMatrixDownward1<Kernel, Point, PointArray>
    (NewNodeL, X, Perm, N0);
  BuildKernelMatrixDownward1<Kernel, Point, PointArray>
    (NewNodeR, X, Perm, N0);

}


//--------------------------------------------------------------------------
template<class Kernel, class Point, class PointArray>
void BMatrix::
BuildKernelMatrixDownward2(Node *mNode, const PointArray &X,
                           const Kernel &mKernel, double lambda) {

  int i = 0;
  Node *childi = NULL;

  if (mNode->NumChild == 0) {

    // A
    PointArray Y;
    X.GetSubset(mNode->start, mNode->n, Y);
    mNode->A.BuildKernelMatrix<Kernel, Point, PointArray>(mKernel, Y);
    mNode->A.AddDiagonal(lambda);

    return;

  }
  else {

    // Recurse on children
    childi = mNode->LeftChild;
    for (i = 0; i < mNode->NumChild; i++, childi = childi->RightSibling) {
      BuildKernelMatrixDownward2<Kernel, Point, PointArray>
        (childi, X, mKernel, lambda);
    }

  }

}


//--------------------------------------------------------------------------
template<class Kernel, class Point, class PointArray>
void BMatrix::
EvalKernelMatAndDoMatVec(const PointArray &X,   // Points (after permut.)
                         const PointArray &Y,   // New points
                         const Kernel &mKernel, // Kernel
                         double lambda,         // Regularization
                         const DVector &w,      // The vector to multiply
                         DVector &z             // z = B'*w
                         ) {

  if (Root == NULL) {
    printf("BMatrix::EvalKernelMatAndDoMatVec. Error: Matrix is empty. Function call takes no effect\n");
    return;
  }
  if (w.GetN() != X.GetN()) {
    printf("BMatrix::EvalKernelMatAndDoMatVec. Error: Length of w does not match number of points in X. Function call takes no effect\n");
  }

  long n = Y.GetN();
  z.Init(n);

  // Do mat-vec for one point at a time
  for (long i = 0; i < n; i++) {
    PointArray y;
    Y.GetSubset(i, 1, y);
    double z0 = 0.0;
    EvalKernelMatAndDoMatVecDownward<Kernel, Point, PointArray>
      (Root, X, y, mKernel, lambda, w, z0);
    z.SetEntry(i, z0);
  }

}


//--------------------------------------------------------------------------
template<class Kernel, class Point, class PointArray>
void BMatrix::
EvalKernelMatAndDoMatVecDownward(Node *mNode, const PointArray &X,
                                 const PointArray &y,
                                 const Kernel &mKernel, double lambda,
                                 const DVector &w, double &z0) {

  // y must be a singleton set

  Node *child = NULL;

  if (mNode->NumChild == 0) {

    // ph is the kernel matrix between the old points (corresponding
    // subset of X) and the new point (a singleton set y)
    PointArray P;
    DMatrix PH;
    DVector ph;
    X.GetSubset(mNode->start, mNode->n, P);
    PH.BuildKernelMatrix<Kernel, Point, PointArray>(mKernel, P, y, lambda);
    PH.GetColumn(0, ph);

    // Multiply ph and the corresponding block of w. The result is z0
    DVector mw;
    w.GetBlock(mNode->start, mNode->n, mw);
    z0 = ph.InProd(mw);

  }
  else {

    // Which child does y land on? The computation here is applicable
    // to only a binary tree constructed by using hyperplane
    // partitioning.
    Point y0;
    y.GetPoint(0, y0);
    double iprod = y0.InProd(mNode->normal);
    if (iprod < mNode->offset) {
      child = mNode->LeftChild;
    }
    else {
      child = mNode->LeftChild->RightSibling;
    }

    // Recurse on that child only
    EvalKernelMatAndDoMatVecDownward<Kernel, Point, PointArray>
      (child, X, y, mKernel, lambda, w, z0);

  }

}


//--------------------------------------------------------------------------
template<class Kernel, class Point, class PointArray>
void BMatrix::
EvalKernelMatAndDoMatMat(const PointArray &X,   // Points (after permut.)
                         const PointArray &Y,   // New points
                         const Kernel &mKernel, // Kernel
                         double lambda,         // Regularization
                         const DMatrix &W,      // The matrix to multiply
                         DMatrix &Z             // Z = B'*W
                         ) {

  if (Root == NULL) {
    printf("BMatrix::EvalKernelMatAndDoMatMat. Error: Matrix is empty. Function call takes no effect\n");
    return;
  }
  if (W.GetM() != X.GetN()) {
    printf("BMatrix::EvalKernelMatAndDoMatMat. Error: Size of W does not match number of points in X. Function call takes no effect\n");
  }

  long n = Y.GetN();
  long m = W.GetN();
  Z.Init(n, m);

  // Do mat-vec for one point at a time
  for (long i = 0; i < n; i++) {
    PointArray y;
    Y.GetSubset(i, 1, y);
    DVector z0(m);
    EvalKernelMatAndDoMatMatDownward<Kernel, Point, PointArray>
      (Root, X, y, mKernel, lambda, W, z0);
    Z.SetRow(i, z0);
  }

}


//--------------------------------------------------------------------------
template<class Kernel, class Point, class PointArray>
void BMatrix::
EvalKernelMatAndDoMatMatDownward(Node *mNode, const PointArray &X,
                                 const PointArray &y,
                                 const Kernel &mKernel, double lambda,
                                 const DMatrix &W, DVector &z0) {

  // y must be a singleton set

  Node *child = NULL;
  int m = W.GetN();

  if (mNode->NumChild == 0) {

    // ph is the kernel matrix between the old points (corresponding
    // subset of X) and the new point (a singleton set y)
    PointArray P;
    DMatrix PH;
    DVector ph;
    X.GetSubset(mNode->start, mNode->n, P);
    PH.BuildKernelMatrix<Kernel, Point, PointArray>(mKernel, P, y, lambda);
    PH.GetColumn(0, ph);

    // Multiply ph and the corresponding block of W. The result is z0
    DMatrix mW;
    W.GetBlock(mNode->start, mNode->n, 0, m, mW);
    mW.MatVec(ph, z0, TRANSPOSE);

  }
  else {

    // Which child does y land on? The computation here is applicable
    // to only a binary tree constructed by using hyperplane
    // partitioning.
    Point y0;
    y.GetPoint(0, y0);
    double iprod = y0.InProd(mNode->normal);
    if (iprod < mNode->offset) {
      child = mNode->LeftChild;
    }
    else {
      child = mNode->LeftChild->RightSibling;
    }

    // Recurse on that child only
    EvalKernelMatAndDoMatMatDownward<Kernel, Point, PointArray>
      (child, X, y, mKernel, lambda, W, z0);

  }

}


#endif

