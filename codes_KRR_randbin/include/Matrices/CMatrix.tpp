#ifndef _CMATRIX_TPP_
#define _CMATRIX_TPP_

//--------------------------------------------------------------------------
template<class Kernel, class Point, class PointArray>
bool CMatrix::
BuildKernelMatrix(PointArray &X, // Points
                  long *Perm,    // Xnew[i] = Xold[Perm[i]]
                  long *iPerm,   // Xnew[iPerm[i]] = Xold[i]
                  const Kernel &mKernel, // Kernel
                  double lambda, // Regularization
                  int r_,        // Rank
                  int N0         // Maximum # of points per leaf node
                  ) {

  if (N0 < r_) {
    printf("CMatrix::BuildKernelMatrix. Error: N0 must >= r. Function call takes no effect.\n");
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
    printf("CMatrix::BuildKernelMatrix. Error: Fail to partition the data set and thus cannot build the matrix.\n");
    Init();
    return false;
  }

  // Find pivots for each nonleaf node and instantiate matrix components
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
void CMatrix::
BuildKernelMatrixDownward1(Node *mNode, PointArray &X,
                           long *Perm, int N0) {

  // Partition the current point set
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
void CMatrix::
BuildKernelMatrixDownward2(Node *mNode, const PointArray &X,
                           const Kernel &mKernel, double lambda) {

  int i = 0;
  Node *childi = NULL;
  Node *parent = mNode->Parent;

  // Leaf
  if (mNode->NumChild == 0) {

    // A
    PointArray Y;
    X.GetSubset(mNode->start, mNode->n, Y);
    mNode->A.BuildKernelMatrix<Kernel, Point, PointArray>(mKernel, Y, lambda);

    if (mNode == Root) {
      return;
    }

    // U
    PointArray P;
    X.GetSubset(parent->pivots, r, P);
    DMatrix UU;
    UU.BuildKernelMatrix<Kernel, Point, PointArray>
      (mKernel, P, Y, BUILD_CMATRIX_REG*mKernel.GetS());
    parent->LU.DGETRS(UU, mNode->U, NORMAL);
    mNode->U.Transpose();

    return;

  }

  // From here on, mNode is not a leaf and thus has children

  // Sample pivot points P
  New_1D_Array<long, int>(&mNode->pivots, r);
  RandPerm(mNode->n, r, mNode->pivots);
  for (i = 0; i < r; i++) {
    mNode->pivots[i] += mNode->start;
  }
  PointArray P;
  X.GetSubset(mNode->pivots, r, P);
  // Compute the kernel matrix PH of pivots
  DMatrix PH;
  PH.BuildKernelMatrix<Kernel, Point, PointArray>
    (mKernel, P, BUILD_CMATRIX_REG*mKernel.GetS());
  // Compute LU factorization of PH
  mNode->LU = PH;
  mNode->LU.DGETRF();

  // Sigma
  mNode->Sigma = PH;

  // W
  if (mNode != Root) {
    PointArray PP;
    X.GetSubset(parent->pivots, r, PP);
    DMatrix HH;
    HH.BuildKernelMatrix<Kernel, Point, PointArray>
      (mKernel, PP, P, BUILD_CMATRIX_REG*mKernel.GetS());
    parent->LU.DGETRS(HH, mNode->W, NORMAL);
    mNode->W.Transpose();
  }

  // Recurse on children
  childi = mNode->LeftChild;
  for (i = 0; i < mNode->NumChild; i++, childi = childi->RightSibling) {
    BuildKernelMatrixDownward2<Kernel, Point, PointArray>
      (childi, X, mKernel, lambda);
  }

}


//--------------------------------------------------------------------------
template<class Kernel, class Point, class PointArray>
void CMatrix::
EvalKernelMatAndDoMatVec(const PointArray &X,   // Points (after permut.)
                         const PointArray &Y,   // New points
                         const Kernel &mKernel, // Kernel
                         double lambda,         // Regularization
                         const DVector &w,      // The vector to multiply
                         DVector &z             // z = K'*w
                         ) {

  if (Root == NULL) {
    printf("CMatrix::EvalKernelMatAndDoMatVec. Error: Matrix is empty. Function call takes no effect\n");
    return;
  }
  if (w.GetN() != X.GetN()) {
    printf("CMatrix::EvalKernelMatAndDoMatVec. Error: Length of w does not match number of points in X. Function call takes no effect\n");
  }

  long n = Y.GetN();
  z.Init(n);

  EvalKernelMatAndDoMatVecInitAugmentedData<Kernel, Point, PointArray>(Root);

  EvalKernelMatAndDoMatVecUpward1<Kernel, Point, PointArray>(Root, w);

  // Do mat-vec for one point at a time
  for (long i = 0; i < n; i++) {
    PointArray y;
    Y.GetSubset(i, 1, y);
    double z0 = 0.0;
    EvalKernelMatAndDoMatVecUpward2<Kernel, Point, PointArray>
      (Root, X, y, mKernel, lambda, w, z0);
    z.SetEntry(i, z0);
  }

  EvalKernelMatAndDoMatVecReleaseAugmentedData<Kernel, Point, PointArray>(Root);

}


//--------------------------------------------------------------------------
template<class Kernel, class Point, class PointArray>
void CMatrix::
EvalKernelMatAndDoMatVecInitAugmentedData(Node *mNode) {

  int i = 0;
  Node *child = NULL;

  if (mNode != Root) {
    mNode->c.Init(r);
    mNode->d.Init(r);
  }

  child = mNode->LeftChild;
  for (i = 0; i < mNode->NumChild; i++, child = child->RightSibling) {
    EvalKernelMatAndDoMatVecInitAugmentedData<Kernel, Point, PointArray>(child);
  }

}


//--------------------------------------------------------------------------
template<class Kernel, class Point, class PointArray>
void CMatrix::
EvalKernelMatAndDoMatVecReleaseAugmentedData(Node *mNode) {

  int i = 0;
  Node *child = NULL;

  if (mNode != Root) {
    mNode->c.ReleaseAllMemory();
    mNode->d.ReleaseAllMemory();
  }

  child = mNode->LeftChild;
  for (i = 0; i < mNode->NumChild; i++, child = child->RightSibling) {
    EvalKernelMatAndDoMatVecReleaseAugmentedData<Kernel, Point, PointArray>(child);
  }

}


//--------------------------------------------------------------------------
template<class Kernel, class Point, class PointArray>
void CMatrix::
EvalKernelMatAndDoMatVecUpward1(Node *mNode, const DVector &w) {

  // In the first upward phase, d is used to store the intermediate
  // mat-vec results computed bottom up, and c is used to store the
  // result from cross-multiply.

  int j = 0, k = 0;
  Node *child = NULL, *sibling = NULL;
  Node *parent = mNode->Parent;

  if (mNode->NumChild == 0) {

    // d_i = U_i' * w_i
    DVector mw;
    w.GetBlock(mNode->start, mNode->n, mw);
    mNode->U.MatVec(mw, mNode->d, TRANSPOSE);

  }
  else {

    child = mNode->LeftChild;
    for (j = 0; j < mNode->NumChild; j++, child = child->RightSibling) {

      // Recurse on children
      EvalKernelMatAndDoMatVecUpward1<Kernel, Point, PointArray>(child, w);
      if (mNode == Root) { continue; }

      // d_i = d_i + W_i' * d_j
      mNode->W.DGEMV(child->d, mNode->d, 1.0, 1.0, TRANSPOSE);

    }

  }

  if (mNode == Root) { return; }

  // c_k = Sigma_p' * d_i
  DVector ck;
  parent->Sigma.MatVec(mNode->d, ck, TRANSPOSE);

  sibling = parent->LeftChild;
  for (k = 0; k < parent->NumChild; k++, sibling = sibling->RightSibling) {
    // get k (k must be unique in a binary tree)
    if (sibling == mNode) { continue; }
    sibling->c = ck;
  }

}


//--------------------------------------------------------------------------
template<class Kernel, class Point, class PointArray>
void CMatrix::
EvalKernelMatAndDoMatVecUpward2(Node *mNode, const PointArray &X,
                                const PointArray &y,
                                const Kernel &mKernel, double lambda,
                                const DVector &w, double &z0) {

  // In the second upward phase, the storage of d is recycled and is
  // used to store the new intermediate mat-vec results computed
  // bottom up.

  // y must be a singleton set

  Node *parent = mNode->Parent;
  Node *child = NULL;

  if (mNode->NumChild == 0) {

    // Initial d_i (at leaf)
    PointArray P;
    DMatrix PH;
    DVector ph;
    if (mNode != Root) {
      X.GetSubset(parent->pivots, r, P);
      PH.BuildKernelMatrix<Kernel, Point, PointArray>
        (mKernel, P, y, BUILD_CMATRIX_REG*mKernel.GetS());
      PH.GetColumn(0, ph);
      parent->LU.DGETRS(ph, mNode->d, NORMAL);
    }

    // ph is the kernel matrix between the old points (corresponding
    // subset of X) and the new point (a singleton set y)
    X.GetSubset(mNode->start, mNode->n, P);
    PH.BuildKernelMatrix<Kernel, Point, PointArray>(mKernel, P, y, lambda);
    PH.GetColumn(0, ph);

    // Multiply ph and the corresponding block of w. The result is
    // the starting z0. It will accumulate later.
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
    EvalKernelMatAndDoMatVecUpward2<Kernel, Point, PointArray>
      (child, X, y, mKernel, lambda, w, z0);

    // d_i = W_i' * d_j
    if (mNode != Root) {
      mNode->W.MatVec(child->d, mNode->d, TRANSPOSE);
    }

  }

  // multiply c_i with d_i and accumulate to z0
  if (mNode != Root) {
    z0 += mNode->c.InProd(mNode->d);
  }

}


//--------------------------------------------------------------------------
template<class Kernel, class Point, class PointArray>
void CMatrix::
EvalKernelMatAndDoMatMat(const PointArray &X,   // Points (after permut.)
                         const PointArray &Y,   // New points
                         const Kernel &mKernel, // Kernel
                         double lambda,         // Regularization
                         const DMatrix &W,      // The matrix to multiply
                         DMatrix &Z             // Z = K'*W
                         ) {

  if (Root == NULL) {
    printf("CMatrix::EvalKernelMatAndDoMatMat. Error: Matrix is empty. Function call takes no effect\n");
    return;
  }
  if (W.GetM() != X.GetN()) {
    printf("CMatrix::EvalKernelMatAndDoMatMat. Error: Size of W does not match number of points in X. Function call takes no effect\n");
    return;
  }

  long n = Y.GetN();
  long m = W.GetN();
  Z.Init(n,m);

  EvalKernelMatAndDoMatMatInitAugmentedData<Kernel, Point, PointArray>(Root, W);

  EvalKernelMatAndDoMatMatUpward1<Kernel, Point, PointArray>(Root, W);

  // Do mat-mat for one point at a time
  for (long i = 0; i < n; i++) {
    PointArray y;
    Y.GetSubset(i, 1, y);
    DVector z0(m);
    EvalKernelMatAndDoMatMatUpward2<Kernel, Point, PointArray>
      (Root, X, y, mKernel, lambda, W, z0);
    Z.SetRow(i, z0);
  }

  EvalKernelMatAndDoMatMatReleaseAugmentedData<Kernel, Point, PointArray>(Root);

}


//--------------------------------------------------------------------------
template<class Kernel, class Point, class PointArray>
void CMatrix::
EvalKernelMatAndDoMatMatInitAugmentedData(Node *mNode, const DMatrix &W) {

  int i = 0;
  Node *child = NULL;
  int m = W.GetN();

  if (mNode != Root) {
    mNode->C.Init(r, m);
    mNode->D.Init(r, m);
    mNode->d.Init(r);
  }

  child = mNode->LeftChild;
  for (i = 0; i < mNode->NumChild; i++, child = child->RightSibling) {
    EvalKernelMatAndDoMatMatInitAugmentedData<Kernel, Point, PointArray>(child, W);
  }

}


//--------------------------------------------------------------------------
template<class Kernel, class Point, class PointArray>
void CMatrix::
EvalKernelMatAndDoMatMatReleaseAugmentedData(Node *mNode) {

  int i = 0;
  Node *child = NULL;

  if (mNode != Root) {
    mNode->C.ReleaseAllMemory();
    mNode->D.ReleaseAllMemory();
    mNode->d.ReleaseAllMemory();
  }

  child = mNode->LeftChild;
  for (i = 0; i < mNode->NumChild; i++, child = child->RightSibling) {
    EvalKernelMatAndDoMatMatReleaseAugmentedData<Kernel, Point, PointArray>(child);
  }

}


//--------------------------------------------------------------------------
template<class Kernel, class Point, class PointArray>
void CMatrix::
EvalKernelMatAndDoMatMatUpward1(Node *mNode, const DMatrix &W) {

  // In the first upward phase, D is used to store the intermediate
  // mat-mat results computed bottom up, and C is used to store the
  // result from cross-multiply.

  int j = 0, k = 0;
  Node *child = NULL, *sibling = NULL;
  Node *parent = mNode->Parent;
  int m = W.GetN();

  if (mNode->NumChild == 0) {

    // D_i = U_i' * W_i
    DMatrix mW;
    W.GetBlock(mNode->start, mNode->n, 0, m, mW);
    mNode->U.MatMat(mW, mNode->D, TRANSPOSE, NORMAL);

  }
  else {

    child = mNode->LeftChild;
    for (j = 0; j < mNode->NumChild; j++, child = child->RightSibling) {

      // Recurse on children
      EvalKernelMatAndDoMatMatUpward1<Kernel, Point, PointArray>(child, W);
      if (mNode == Root) { continue; }

      // D_i = D_i + W_i' * D_j
      mNode->W.DGEMM(child->D, mNode->D, 1.0, 1.0, TRANSPOSE, NORMAL);

    }

  }

  if (mNode == Root) { return; }

  // C_k = Sigma_p' * D_i
  DMatrix Ck;
  parent->Sigma.MatMat(mNode->D, Ck, TRANSPOSE, NORMAL);

  sibling = parent->LeftChild;
  for (k = 0; k < parent->NumChild; k++, sibling = sibling->RightSibling) {
    // get k (k must be unique in a binary tree)
    if (sibling == mNode) { continue; }
    sibling->C = Ck;
  }

}


//--------------------------------------------------------------------------
template<class Kernel, class Point, class PointArray>
void CMatrix::
EvalKernelMatAndDoMatMatUpward2(Node *mNode, const PointArray &X,
                                const PointArray &y,
                                const Kernel &mKernel, double lambda,
                                const DMatrix &W, DVector &z0) {

  // In the second upward phase, the storage of d is used to store the
  // new intermediate mat-mat results computed bottom up.

  // y must be a singleton set

  Node *parent = mNode->Parent;
  Node *child = NULL;
  int m = W.GetN();

  if (mNode->NumChild == 0) {

    // Initial d_i (at leaf)
    PointArray P;
    DMatrix PH;
    DVector ph;
    if (mNode != Root) {
      X.GetSubset(parent->pivots, r, P);
      PH.BuildKernelMatrix<Kernel, Point, PointArray>
        (mKernel, P, y, BUILD_CMATRIX_REG*mKernel.GetS());
      PH.GetColumn(0, ph);
      parent->LU.DGETRS(ph, mNode->d, NORMAL);
    }

    // ph is the kernel matrix between the old points (corresponding
    // subset of X) and the new point (a singleton set y)
    X.GetSubset(mNode->start, mNode->n, P);
    PH.BuildKernelMatrix<Kernel, Point, PointArray>(mKernel, P, y, lambda);
    PH.GetColumn(0, ph);

    // Multiply ph and the corresponding block of W. The result is
    // the starting z0. It will accumulate later.
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
    EvalKernelMatAndDoMatMatUpward2<Kernel, Point, PointArray>
      (child, X, y, mKernel, lambda, W, z0);

    // d_i = W_i' * d_j
    if (mNode != Root) {
      mNode->W.MatVec(child->d, mNode->d, TRANSPOSE);
    }

  }

  // multiply C_i with d_i and accumulate to z0
  if (mNode != Root) {
    mNode->C.DGEMV(mNode->d, z0, 1.0, 1.0, TRANSPOSE);
  }

}


#endif

