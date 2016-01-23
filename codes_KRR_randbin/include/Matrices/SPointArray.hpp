// The SPointArray class implements a set of sparse data points; see
// the SPoint class for an individual sparse data point. The point set
// is treated as a sparse matrix, which is represented by three
// numbers and three arrays. Let the matrix have a size N*d with nnz
// nonzero elements, where N is the number of points and d is the
// dimension of the points. The three arrays are start, idx, and X,
// which means respectively, the starting location of a new point, the
// index of the nonzero elements, and the values of the nonzero
// elements. Such a storage format is consistent with the compressed
// sparse row (CSR) matrix format. For more details, see the class
// definition below.

#ifndef _SPOINT_ARRAY_
#define _SPOINT_ARRAY_

#include "SPoint.hpp"
#include "DPoint.hpp"
#include "DMatrix.hpp"

class SPointArray {

public:

  SPointArray();
  SPointArray(long N_, int d_, long nnz_);
  SPointArray(const SPointArray& G);
  SPointArray& operator= (const SPointArray& G);
  ~SPointArray();

  void Init(void);
  void Init(long N_, int d_, long nnz_); // DID NOT initialized with zero
  void ReleaseAllMemory(void);
  void DeepCopy(const SPointArray &G);

  //-------------------- Utilities --------------------

  // Get dimension
  int GetD(void) const;

  // Get number of points
  long GetN(void) const;

  // Get nnz
  long GetNNZ(void) const;

  // Get the i-th point
  void GetPoint(long i, SPoint &x) const;
  void GetPoint(long i, DPoint &x) const;

  // Get a consecutive chunk
  void GetSubset(long istart, long n, SPointArray &Y) const;

  // Get a subset
  void GetSubset(long *iidx, long n, SPointArray &Y) const;

  // Get the pointer to start
  long* GetPointerStart(void) const;

  // Get the pointer to idx
  int* GetPointerIdx(void) const;

  // Get the pointer to X
  double* GetPointerX(void) const;

  // Print the point set
  void PrintPointArray(const char *name) const;

  //-------------------- Computations --------------------

  // Center of the point set
  void Center(DPoint &c) const;

  // Root mean squared distances between the center and the points
  double StdvDist(void) const;

  // y = X*b (size of X is N*d)
  void MatVec(const DVector &b, DVector &y, MatrixMode ModeA) const;
  void MatMat(const DMatrix &B, DMatrix &Y,
              MatrixMode ModeA, MatrixMode ModeB) const;

  // Bipartition and permute the points by using a hyperplane with
  // random orientation.
  //
  // This routine partitions the data points (indexed from start to
  // start+n-1) into two equal halves randomly. Let the dividing
  // hyperplane have a normal direction 'normal'. All the points are
  // projected along this direction and a median (named 'offset') of
  // the projected values is computed. The hyperplane equation is thus
  // h(x) = normal'*x - offset = 0. In the nondegenerate case, there
  // are m1 = floor(n/2) points x satisfying h(x) < 0 and the rest
  // satisfying h(x) > 0. The routine permutes the points so that
  // those indexed from start to start+m1-1 correspond to h(x) < 0 and
  // the rest corresponds to h(x) > 0. The return value of the routine
  // is m1. Additionally, if the permutation array perm is not NULL,
  // then it records the reordering of the points.
  long RandomBipartition(long istart, long n, long N0, long *perm,
                         DPoint &normal, double &offset);

protected:

private:

  long N;      // Number of points
  int d;       // Dimension
  long nnz;    // Total number of nonzeros
  long *start; //
  int *idx;    //
  double *X;   //

  // The three arrays 'start', 'idx', and 'X', altogether, are used to
  // represent points in a manner consistent with the CSR format of a
  // sparse matrix.
  //
  // 'start' has a length N+1, where start[i] means the location of
  // idx (and of X) where the i-th point begins, for i = 0:N-1. For
  // convenience, start[N] = nnz.
  //
  // 'idx' has a length nnz, where the segment idx[start[i]] to
  // idx[start[i+1]-1] stores the coordinates of the nonzero elements
  // of the i-th point. All 'idx' entries must be < d.
  //
  // 'X' has a length nnz, where the segment X[start[i]] to
  // X[start[i+1]-1] stores the values of the nonzero elements of the
  // i-th point.

};

#endif
