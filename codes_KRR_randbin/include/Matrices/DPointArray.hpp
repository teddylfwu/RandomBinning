// The DPointArray class implements a set of dense data points; see
// the DPoint class for an individual dense data point. The point set
// is treated as a dense matrix of size N*d, where N is the number of
// points and d is the dimension of the points.

#ifndef _DPOINT_ARRAY_
#define _DPOINT_ARRAY_

#include "DPoint.hpp"
#include "DMatrix.hpp"

class DPointArray {

public:

  DPointArray();
  DPointArray(long N_, int d_);
  DPointArray(const DPointArray& G);
  DPointArray& operator= (const DPointArray& G);
  ~DPointArray();

  void Init(void);
  void Init(long N_, int d_); // Initialized as a zero matrix
  void ReleaseAllMemory(void);
  void DeepCopy(const DPointArray &G);

  //-------------------- Utilities --------------------

  // Get dimension
  int GetD(void) const;

  // Get number of points
  long GetN(void) const;

  // Get the i-th point
  void GetPoint(long i, DPoint &x) const;

  // Get a consecutive chunk
  void GetSubset(long start, long n, DPointArray &Y) const;

  // Get a subset
  void GetSubset(long *idx, long n, DPointArray &Y) const;

  // Get the pointer
  double* GetPointer(void) const;

  // Generate points with uniformly random coordinates inside the unit box
  void SetUniformRandom01(void);

  // Generate points uniformly random on the unit sphere
  void SetUniformSphere(void);

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
  long RandomBipartition(long start, long n, long N0, long *perm,
                         DPoint &normal, double &offset);

protected:

private:

  long N;    // Number of points
  int d;     // Dimension
  double *X; // Points

  // X is an array of length N*d, where X[0] to X[N-1] give the first
  // coordinate value of the points, X[N] to X[2N-1] give the second
  // coordinate values, etc.

};

#endif
