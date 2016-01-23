// The SPoint class implements a sparse data point. The point is
// treated as a sparse vector, where d is the length, nnz is the
// number of nonzeros, x is an array storing the nonzero elements, and
// idx is an array storing the indices of these nonzeros. For more
// details, see the class definition below.

#ifndef _SPOINT_
#define _SPOINT_

#include "../Misc/Common.hpp"
#include "DPoint.hpp"

class SPoint {

public:

  SPoint();
  SPoint(int d_, int nnz_);
  SPoint(const SPoint& G);
  SPoint& operator= (const SPoint& G);
  ~SPoint();

  void Init(void);
  void Init(int d_, int nnz_); // DID NOT initialized with zero
  void ReleaseAllMemory(void);
  void DeepCopy(const SPoint &G);

  //-------------------- Utilities --------------------

  // Get dimension
  int GetD(void) const;

  // Get nnz
  int GetNNZ(void) const;

  // Get the pointer to idx
  int* GetPointerIdx(void) const;

  // Get the pointer to x
  double* GetPointerX(void) const;

  // Print the data point
  void PrintPoint(const char *name) const;

  //-------------------- Computations --------------------

  // x'*y
  double InProd(const DPoint &y) const;
  double InProd(const SPoint &y) const;

  // sum_i |x_i-y_i|
  double Dist1(const DPoint &y) const;
  double Dist1(const SPoint &y) const;

  // sum_i |x_i-y_i|
  double Dist2(const DPoint &y) const;
  double Dist2(const SPoint &y) const;

protected:

private:

  int d;     // Dimension
  int nnz;   // Number of nonzeros
  int *idx;  // Indices (Must be sorted increasingly)
  double *x; // Values

  // The points are organized in a sparse format. For example, if d =
  // 10, idx = {0, 3, 9}, and x = {1.0, 2.1, 6.3}; then the point has
  // a dimension 10, nnz = 3, x[0] = 1.0, x[3] = 2.1, x[9] = 6.3, and
  // the rest of x is 0.

};

#endif
