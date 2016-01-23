// The DPoint class implements a dense data point. The point is
// treated as a dense vector.

#ifndef _DPOINT_
#define _DPOINT_

#include "../Misc/Common.hpp"

class DPoint {

public:

  DPoint();
  DPoint(int d_);
  DPoint(const DPoint& G);
  DPoint& operator= (const DPoint& G);
  ~DPoint();

  void Init(void);
  void Init(int d_); // Initialized as a zero vector
  void ReleaseAllMemory(void);
  void DeepCopy(const DPoint &G);

  //-------------------- Utilities --------------------

  // Get dimension
  int GetD(void) const;

  // Get the pointer
  double* GetPointer(void) const;

  // Set data
  void SetPoint(const double *x_, int d_);

  // x = randn()
  void SetStandardNormal(void);

  // Print the data point
  void PrintPoint(const char *name) const;

  //-------------------- Computations --------------------

  // x = x / norm(x,2)
  void Normalize(void);

  // x'*y
  double InProd(const DPoint &y) const;

  // sum_i |x_i-y_i|
  double Dist1(const DPoint &y) const;

  // sum_i |x_i-y_i|^2
  double Dist2(const DPoint &y) const;
  
  // x = x-y  or  z = x-y
  void Subtract(const DPoint &y);
  void Subtract(const DPoint &y, DPoint &z) const;

  // x = (x+y)/2  or  z = (x+y)/2
  void AverageWith(const DPoint &y);
  void AverageWith(const DPoint &y, DPoint &z) const;

protected:

private:

  int d;     // Dimension
  double *x; // Point data

};

#endif
