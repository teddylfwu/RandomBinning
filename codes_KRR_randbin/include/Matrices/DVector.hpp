// The DVector class implements a dense vector in double precision. A
// substantial portion of the methods in this class are wrappers of
// BLAS1 routines. The implementation tends to be coherent with Matlab
// constructs.

#ifndef _DVECTOR_
#define _DVECTOR_

#include "../Misc/Common.hpp"

class DVector {

public:

  DVector();
  DVector(long N_);
  DVector(const DVector &G);
  DVector& operator= (const DVector &G);
  ~DVector();

  void Init(void);
  void Init(long N_); // Initialized as a zero vector
  void ReleaseAllMemory(void);
  void DeepCopy(const DVector &G);

  //-------------------- Utilities --------------------

  // Get length
  long GetN(void) const;

  // Get a(i)
  double GetEntry(long i) const;
  // b = a(RowStart:RowStart+nRow-1)
  void GetBlock(long RowStart, long nRow, DVector &b) const;
  void GetBlock(long RowStart, long nRow, double *b) const;
  // b = a(idx)
  void GetBlock(long *idx, long n, DVector &b) const;
  // Get the double* pointer
  double* GetPointer(void) const;

  // Set a(i) = b
  void SetEntry(long i, double b);
  // a(RowStart:RowStart+nRow-1) = b
  void SetBlock(long RowStart, long nRow, const DVector &b);
  void SetBlock(long RowStart, long nRow, const double *b);

  // a = c
  void SetConstVal(double c);
  // a = rand()
  void SetUniformRandom01(void);
  // a = randn()
  void SetStandardNormal(void);
  // a = trnd(1); each element is a student-t of degree 1
  void SetStudentT1(void);
  // The whole vector is a multivariate student-t of degree 1
  void SetMultivariateStudentT1(void);

  // Permutation
  // a(i) = a(Perm(i))
  void Permute(const long *Perm, long N_);
  // a(iPerm(i)) = a(i)
  void iPermute(const long *iPerm, long N_);
  // Note the following two sets of calling patterns:
  //   (1)  Permute(Perm, N) is equivalent to iPermute(iPerm, N);
  //   (2)  Permute(iPerm, N) is equivalent to iPermute(Perm, N).
  // Moreover, the effect of (1) is opposite to that of (2)

  // Sorting
  // a = sort(a, type)
  void Sort(SortType type);
  // a = sort(a, type). Sort accoding to |a|
  void SortByMagnitude(SortType type);

  // Find elements larger than tol. The array idx contains the indices
  // of the elements that satisfy the requirement. The return value is
  // the number of such elements. Memory of idx must be
  // preallocated. For safety, it is recommended that idx has the
  // length as the vector itself.
  long FindLargerThan(double tol, long *idx) const;

  // Print the vector in the Matlab form
  void PrintVectorMatlabForm(const char *name) const;

  //-------------------- Vector computations --------------------

  // a = -a  or  b = -a
  void Negate(void);
  void Negate(DVector &b) const;

  // a = a + b  or  c = a + b
  void Add(double b);
  void Add(const DVector &b);
  void Add(double b, DVector &c) const;
  void Add(const DVector &b, DVector &c) const;

  // a = a - b  or  c = a - b
  void Subtract(double b);
  void Subtract(const DVector &b);
  void Subtract(double b, DVector &c) const;
  void Subtract(const DVector &b, DVector &c) const;

  // a = a * b  or  c = a * b
  void Multiply(double b);
  void Multiply(double b, DVector &c) const;
  // a = a .* b  or  c = a .* b
  void Multiply(const DVector &b);
  void Multiply(const DVector &b, DVector &c) const;

  // a = a / b  or  a = a / b
  void Divide(double b);
  void Divide(double b, DVector &c) const;
  // a = a ./ b  or  a = a ./ b
  void Divide(const DVector &b);
  void Divide(const DVector &b, DVector &c) const;

  // a' * b
  double InProd(const DVector &b) const;

  // a = abs(a)  or  b = abs(a)
  void Abs(void);
  void Abs(DVector &b) const;

  // a = sqrt(a)  or  b = sqrt(a)
  void Sqrt(void);
  void Sqrt(DVector &b) const;

  // a = 1./a  or  b = 1./a
  void Inv(void);
  void Inv(DVector &b) const;

  // norm(a,2)
  double Norm2(void) const;

  // min(a). idx is the location of the min
  double Min(void) const;
  double Min(long &idx) const;

  // max(a). idx is the location of the max
  double Max(void) const;
  double Max(long &idx) const;

  // sum(a)
  double Sum(void) const;

  // mean(a)
  double Mean(void) const;

protected:

private:

  long N;    // Vector length
  double *a; // Vector data

};

#endif
