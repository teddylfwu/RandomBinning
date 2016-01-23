// The Fourier class implements the Fourier method for performing
// approximate kernel ridge regression.
//
// Given training points X (n*d), labels y (n*1), and rank r, the
// Fourier feature Z (n*r) is computed in the following way:
//
//   kernel matrix Phi approx= Z*Z',  Z = cos(X*w+b) * sqrt(2/r),
//
// where each column of w is iid sampled from the Fourier transform of
// the kernel and each entry of b is sampled from Uniform[0,2*pi].
//
// For testing points X0, the output y0 is computed as
//
//   y0 = Z0 * inv(Z'*Z + lambda) * Z'*(y-ymean) + ymean.
//
// We let c = inv(Z'*Z + lambda) * Z'*(y-ymean); then y0 = Z0*c + ymean.

#ifndef _FOURIER_
#define _FOURIER_

#include "../Matrices/DMatrix.hpp"

template<class Kernel, class Point, class PointArray>
class Fourier {

public:

  Fourier();
  Fourier(const Fourier &G);
  Fourier& operator= (const Fourier &G);
  ~Fourier();

  void Init(void);
  void ReleaseAllMemory(void);
  void DeepCopy(const Fourier &G);

  // Training. Kernel should not contain the regularization term lambda.
  //
  // The output variable mem_est is the estimated memory consumption
  // per data point according to a simplified memory metric model
  void Train(const Kernel &mKernel, double lambda,
             const PointArray &Xtrain,
             const DVector &ytrain,
             long r, unsigned Seed, double &mem_est);

  // Testing.
  void Test(const Kernel &mKernel,
            const PointArray &Xtest,
            DVector &ytest_predict, long r) const;

  // Alternative training and testing functions. They are designed
  // particularly for multiclass classification. The intermediate
  // results Z and ZZ will be stored after training and be used in
  // testing. The storage of Z and ZZ allows the testing function to
  // be called multiple times, each for a set of ytrain vectors.
  void TrainAlt(const Kernel &mKernel, double lambda,
                const PointArray &Xtrain,
                long r, unsigned Seed, double &mem_est);

  void TestAlt(const Kernel &mKernel,
               const PointArray &Xtest,
               const DMatrix &Ytrain,
               DMatrix &Ytest_predict, long r);

protected:

private:

  // Data resulting from training and being used for testing
  DMatrix w;
  DVector b;
  DVector c;    // Stored in Train but not stored in TrainAlt
  double ymean; // Stored in Train but not stored in TrainAlt
  DMatrix Z;    // Not stored in Train but stored in TrainAlt
  DMatrix ZZ;   // Not stored in Train but stored in TrainAlt

  // Note that w and b are random and could have been regenerated on
  // the fly in the testing phase, provided that the same seed is used
  // for the random number generator. This on-the-fly generation saves
  // the storage of w and b. For some kernels, the generation of w is
  // sufficiently fast; hence it is worthwhile to rid the storage of
  // w. In this implementation, however, we maintain w in storage for
  // simplicity.

};

#include "Fourier.tpp"

#endif
