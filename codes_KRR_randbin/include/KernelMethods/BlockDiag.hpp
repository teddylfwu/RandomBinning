// The BlockDiag class implements the kernel ridge regression by using
// block-diagonal approximation.
//
//   y0 = K0 * inv(K + lambda) * (y-ymean) + ymean.
//
// We let c = inv(K + lambda) * (y-ymean); then y0 = K0*c + ymean.

#ifndef _BlockDiag_
#define _BlockDiag_

#include "../Matrices/BMatrix.hpp"

template<class Kernel, class Point, class PointArray>
class BlockDiag {

public:

  BlockDiag();
  BlockDiag(const BlockDiag &G);
  BlockDiag& operator= (const BlockDiag &G);
  ~BlockDiag();

  void Init(void);
  void ReleaseAllMemory(void);
  void DeepCopy(const BlockDiag &G);

  // Training. Kernel should not contain the regularization term lambda.
  //
  // If verbose is set true, some diagnostic information regarding the
  // computation will be printed.
  //
  // The routine returns a boolean value indicating whether training
  // is successful. Training fails only when the construction of the
  // block-diagonal matrix fails.
  //
  // The output variable mem_est is the estimated memory consumption
  // per data point according to a simplified memory metric model
  bool Train(const Kernel &mKernel, double lambda,
             PointArray &Xtrain,      // Will be permuted
             DVector &ytrain,         // Will be accordingly permuted
             long *Perm, long *iPerm, // See BMatrix::BuildKernelMatrix
             long r, unsigned Seed, bool verbose, double &mem_est);

  // Testing.
  void Test(const Kernel &mKernel, double lambda,
            const PointArray &Xtrain, // Must use permuted Xtrain
                                      // after training
            const PointArray &Xtest,
            DVector &ytest_predict);

  // Alternative training and testing functions. They are designed
  // particularly for multiclass classification. The intermediate
  // result invK will be stored after training and be used in
  // testing. The storage of invK allows the testing function to be
  // called multiple times, each for a set of ytrain vectors.
  bool TrainAlt(const Kernel &mKernel, double lambda,
                PointArray &Xtrain, // Will be permuted
                long *Perm, long *iPerm,
                long r, unsigned Seed, bool verbose, double &mem_est);

  void TestAlt(const Kernel &mKernel, double lambda,
               const PointArray &Xtrain, // Must use permuted Xtrain
                                         // after training
               const PointArray &Xtest,
               DMatrix &Ytrain, // Will be accordingly permuted if the
                                // flag PermuteYtrain is set
               const long *Perm, const long *iPerm,
               bool PermuteYtrain,
               DMatrix &Ytest_predict);

protected:

private:

  // Data resulting from training and being used for testing
  BMatrix K;
  DVector c;    // Stored in Train but not stored in TrainAlt
  double ymean; // Stored in Train but not stored in TrainAlt
  BMatrix invK; // Not stored in Train but stored in TrainAlt

};

#include "BlockDiag.tpp"

#endif
