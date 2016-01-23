// The RLCM class implements the kernel ridge regression by using
// recursively low-rank compressed matrices.
//
//   y0 = K0 * inv(K + lambda) * (y-ymean) + ymean.
//
// We let c = inv(K + lambda) * (y-ymean); then y0 = K0*c + ymean.

#ifndef _RLCM_
#define _RLCM_

#include "../Matrices/CMatrix.hpp"
#include "../Solvers/PCG.hpp"

template<class Kernel, class Point, class PointArray>
class RLCM {

public:

  RLCM();
  RLCM(const RLCM &G);
  RLCM& operator= (const RLCM &G);
  ~RLCM();

  void Init(void);
  void ReleaseAllMemory(void);
  void DeepCopy(const RLCM &G);

  // Training. Kernel should not contain the regularization term lambda.
  //
  // If verbose is set true, some diagnostic information regarding the
  // computation will be printed.
  //
  // The routine returns a boolean value indicating whether training
  // is successful. Training fails only when the construction of the
  // recursively low-rank compressed matrix fails.
  //
  // The output variable mem_est is the estimated memory consumption
  // per data point according to a simplified memory metric model
  bool Train(const Kernel &mKernel, double lambda,
             PointArray &Xtrain,      // Will be permuted
             DVector &ytrain,         // Will be accordingly permuted
             long *Perm, long *iPerm, // See CMatrix::BuildKernelMatrix
             long r, unsigned Seed, bool verbose, double &mem_est);

  // Testing.
  void Test(const Kernel &mKernel, double lambda,
            const PointArray &Xtrain, // Must use permuted Xtrain
                                      // after training
            const PointArray &Xtest,
            DVector &ytest_predict);

  double TrainPerf(const DVector &ytrain_truth, const DVector &ytrain_predict, 
        const int NumClasses);

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
               DVector &ytrain_truth, //lingfei: for computing training error
               const long *Perm, const long *iPerm,
               bool PermuteYtrain,
               bool verbose,
               DMatrix &Ytest_predict);

  double TrainPerf(const DVector &ytrain_truth, const DMatrix &Ytrain_predict,
        const int NumClasses);

protected:

private:

  // Data resulting from training and being used for testing
  CMatrix K;
  DVector c;    // Stored in Train but not stored in TrainAlt
  double ymean; // Stored in Train but not stored in TrainAlt
  CMatrix invK; // Not stored in Train but stored in TrainAlt

};

#include "RLCM.tpp"

#endif
