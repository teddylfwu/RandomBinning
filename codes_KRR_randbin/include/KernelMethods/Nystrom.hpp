// The Nystrom class implements the Nystrom method for performing
// approximate kernel ridge regression.
//
// Given training points X (n*d), labels y (n*1), and rank r, the
// Nystrom feature Z (n*r) is computed in the following way:
//
//   kernel matrix Phi approx= Z*Z' = K1*pinv(K2)*K1,
//
// where K1 is the data*center matrix and K2 is the center*center
// matrix. The r centers are found by using random sampling.
//
// For testing points X0, the output y0 is computed as
//
//   y0 = Z0 * inv(Z'*Z + lambda) * Z'*(y-ymean) + ymean.
//
// We let c = inv(Z'*Z + lambda) * Z'*(y-ymean); then y0 = Z0*c + ymean.

#ifndef _NYSTROM_
#define _NYSTROM_

#include "../Matrices/DMatrix.hpp"

template<class Kernel, class Point, class PointArray>
class Nystrom {

public:

  Nystrom();
  Nystrom(const Nystrom &G);
  Nystrom& operator= (const Nystrom &G);
  ~Nystrom();

  void Init(void);
  void ReleaseAllMemory(void);
  void DeepCopy(const Nystrom &G);

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
            DVector &ytest_predict) const;

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
               DMatrix &Ytest_predict);

protected:

private:

  // Data resulting from training and being used for testing
  PointArray Y; // Centers
  DMatrix K3;   // sqrt(pinv(K2))
  DVector c;    // Stored in Train but not stored in TrainAlt
  double ymean; // Stored in Train but not stored in TrainAlt
  DMatrix Z;    // Not stored in Train but stored in TrainAlt
  DMatrix ZZ;   // Not stored in Train but stored in TrainAlt

};

#include "Nystrom.tpp"

#endif
