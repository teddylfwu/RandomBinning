// The Standard class implements the standard kernel ridge regression.
//
//   y0 = K0 * inv(K + lambda) * (y-ymean) + ymean.
//
// We let c = inv(K + lambda) * (y-ymean); then y0 = K0*c + ymean.

#ifndef _STANDARD_
#define _STANDARD_

#include "../Matrices/DMatrix.hpp"

template<class Kernel, class Point, class PointArray>
class Standard {

public:

  Standard();
  Standard(const Standard &G);
  Standard& operator= (const Standard &G);
  ~Standard();

  void Init(void);
  void ReleaseAllMemory(void);
  void DeepCopy(const Standard &G);

  // Training. Kernel should not contain the regularization term lambda.
  //
  // The output variable mem_est is the estimated memory consumption
  // per data point according to a simplified memory metric model
  void Train(const Kernel &mKernel, double lambda,
             const PointArray &Xtrain,
             const DVector &ytrain,
             double &mem_est);

  // Testing. Because the kernel matrix between Xtrain and Xtest can
  // be huge, Xtest is split into batches of size Budget/Xtrain.GetN()
  // when doing the computation.
  void Test(const Kernel &mKernel,
            const PointArray &Xtrain,
            const PointArray &Xtest,
            DVector &ytest_predict,
            long Budget) const;

  // Alternative training and testing functions. They are designed
  // particularly for multiclass classification. The intermediate
  // result LU will be stored after training and be used in
  // testing. The storage of LU allows the testing function to be
  // called multiple times, each for a set of ytrain vectors.
  void TrainAlt(const Kernel &mKernel, double lambda,
                const PointArray &Xtrain,
                double &mem_est);

  void TestAlt(const Kernel &mKernel,
               const PointArray &Xtrain,
               const PointArray &Xtest,
               const DMatrix &Ytrain,
               DMatrix &Ytest_predict,
               long Budget);

protected:

private:

  // Data resulting from training and being used for testing
  DVector c;    // Stored in Train but not stored in TrainAlt
  double ymean; // Stored in Train but not stored in TrainAlt
  DMatrix LU;   // Not stored in Train but stored in TrainAlt

};

#include "Standard.tpp"

#endif
