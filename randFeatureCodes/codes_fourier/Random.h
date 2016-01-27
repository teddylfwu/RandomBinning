#ifndef RANDOM_H_
#define RANDOM_H_

#define TWO_PI 6.28318530717958647692
void UniformRandom01(double *a, int n);
void StandardNormal(double *a, int n); 
void StudentT1(double *a, int n); 
void MultivariateStudentT1(double *a, int n); 

#endif // RANDOM_H_
