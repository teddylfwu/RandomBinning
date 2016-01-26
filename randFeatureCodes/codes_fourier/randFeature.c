#include <math.h>
#include <time.h>
#include <stdio.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#include "ReadLibSVM.h"
#include "WriteLibSVM.h"
#include "Feature_matrix.h"
#include "Random.h"

typedef enum { Gaussian, Laplace, ProdLaplace } KERNEL;

int main(int argc, char* argv[]){

    if(argc < 1+7){
        printf("Usage: [inTrain] [inTest] [outTrain] [outTest] [r] [sigma] [KernelType]\n");
        printf("KernelType: {Gauss, Laplace, ProdLaplace}\n");
        return -1;
    }
    
    int i; //loop index
    int idx = 0;
    char* trainFile = argv[++idx];
    char* testFile = argv[++idx];
    char* trainOut = argv[++idx];
    char* testOut = argv[++idx];
    int r = atoi(argv[++idx]);
    double sigma = atof(argv[++idx]);
    char* Kernel = argv[++idx];                                                                                                  
    
    KERNEL mKernel;
    if (strcmp(Kernel, "Gauss") == 0) {
        mKernel = Gaussian;
    }
    else if (strcmp(Kernel, "Laplace") == 0) {
        mKernel = Laplace;
    }
    else if (strcmp(Kernel, "ProdLaplace") == 0) {
        mKernel = ProdLaplace;
    }
    else {
        printf("KRR_OneVsOne. Error: unidentified kernel type!\n");
        return -1;
    }

    // Read in X = Xtrain (n*d), y = ytrain (n*1),
    //     and X0 = Xtest (m*d), y0 = ytest (m*1)
    double *Xtrain = NULL, *Xtest = NULL;
    double *ytrain = NULL, *ytest = NULL;
    int n = 0, m = 0, d = 0;
    if (ReadLibSVM(trainFile, &Xtrain, &ytrain, &d, &n) == -1) {
        return -1;
    }
    if (ReadLibSVM(testFile, &Xtest, &ytest, &d, &m) == -1) {
        return -1;
    }
    printf("Generate fourier feature: finish loading data\n");
    
    // Seed the random number genrator
    int Seed = 0;  
    srandom(Seed);
    // Generate random numbers w and b
    // w = random_distribution(d*r)/sigma
    // b = rand(1,r)*2*pi
    double *w = (double *)malloc(d*r*sizeof(double)); 
    double *b = (double *)malloc(r*sizeof(double));
    switch (mKernel) {
        case Gaussian:
            StandardNormal(w, d*r);
            break;
        case Laplace:
            for (i = 0; i < r; i++) {
                MultivariateStudentT1(w+i*d, d);
            }
            break;
        case ProdLaplace:
            StudentT1(w, d*r);
            break;
    }
    #pragma omp parallel for private(i)
    for (i = 0; i < d*r; i++) {
        w[i] /= sigma;
    }
 
    // make sure w is the same as other codes   
    /*printf("w = \n");
    for (i = 0;i < r; i++){
        for (j = 0;j< d; j++){
            printf(" %g", w[d*j+i]);
        }                                                                                                                               
        printf("\n");
    }*/

    UniformRandom01(b, r);
    #pragma omp parallel for private(i)
    for (i = 0; i < r; i++) {
        b[i] *= TWO_PI;
    }

    // make sure b is the same as other codes   
    /*printf("b = \n");
    for (i = 0;i < r; i++){
        printf(" %g\n", b[i]);
    }*/

    printf("Generate fourier feature: finish generating w and b\n");

    // Generate feature matrix Z given X. Size n*r
    double *Ztrain = (double *)malloc(n*r*sizeof(double));
    ComputeFeatureMatrix(n, d, r, Xtrain, w, b, Ztrain, sigma);
    double *Ztest = (double *)malloc(m*r*sizeof(double));
    ComputeFeatureMatrix(n, d, r, Xtest, w, b, Ztest, sigma);

    printf("Generate fourier feature: finish computing feature matrix\n");
    
    // Write fourier feature matrix in LibSVM format
    if (WriteLibSVM(trainOut, Ztrain, ytrain, r, n) == -1){
        return -1;
    }
    if (WriteLibSVM(testOut, Ztest, ytest, r, m) == -1){
        return -1;
    }

    printf("Generate fourier feature: finish storing data\n");
    
    free(Xtrain);
    free(ytrain);
    free(Xtest);
    free(ytest);
    free(w);
    free(b);
    //No idea why have double-free problems for Ztrain and Ztest, resolve it later
    //free(Ztrain);
    //free(Ztest);

    return 0;
}
