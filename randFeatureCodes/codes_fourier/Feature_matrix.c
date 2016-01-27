#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <string.h>
#include "Feature_matrix.h"

void dgemm_(char *TRANSA, char *TRANSB, int *M, int *N, int *K, double *ALPHA, double *A, int *LDA, double *B, int *LDB, double *BETA, double *C, int *LDC);

void ComputeFeatureMatrix(int n, int d, int r, double *X, double *w, double *b, double *Z, double sigma)
{
    
    // Z = cos(X*w+repmat(b,n,1)) * sqrt(2/r) 
    // Assume that memory space for Z has been allocated
    // Initialize Z as repmat(b,n,1), where b is a row vector of length r
    int i, j;
    #pragma omp parallel for private(i, j)
    for (i = 0; i < n; i++) {
        for (j = 0; j < r; j++) {
            Z[i+j*n] = b[j];
        }
    }
    
    // Z = X*w + Z
    double ONE = 1.0;
    char TRANS_N = 'N';
    dgemm_(&TRANS_N, &TRANS_N, &n, &r, &d, &ONE, X, &n, w, &d, &ONE, Z, &n);
   
    // Z = cos(Z) * sqrt(2/r) 
    double two = 2;
    double fac = sqrt(two/r);
    #pragma omp parallel for private(i)
    for (i = 0; i < n*r; i++) {
        Z[i] = cos(Z[i]) * fac;
    }
}
