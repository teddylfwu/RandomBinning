// This file contains miscellaneous stuffs supporting the
// implementation of other classes.

#ifndef _COMMON_
#define _COMMON_

#include <math.h>
#include <time.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <limits.h>
#include <typeinfo>
#ifdef USE_OPENBLAS
#include <cblas.h>
#elif USE_ESSL
#define _ESV_COMPLEX99_
#include <essl.h>
#include <omp.h>
#elif USE_OPENMP
#include <omp.h>
#endif

enum MatrixMode { NORMAL, TRANSPOSE, CONJ_TRANS };
enum SelectType { LHP, RHP };
enum SortType { ASCEND, DESCEND };

#define IF_NUMERIC_TYPE                                             \
    if (typeid(T) == typeid(int) || typeid(T) == typeid(long) ||    \
        typeid(T) == typeid(float) || typeid(T) == typeid(double))

typedef int LOGICAL;
#define FTRUE  1
#define FFALSE 0

#define EPS 2.220446049250313e-16

#define PREPARE_CLOCK(is_timing)                 \
    struct timespec time_start, time_end;        \
    double ELAPSED_TIME = 0.0;                   \
    bool timing = is_timing;

#define START_CLOCK                                 \
    if (timing) {                                   \
      clock_gettime(CLOCK_MONOTONIC, &time_start);  \
    }

#define END_CLOCK                                                       \
    if (timing) {                                                       \
      clock_gettime(CLOCK_MONOTONIC, &time_end);                        \
      ELAPSED_TIME = time_end.tv_sec - time_start.tv_sec;               \
      ELAPSED_TIME += (time_end.tv_nsec - time_start.tv_nsec) / 1e9;    \
    }

// Mac and Liux os systems both permit using gettimeofday to get timing. 
/*
#define PREPARE_CLOCK(is_timing)                \
    struct timeval time_start, time_end;        \
    double ELAPSED_TIME = 0.0;                  \
    bool timing = is_timing;
    
#define START_CLOCK                             \
    if (timing) {                               \
       gettimeofday(&time_start, NULL);         \
    }

#define END_CLOCK                                                                                   \
    if (timing) {                                                                                   \
       gettimeofday(&time_end, NULL);                                                               \
       ELAPSED_TIME = (double) time_end.tv_sec - (double) time_start.tv_sec;                        \
       ELAPSED_TIME += ((double) time_end.tv_usec - (double) time_start.tv_usec) / (double) 1E6;    \
    }
*/

// Hard-coded parameters
#define TWOMEANS_NUMTRIAL 5
#define TWOMEANS_MAXITER  5
#define TWOMEANS_TOL      1e-3
#define BUILD_CMATRIX_REG 1e-8
#define RLCM_PCG_MAXIT    10
#define RLCM_PCG_RTOL     1e-8


//--------------------------------------------------------------------------
// x^2
double Square(double x);

// Generate [0,1] uniform random numbers
void UniformRandom01(double *a, long n);

// Generate Gaussian random numbers
void StandardNormal(double *a, long n);

// Generate student-t of degree 1 random numbers
void StudentT1(double *a, long n);

// Generate a random vector of multivariate student-t of degree 1
void MultivariateStudentT1(double *a, long n);

// Generate k random integers in [0,n) without repetition
void RandPerm(long n, long k, long *a);

// Needed by qsort
int CompareNaturalOrderLess(const void *x, const void *y);
int CompareNaturalOrderGreater(const void *x, const void *y);
int CompareByMagnitudeLess(const void *x, const void *y);
int CompareByMagnitudeGreater(const void *x, const void *y);

// Needed by DGEES
LOGICAL SelectLeftHalfPlane(double *WR, double *WI);
LOGICAL SelectRightHalfPlane(double *WR, double *WI);


//--------------------------------------------------------------------------
// BLAS and LAPACK
extern "C" {
  void dgemv_(char *TRANS, int *M, int *N, double *ALPHA, double *A, int *LDA, double *X, int *INCX, double *BETA, double *Y, int *INCY);
  void dgemm_(char *TRANSA, char *TRANSB, int *M, int *N, int *K, double *ALPHA, double *A, int *LDA, double *B, int *LDB, double *BETA, double *C, int *LDC);
  void dgesvx_(char *FACT, char *TRANS, int *N, int *NRHS, double *A, int *LDA, double *AF, int *LDAF, int *IPIV, char *EQUED, double *R, double *C, double *B, int *LDB, double *X, int *LDX, double *RCOND, double *FERR, double *BERR, double *WORK, int *IWORK, int *INFO);
  void dgetrf_(int *M, int *N, double *A, int *LDA, int *IPIV, int *INFO);
  //int dgetrf_(int *M, int *N, double *A, int *LDA, int *IPIV, int *INFO);
  void dgetrs_(char *TRANS, int *N, int *NRHS, double *A, int *LDA, int *IPIV, double *B, int *LDB, int *INFO);
  //int dgetrs_(char *TRANS, int *N, int *NRHS, double *A, int *LDA, int *IPIV, double *B, int *LDB, int *INFO);
  double dlange_(char *NORM, int *M, int *N, double *A, int *LDA, double *WORK);
  void dgecon_(char *NORM, int *N, double *A, int *LDA, double *ANORM, double *RCOND, double *WORK, int *IWORK, int *INFO);
  void dgesvd_(char *JOBU, char *JOBVT, int *M, int *N, double *A, int *LDA, double *S, double *U, int *LDU, double *VT, int *LDVT, double *WORK, int *LWORK, int *INFO);
  void dgelss_(int *M, int *N, int *NRHS, double *A, int *LDA, double *B, int *LDB, double *S, double *RCOND, int *RANK, double *WORK, int *LWORK, int *INFO);
  void dpotrf_(char *UPLO, int *N, double *A, int *LDA, int *INFO);
  //int dpotrf_(char *UPLO, int *N, double *A, int *LDA, int *INFO);
  void dsyev_(char *JOBZ, char *UPLO, int *N, double *A, int *LDA, double *W, double *WORK, int *LWORK, int *INFO);
  void dggev_(char *JOBVL, char *JOBVR, int *N, double *A, int *LDA, double *B, int *LDB, double *ALPHAR, double *ALPHAI, double *BETA, double *VL, int *LDVL, double *VR, int *LDVR, double *WORK, int *LWORK, int *INFO);
  void dgees_(char *JOBVS, char *SORT, LOGICAL (*SELECT)(double *WR, double *WI), int *N, double *A, int *LDA, int *SDIM, double *WR, double *WI, double *VS, int *LDVS, double *WORK, int *LWORK, LOGICAL *BWORK, int *INFO);
  void dgeqrf_(int *M, int *N, double *A, int *LDA, double *TAU, double *WORK, int *LWORK, int *INFO);
  void dorgqr_(int *M, int *N, int *K, double *A, int *LDA, double *TAU, double *WORK, int *LWORK, int *INFO);
  void dgeqp3_(int *M, int *N, double *A, int *LDA, int *JPVT, double *TAU, double *WORK, int *LWORK, int *INFO);
}

//--------------------------------------------------------------------------
// Templated new and delete. For all the delete functions, the pointer
// is set to NULL. For all the new functions, the allocated memory is
// set to zero if the data type is numeric.
//
// Provides the following interfaces:
//
// 1a. Delete a 1D array
//   void Delete_1D_Array(T **a);
//
// 1b. New a 1D array of length d1
//   void New_1D_Array(T **a, T1 d1);
//
// 2a. Delete a 2D array (first dimension d1)
//   void Delete_2D_Array(T ***a, T1 d1);
//
// 2b. New a 2D array of size d1*d2
//   void New_2D_Array(T ***a, T1 d1, T2 d2);
//
// 3a. Delete a 3D array (first dimension d1, second dimension d2)
//   void Delete_3D_Array(T ****a, T1 d1, T2 d2);
//
// 3b. New a 3D array of size d1*d2*d3
//   void New_3D_Array(T ****a, T1 d1, T2 d2, T3 d3);

template <class T>
void Delete_1D_Array(T **a) {
  delete [] (*a);
  (*a) = NULL;
}

template <class T, class T1>
void New_1D_Array(T **a, T1 d1) {
  (*a) = new T [d1];
  IF_NUMERIC_TYPE{ memset((*a), 0, d1*sizeof(T)); }
}

template <class T, class T1>
void Delete_2D_Array(T ***a, T1 d1) {
  if (*a) {
    for (T1 i = 0; i < d1; i++) {
      delete [] (*a)[i];
    }
    delete [] (*a);
    (*a) = NULL;
  }
}

template <class T, class T1, class T2>
void New_2D_Array(T ***a, T1 d1, T2 d2) {
  (*a) = new T * [d1];
  for (T1 i = 0; i < d1; i++) {
    (*a)[i] = new T [d2];
    IF_NUMERIC_TYPE{ memset((*a)[i], 0, d2*sizeof(T)); }
  }
}

template <class T, class T1, class T2>
void Delete_3D_Array(T ****a, T1 d1, T2 d2) {
  if (*a) {
    for (T1 i = 0; i < d1; i++) {
      if ((*a)[i]) {
        for (T2 j = 0; j < d2; j++) {
          delete [] (*a)[i][j];
        }
        delete [] (*a)[i];
      }
    }
    delete [] (*a);
    (*a) = NULL;
  }
}

template <class T, class T1, class T2, class T3>
void New_3D_Array(T ****a, T1 d1, T2 d2, T3 d3) {
  (*a) = new T ** [d1];
  for (T1 i = 0; i < d1; i++) {
    (*a)[i] = new T * [d2];
    for (T2 j = 0; j < d2; j++) {
      (*a)[i][j] = new T [d3];
      IF_NUMERIC_TYPE{ memset((*a)[i][j], 0, d3*sizeof(T)); }
    }
  }
}


//--------------------------------------------------------------------------
// Templated swap
template <class T>
void Swap(T &a, T &b) {
  T tmp = a;
  a = b;
  b = tmp;
}


#endif
