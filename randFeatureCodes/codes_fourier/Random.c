#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "omp.h"
#include "Random.h"

//--------------------------------------------------------------------------
void UniformRandom01(double *a, int n) {
    int i;
    for (i = 0; i < n; i++) {
        a[i] = (double)random()/RAND_MAX;
    }
}

//--------------------------------------------------------------------------
// Box-Muller: U, V from uniform[0,1]
// X = sqrt(-2.0*log(U))*cos(2.0*M_PI*V)
// Y = sqrt(-2.0*log(U))*sin(2.0*M_PI*V)
void StandardNormal(double *a, int n) {
    int i;
    for (i = 0; i < n/2; i++) {
        double U = (double)random()/RAND_MAX;
        double V = (double)random()/RAND_MAX;
        double common1 = sqrt(-2.0*log(U));
        double common2 = TWO_PI*V;
        a[2*i] = common1 * cos(common2);
        a[2*i+1] = common1 * sin(common2);
    }
    if (n%2 == 1) {
        double U = (double)random()/RAND_MAX;
        double V = (double)random()/RAND_MAX;
        double common1 = sqrt(-2.0*log(U));
        double common2 = TWO_PI*V;
        a[n-1] = common1 * cos(common2);
    }
}

//--------------------------------------------------------------------------
// Let X and Y be independent standard normal. Then X/fabs(Y) follows
// student t with 1 degree of freedom.
void StudentT1(double *a, int n) {
    int i;
    for (i = 0; i < n; i++) {
        double V = (double)random()/RAND_MAX;
        a[i] = tan(2.0*M_PI*V); // Why is cot not in math.h??? :-(
        if (V > 0.5) {
            a[i] = -a[i];
        }
    }
}
    
//--------------------------------------------------------------------------
// Let X and Y be independent standard normal. Then X/fabs(Y) follows
// student t with 1 degree of freedom.
void MultivariateStudentT1(double *a, int n) {
    StandardNormal(a, n);
    double b = 0.0;
    StandardNormal(&b, 1);
    b = fabs(b);
    int i;
    for (i = 0; i < n; i++) {
        a[i] /= b;
    }
}

