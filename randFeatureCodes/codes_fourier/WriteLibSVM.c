
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "WriteLibSVM.h"

int WriteLibSVM(const char *filename, double *X, double *y, int d, int n) {

  FILE *fp = NULL;
  fp = fopen(filename, "w");
  if (fp == NULL) {
    printf("WriteData. Error: Cannot open file %s. Function call takes no effect.\n", filename);
    return -1;
  }

  // LibSVM index starts from 1
  long i,j;
  // prevent overflow when (*n)*d > int_max
  long ln = (long)n;
  long ld = (long)d;
  for (i = 0;i < ln; i++){
    fprintf(fp, "%f", y[i]);
    for (j = 0;j< ld; j++){
      fprintf(fp, " %ld:%g", j+1, X[ln*j+i]);      
    }
    fprintf(fp, "\n");
  }

  fclose(fp);
  return 0;

}

