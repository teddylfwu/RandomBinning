
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ReadLibSVM.h"

int ReadLibSVM(const char *filename, double **X, double **y, int *d, int *n) {

  if (VerifyFileFormat(filename, d, n) == -1) {
    return -1;
  }
  
  // prevent overflow when (*n)*d > int_max
  long ln = (long)(*n);
  long ld = (long)(*d);

  FILE *fp = NULL;
  fp = fopen(filename, "r");
  
  // Allocate memory for X and y
  (*X) = (double *)malloc(ln*ld*sizeof(double));
  (*y) = (double *)malloc(ln*sizeof(double));

  // Read the file again and populate X and y
  long i = 0;
  long numi = 0;
  double numf = 0.0;
  char *line = NULL, *str = NULL, *saveptr = NULL, *subtoken = NULL;
  ssize_t read = 0;
  size_t len = 0;
  while ((read = getline(&line, &len, fp)) != -1) { // Read in a line
    long cnt = 0;
    for (str = line; ; str = NULL) {
      subtoken = strtok_r(str, ": ", &saveptr); // Tokenize the line
      if (subtoken == NULL || !strcmp(subtoken,"\n")) {
        break;
      }
      else {
        cnt++;
      }
      if (cnt == 1) {
        (*y)[i] = atof(subtoken);
      }
      else if (cnt%2 == 0) {
        numi = atoi(subtoken);
      }
      else {
        numf = atof(subtoken);
        (*X)[i+ln*(numi-1)] = numf; // LibSVM index starts from 1
      }
    }
    i++;
  }
  if (line) {
    free(line);
  }

  fclose(fp);
  return 0;

}


//--------------------------------------------------------------------------
int VerifyFileFormat(const char *filename, int *d, int *n) {

  FILE *fp = NULL;
  fp = fopen(filename, "r");
  if (fp == NULL) {
    printf("VerifyFileFormat. Error: Cannot open file %s. Function call takes no effect.\n", filename);
    return -1;
  }

  (*n) = 0;
  (*d) = 0;
  char *line = NULL, *str = NULL, *saveptr = NULL, *subtoken = NULL;
  ssize_t read = 0;
  size_t len = 0;
  while ((read = getline(&line, &len, fp)) != -1) { // Read in a line
    (*n)++;
    int cnt = 0, maxdim = 0;
    for (str = line; ; str = NULL) {
      subtoken = strtok_r(str, ": ", &saveptr); // Tokenize the line
      if (subtoken == NULL || !strcmp(subtoken,"\n")) {
        break;
      }
      else {
        cnt++;
      }
      if (cnt%2 == 0) { // Verify format
        int num = atoi(subtoken);
        if (maxdim > num) {
          printf("VerifyFileFormat. Error: Indices in line %d are not in the ascending order. Stop reading data. Function call takes no effect.\n", (*n));
          fclose(fp);
          return -1;
        }
        else {
          maxdim = num;
          *d = maxdim;
        }
      }
    }
    if (cnt%2 != 1) { // Verify format
      printf("VerifyFileFormat. Error: Line %d does not conform with a LibSVM format. Stop reading data. Function call takes no effect.\n", (*n));
      fclose(fp);
      return -1;
    }
  }
  if (line) {
    free(line);
  }

  fclose(fp);
  if ((*n) == 0) {
    printf("VerifyFileFormat. Error: Empty file!\n");
    return -1;
  }
  return 0;

}

