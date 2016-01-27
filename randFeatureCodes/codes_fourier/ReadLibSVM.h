
#ifndef READLIBSVM_H
#define READLIBSVM_H

int ReadLibSVM(const char *filename, double **X, double **y, int *d, int *n);
int VerifyFileFormat(const char *filename, int *d, int *n);

#endif
