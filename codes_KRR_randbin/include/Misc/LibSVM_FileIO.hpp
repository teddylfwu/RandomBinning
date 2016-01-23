#ifndef _LIBSVM_FILEIO_
#define _LIBSVM_FILEIO_

#include "../Matrices/DPointArray.hpp"
#include "../Matrices/SPointArray.hpp"

// Read a data file in the LibSVM format. The points are stored in X
// and the labels are stored in y. The sizse of X is N*d, where N
// (number of points) is inferred from the data file, and d (dimension
// of points) is a mandatoray input. One can to choose store the data
// points in a dense format (DPointArray) or in a sparse format
// (SPointArray).

bool ReadData(const char *path_and_filename, DPointArray &X, DVector &y, int d);
bool ReadData(const char *path_and_filename, SPointArray &X, DVector &y, int d);

// Subroutine called by ReadData(). Verify the file format and count N
// and nnz.
bool VerifyFileFormat(const char *path_and_filename, int d, long &N, long &nnz);

#endif
