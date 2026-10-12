/* HDF5 filter SZ3 (id 32024), implemented by sz3-rust; the C part of SZ3's H5Z_SZ3.hpp. */
#ifndef SZ3_H5Z_SZ3_H
#define SZ3_H5Z_SZ3_H

#define H5Z_FILTER_SZ3 32024

#include "hdf5.h"

#define H5Z_SZ3_EB_ABS 0
#define H5Z_SZ3_EB_REL 1
#define H5Z_SZ3_EB_PSNR 2
#define H5Z_SZ3_EB_L2NORM 3
#define H5Z_SZ3_EB_ABS_AND_REL 4
#define H5Z_SZ3_EB_ABS_OR_REL 5

#define H5Z_SZ3_ALGO_LORENZO_REG 0
#define H5Z_SZ3_ALGO_INTERP_LORENZO 1
#define H5Z_SZ3_ALGO_INTERP 2
#define H5Z_SZ3_ALGO_NOPRED 3
#define H5Z_SZ3_ALGO_LOSSLESS 4
#define H5Z_SZ3_ALGO_BIOMD 5
/* Removed in 3.4.0; H5Pset_sz3 refuses it. Use H5Z_SZ3_ALGO_BIOMD. */
#define H5Z_SZ3_ALGO_BIOMDXTC 6

#ifdef __cplusplus
extern "C" {
#endif
/* Every bound must be finite and not negative; errorBoundMode decides which are used. Returns 1, or -1 with the
 * reason on the HDF5 error stack. */
herr_t H5Pset_sz3(hid_t propertyList, int cmprAlgo, int errorBoundMode, double absErrorBound, double relErrorBound,
                  double psnrErrorBound, double l2normErrorBound);
#ifdef __cplusplus
}
#endif

#endif /* SZ3_H5Z_SZ3_H */
