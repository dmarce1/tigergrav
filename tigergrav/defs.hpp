#pragma once



#define NDIM    3
#define NCHILD  2

#define ERROR() printf( "ERROR %s %i\n", __FILE__, __LINE__)

//#define INDIRECT_DOUBLE
//#define BALANCED_TREE

#ifdef INDIRECT_DOUBLE
using ireal = double;
#define isimd_vector simd_dvector
#define ISIMD_LEN SIMD_DLEN
#else
using ireal = float;
#define isimd_vector  simd_svector
#define ISIMD_LEN SIMD_SLEN
#endif
