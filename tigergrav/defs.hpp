#pragma once



#define NDIM    3
#define NCHILD  2

#define ERROR() printf( "ERROR %s %i\n", __FILE__, __LINE__)

//#define INDIRECT_DOUBLE
//#ifdef __AVX512F__
//#define USE_AVX512
//#warning 'compiling for AVX512'
//#else
#define USE_AVX2
//#endif




#ifdef INDIRECT_DOUBLE
using ireal = double;
#define simd_real simd_double
#define SIMD_REAL_LEN SIMD_DOUBLE_LEN
#else
using ireal = float;
#define simd_real  simd_float
#define SIMD_REAL_LEN SIMD_FLOAT_LEN
#endif
