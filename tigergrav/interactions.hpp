#pragma once

#include <tigergrav/green.hpp>

#ifdef __CUDA_ARCH__
__device__ const cuda_ewald_const& cuda_get_const();
#endif
// 43009,703
template<class DOUBLE, class SINGLE> // 986 // 251936
CUDA_EXPORT inline int multipole_interaction(expansion<DOUBLE> &L, const multipole<SINGLE> &M, vect<SINGLE> dX, bool ewald) { // 670/700 + 418 * NREAL + 50 * NFOUR
	int flop = 0;
	expansion<SINGLE> D;
	flop += ewald ? green_ewald(D,dX) : green_direct(D,dX);
	auto &L0 = L;
	for (int i = 0; i < LP; i++) {
		L[i] = fma( M[0], D[i], L[i]);
	}
	flop += 2 * LP;
	L[0] = fma(M[1], D[4] * float(5.000000000e-01), L[0]);
	L[1] = fma(M[1], D[10] * float(5.000000000e-01), L[1]);
	L[0] = fma(-M[7], D[10] * float(1.666666670e-01), L[0]);
	L[1] = fma(-M[7], D[20] * float(1.666666670e-01), L[1]);
	L[0] = fma(-M[8], D[11] * float(5.000000000e-01), L[0]);
	L[1] = fma(-M[8], D[21] * float(5.000000000e-01), L[1]);
	L[0] = fma(-M[9], D[12] * float(5.000000000e-01), L[0]);
	L[1] = fma(-M[9], D[22] * float(5.000000000e-01), L[1]);
	L[0] = fma(M[2], D[5], L[0]);
	L[1] = fma(M[2], D[11], L[1]);
	L[0] = fma(-M[10], D[13] * float(5.000000000e-01), L[0]);
	L[1] = fma(-M[10], D[23] * float(5.000000000e-01), L[1]);
	L[0] = fma(-M[11], D[14], L[0]);
	L[1] = fma(-M[11], D[24], L[1]);
	L[0] = fma(M[3], D[6], L[0]);
	L[1] = fma(M[3], D[12], L[1]);
	L[0] = fma(-M[12], D[15] * float(5.000000000e-01), L[0]);
	L[1] = fma(-M[12], D[25] * float(5.000000000e-01), L[1]);
	L[0] = fma(M[4], D[7] * float(5.000000000e-01), L[0]);
	L[1] = fma(M[4], D[13] * float(5.000000000e-01), L[1]);
	L[0] = fma(-M[13], D[16] * float(1.666666670e-01), L[0]);
	L[1] = fma(-M[13], D[26] * float(1.666666670e-01), L[1]);
	L[0] = fma(-M[14], D[17] * float(5.000000000e-01), L[0]);
	L[1] = fma(-M[14], D[27] * float(5.000000000e-01), L[1]);
	L[0] = fma(M[5], D[8], L[0]);
	L[1] = fma(M[5], D[14], L[1]);
	L[0] = fma(-M[15], D[18] * float(5.000000000e-01), L[0]);
	L[1] = fma(-M[15], D[28] * float(5.000000000e-01), L[1]);
	L[0] = fma(M[6], D[9] * float(5.000000000e-01), L[0]);
	L[1] = fma(M[6], D[15] * float(5.000000000e-01), L[1]);
	L[0] = fma(-M[16], D[19] * float(1.666666670e-01), L[0]);
	L[1] = fma(-M[16], D[29] * float(1.666666670e-01), L[1]);
	L[2] = fma(M[1], D[11] * float(5.000000000e-01), L[2]);
	L[3] = fma(M[1], D[12] * float(5.000000000e-01), L[3]);
	L[2] = fma(-M[7], D[21] * float(1.666666670e-01), L[2]);
	L[3] = fma(-M[7], D[22] * float(1.666666670e-01), L[3]);
	L[2] = fma(-M[8], D[23] * float(5.000000000e-01), L[2]);
	L[3] = fma(-M[8], D[24] * float(5.000000000e-01), L[3]);
	L[2] = fma(-M[9], D[24] * float(5.000000000e-01), L[2]);
	L[3] = fma(-M[9], D[25] * float(5.000000000e-01), L[3]);
	L[2] = fma(M[2], D[13], L[2]);
	L[3] = fma(M[2], D[14], L[3]);
	L[2] = fma(-M[10], D[26] * float(5.000000000e-01), L[2]);
	L[3] = fma(-M[10], D[27] * float(5.000000000e-01), L[3]);
	L[2] = fma(-M[11], D[27], L[2]);
	L[3] = fma(-M[11], D[28], L[3]);
	L[2] = fma(M[3], D[14], L[2]);
	L[3] = fma(M[3], D[15], L[3]);
	L[2] = fma(-M[12], D[28] * float(5.000000000e-01), L[2]);
	L[3] = fma(-M[12], D[29] * float(5.000000000e-01), L[3]);
	L[2] = fma(M[4], D[16] * float(5.000000000e-01), L[2]);
	L[3] = fma(M[4], D[17] * float(5.000000000e-01), L[3]);
	L[2] = fma(-M[13], D[30] * float(1.666666670e-01), L[2]);
	L[3] = fma(-M[13], D[31] * float(1.666666670e-01), L[3]);
	L[2] = fma(-M[14], D[31] * float(5.000000000e-01), L[2]);
	L[3] = fma(-M[14], D[32] * float(5.000000000e-01), L[3]);
	L[2] = fma(M[5], D[17], L[2]);
	L[3] = fma(M[5], D[18], L[3]);
	L[2] = fma(-M[15], D[32] * float(5.000000000e-01), L[2]);
	L[3] = fma(-M[15], D[33] * float(5.000000000e-01), L[3]);
	L[2] = fma(M[6], D[18] * float(5.000000000e-01), L[2]);
	L[3] = fma(M[6], D[19] * float(5.000000000e-01), L[3]);
	L[2] = fma(-M[16], D[33] * float(1.666666670e-01), L[2]);
	L[3] = fma(-M[16], D[34] * float(1.666666670e-01), L[3]);
	L[4] = fma(M[1], D[20] * float(5.000000000e-01), L[4]);
	L[5] = fma(M[1], D[21] * float(5.000000000e-01), L[5]);
	L[4] = fma(M[2], D[21], L[4]);
	L[5] = fma(M[2], D[23], L[5]);
	L[4] = fma(M[3], D[22], L[4]);
	L[5] = fma(M[3], D[24], L[5]);
	L[4] = fma(M[4], D[23] * float(5.000000000e-01), L[4]);
	L[5] = fma(M[4], D[26] * float(5.000000000e-01), L[5]);
	L[4] = fma(M[5], D[24], L[4]);
	L[5] = fma(M[5], D[27], L[5]);
	L[4] = fma(M[6], D[25] * float(5.000000000e-01), L[4]);
	L[5] = fma(M[6], D[28] * float(5.000000000e-01), L[5]);
	L[6] = fma(M[1], D[22] * float(5.000000000e-01), L[6]);
	L[7] = fma(M[1], D[23] * float(5.000000000e-01), L[7]);
	L[6] = fma(M[2], D[24], L[6]);
	L[7] = fma(M[2], D[26], L[7]);
	L[6] = fma(M[3], D[25], L[6]);
	L[7] = fma(M[3], D[27], L[7]);
	L[6] = fma(M[4], D[27] * float(5.000000000e-01), L[6]);
	L[7] = fma(M[4], D[30] * float(5.000000000e-01), L[7]);
	L[6] = fma(M[5], D[28], L[6]);
	L[6] = fma(M[6], D[29] * float(5.000000000e-01), L[6]);
	L[7] = fma(M[5], D[31], L[7]);
	L[7] = fma(M[6], D[32] * float(5.000000000e-01), L[7]);
	L[8] = fma(M[1], D[24] * float(5.000000000e-01), L[8]);
	L[8] = fma(M[2], D[27], L[8]);
	L[9] = fma(M[1], D[25] * float(5.000000000e-01), L[9]);
	L[8] = fma(M[3], D[28], L[8]);
	L[9] = fma(M[2], D[28], L[9]);
	L[8] = fma(M[4], D[31] * float(5.000000000e-01), L[8]);
	L[9] = fma(M[3], D[29], L[9]);
	L[8] = fma(M[5], D[32], L[8]);
	L[9] = fma(M[4], D[32] * float(5.000000000e-01), L[9]);
	L[8] = fma(M[6], D[33] * float(5.000000000e-01), L[8]);
	L[9] = fma(M[5], D[33], L[9]);
	L[9] = fma(M[6], D[34] * float(5.000000000e-01), L[9]);
	flop += 306;
	return flop;
}

template<class DOUBLE, class SINGLE> // 401 / 251351
inline int multipole_interaction(expansion<DOUBLE> &L, const SINGLE &M, vect<SINGLE> dX, bool ewald = false) { // 390 / 47301
	static const expansion_factors<SINGLE> expansion_factor;
	int flop = 0;
	expansion<SINGLE> D;
	flop += ewald ? green_ewald(D,dX) : green_direct(D,dX);
	// 175
	for (int i = 0; i < LP; i++) {
		L[i] = fma( M, D[i], L[i]);
	}
	flop += 2 * LP;
	return flop;
}

template<class DOUBLE, class SINGLE> // 516 / 251466
CUDA_EXPORT inline int multipole_interaction(vect<DOUBLE> &g, DOUBLE &phi, const multipole<SINGLE> &M, vect<SINGLE> dX, bool ewald = false) { // 517 / 47428
	int flop = 216;
	expansion<SINGLE> D;
	flop += ewald ? green_ewald(D,dX) : green_direct(D,dX);
	SINGLE L[4];

	for (int i = 0; i < 4; i++) {
		L[i] = M[0] * D[i];
	}
	flop += 4;
	L[0] = fma(M[1], D[4] * float(5.000000000e-01), L[0]);
	L[1] = fma(M[1], D[10] * float(5.000000000e-01), L[1]);
	L[0] = fma(-M[7], D[10] * float(1.666666670e-01), L[0]);
	L[1] = fma(-M[7], D[20] * float(1.666666670e-01), L[1]);
	L[0] = fma(-M[8], D[11] * float(5.000000000e-01), L[0]);
	L[1] = fma(-M[8], D[21] * float(5.000000000e-01), L[1]);
	L[0] = fma(-M[9], D[12] * float(5.000000000e-01), L[0]);
	L[1] = fma(-M[9], D[22] * float(5.000000000e-01), L[1]);
	L[0] = fma(M[2], D[5], L[0]);
	L[1] = fma(M[2], D[11], L[1]);
	L[0] = fma(-M[10], D[13] * float(5.000000000e-01), L[0]);
	L[1] = fma(-M[10], D[23] * float(5.000000000e-01), L[1]);
	L[0] = fma(-M[11], D[14], L[0]);
	L[1] = fma(-M[11], D[24], L[1]);
	L[0] = fma(M[3], D[6], L[0]);
	L[1] = fma(M[3], D[12], L[1]);
	L[0] = fma(-M[12], D[15] * float(5.000000000e-01), L[0]);
	L[1] = fma(-M[12], D[25] * float(5.000000000e-01), L[1]);
	L[0] = fma(M[4], D[7] * float(5.000000000e-01), L[0]);
	L[1] = fma(M[4], D[13] * float(5.000000000e-01), L[1]);
	L[0] = fma(-M[13], D[16] * float(1.666666670e-01), L[0]);
	L[1] = fma(-M[13], D[26] * float(1.666666670e-01), L[1]);
	L[0] = fma(-M[14], D[17] * float(5.000000000e-01), L[0]);
	L[1] = fma(-M[14], D[27] * float(5.000000000e-01), L[1]);
	L[0] = fma(M[5], D[8], L[0]);
	L[1] = fma(M[5], D[14], L[1]);
	L[0] = fma(-M[15], D[18] * float(5.000000000e-01), L[0]);
	L[1] = fma(-M[15], D[28] * float(5.000000000e-01), L[1]);
	L[0] = fma(M[6], D[9] * float(5.000000000e-01), L[0]);
	L[1] = fma(M[6], D[15] * float(5.000000000e-01), L[1]);
	L[0] = fma(-M[16], D[19] * float(1.666666670e-01), L[0]);
	L[1] = fma(-M[16], D[29] * float(1.666666670e-01), L[1]);
	L[2] = fma(M[1], D[11] * float(5.000000000e-01), L[2]);
	L[3] = fma(M[1], D[12] * float(5.000000000e-01), L[3]);
	L[2] = fma(-M[7], D[21] * float(1.666666670e-01), L[2]);
	L[3] = fma(-M[7], D[22] * float(1.666666670e-01), L[3]);
	L[2] = fma(-M[8], D[23] * float(5.000000000e-01), L[2]);
	L[3] = fma(-M[8], D[24] * float(5.000000000e-01), L[3]);
	L[2] = fma(-M[9], D[24] * float(5.000000000e-01), L[2]);
	L[3] = fma(-M[9], D[25] * float(5.000000000e-01), L[3]);
	L[2] = fma(M[2], D[13], L[2]);
	L[3] = fma(M[2], D[14], L[3]);
	L[2] = fma(-M[10], D[26] * float(5.000000000e-01), L[2]);
	L[3] = fma(-M[10], D[27] * float(5.000000000e-01), L[3]);
	L[2] = fma(-M[11], D[27], L[2]);
	L[3] = fma(-M[11], D[28], L[3]);
	L[2] = fma(M[3], D[14], L[2]);
	L[3] = fma(M[3], D[15], L[3]);
	L[2] = fma(-M[12], D[28] * float(5.000000000e-01), L[2]);
	L[3] = fma(-M[12], D[29] * float(5.000000000e-01), L[3]);
	L[2] = fma(M[4], D[16] * float(5.000000000e-01), L[2]);
	L[3] = fma(M[4], D[17] * float(5.000000000e-01), L[3]);
	L[2] = fma(-M[13], D[30] * float(1.666666670e-01), L[2]);
	L[3] = fma(-M[13], D[31] * float(1.666666670e-01), L[3]);
	L[2] = fma(-M[14], D[31] * float(5.000000000e-01), L[2]);
	L[3] = fma(-M[14], D[32] * float(5.000000000e-01), L[3]);
	L[2] = fma(M[5], D[17], L[2]);
	L[3] = fma(M[5], D[18], L[3]);
	L[2] = fma(-M[15], D[32] * float(5.000000000e-01), L[2]);
	L[3] = fma(-M[15], D[33] * float(5.000000000e-01), L[3]);
	L[2] = fma(M[6], D[18] * float(5.000000000e-01), L[2]);
	L[3] = fma(M[6], D[19] * float(5.000000000e-01), L[3]);
	L[2] = fma(-M[16], D[33] * float(1.666666670e-01), L[2]);
	L[3] = fma(-M[16], D[34] * float(1.666666670e-01), L[3]);
	phi = L[0];
	for (int dim = 0; dim < NDIM; dim++) {
		g[dim] = -L[1 + dim];
	}
	flop += 3;
	return flop;
}
