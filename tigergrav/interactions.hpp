#pragma once

#include <tigergrav/green.hpp>

#ifdef __CUDA_ARCH__
__device__ const cuda_ewald_const& cuda_get_const();
#endif
// 43009,703
template<class DOUBLE, class SINGLE> // 986 // 251936
CUDA_EXPORT inline int multipole_interaction(expansion<DOUBLE> &L, const multipole<SINGLE> &M, vect<SINGLE> dX, bool ewald, bool do_phi) { // 670/700 + 418 * NREAL + 50 * NFOUR
	int flop = 0;
	expansion<SINGLE> D;
	flop += ewald ? green_ewald(D, dX) : green_direct(D, dX);
	auto &L0 = L;
	for (int i = 0; i < LP; i++) {
		L[i] = fma(M[0], D[i], L[i]);
	}
	flop += 2 * LP;
	// 18 + 84 * 2 + 30 + 15 = 231
	// 16 * 2 + 10 + 6 = = 48
	const auto halfD11 = float(0.5) * D[11];
	const auto halfD12 = float(0.5) * D[12];
	const auto halfD13 = float(0.5) * D[13];
	const auto halfD15 = float(0.5) * D[15];
	const auto halfD17 = float(0.5) * D[17];
	const auto halfD18 = float(0.5) * D[18];						// 6
	if (do_phi) {
		L[0] = fma(M[1], D[4] * float(5.000000000e-01), L[0]);
		L[0] = fma(-M[7], D[10] * float(1.666666670e-01), L[0]);    // 6
		L[0] = fma(-M[8], halfD11, L[0]);
		L[0] = fma(-M[9], halfD12, L[0]);
		L[0] = fma(M[2], D[5], L[0]);
		L[0] = fma(-M[10], halfD13, L[0]);
		L[0] = fma(-M[11], D[14], L[0]);
		L[0] = fma(M[3], D[6], L[0]);
		L[0] = fma(-M[12], halfD15, L[0]);                          // 14
		L[0] = fma(M[4], D[7] * float(5.000000000e-01), L[0]);
		L[0] = fma(-M[13], D[16] * float(1.666666670e-01), L[0]);   // 6
		L[0] = fma(-M[14], halfD17, L[0]);
		L[0] = fma(M[5], D[8], L[0]);
		L[0] = fma(-M[15], halfD18, L[0]);                          // 6
		L[0] = fma(M[6], D[9] * float(5.000000000e-01), L[0]);
		L[0] = fma(-M[16], D[19] * float(1.666666670e-01), L[0]);   // 6
	}
	const auto halfD21 = float(0.5) * D[21];
	const auto halfD22 = float(0.5) * D[22];
	const auto halfD23 = float(0.5) * D[23];
	const auto halfD24 = float(0.5) * D[24];
	const auto halfD25 = float(0.5) * D[25];
	const auto halfD26 = float(0.5) * D[26];
	const auto halfD27 = float(0.5) * D[27];
	const auto halfD28 = float(0.5) * D[28];
	const auto halfD29 = float(0.5) * D[29];
	const auto halfD31 = float(0.5) * D[31];
	const auto halfD32 = float(0.5) * D[32];
	const auto halfD33 = float(0.5) * D[33];
	L[1] = fma(M[1], D[10] * float(5.000000000e-01), L[1]);
	L[1] = fma(-M[7], D[20] * float(1.666666670e-01), L[1]);
	L[1] = fma(-M[8], halfD21, L[1]);
	L[1] = fma(-M[9], halfD22, L[1]);
	L[1] = fma(M[2], D[11], L[1]);
	L[1] = fma(-M[10], halfD23, L[1]);
	L[1] = fma(-M[11], D[24], L[1]);
	L[1] = fma(M[3], D[12], L[1]);
	L[1] = fma(-M[12], halfD25, L[1]);
	L[1] = fma(M[4], halfD13, L[1]);
	L[1] = fma(-M[13], D[26] * float(1.666666670e-01), L[1]);
	L[1] = fma(-M[14], halfD27, L[1]);
	L[1] = fma(M[5], D[14], L[1]);
	L[1] = fma(-M[15], halfD28, L[1]);
	L[1] = fma(M[6], halfD15, L[1]);
	L[1] = fma(-M[16], D[29] * float(1.666666670e-01), L[1]);
	L[2] = fma(M[1], halfD11, L[2]);
	L[3] = fma(M[1], halfD12, L[3]);
	L[2] = fma(-M[7], D[21] * float(1.666666670e-01), L[2]);
	L[3] = fma(-M[7], D[22] * float(1.666666670e-01), L[3]);
	L[2] = fma(-M[8], halfD23, L[2]);
	L[3] = fma(-M[8], halfD24, L[3]);
	L[2] = fma(-M[9], halfD24, L[2]);
	L[3] = fma(-M[9], halfD25, L[3]);
	L[2] = fma(M[2], D[13], L[2]);
	L[3] = fma(M[2], D[14], L[3]);
	L[2] = fma(-M[10], halfD26, L[2]);
	L[3] = fma(-M[10], halfD27, L[3]);
	L[2] = fma(-M[11], D[27], L[2]);
	L[3] = fma(-M[11], D[28], L[3]);
	L[2] = fma(M[3], D[14], L[2]);
	L[3] = fma(M[3], D[15], L[3]);
	L[2] = fma(-M[12], halfD28, L[2]);
	L[3] = fma(-M[12], halfD29, L[3]);
	L[2] = fma(M[4], D[16] * float(5.000000000e-01), L[2]);
	L[3] = fma(M[4], halfD17, L[3]);
	L[2] = fma(-M[13], D[30] * float(1.666666670e-01), L[2]);
	L[3] = fma(-M[13], D[31] * float(1.666666670e-01), L[3]);
	L[2] = fma(-M[14], halfD31, L[2]);
	L[3] = fma(-M[14], halfD32, L[3]);
	L[2] = fma(M[5], D[17], L[2]);
	L[3] = fma(M[5], D[18], L[3]);
	L[2] = fma(-M[15], halfD32, L[2]);
	L[3] = fma(-M[15], halfD33, L[3]);
	L[2] = fma(M[6], halfD18, L[2]);
	L[3] = fma(M[6], D[19] * float(5.000000000e-01), L[3]);
	L[2] = fma(-M[16], D[33] * float(1.666666670e-01), L[2]);
	L[3] = fma(-M[16], D[34] * float(1.666666670e-01), L[3]);
	L[4] = fma(M[1], D[20] * float(5.000000000e-01), L[4]);
	L[5] = fma(M[1], halfD21, L[5]);
	L[4] = fma(M[2], D[21], L[4]);
	L[5] = fma(M[2], D[23], L[5]);
	L[4] = fma(M[3], D[22], L[4]);
	L[5] = fma(M[3], D[24], L[5]);
	L[4] = fma(M[4], halfD23, L[4]);
	L[5] = fma(M[4], halfD26, L[5]);
	L[4] = fma(M[5], D[24], L[4]);
	L[5] = fma(M[5], D[27], L[5]);
	L[4] = fma(M[6], halfD25, L[4]);
	L[5] = fma(M[6], halfD28, L[5]);
	L[6] = fma(M[1], halfD22, L[6]);
	L[7] = fma(M[1], halfD23, L[7]);
	L[6] = fma(M[2], D[24], L[6]);
	L[7] = fma(M[2], D[26], L[7]);
	L[6] = fma(M[3], D[25], L[6]);
	L[7] = fma(M[3], D[27], L[7]);
	L[6] = fma(M[4], halfD27, L[6]);
	L[7] = fma(M[4], D[30] * float(5.000000000e-01), L[7]);
	L[6] = fma(M[5], D[28], L[6]);
	L[6] = fma(M[6], halfD29, L[6]);
	L[7] = fma(M[5], D[31], L[7]);
	L[7] = fma(M[6], halfD32, L[7]);
	L[8] = fma(M[1], halfD24, L[8]);
	L[8] = fma(M[2], D[27], L[8]);
	L[9] = fma(M[1], halfD25, L[9]);
	L[8] = fma(M[3], D[28], L[8]);
	L[9] = fma(M[2], D[28], L[9]);
	L[8] = fma(M[4], halfD31, L[8]);
	L[9] = fma(M[3], D[29], L[9]);
	L[8] = fma(M[5], D[32], L[8]);
	L[9] = fma(M[4], halfD32, L[9]);
	L[8] = fma(M[6], halfD33, L[8]);
	L[9] = fma(M[5], D[33], L[9]);
	L[9] = fma(M[6], D[34] * float(5.000000000e-01), L[9]);
	flop += do_phi ? 349 : 301;
	return flop;
}

template<class DOUBLE, class SINGLE> // 401 / 251351
inline int multipole_interaction(expansion<DOUBLE> &L, const SINGLE &M, vect<SINGLE> dX, bool ewald, bool do_phi) { // 390 / 47301
	static const expansion_factors<SINGLE> expansion_factor;
	int flop = 0;
	expansion<SINGLE> D;
	flop += ewald ? green_ewald(D, dX) : green_direct(D, dX);
	// 175
	for (int i = 0; i < LP; i++) {
		L[i] = fma(M, D[i], L[i]);
	}
	flop += 2 * LP;
	return flop;
}

template<class DOUBLE, class SINGLE> // 516 / 251466
CUDA_EXPORT inline int multipole_interaction(vect<DOUBLE> &g, DOUBLE &phi, const multipole<SINGLE> &M, vect<SINGLE> dX, bool ewald, bool do_phi) { // 517 / 47428
	int flop = do_phi ? 216 : 172;
	expansion<SINGLE> D;
	flop += ewald ? green_ewald(D, dX) : green_direct(D, dX);
	SINGLE L[4];

	for (int i = 0; i < 4; i++) {
		L[i] = M[0] * D[i];
	}
	const auto halfD11 = float(0.5) * D[11];
	const auto halfD12 = float(0.5) * D[12];
	const auto halfD13 = float(0.5) * D[13];
	const auto halfD15 = float(0.5) * D[15];
	const auto halfD17 = float(0.5) * D[17];
	const auto halfD18 = float(0.5) * D[18];
	flop += 4;
	if (do_phi) {
		L[0] = fma(M[1], D[4] * float(5.000000000e-01), L[0]);
		L[0] = fma(-M[7], D[10] * float(1.666666670e-01), L[0]);
		L[0] = fma(-M[8], halfD11, L[0]);
		L[0] = fma(-M[9], halfD12, L[0]);
		L[0] = fma(M[2], D[5], L[0]);
		L[0] = fma(-M[10], halfD13, L[0]);
		L[0] = fma(-M[11], D[14], L[0]);
		L[0] = fma(M[3], D[6], L[0]);
		L[0] = fma(-M[12], halfD15, L[0]);
		L[0] = fma(M[4], D[7] * float(5.000000000e-01), L[0]);
		L[0] = fma(-M[13], D[16] * float(1.666666670e-01), L[0]);
		L[0] = fma(-M[14], halfD17, L[0]);
		L[0] = fma(M[5], D[8], L[0]);
		L[0] = fma(-M[15], halfD18, L[0]);
		L[0] = fma(M[6], D[9] * float(5.000000000e-01), L[0]);
		L[0] = fma(-M[16], D[19] * float(1.666666670e-01), L[0]);
	}
	const auto halfD21 = float(0.5) * D[21];
	const auto halfD22 = float(0.5) * D[22];
	const auto halfD23 = float(0.5) * D[23];
	const auto halfD24 = float(0.5) * D[24];
	const auto halfD25 = float(0.5) * D[25];
	const auto halfD26 = float(0.5) * D[26];
	const auto halfD27 = float(0.5) * D[27];
	const auto halfD28 = float(0.5) * D[28];
	const auto halfD29 = float(0.5) * D[29];
	const auto halfD31 = float(0.5) * D[31];
	const auto halfD32 = float(0.5) * D[32];
	const auto halfD33 = float(0.5) * D[33];
	L[1] = fma(M[1], D[10] * float(5.000000000e-01), L[1]);
	L[1] = fma(-M[7], D[20] * float(1.666666670e-01), L[1]);
	L[1] = fma(-M[8], halfD21, L[1]);
	L[1] = fma(-M[9], halfD22, L[1]);
	L[1] = fma(M[2], D[11], L[1]);
	L[1] = fma(-M[10], halfD23, L[1]);
	L[1] = fma(-M[11], D[24], L[1]);
	L[1] = fma(M[3], D[12], L[1]);
	L[1] = fma(-M[12], halfD25, L[1]);
	L[1] = fma(M[4], halfD13, L[1]);
	L[1] = fma(-M[13], D[26] * float(1.666666670e-01), L[1]);
	L[1] = fma(-M[14], halfD27, L[1]);
	L[1] = fma(M[5], D[14], L[1]);
	L[1] = fma(-M[15], halfD28, L[1]);
	L[1] = fma(M[6], halfD15, L[1]);
	L[1] = fma(-M[16], D[29] * float(1.666666670e-01), L[1]);
	L[2] = fma(M[1], halfD11, L[2]);
	L[3] = fma(M[1], halfD12, L[3]);
	L[2] = fma(-M[7], D[21] * float(1.666666670e-01), L[2]);
	L[3] = fma(-M[7], D[22] * float(1.666666670e-01), L[3]);
	L[2] = fma(-M[8], halfD23, L[2]);
	L[3] = fma(-M[8], halfD24, L[3]);
	L[2] = fma(-M[9], halfD24, L[2]);
	L[3] = fma(-M[9], halfD25, L[3]);
	L[2] = fma(M[2], D[13], L[2]);
	L[3] = fma(M[2], D[14], L[3]);
	L[2] = fma(-M[10], halfD26, L[2]);
	L[3] = fma(-M[10], halfD27, L[3]);
	L[2] = fma(-M[11], D[27], L[2]);
	L[3] = fma(-M[11], D[28], L[3]);
	L[2] = fma(M[3], D[14], L[2]);
	L[3] = fma(M[3], D[15], L[3]);
	L[2] = fma(-M[12], halfD28, L[2]);
	L[3] = fma(-M[12], halfD29, L[3]);
	L[2] = fma(M[4], D[16] * float(5.000000000e-01), L[2]);
	L[3] = fma(M[4], halfD17, L[3]);
	L[2] = fma(-M[13], D[30] * float(1.666666670e-01), L[2]);
	L[3] = fma(-M[13], D[31] * float(1.666666670e-01), L[3]);
	L[2] = fma(-M[14], halfD31, L[2]);
	L[3] = fma(-M[14], halfD32, L[3]);
	L[2] = fma(M[5], D[17], L[2]);
	L[3] = fma(M[5], D[18], L[3]);
	L[2] = fma(-M[15], halfD32, L[2]);
	L[3] = fma(-M[15], halfD33, L[3]);
	L[2] = fma(M[6], halfD18, L[2]);
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
