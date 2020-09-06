#pragma once

#include <tigergrav/green.hpp>

#ifdef __CUDA_ARCH__
__device__ const cuda_ewald_const& cuda_get_const();
#endif
// 43009,703
template<class DOUBLE, class SINGLE> // 986 // 251936
CUDA_EXPORT inline void multipole_interaction(expansion<DOUBLE> &Lacc, const multipole<SINGLE> &M2, vect<SINGLE> dX, bool ewald) { // 670/700 + 418 * NREAL + 50 * NFOUR
#ifdef __CUDA_ARCH__
	const expansion<float>& expansion_factor = cuda_get_const().exp_factors;
#else
	static const float efs[LP + 1] = { 1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 5.00000000e-01, 1.00000000e+00, 1.00000000e+00,
			5.00000000e-01, 1.00000000e+00, 5.00000000e-01, 1.66666667e-01, 5.00000000e-01, 5.00000000e-01, 5.00000000e-01, 1.00000000e+00, 5.00000000e-01,
			1.66666667e-01, 5.00000000e-01, 5.00000000e-01, 1.66666667e-01, 4.16666667e-02, 1.66666667e-01, 1.66666667e-01, 2.50000000e-01, 5.00000000e-01,
			2.50000000e-01, 1.66666667e-01, 5.00000000e-01, 5.00000000e-01, 1.66666667e-01, 4.16666667e-02, 1.66666667e-01, 2.50000000e-01, 1.66666667e-01,
			4.16666667e-02, 0.0 };
	const expansion<float> &expansion_factor = *reinterpret_cast<const expansion<float>*>(efs);
#endif

	const expansion<SINGLE> D = ewald ? green_ewald(dX) : green_direct(dX);
	expansion<SINGLE> L;

	// 760
	auto& L0 = L;
	L0() = M2() * D();																// 5
	for (int a = 0; a < 3; a++) {
		auto &La = L(a);
		La = M2() * D(a);
	}

	for (int a = 0; a < 3; a++) {
		for (int b = a; b < 3; b++) {
			auto &Lab = L(a, b);
			Lab = M2() * D(a, b);												// 30
		}
	}

	for (int a = 0; a < 3; a++) {
		for (int b = a; b < 3; b++) {
			for (int c = b; c < 3; c++) {
				auto &Labc = L(a, b, c);
				Labc = M2() * D(a, b, c);										// 50
			}
		}
	}
	for (int a = 0; a < 3; a++) {
		for (int b = a; b < 3; b++) {
			for (int c = b; c < 3; c++) {
				for (int d = c; d < 3; d++) {
					auto &Labcd = L(a, b, c, d);
					Labcd = M2() * D(a, b, c, d);								// 75
				}
			}
		}
	}

	for (int a = 0; a < 3; a++) {
		for (int b = a; b < 3; b++) {
			L0() = fma(M2(a, b) * D(a, b), expansion_factor(a, b), L0());						// 36
			for (int c = b; c < 3; c++) {
				L0() =fma( -M2(a, b, c) * D(a, b, c), expansion_factor(a, b, c), L0());			// 60
			}
		}
	}

	for (int a = 0; a < 3; a++) {
		auto &La = L(a);
		for (int b = 0; b < 3; b++) {
			for (int c = b; c < 3; c++) {
				La = fma(M2(c, b) * D(a, b, c),expansion_factor(c, b), La);				// 108
				for (int d = c; d < 3; d++) {
					La = fma(-M2(b, c, d) * D(a, b, c, d), expansion_factor(b, c, d), La);	//180
				}
			}
		}
	}

	for (int a = 0; a < 3; a++) {
		for (int b = a; b < 3; b++) {
			auto &Lab = L(a, b);
			for (int c = 0; c < 3; c++) {
				for (int d = c; d < 3; d++) {
					Lab = fma(M2(c, d) * D(a, b, c, d),  expansion_factor(c, d), Lab);	 // 216
				}
			}
		}
	}
	for( int i = 0; i < LP; i++) {
		Lacc[i] += L[i];
	}
}

template<class DOUBLE, class SINGLE> // 401 / 251351
inline void multipole_interaction(expansion<DOUBLE> &L, const SINGLE &M, vect<SINGLE> dX, bool ewald = false) { // 390 / 47301
	static const expansion_factors<SINGLE> expansion_factor;
	const expansion<SINGLE> D = ewald ? green_ewald(dX) : green_direct(dX);
	// 175
	auto &L0 = L();
	L0 += M * D();													// 5
	for (int a = 0; a < 3; a++) {
		auto &La = L(a);
		La += M * D(a);												// 15
		for (int b = a; b < 3; b++) {
			auto &Lab = L(a, b);
			Lab += M * D(a, b);										// 30
			for (int c = b; c < 3; c++) {
				auto &Labc = L(a, b, c);
				Labc += M * D(a, b, c);								// 50
				for (int d = c; d < 3; d++) {
					auto &Labcd = L(a, b, c, d);
					Labcd += M * D(a, b, c, d);						// 75
				}
			}
		}
	}
}

template<class DOUBLE, class SINGLE> // 516 / 251466
CUDA_EXPORT inline void multipole_interaction(vect<DOUBLE> &g, DOUBLE &phi, const multipole<SINGLE> &M, vect<SINGLE> dX, bool ewald = false) { // 517 / 47428
	static const float efs[LP + 1] = { 1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 5.00000000e-01, 1.00000000e+00, 1.00000000e+00,
			5.00000000e-01, 1.00000000e+00, 5.00000000e-01, 1.66666667e-01, 5.00000000e-01, 5.00000000e-01, 5.00000000e-01, 1.00000000e+00, 5.00000000e-01,
			1.66666667e-01, 5.00000000e-01, 5.00000000e-01, 1.66666667e-01, 4.16666667e-02, 1.66666667e-01, 1.66666667e-01, 2.50000000e-01, 5.00000000e-01,
			2.50000000e-01, 1.66666667e-01, 5.00000000e-01, 5.00000000e-01, 1.66666667e-01, 4.16666667e-02, 1.66666667e-01, 2.50000000e-01, 1.66666667e-01,
			4.16666667e-02, 0.0 };
	const expansion<float> &expansion_factor = *reinterpret_cast<const expansion<float>*>(efs);
	const expansion<SINGLE> D = ewald ? green_ewald(dX) : green_direct(dX);

	//290
	auto &ffirst = phi;
	ffirst = M() * D();																// 5
	for (int a = 0; a < 3; a++) {
		for (int b = a; b < 3; b++) {
			ffirst += M(a, b) * D(a, b) * expansion_factor(a, b);					// 36
		}
	}
	for (int a = 0; a < 3; a++) {
		for (int b = 0; b < 3; b++) {
			for (int c = b; c < 3; c++) {
				ffirst -= M(a, b, c) * D(a, b, c) * expansion_factor(a, b, c);		// 60
			}
		}
	}
	g = DOUBLE(0);
	for (int a = 0; a < 3; a++) {
		g[a] -= M() * D(a);													// 15
	}
	for (int a = 0; a < 3; a++) {
		for (int b = 0; b < 3; b++) {
			for (int c = b; c < 3; c++) {
				g[a] -= M(c, b) * D(a, b, c) * expansion_factor(c, b);		// 108
			}
		}
	}
	for (int a = 0; a < 3; a++) {
		for (int b = 0; b < 3; b++) {
			for (int c = b; c < 3; c++) {
				for (int d = c; d < 3; d++) {
					auto &fseconda = g[a];
					fseconda += M(c, b, d) * D(a, b, c, d) * expansion_factor(b, c, d); // 180
				}
			}
		}
	}
}
