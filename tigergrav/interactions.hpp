#pragma once

#include <tigergrav/green.hpp>


// 43009,703
template<class DOUBLE, class SINGLE> // 986 // 251936
CUDA_EXPORT inline void multipole_interaction(expansion<DOUBLE> &L, const multipole<SINGLE> &M2, vect<SINGLE> dX, bool ewald) { // 670/700 + 418 * NREAL + 50 * NFOUR
	static const float efs[LP+1] = { 1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 5.00000000e-01, 1.00000000e+00, 1.00000000e+00,
			5.00000000e-01, 1.00000000e+00, 5.00000000e-01, 1.66666667e-01, 5.00000000e-01, 5.00000000e-01, 5.00000000e-01, 1.00000000e+00, 5.00000000e-01,
			1.66666667e-01, 5.00000000e-01, 5.00000000e-01, 1.66666667e-01, 4.16666667e-02, 1.66666667e-01, 1.66666667e-01, 2.50000000e-01, 5.00000000e-01,
			2.50000000e-01, 1.66666667e-01, 5.00000000e-01, 5.00000000e-01, 1.66666667e-01, 4.16666667e-02, 1.66666667e-01, 2.50000000e-01, 1.66666667e-01,
			4.16666667e-02, 0.0 };
	const expansion<SINGLE>& expansion_factor = *reinterpret_cast<const expansion<SINGLE>*>(efs);
	expansion<SINGLE> D;
	if (ewald) {
		D = green_ewald(dX);		// 251176
	} else {
		D = green_direct(dX);        // 226
	}

	// 760
	auto &L0 = L();
	L0 += M2() * D();																// 5
	for (int a = 0; a < 3; a++) {
		for (int b = a; b < 3; b++) {
			L0 += M2(a, b) * D(a, b) * expansion_factor(a, b);						// 36
			for (int c = b; c < 3; c++) {
				L0 -= M2(a, b, c) * D(a, b, c) * expansion_factor(a, b, c);			// 60
			}
		}
	}
	for (int a = 0; a < 3; a++) {
		auto &La = L(a);
		La += M2() * D(a);
		for (int b = 0; b < 3; b++) {
			for (int c = b; c < 3; c++) {
				La += M2(c, b) * D(a, b, c) * expansion_factor(c, b);				// 108
				for (int d = c; d < 3; d++) {
					La -= M2(b, c, d) * D(a, b, c, d) * expansion_factor(b, c, d);	//180
				}
			}
		}
	}

	for (int a = 0; a < 3; a++) {
		for (int b = a; b < 3; b++) {
			auto &Lab = L(a, b);
			Lab += M2() * D(a, b);												// 30
			for (int c = 0; c < 3; c++) {
				for (int d = c; d < 3; d++) {
					Lab += M2(c, d) * D(a, b, c, d) * expansion_factor(c, d);	 // 216
				}
			}
		}
	}

	for (int a = 0; a < 3; a++) {
		for (int b = a; b < 3; b++) {
			for (int c = b; c < 3; c++) {
				auto &Labc = L(a, b, c);
				Labc += M2() * D(a, b, c);										// 50
			}
		}
	}
	for (int a = 0; a < 3; a++) {
		for (int b = a; b < 3; b++) {
			for (int c = b; c < 3; c++) {
				for (int d = c; d < 3; d++) {
					auto &Labcd = L(a, b, c, d);
					Labcd += M2() * D(a, b, c, d);								// 75
				}
			}
		}
	}
}

template<class DOUBLE, class SINGLE> // 401 / 251351
inline void multipole_interaction(expansion<DOUBLE> &L, const SINGLE &M, vect<SINGLE> dX, bool ewald = false) { // 390 / 47301
	static const expansion_factors<SINGLE> expansion_factor;
	expansion<SINGLE> D;
	if (ewald) {
		D = green_ewald(dX);		// 251175
	} else {
		D = green_direct(dX);          // 226
	}

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
inline void multipole_interaction(std::pair<DOUBLE, vect<DOUBLE>> &f, const multipole<SINGLE> &M, vect<SINGLE> dX, bool ewald = false) { // 517 / 47428
	static const expansion_factors<SINGLE> expansion_factor;
	expansion<SINGLE> D;
	if (ewald) {
		D = green_ewald(dX);				// 251176
	} else {
		D = green_direct(dX);				// 226
	}

	//290
	auto &ffirst = f.first;
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
	f.second = DOUBLE(0);
	for (int a = 0; a < 3; a++) {
		f.second[a] -= M() * D(a);													// 15
	}
	for (int a = 0; a < 3; a++) {
		for (int b = 0; b < 3; b++) {
			for (int c = b; c < 3; c++) {
				f.second[a] -= M(c, b) * D(a, b, c) * expansion_factor(c, b);		// 108
			}
		}
	}
	for (int a = 0; a < 3; a++) {
		for (int b = 0; b < 3; b++) {
			for (int c = b; c < 3; c++) {
				for (int d = c; d < 3; d++) {
					auto &fseconda = f.second[a];
					fseconda += M(c, b, d) * D(a, b, c, d) * expansion_factor(b, c, d); // 180
				}
			}
		}
	}
}
