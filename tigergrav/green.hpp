#pragma once

constexpr float EWALD_REAL_N2 = 17;
constexpr float EWALD_FOUR_N2 = 9;

#include <tigergrav/expansion.hpp>

struct ewald_indices: public std::vector<vect<float>> {
	ewald_indices(int n2max) {
		const int nmax = sqrt(n2max) + 1;
		vect<float> h;
		for (int i = -nmax; i <= nmax; i++) {
			for (int j = -nmax; j <= nmax; j++) {
				for (int k = -nmax; k <= nmax; k++) {
					if (i * i + j * j + k * k <= n2max) {
						h[0] = i;
						h[1] = j;
						h[2] = k;
						this->push_back(h);
					}
				}
			}
		}
		std::sort(this->begin(), this->end(), [](vect<float> &a, vect<float> &b) {
			return a.dot(a) > b.dot(b);
		});
	}
};

struct periodic_parts: public std::vector<expansion<float>> {
	periodic_parts() {
		static const ewald_indices indices(EWALD_FOUR_N2);
		for (auto i : indices) {
			vect<float> h = i;
			const float h2 = h.dot(h);                     // 5 OP
			expansion<float> D;
			D = 0.0;
			if (h2 > 0) {
				const float hinv = 1.0 / h2;                  // 1 OP
				const float c0 = 1.0 / h2 * exp(-M_PI * M_PI * h2 / 4.0);
				D() = -(1.0 / M_PI) * c0;
				for (int a = 0; a < NDIM; a++) {
					D(a) = 2.0 * h[a] * c0;
					for (int b = 0; b <= a; b++) {
						D(a, b) = 4.0 * M_PI * h[a] * h[b] * c0;
						for (int c = 0; c <= b; c++) {
							D(a, b, c) = -8.0 * M_PI * M_PI * h[a] * h[b] * h[c] * c0;
							for (int d = 0; d <= c; d++) {
								D(a, b, c, d) = -16.0 * M_PI * M_PI * M_PI * h[a] * h[b] * h[c] * h[d] * c0;
							}
						}
					}

				}
			}
			this->push_back(D);
		}
	}
};


template<class SINGLE>  // 167
void green_deriv_direct(expansion<SINGLE> &D, const SINGLE &d0, const SINGLE &d1, const SINGLE &d2, const SINGLE &d3, const SINGLE &d4,
		const vect<SINGLE> &dx) {
	static const SINGLE two(2.0);

	D() = d0;													// 1
	for (int a = 0; a < NDIM; a++) {
		auto &Da = D(a);
		Da = dx[a] * d1;										// 3
		for (int b = 0; b <= a; b++) {
			auto &Dab = D(a, b);
			const auto dxadxb = dx[a] * dx[b];					// 6
			Dab = dxadxb * d2;									// 6
			for (int c = 0; c <= b; c++) {
				auto &Dabc = D(a, b, c);
				const auto dxadxbdxc = dxadxb * dx[c];			// 10
				Dabc = dxadxbdxc * d3;							// 10
				for (int d = 0; d <= c; d++) {
					auto &Dabcd = D(a, b, c, d);
					Dabcd = dxadxbdxc * dx[d] * d4;				// 30
				}
			}
		}
	}
	for (int a = 0; a < NDIM; a++) {
		auto &Daa = D(a, a);
		auto &Daaa = D(a, a, a);
		auto &Daaaa = D(a, a, a, a);
		Daa += d1;												// 3
		Daaa = fmadd(dx[a], d2, Daaa);							// 3
		Daaaa = fmadd(dx[a] * dx[a], d3, Daaaa);				// 6
		Daaaa = fmadd(two, d2, Daaaa);							// 3
		for (int b = 0; b <= a; b++) {
			auto &Daab = D(a, a, b);
			auto &Dabb = D(a, b, b);
			auto &Daaab = D(a, a, a, b);
			auto &Daabb = D(a, a, b, b);
			auto &Dabbb = D(a, b, b, b);
			const auto dxadxb = dx[a] * dx[b];					// 6
			Daab = fmadd(dx[b], d2, Daab);						// 6
			Dabb = fmadd(dx[a], d2, Dabb);						// 6
			Daaab = fmadd(dxadxb, d3, Daaab);					// 6
			Dabbb = fmadd(dxadxb, d3, Dabbb);					// 6
			Daabb += d2;										// 6
			for (int c = 0; c <= b; c++) {
				auto &Daabc = D(a, a, b, c);
				auto &Dabcc = D(a, b, c, c);
				auto &Dabbc = D(a, b, b, c);
				Daabc = fmadd(dx[b], dx[c] * d3, Daabc);		// 20
				Dabcc = fmadd(dxadxb, d3, Dabcc);				// 10
				Dabbc = fmadd(dx[a], dx[c] * d3, Dabbc);		// 20
			}
		}
	}

}

template<class DOUBLE, class SINGLE>  // 576
void green_deriv_ewald(expansion<DOUBLE> &D, const SINGLE &d0, const SINGLE &d1, const SINGLE &d2, const SINGLE &d3, const SINGLE &d4, const vect<SINGLE> &dx) {
	static const SINGLE two(2.0);
	D() += d0;													//  4
	for (int a = 0; a < NDIM; a++) {
		auto &Da = D(a);
		Da += dx[a] * d1;										// 15
		auto &Daa = D(a, a);
		auto &Daaa = D(a, a, a);
		auto &Daaaa = D(a, a, a, a);
		Daa += d1;												// 12
		Daaa += dx[a] * d2;										// 15
		Daaaa += dx[a] * dx[a] * d3;							// 18
		Daaaa += two * d2;										// 12
		for (int b = 0; b <= a; b++) {
			auto &Dab = D(a, b);
			auto &Daab = D(a, a, b);
			auto &Dabb = D(a, b, b);
			auto &Daaab = D(a, a, a, b);
			auto &Daabb = D(a, a, b, b);
			auto &Dabbb = D(a, b, b, b);
			const auto dxadxb = dx[a] * dx[b];					// 6
			Dab += dxadxb * d2;									// 30
			Daab += dx[b] * d2;									// 30
			Dabb += dx[a] * d2;									// 30
			Daaab += dxadxb * d3;								// 30
			Dabbb += dxadxb * d3;								// 30
			Daabb += d2;										// 24
			for (int c = 0; c <= b; c++) {
				auto &Dabc = D(a, b, c);
				const auto dxadxbdxc = dxadxb * dx[c];			// 10
				Dabc += dxadxbdxc * d3;							// 50
				auto &Daabc = D(a, a, b, c);
				auto &Dabcc = D(a, b, c, c);
				auto &Dabbc = D(a, b, b, c);
				Daabc += dx[b] * dx[c] * d3;					// 60
				Dabcc += dxadxb * d3;							// 50
				Dabbc += dx[a] * dx[c] * d3;					// 60
				for (int d = 0; d <= c; d++) {
					auto &Dabcd = D(a, b, c, d);
					Dabcd += dxadxbdxc * dx[d] * d4;			// 90
				}
			}
		}
	}

}

template<class T>
inline expansion<T> green_direct(const vect<T> &dX) {		// 59  + 167 = 226
	static const T r0 = 1.0e-9;
//	static const T H = options::get().soft_len;
	static const T nthree(-3.0);
	static const T nfive(-5.0);
	static const T nseven(-7.0);
	const T r2 = dX.dot(dX);		// 5
	const T r = sqrt(r2);		// 7
	const T rinv = (r > r0) / max(r, r0);	// 38
	const T r2inv = rinv * rinv;			// 1
	const T d0 = -rinv;						// 1
	const T d1 = -d0 * r2inv;				// 2
	const T d2 = nthree * d1 * r2inv;		// 2
	const T d3 = nfive * d2 * r2inv;		// 2
	const T d4 = nseven * d3 * r2inv;		// 2
	expansion<T> D;
	green_deriv_direct(D, d0, d1, d2, d3, d4, dX);		// 167
	return D;
}


template<class T>
inline expansion<T> green_ewald(const vect<T> &X) {		// 251176
	static const periodic_parts periodic;
	expansion<simd_double> D;
	D = 0.0;
	vect<T> n;
	vect<float> h;
	static const ewald_indices indices_real(EWALD_REAL_N2);
	static const ewald_indices indices_four(EWALD_FOUR_N2);
	static const T three(3.0);
	static const T fouroversqrtpi(4.0 / sqrt(M_PI));
	static const T two(2.0);
	static const T eight(8.0);
	static const T fifteen(15.0);
	static const T thirtyfive(35.0);
	static const T fourty(40.0);
	static const T fiftysix(56.0);
	static const T sixtyfour(64.0);
	static const T onehundredfive(105.0);
	static const simd_float rcut(1.0e-6);
//	printf("%i %i\n", indices_real.size(), indices_four.size());
	const T r = abs(X);															// 5
	const simd_float zmask = r > rcut;											// 2
	for (int i = 0; i < indices_real.size(); i++) {			// 739 * 305 		// 225395
		h = indices_real[i];
		n = h;
		const vect<T> dx = X - n;				// 3
		const T r2 = dx.dot(dx);				// 5
		const T r4 = r2 * r2;					// 1
		const T r6 = r2 * r4;					// 1
		const T r = sqrt(r2);					// 7
		const T mask = (r < 3.6) * zmask;		// 3
		const T rinv = mask / max(r, rcut);		// 36
		const T r2inv = rinv * rinv;			// 1
		const T r3inv = r2inv * rinv;			// 1
		const T r5inv = r2inv * r3inv;			// 1
		const T r7inv = r2inv * r5inv;			// 1
		const T r9inv = r2inv * r7inv;			// 1
		T expfac;
		const T erfc = erfcexp(two * r, &expfac);			// 76
		const T expfactor = fouroversqrtpi * r * expfac; 	// 2
		const T d0 = -erfc * rinv;							// 2
		const T d1 = (expfactor + erfc) * r3inv;			// 2
		const T d2 = -fmadd(expfactor, fmadd(eight, T(r2), three), three * erfc) * r5inv;		// 5
		const T d3 = fmadd(expfactor, (fifteen + fmadd(fourty, T(r2), sixtyfour * T(r4))), fifteen * erfc) * r7inv;		// 6
		const T d4 = -fmadd(expfactor, fmadd(eight * T(r2), (thirtyfive + fmadd(fiftysix, r2, sixtyfour * r4)), onehundredfive), onehundredfive * erfc) * r9inv;// 9
		green_deriv_ewald(D, d0, d1, d2, d3, d4, dx);			// 576
	}
	for (int i = 0; i < indices_four.size(); i++) {		// 207 * 123 = 			// 25461
		h = indices_four[i];
		const auto H = periodic[i];
		T hdotdx = X[0] * h[0];		// 1
		for (int a = 1; a < NDIM; a++) {
			hdotdx += X[a] * h[a];								// 4
		}
		static const T twopi = 2.0 * M_PI;
		const T omega = twopi * hdotdx;							// 1
		T co, si;
		sincos(omega, &si, &co);								// 25
		si *= zmask;											// 1
		co *= zmask;											// 1
		D() += H() * co;										// 5
		for (int a = 0; a < NDIM; a++) {
			D(a) += H(a) * si;									// 15
			for (int b = 0; b <= a; b++) {
				D(a, b) += H(a, b) * co;						// 30
				for (int c = 0; c <= b; c++) {
					D(a, b, c) += H(a, b, c) * si;				// 50
					for (int d = 0; d <= c; d++) {
						D(a, b, c, d) += H(a, b, c, d) * co; 	// 75
					}
				}

			}
		}
	}
	expansion<T> rcD;
	for (int i = 0; i < LP; i++) {
		rcD[i] = D[i];																	// 70
	}
	const auto D1 = green_direct(X);													// 167
	const T rinv = -zmask * D1();														// 2
	rcD() = T(M_PI / 4.0) + rcD() + rinv;												// 2
	rcD() = fmadd(T(2.8372975), (T(1) - zmask), +rcD() * zmask);						// 4
	for (int a = 0; a < NDIM; a++) {
		rcD(a) = zmask * (rcD(a) - D1(a));												// 6
		for (int b = 0; b <= a; b++) {
			rcD(a, b) = zmask * (rcD(a, b) - D1(a, b));									// 12
			for (int c = 0; c <= b; c++) {
				rcD(a, b, c) = zmask * (rcD(a, b, c) - D1(a, b, c));					// 20
				for (int d = 0; d <= c; d++) {
					rcD(a, b, c, d) = zmask * (rcD(a, b, c, d) - D1(a, b, c, d));		// 30
				}
			}
		}
	}
	return rcD;
}
