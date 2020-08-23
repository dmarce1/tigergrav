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

template<class T>
inline expansion<T> green_ewald(const vect<T> &X) {		// 371 + 763 * NREAL + 66 * NFOUR
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

	const T r = abs(X);															// 5S
	const simd_float zmask = r > rcut;											// 1S
	for (int i = 0; i < indices_real.size(); i++) {		// ((311S  + 226D) X NREAL
		h = indices_real[i];
		n = h;
		const vect<T> dx = X - n;				// 3S
		const T r2 = dx.dot(dx);				// 5S
		const T r4 = r2 * r2;					// 1S
		const T r6 = r2 * r4;					// 1S
		const T r = sqrt(r2);					// 7S
		const T mask = (r < 3.6) * zmask;		// 3S
		const T rinv = mask / max(r, rcut);		//36S
		const T r2inv = rinv * rinv;			// 1S
		const T r3inv = r2inv * rinv;			// 1S
		const T r5inv = r2inv * r3inv;			// 1S
		const T r7inv = r2inv * r5inv;			// 1S
		const T r9inv = r2inv * r7inv;			// 1S
		T expfac;
		const T erfc = erfcexp(two * r, &expfac);			// 82S
		const T expfactor = fouroversqrtpi * r * expfac;		// 2S
		const T d0 = -erfc * rinv;								// 2S
		const T d1 = (expfactor + erfc) * r3inv;		// 2S
		const T d2 = -fmadd(expfactor, fmadd(eight, T(r2), three), three * erfc) * r5inv;		// 7S
		const T d3 = fmadd(expfactor, (fifteen + fmadd(fourty, T(r2), sixtyfour * T(r4))), fifteen * erfc) * r7inv;		// 8S
		const T d4 = -fmadd(expfactor, fmadd(eight * T(r2), (thirtyfive + fmadd(fiftysix, r2, sixtyfour * r4)), onehundredfive), onehundredfive * erfc) * r9inv;// 12S
		green_deriv(D, d0, d1, d2, d3, d4, dx);			// 135 S // 226 D // 227C
	}
	for (int i = 0; i < indices_four.size(); i++) {		// (46S + 10D) x NFOUR
		h = indices_four[i];
		const auto H = periodic[i];
		T hdotdx = X[0] * h[0];		// 1
		for (int a = 1; a < NDIM; a++) {
			hdotdx += X[a] * h[a];								// 4S
		}
		static const T twopi = 2.0 * M_PI;
		const T omega = twopi * hdotdx;							// 1S
		T co, si;
		sincos(omega, &si, &co);								//34S
		si *= zmask;											// 1S
		co *= zmask;											// 1S
		D() += H() * co;										// 1S / 2D
		for (int a = 0; a < NDIM; a++) {
			D(a) += H(a) * si;									// 1S / 2D
			for (int b = 0; b <= a; b++) {
				D(a, b) += H(a, b) * co;						// 1S / 2D
				for (int c = 0; c <= b; c++) {
					D(a, b, c) += H(a, b, c) * si;				// 1S / 2D
					for (int d = 0; d <= c; d++) {
						D(a, b, c, d) += H(a, b, c, d) * co; 	// 1S / 2D
					}
				}

			}
		}
	}
	expansion<T> rcD;
	for (int i = 0; i < LP; i++) {
		rcD[i] = D[i];																	//    / 3D
	}
	const auto D1 = green_direct(X);													// 286S
	const T rinv = -zmask * D1();														// 2S
	rcD() = T(M_PI / 4.0) + rcD() + rinv;												// 2S
	rcD() = fmadd(T(2.8372975), (T(1) - zmask), +rcD() * zmask);						// 4S
	for (int a = 0; a < NDIM; a++) {
		rcD(a) = zmask * (rcD(a) - D1(a));												// 6S
		for (int b = 0; b <= a; b++) {
			rcD(a, b) = zmask * (rcD(a, b) - D1(a, b));									// 12S
			for (int c = 0; c <= b; c++) {
				rcD(a, b, c) = zmask * (rcD(a, b, c) - D1(a, b, c));					// 20S
				for (int d = 0; d <= c; d++) {
					rcD(a, b, c, d) = zmask * (rcD(a, b, c, d) - D1(a, b, c, d));		// 30S
				}
			}
		}
	}
	return rcD;
}

template<class DOUBLE, class SINGLE>  // 135 S // 226 D // 227C
void green_deriv(expansion<DOUBLE> &D, const SINGLE &d0, const SINGLE &d1, const SINGLE &d2, const SINGLE &d3, const SINGLE &d4, const vect<SINGLE> &dx) {
	static const SINGLE two(2.0);
	D() += d0;													//    / 2D / 1
	for (int a = 0; a < NDIM; a++) {
		auto &Da = D(a);
		auto &Daa = D(a, a);
		auto &Daaa = D(a, a, a);
		auto &Daaaa = D(a, a, a, a);
		Da += dx[a] * d1;										// 3S / 6D / 3
		Daa += d1;												//    / 6D / 3
		Daaa += dx[a] * d2;										// 3S / 6D / 6
		Daaaa += dx[a] * dx[a] * d3;							// 6S / 6D / 9
		Daaaa += two * d2;										// 3S / 6D / 6
		for (int b = 0; b <= a; b++) {
			auto &Dab = D(a, b);
			auto &Daab = D(a, a, b);
			auto &Dabb = D(a, b, b);
			auto &Daaab = D(a, a, a, b);
			auto &Daabb = D(a, a, b, b);
			auto &Dabbb = D(a, b, b, b);
			const auto dxadxb = dx[a] * dx[b];					//    / 12D / 3
			Dab += dxadxb * d2;									// 6S / 12D / 9
			Daab += dx[b] * d2;									// 6S / 12D / 9
			Dabb += dx[a] * d2;									// 6S / 12D / 9
			Daaab += dxadxb * d3;								// 6S / 12D / 9
			Dabbb += dxadxb * d3;								// 6S / 12D / 9
			Daabb += d2;										//    / 12D / 6
			for (int c = 0; c <= b; c++) {
				auto &Dabc = D(a, b, c);
				auto &Daabc = D(a, a, b, c);
				auto &Dabcc = D(a, b, c, c);
				auto &Dabbc = D(a, b, b, c);
				const auto dxadxbdxc = dxadxb * dx[c];
				Dabc += dxadxbdxc * d3;							// 10S / 20D / 20
				Daabc += dx[b] * dx[c] * d3;					// 20S / 20D / 30
				Dabcc += dxadxb * d3;							// 10S / 20D / 20
				Dabbc += dx[a] * dx[c] * d3;					// 20S / 20D / 30
				for (int d = 0; d <= c; d++) {
					auto &Dabcd = D(a, b, c, d);
					Dabcd += dxadxbdxc * dx[d] * d4;			// 30S / 30D // 45
				}
			}
		}
	}

}

template<class T>
inline expansion<T> green_direct(const vect<T> &dX) {		// 286
	static const T r0 = 1.0e-9;
//	static const T H = options::get().soft_len;
	static const T nthree(-3.0);
	static const T nfive(-5.0);
	static const T nseven(-7.0);
	const T r2 = dX.dot(dX);			// 5
	const T r = sqrt(r2);				// 7
	const T rinv = (r > r0) / max(r, r0);		// 37
	const T r2inv = rinv * rinv;		// 1
	const T d0 = -rinv;					// 1
	const T d1 = -d0 * r2inv;			// 2
	const T d2 = nthree * d1 * r2inv;		// 2
	const T d3 = nfive * d2 * r2inv;		// 2
	const T d4 = nseven * d3 * r2inv;		// 2
	expansion<T> D;
	D = 0.0;
	green_deriv(D, d0, d1, d2, d3, d4, dX); // 135 S // 226 D // 227C
	return D;
}
