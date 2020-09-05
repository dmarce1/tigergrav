#pragma once

constexpr float EWALD_REAL_N2 = 18;
constexpr float EWALD_FOUR_N2 = 9;

#include <tigergrav/expansion.hpp>

struct ewald_indices: public std::vector<vect<float>> {
	ewald_indices(int n2max, bool nozero) {
		const int nmax = sqrt(n2max) + 1;
		vect<float> h;
		for (int i = -nmax; i <= nmax; i++) {
			for (int j = -nmax; j <= nmax; j++) {
				for (int k = -nmax; k <= nmax; k++) {
					if (i * i + j * j + k * k <= n2max) {
						h[0] = i;
						h[1] = j;
						h[2] = k;
						if (!nozero || h.dot(h) > 0) {
							this->push_back(h);
						}
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
		static const ewald_indices indices(EWALD_FOUR_N2, true);
		for (auto i : indices) {
			vect<float> h = i;
			const float h2 = h.dot(h);                     // 5 OP
			expansion<float> D;
			D = 0.0;
			if (h2 > 0) {
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
CUDA_EXPORT void green_deriv_direct(expansion<SINGLE> &D, const SINGLE &d0, const SINGLE &d1, const SINGLE &d2, const SINGLE &d3, const SINGLE &d4,
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
		Daaa = fma(dx[a], d2, Daaa);							// 3
		Daaaa = fma(dx[a] * dx[a], d3, Daaaa);				// 6
		Daaaa = fma(two, d2, Daaaa);							// 3
		for (int b = 0; b <= a; b++) {
			auto &Daab = D(a, a, b);
			auto &Dabb = D(a, b, b);
			auto &Daaab = D(a, a, a, b);
			auto &Daabb = D(a, a, b, b);
			auto &Dabbb = D(a, b, b, b);
			const auto dxadxb = dx[a] * dx[b];					// 6
			Daab = fma(dx[b], d2, Daab);						// 6
			Dabb = fma(dx[a], d2, Dabb);						// 6
			Daaab = fma(dxadxb, d3, Daaab);					// 6
			Dabbb = fma(dxadxb, d3, Dabbb);					// 6
			Daabb += d2;										// 6
			for (int c = 0; c <= b; c++) {
				auto &Daabc = D(a, a, b, c);
				auto &Dabcc = D(a, b, c, c);
				auto &Dabbc = D(a, b, b, c);
				Daabc = fma(dx[b], dx[c] * d3, Daabc);		// 20
				Dabcc = fma(dxadxb, d3, Dabcc);				// 10
				Dabbc = fma(dx[a], dx[c] * d3, Dabbc);		// 20
			}
		}
	}

}

template<class DOUBLE, class SINGLE>  // 576
CUDA_EXPORT void green_deriv_ewald(expansion<DOUBLE> &D, const SINGLE &d0, const SINGLE &d1, const SINGLE &d2, const SINGLE &d3, const SINGLE &d4,
		const vect<SINGLE> &dx) {
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
CUDA_EXPORT inline expansion<T> green_direct(const vect<T> &dX) {		// 59  + 167 = 226
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

#ifdef __CUDA_ARCH__

__device__ const cuda_ewald_const& cuda_get_const();

CUDA_EXPORT expansion<float> green_ewald(const vect<float> &X) {
	const auto& cuda_const = cuda_get_const();
	const auto& four_indices = cuda_const.four_indices;
	const auto& real_indices = cuda_const.real_indices;
	const auto& hparts = cuda_const.periodic_parts;
	static const float three(3.0);
	const float fouroversqrtpi(4.0 / sqrt(M_PI));
	static const float two(2.0);
	static const float eight(8.0);
	static const float fifteen(15.0);
	static const float thirtyfive(35.0);
	static const float fourty(40.0);
	static const float fiftysix(56.0);
	static const float sixtyfour(64.0);
	static const float onehundredfive(105.0);
	static const float rcut(1.0e-6);
	const float r = abs(X);
	const float zmask = r > rcut;											// 2
	expansion<double> D;
	D = 0.0;
	for( auto n : real_indices) {
	//	printf( "n = %e %e %e\n", n[0], n[1], n[2]);
					const vect<float> dx = X - vect<float>(n);				// 3
				const float r2 = dx.dot(dx);				// 5
				if (r2 < 12.96) {
					const float r = sqrt(r2);					// 7
					const float r4 = r2 * r2;					// 1
					const float cmask = 1.0 - (n.dot(n) > 0.0);
					const float mask = (1.0 - (1.0 - zmask) * cmask);
					const float rinv = mask / max(r, rcut);		// 36
					const float r2inv = rinv * rinv;			// 1
					const float r3inv = r2inv * rinv;			// 1
					const float r5inv = r2inv * r3inv;			// 1
					const float r7inv = r2inv * r5inv;			// 1
					const float r9inv = r2inv * r7inv;			// 1
					const float erfc0 = erfcf(two * r);			// 76
					const float exp0 = expf(-two * two * r * r);
					const float expfactor = fouroversqrtpi * r * exp0; 	// 2
					const float d0 = -erfc0 * rinv;							// 2
					const float d1 = (expfactor + erfc0) * r3inv;			// 2
					const float d2 = -fma(expfactor, fma(eight, r2, three), three * erfc0) * r5inv;		// 5
					const float d3 = fma(expfactor, (fifteen + fma(fourty, r2, sixtyfour * r4)), fifteen * erfc0) * r7inv;		// 6
					const float d4 = -fma(expfactor, fma(eight * r2, (thirtyfive + fma(fiftysix, r2, sixtyfour * r4)), onehundredfive), onehundredfive * erfc0)
							* r9inv;		// 9
					green_deriv_ewald(D, d0, d1, d2, d3, d4, dx);			// 576
		}
	}
	static const float twopi = 2.0 * M_PI;

	for( int i = 0; i < EWALD_NFOUR; i++ ) {
		const auto & h = four_indices[i];
		const auto & hpart = hparts[i];
//		printf( "H = %e %e %e\n", h[0], h[1], h[2]);
					const float h2 = h.dot( h);
					const float hdotx = h.dot(X);
						const float co = cosf(twopi * hdotx);
						const float so = sinf(twopi * hdotx);
						D() += hpart() * co;
						for (int a = 0; a < NDIM; a++) {
							D(a) += hpart(a) * so;
							for (int b = 0; b <= a; b++) {
								D(a, b) += hpart(a,b) * co;
								for (int c = 0; c <= b; c++) {
									D(a, b, c) += hpart(a,b,c) * so;
									for (int d = 0; d <= c; d++) {
										D(a, b, c, d) += hpart(a,b,c,d)* co;
									}
								}
							}
					}
	}

	expansion<float> rcD;

	for (int i = 0; i < LP; i++) {
		rcD[i] =D[i];																	// 70
	}
	const auto D1 = green_direct(X);													// 167
	const float rinv = -D1();														// 2
	rcD() = (M_PI / 4.0) + rcD() + zmask * rinv;												// 2
	for (int a = 0; a < NDIM; a++) {
		rcD(a) = (rcD(a) - zmask * D1(a));												// 6
		for (int b = 0; b <= a; b++) {
			rcD(a, b) = (rcD(a, b) - zmask * D1(a, b));									// 12
			for (int c = 0; c <= b; c++) {
				rcD(a, b, c) = (rcD(a, b, c) - zmask * D1(a, b, c));					// 20
				for (int d = 0; d <= c; d++) {
					rcD(a, b, c, d) = (rcD(a, b, c, d) - zmask * D1(a, b, c, d));		// 30
				}
			}
		}
	}

	return rcD;

}
#else

template<class T>
inline expansion<T> green_ewald(const vect<T> &X) {		// 251176
	static const periodic_parts periodic;
	expansion<simd_double> D;
	D = 0.0;
	vect<T> n;
	vect<float> h;
	static const ewald_indices indices_real(EWALD_REAL_N2, false);
	static const ewald_indices indices_four(EWALD_FOUR_N2, true);
//	printf( "%i %i\n", indices_real.size(), indices_four.size());
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
		const T cmask = T(1) - (n.dot(n) > 0.0);
		const T mask = (T(1) - (T(1) - zmask) * cmask) * (r < 3.6);
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
		const T d2 = -fma(expfactor, fma(eight, T(r2), three), three * erfc) * r5inv;		// 5
		const T d3 = fma(expfactor, (fifteen + fma(fourty, T(r2), sixtyfour * T(r4))), fifteen * erfc) * r7inv;		// 6
		const T d4 = -fma(expfactor, fma(eight * T(r2), (thirtyfive + fma(fiftysix, r2, sixtyfour * r4)), onehundredfive), onehundredfive * erfc) * r9inv;	// 9
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
	const T rinv = -D1();														// 2
	rcD() = T(M_PI / 4.0) + rcD() + zmask * rinv;												// 2
	for (int a = 0; a < NDIM; a++) {
		rcD(a) = (rcD(a) - zmask * D1(a));												// 6
		for (int b = 0; b <= a; b++) {
			rcD(a, b) = (rcD(a, b) - zmask * D1(a, b));									// 12
			for (int c = 0; c <= b; c++) {
				rcD(a, b, c) = (rcD(a, b, c) - zmask * D1(a, b, c));					// 20
				for (int d = 0; d <= c; d++) {
					rcD(a, b, c, d) = (rcD(a, b, c, d) - zmask * D1(a, b, c, d));		// 30
				}
			}
		}
	}
	return rcD;
}
#endif
