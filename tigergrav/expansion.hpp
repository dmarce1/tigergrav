/*  
 Copyright (c) 2016 Dominic C. Marcello

 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef EXPAN222SION_H_
#define EXPAN222SION_H_

#include <tigergrav/multipole.hpp>
#include <tigergrav/options.hpp>
#include <tigergrav/simd.hpp>

#include <algorithm>
#include <vector>

constexpr int LP = 35;

//constexpr float EWALD_REAL_N2 = 5;
//constexpr float EWALD_FOUR_N2 = 10;
constexpr float EWALD_REAL_N2 = 17;
constexpr float EWALD_FOUR_N2 = 9;
//36 8.55046
//38 9.06660
//40 1.01544
//42 7.19136
//44 8.73753
//46 1.01339
//50 7.953

struct force {
	double phi;
	vect<double> g;
	template<class A>
	void serialize(A &&arc, unsigned) {
		arc & phi;
		arc & g;
	}
};

template<class T>
class expansion: public std::array<T, LP> {

public:
	expansion<T>& operator*=(T r) {
		for (int i = 0; i != LP; ++i) {
			(*this)[i] *= r;
		}
		return *this;
	}
	expansion();
	T operator ()() const;
	T& operator ()();
	T operator ()(int i) const;
	T& operator ()(int i);
	T operator ()(int i, int j) const;
	T& operator ()(int i, int j);
	T operator ()(int i, int j, int k) const;
	T& operator ()(int i, int j, int k);
	T operator ()(int i, int j, int k, int l) const;
	T& operator ()(int i, int j, int k, int l);
	expansion<T>& operator =(const expansion<T> &expansion);
	expansion<T>& operator =(T expansion);
	force translate_L2(const vect<T> &dX) const;
	expansion<T> operator<<(const vect<T> &dX) const;
	expansion<T>& operator<<=(const vect<T> &dX);
	std::array<T, LP>& operator +=(const std::array<T, LP> &vec);
	std::array<T, LP>& operator -=(const std::array<T, LP> &vec);
	void compute_D(const vect<T> &Y);
	void invert();
	std::array<expansion<T>, NDIM> get_derivatives() const;
};

template<class T>
inline expansion<T>::expansion() {
}

template<class T>
inline T expansion<T>::operator ()() const {
	return (*this)[0];
}
template<class T>
inline T& expansion<T>::operator ()() {
	return (*this)[0];
}

template<class T>
inline T expansion<T>::operator ()(int i) const {
	return (*this)[1 + i];
}
template<class T>
inline T& expansion<T>::operator ()(int i) {
	return (*this)[1 + i];
}

template<class T>
inline T expansion<T>::operator ()(int i, int j) const {
	static constexpr size_t map2[3][3] = { { 0, 1, 2 }, { 1, 3, 4 }, { 2, 4, 5 } };
	return (*this)[4 + map2[i][j]];
}
template<class T>
inline T& expansion<T>::operator ()(int i, int j) {
	static constexpr size_t map2[3][3] = { { 0, 1, 2 }, { 1, 3, 4 }, { 2, 4, 5 } };
	return (*this)[4 + map2[i][j]];
}

template<class T>
inline T expansion<T>::operator ()(int i, int j, int k) const {
	static constexpr size_t map3[3][3][3] = { { { 0, 1, 2 }, { 1, 3, 4 }, { 2, 4, 5 } }, { { 1, 3, 4 }, { 3, 6, 7 }, { 4, 7, 8 } }, { { 2, 4, 5 }, { 4, 7, 8 },
			{ 5, 8, 9 } } };

	return (*this)[10 + map3[i][j][k]];
}
template<class T>
inline T& expansion<T>::operator ()(int i, int j, int k) {
	static constexpr size_t map3[3][3][3] = { { { 0, 1, 2 }, { 1, 3, 4 }, { 2, 4, 5 } }, { { 1, 3, 4 }, { 3, 6, 7 }, { 4, 7, 8 } }, { { 2, 4, 5 }, { 4, 7, 8 },
			{ 5, 8, 9 } } };

	return (*this)[10 + map3[i][j][k]];
}

template<class T>
inline T& expansion<T>::operator ()(int i, int j, int k, int l) {
	static constexpr size_t map4[3][3][3][3] = { { { { 0, 1, 2 }, { 1, 3, 4 }, { 2, 4, 5 } }, { { 1, 3, 4 }, { 3, 6, 7 }, { 4, 7, 8 } }, { { 2, 4, 5 }, { 4, 7,
			8 }, { 5, 8, 9 } } }, { { { 1, 3, 4 }, { 3, 6, 7 }, { 4, 7, 8 } }, { { 3, 6, 7 }, { 6, 10, 11 }, { 7, 11, 12 } }, { { 4, 7, 8 }, { 7, 11, 12 }, { 8,
			12, 13 } } }, { { { 2, 4, 5 }, { 4, 7, 8 }, { 5, 8, 9 } }, { { 4, 7, 8 }, { 7, 11, 12 }, { 8, 12, 13 } }, { { 5, 8, 9 }, { 8, 12, 13 },
			{ 9, 13, 14 } } } };
	return (*this)[20 + map4[i][j][k][l]];
}

template<class T>
inline T expansion<T>::operator ()(int i, int j, int k, int l) const {
	static constexpr size_t map4[3][3][3][3] = { { { { 0, 1, 2 }, { 1, 3, 4 }, { 2, 4, 5 } }, { { 1, 3, 4 }, { 3, 6, 7 }, { 4, 7, 8 } }, { { 2, 4, 5 }, { 4, 7,
			8 }, { 5, 8, 9 } } }, { { { 1, 3, 4 }, { 3, 6, 7 }, { 4, 7, 8 } }, { { 3, 6, 7 }, { 6, 10, 11 }, { 7, 11, 12 } }, { { 4, 7, 8 }, { 7, 11, 12 }, { 8,
			12, 13 } } }, { { { 2, 4, 5 }, { 4, 7, 8 }, { 5, 8, 9 } }, { { 4, 7, 8 }, { 7, 11, 12 }, { 8, 12, 13 } }, { { 5, 8, 9 }, { 8, 12, 13 },
			{ 9, 13, 14 } } } };
	return (*this)[20 + map4[i][j][k][l]];
}

template<class T>
inline expansion<T>& expansion<T>::operator =(const expansion<T> &expansion) {
	for (int i = 0; i < LP; i++) {
		(*this)[i] = expansion[i];
	}
	return *this;
}

template<class T>
inline expansion<T>& expansion<T>::operator =(T expansion) {
	for (int i = 0; i < LP; i++) {
		(*this)[i] = expansion;
	}
	return *this;
}

template<class T>
inline expansion<T> expansion<T>::operator<<(const vect<T> &dX) const {
	expansion you = *this;
	you <<= dX;
	return you;
}

template<class T>
struct expansion_factors: public expansion<T> {
	expansion_factors() {
		for (int i = 0; i < LP; i++) {
			(*this)[i] = T(0.0);
		}
		(*this)() += T(1);
		for (int a = 0; a < NDIM; ++a) {
			(*this)(a) += T(1.0);
			for (int b = 0; b < NDIM; ++b) {
				(*this)(a, b) += T(0.5);
				for (int c = 0; c < NDIM; ++c) {
					(*this)(a, b, c) += T(1.0 / 6.0);
					for (int d = 0; d < NDIM; ++d) {
						(*this)(a, b, c, d) += T(1.0 / 24.0);
					}
				}
			}
		}
	}
};

template<class T>
inline expansion<T>& expansion<T>::operator<<=(const vect<T> &dX) {
	const static expansion_factors<T> factor;
	expansion<T> &me = *this;
	for (int a = 0; a < 3; a++) {
		me() += me(a) * dX[a];
		for (int b = 0; b <= a; b++) {
			me() += me(a, b) * dX[a] * dX[b] * factor(a, b);
			for (int c = 0; c <= b; c++) {
				me() += me(a, b, c) * dX[a] * dX[b] * dX[c] * factor(a, b, c);
				for (int d = 0; d <= c; d++) {
					me() += me(a, b, c, d) * dX[a] * dX[b] * dX[c] * dX[d] * factor(a, b, c, d);
				}
			}
		}
	}
	for (int a = 0; a < 3; a++) {
		for (int b = 0; b < 3; b++) {
			me(a) += me(a, b) * dX[b];
			for (int c = 0; c <= b; c++) {
				me(a) += me(a, b, c) * dX[b] * dX[c] * factor(b, c);
				for (int d = 0; d <= c; d++) {
					me(a) += me(a, b, c, d) * dX[b] * dX[c] * dX[d] * factor(b, c, d);
				}
			}
		}
	}
	for (int a = 0; a < 3; a++) {
		for (int b = 0; b <= a; b++) {
			for (int c = 0; c < NDIM; c++) {
				me(a, b) += me(a, b, c) * dX[c];
				for (int d = 0; d <= c; d++) {
					me(a, b) += me(a, b, c, d) * dX[c] * dX[d] * factor(c, d);
				}
			}
		}
	}

	for (int a = 0; a < 3; a++) {
		for (int b = 0; b <= a; b++) {
			for (int c = 0; c <= b; c++) {
				for (int d = 0; d < 3; d++) {
					me(a, b, c) += me(a, b, c, d) * dX[d];
				}
			}
		}
	}

	return me;
}

template<class T>
inline force expansion<T>::translate_L2(const vect<T> &dX) const {
	const static expansion_factors<T> factor;

	const auto &me = *this;
	force f;
	f.phi = (*this)();
	for (int a = 0; a < 3; a++) {
		f.phi += me(a) * dX[a];
		for (int b = a; b < 3; b++) {
			f.phi += me(a, b) * dX[a] * dX[b] * factor(a, b);
			for (int c = b; c < 3; c++) {
				f.phi += me(a, b, c) * dX[a] * dX[b] * dX[c] * factor(a, b, c);
				for (int d = c; d < 3; d++) {
					f.phi += me(a, b, c, d) * dX[a] * dX[b] * dX[c] * dX[d] * factor(a, b, c, d);
				}
			}
		}
	}
	for (int a = 0; a < 3; a++) {
	}
	for (int a = 0; a < 3; a++) {
		f.g[a] = -(*this)(a);
		for (int b = 0; b < 3; b++) {
			f.g[a] -= me(a, b) * dX[b];
			for (int c = b; c < 3; c++) {
				f.g[a] -= me(a, b, c) * dX[b] * dX[c] * factor(b, c);
				for (int d = c; d < 3; d++) {
					f.g[a] -= me(a, b, c, d) * dX[b] * dX[c] * dX[d] * factor(b, c, d);
				}
			}
		}
	}
	return f;
}

template<class T>
inline std::array<T, LP>& expansion<T>::operator -=(const std::array<T, LP> &vec) {
	for (int i = 0; i < LP; i++) {
		(*this)[i] -= vec[i];
	}
	return *this;
}

//void expansion::compute_D(const vect<T>& Y) {
//}

template<class T>
inline void expansion<T>::invert() {
	expansion<T> &me = *this;
	for (int a = 0; a < 3; a++) {
		me(a) = -me(a);
	}
	for (int a = 0; a < 3; a++) {
		for (int b = a; b < 3; b++) {
			for (int c = b; c < 3; c++) {
				me(a, b, c) = -me(a, b, c);
			}
		}
	}
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
inline expansion<T> green_direct(const vect<T> &dX) {		// 227

//	static const T H = options::get().soft_len;
	const float tiny = std::numeric_limits<float>::min() * 10.0;
	static const T nthree(-3.0);
	static const T nfive(-5.0);
	static const T nseven(-7.0);
	const T r2 = dX.dot(dX);			// 5
	const T r = sqrt(r2);				// 1
	const T rinv = r / (r * r + tiny);	// 3
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
inline expansion<T> green_ewald(const vect<T> &X) {		// (311S + 3D) + (237S  + 226D) * NREAL + (44S + 10D) * NFOUR
	static const periodic_parts periodic;
	expansion<simd_double> D;
	D = 0.0;
	const float huge = std::numeric_limits<float>::max() / 100.0;
	const float tiny = std::numeric_limits<float>::min() * 10.0;
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
	for (int i = 0; i < indices_real.size(); i++) {		// ((237S  + 226D) X NREAL
		h = indices_real[i];
		n = h;
		const vect<T> dx = X - n;				// 3S
		const T r2 = dx.dot(dx);				// 5S
		const T r4 = r2 * r2;					// 1S
		const T r6 = r2 * r4;					// 1S
		const T r = sqrt(r2);					// 1S
		const T mask = r < 3.6;
		const T rinv = mask * r / (r2 + tiny);	// 2S
		const T r2inv = rinv * rinv;			// 1S
		const T r3inv = r2inv * rinv;			// 1S
		const T r5inv = r2inv * r3inv;			// 1S
		const T r7inv = r2inv * r5inv;			// 1S
		const T r9inv = r2inv * r7inv;			// 1S
		T expfac;
		const T erfc = erfcexp(two * r, &expfac);			// 51S
		const T expfactor = fouroversqrtpi * r * expfac;		// 2S
		const T d0 = -erfc * rinv;								// 2S
		const T d1 = (expfactor + erfc) * r3inv;		// 2S
		const T d2 = -fmadd(expfactor, fmadd(eight, T(r2), three), three * erfc) * r5inv;		// 7S
		const T d3 = fmadd(expfactor, (fifteen + fmadd(fourty, T(r2), sixtyfour * T(r4))), fifteen * erfc) * r7inv;		// 8S
		const T d4 = -fmadd(expfactor, fmadd(eight * T(r2), (thirtyfive + fmadd(fiftysix, r2, sixtyfour * r4)), onehundredfive), onehundredfive * erfc) * r9inv;// 12S
		green_deriv(D, d0, d1, d2, d3, d4, dx);			// 135 S // 226 D // 227C
	}
	for (int i = 0; i < indices_four.size(); i++) {		// (44S + 10D) x NFOUR
		h = indices_four[i];
		const auto H = periodic[i];
		T hdotdx = X[0] * h[0];		// 1
		for (int a = 1; a < NDIM; a++) {
			hdotdx += X[a] * h[a];								// 4S
		}
		static const T twopi = 2.0 * M_PI;
		const T omega = twopi * hdotdx;							// 1S
		T co, si;
		sincos(omega, &si, &co);								// 34S
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
		rcD[i] = D[i];															//    / 3D
	}
	const auto D1 = green_direct(X);											// 227S
	const T r = abs(X);															// 5S
	const T rinv = r / fmadd(r, r, tiny);										// 3S
	rcD() = T(M_PI / 4.0) + rcD() + rinv;											// 2S
	const T sw = min(T(huge) * r, T(1.0));										// 2S
	rcD() = fmadd(T(2.8372975), (T(1) - sw), +rcD() * sw);							// 4S
	for (int a = 0; a < NDIM; a++) {
		rcD(a) = sw * (rcD(a) - D1(a));												// 6S
		for (int b = 0; b <= a; b++) {
			rcD(a, b) = sw * (rcD(a, b) - D1(a, b));								// 12S
			for (int c = 0; c <= b; c++) {
				rcD(a, b, c) = sw * (rcD(a, b, c) - D1(a, b, c));					// 20S
				for (int d = 0; d <= c; d++) {
					rcD(a, b, c, d) = sw * (rcD(a, b, c, d) - D1(a, b, c, d));		// 30S
				}
			}
		}
	}
	return rcD;
}

// 43009,703
template<class DOUBLE, class SINGLE>
inline void multipole_interaction(expansion<DOUBLE> &L, const multipole<SINGLE> &M2, vect<SINGLE> dX, bool ewald = false) { // 670/700 + 418 * NREAL + 50 * NFOUR
	static const expansion_factors<SINGLE> expansion_factor;
	expansion<SINGLE> D;
	if (ewald) {
		D = green_ewald(dX);		// // (311S + 3D) + (237S  + 226D) * NREAL + (44S + 10D) * NFOUR
	} else {
		D = green_direct(dX);        // 227S
	}

	auto &L0 = L();
	L0 += M2() * D();																// 1 S / 2D
	for (int a = 0; a < 3; a++) {
		for (int b = a; b < 3; b++) {
			L0 += M2(a, b) * D(a, b) * expansion_factor(a, b);						// 18
			for (int c = b; c < 3; c++) {
				L0 -= M2(a, b, c) * D(a, b, c) * expansion_factor(a, b, c);			// 30
			}
		}
	}
	for (int a = 0; a < 3; a++) {
		auto &La = L(a);
		La += M2() * D(a);
		for (int b = 0; b < 3; b++) {
			for (int c = b; c < 3; c++) {
				La += M2(c, b) * D(a, b, c) * expansion_factor(c, b);				// 54
				for (int d = c; d < 3; d++) {
					La -= M2(b, c, d) * D(a, b, c, d) * expansion_factor(b, c, d);	// 90
				}
			}
		}
	}

	for (int a = 0; a < 3; a++) {
		for (int b = a; b < 3; b++) {
			auto &Lab = L(a, b);
			Lab += M2() * D(a, b);													// 12
			for (int c = 0; c < 3; c++) {
				for (int d = c; d < 3; d++) {
					Lab += M2(c, d) * D(a, b, c, d) * expansion_factor(c, d);		// 108
				}
			}
		}
	}

	for (int a = 0; a < 3; a++) {
		for (int b = a; b < 3; b++) {
			for (int c = b; c < 3; c++) {
				auto &Labc = L(a, b, c);
				Labc += M2() * D(a, b, c);										// 20
			}
		}
	}
	for (int a = 0; a < 3; a++) {
		for (int b = a; b < 3; b++) {
			for (int c = b; c < 3; c++) {
				for (int d = c; d < 3; d++) {
					auto &Labcd = L(a, b, c, d);
					Labcd += M2() * D(a, b, c, d);								// 30
				}
			}
		}
	}
}

// 42707,401
template<class DOUBLE, class SINGLE>
inline void multipole_interaction(expansion<DOUBLE> &L, const SINGLE &M, vect<SINGLE> dX, bool ewald = false) { // 390 / 47301
	static const expansion_factors<SINGLE> expansion_factor;
	expansion<SINGLE> D;
	if (ewald) {
		D = green_ewald(dX);		// (311S + 3D) + (237S  + 226D) * NREAL + (44S + 10D) * NFOUR
	} else {
		D = green_direct(dX);          // 227S
	}

	auto &L0 = L();
	L0 += M * D();
	for (int a = 0; a < 3; a++) {
		auto &La = L(a);
		La += M * D(a);
		for (int b = a; b < 3; b++) {
			auto &Lab = L(a, b);
			Lab += M * D(a, b);													// 12
			for (int c = b; c < 3; c++) {
				auto &Labc = L(a, b, c);
				Labc += M * D(a, b, c);										// 20
				for (int d = c; d < 3; d++) {
					auto &Labcd = L(a, b, c, d);
					Labcd += M * D(a, b, c, d);								// 30
				}
			}
		}
	}
}

//42826, 520
template<class T>
inline std::pair<T, vect<T>> multipole_interaction(const multipole<T> &M, vect<T> dX, bool ewald = false) { // 517 / 47428
	static const expansion_factors<T> expansion_factor;
	expansion<T> D;
	if (ewald) {
		D = green_ewald(dX);				//(311S + 3D) + (237S  + 226D) * NREAL + (44S + 10D) * NFOUR
	} else {
		D = green_direct(dX);				// 227S
	}

	std::pair<T, vect<T>> f;
	auto &ffirst = f.first;
	ffirst = 0.0;
	ffirst = fmadd(M(), D(), ffirst);															// 1
	for (int a = 0; a < 3; a++) {
		for (int b = a; b < 3; b++) {
			ffirst = fmadd(M(a, b) * D(a, b), expansion_factor(a, b), ffirst);					// 18
		}
	}
	for (int a = 0; a < 3; a++) {
		for (int b = 0; b < 3; b++) {
			for (int c = b; c < 3; c++) {
				ffirst -= M(a, b, c) * D(a, b, c) * expansion_factor(a, b, c);		// 30
			}
		}
	}
	f.second = vect<float>(0);
	for (int a = 0; a < 3; a++) {
		f.second[a] -= M() * D(a);													// 6
	}
	for (int a = 0; a < 3; a++) {
		for (int b = 0; b < 3; b++) {
			for (int c = b; c < 3; c++) {
				f.second[a] -= M(c, b) * D(a, b, c) * expansion_factor(c, b);		// 36
			}
		}
	}
	for (int a = 0; a < 3; a++) {
		for (int b = 0; b < 3; b++) {
			for (int c = b; c < 3; c++) {
				for (int d = c; d < 3; d++) {
					auto &fseconda = f.second[a];
					fseconda = fmadd(M(c, b, d) * D(a, b, c, d), expansion_factor(b, c, d), fseconda); // 90
				}
			}
		}
	}
	return f;

}

/* namespace fmmx */
#endif /* expansion_H_ */
