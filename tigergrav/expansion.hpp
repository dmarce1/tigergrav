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

constexpr int LP = 35;

struct force {
	float phi;
	vect<float> g;
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
inline expansion<T>& expansion<T>::operator<<=(const vect<T> &dX) {
	expansion<T> &me = *this;
	for (int a = 0; a < 3; a++) {
		me() += me(a) * dX[a];
	}
	for (int a = 0; a < 3; a++) {
		for (int b = 0; b < 3; b++) {
			me() += me(a, b) * dX[a] * dX[b] * (0.5);
		}
	}
	for (int a = 0; a < 3; a++) {
		for (int b = 0; b < 3; b++) {
			me(a) += me(a, b) * dX[b];
		}
	}
	for (int a = 0; a < 3; a++) {
		for (int b = 0; b < 3; b++) {
			for (int c = 0; c < 3; c++) {
				me() += me(a, b, c) * dX[a] * dX[b] * dX[c] * (1.0 / 6.0);
			}
		}
	}
	for (int a = 0; a < 3; a++) {
		for (int b = 0; b < 3; b++) {
			for (int c = 0; c < 3; c++) {
				me(a) += me(a, b, c) * dX[b] * dX[c] * (0.5);
			}
		}
	}
	for (int a = 0; a < 3; a++) {
		for (int b = 0; b < 3; b++) {
			for (int c = a; c < 3; c++) {
				me(a, c) += me(a, b, c) * dX[b];
			}
		}
	}

	for (int a = 0; a < 3; a++) {
		for (int b = 0; b < 3; b++) {
			for (int c = 0; c < 3; c++) {
				for (int d = 0; d < 3; d++) {
					me() += me(a, b, c, d) * dX[a] * dX[b] * dX[c] * dX[d] * (1.0 / 24.0);
				}
			}
		}
	}

	for (int a = 0; a < 3; a++) {
		for (int b = 0; b < 3; b++) {
			for (int c = 0; c < 3; c++) {
				for (int d = 0; d < 3; d++) {
					me(a) += me(a, b, c, d) * dX[b] * dX[c] * dX[d] * (1.0 / 6.0);
				}
			}
		}
	}

	for (int a = 0; a < 3; a++) {
		for (int b = a; b < 3; b++) {
			for (int c = 0; c < 3; c++) {
				for (int d = 0; d < 3; d++) {
					me(a, b) += me(a, b, c, d) * dX[c] * dX[d] * (0.5);
				}
			}
		}
	}

	for (int a = 0; a < 3; a++) {
		for (int b = a; b < 3; b++) {
			for (int c = b; c < 3; c++) {
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
	const auto &me = *this;
	force f;
	f.phi = (*this)();
	for (int a = 0; a < 3; a++) {
		f.phi += me(a) * dX[a];
	}
	for (int a = 0; a < 3; a++) {
		for (int b = 0; b < 3; b++) {
			f.phi += me(a, b) * dX[a] * dX[b] * (0.5);
		}
	}
	for (int a = 0; a < 3; a++) {
		f.g[a] = -(*this)(a);
		for (int b = 0; b < 3; b++) {
			f.g[a] -= me(a, b) * dX[b];
		}
	}
	for (int a = 0; a < 3; a++) {
		for (int b = 0; b < 3; b++) {
			for (int c = 0; c < 3; c++) {
				f.phi += me(a, b, c) * dX[a] * dX[b] * dX[c] * (1.0 / 6.0);
			}
		}
	}
	for (int a = 0; a < 3; a++) {
		for (int b = 0; b < 3; b++) {
			for (int c = 0; c < 3; c++) {
				f.g[a] -= me(a, b, c) * dX[b] * dX[c] * (0.5);
			}
		}
	}
	for (int a = 0; a < 3; a++) {
		for (int b = 0; b < 3; b++) {
			for (int c = 0; c < 3; c++) {
				for (int d = 0; d < 3; d++) {
					f.phi += me(a, b, c, d) * dX[a] * dX[b] * dX[c] * dX[d] * (1.0 / 24.0);
				}
			}
		}
	}
	for (int a = 0; a < 3; a++) {
		for (int b = 0; b < 3; b++) {
			for (int c = 0; c < 3; c++) {
				for (int d = 0; d < 3; d++) {
					f.g[a] -= me(a, b, c, d) * dX[b] * dX[c] * dX[d] * (1.0 / 6.0);
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

template<class T>
inline expansion<T> green_direct(const vect<T> &dX) {		// 339 OPS

	static const T H = options::get().soft_len;
	const double tiny = std::numeric_limits<float>::min() * 10.0;

	const T r2 = dX.dot(dX);						// 6
	const T r = sqrt(r2);
	const T rinv = r / (r * r + H * H + tiny);
	const T r2inv = rinv * rinv;
	const T d0 = -rinv;								// 2
	const T d1 = -d0 * r2inv;								// 2
	const T d2 = -3.0 * d1 * r2inv;							// 3
	const T d3 = -5.0 * d2 * r2inv;							// 3
	const T d4 = -7.0 * d3 * r2inv;							// 3

	expansion<T> D;
	D = 0.0;
	D() += d0;												// 1
	for (int i = 0; i < NDIM; i++) {
		D(i) += dX[i] * d1;									// 6
		D(i, i) += d1;										// 3
		D(i, i, i) += dX[i] * d2;							// 6
		D(i, i, i, i) += dX[i] * dX[i] * d3;				// 9
		D(i, i, i, i) += 2.0 * d2;							// 6
		for (int j = 0; j <= i; j++) {
			D(i, j) += dX[i] * dX[j] * d2;					// 18
			D(i, i, j) += dX[j] * d2;						// 12
			D(i, j, j) += dX[i] * d2;						// 12
			D(i, i, i, j) += dX[i] * dX[j] * d3;			// 18
			D(i, j, j, j) += dX[i] * dX[j] * d3;			// 18
			D(i, i, j, j) += d2;							// 6
			for (int k = 0; k <= j; k++) {
				D(i, j, k) += dX[i] * dX[j] * dX[k] * d3;	// 40
				D(i, i, j, k) += dX[j] * dX[k] * d3;		// 30
				D(i, j, k, k) += dX[i] * dX[j] * d3;		// 30
				D(i, j, j, k) += dX[i] * dX[k] * d3;		// 30
				for (int l = 0; l <= k; l++) {
					D(i, j, k, l) += dX[i] * dX[j] * dX[k] * dX[l] * d4;	// 75
				}
			}
		}
	}

	return D;
}

template<class T>
inline expansion<T> green_ewald(const vect<T> &X) {		// 339 OPS
	constexpr int nmax = 2;

	const double huge = std::numeric_limits<float>::max() / 10.0 / (nmax * nmax * nmax);
	const double tiny = std::numeric_limits<float>::min() * 10.0;

	expansion<T> D;
	D = green_direct(X);
	for (int i = 0; i < LP; i++) {
		D[i] = -D[i];
	}
	vect<double> n;
	D() += T(M_PI / 4.0);
	for (int a = -nmax; a <= +nmax; a++) {
		n[0] = a;
		for (int b = -nmax; b <= +nmax; b++) {
			n[1] = b;
			for (int c = -nmax; c <= +nmax; c++) {
				n[2] = c;

				vect<double> h = n;
				vect<T> hv = h;
				const double h2 = h.dot(h);
				if (h2 > 0.0) {
					const T hdotx = hv.dot(X);
					const double h2inv = 1.0 / h2;
					T s = sin(2.0 * M_PI * hdotx);
					T c = cos(2.0 * M_PI * hdotx);
					s = s * h2inv;
					c = c * h2inv;
					const T a0 = (-1.0 / M_PI) * h2inv * exp(-M_PI * M_PI * h2 / 4.0);
					D() += a0 * c;
					for (int i = 0; i < NDIM; i++) {
						D(i) += -(2.0 * M_PI) * a0 * h[i] * s;
						for (int j = 0; j <= i; j++) {
							D(i, j) += -pow(2.0 * M_PI, 2) * a0 * h[i] * h[j] * c;
							for (int k = 0; k <= j; k++) {
								D(i, j, k) += +pow(2.0 * M_PI, 3) * a0 * h[i] * h[j] * h[k] * s;
								for (int l = 0; l <= k; l++) {
									D(i, j, k, l) += +pow(2.0 * M_PI, 4) * a0 * h[i] * h[j] * h[k] * h[l] * c;
								}
							}
						}
					}

				}
				vect<T> nv = n;
				vect<T> dX = X - nv;
				const T r2 = dX.dot(dX);
				const T r = sqrt(r2);
				const T r4 = r2 * r2;
				const T r6 = r2 * r4;
				const T rinv = r / (r * r + tiny);
				const T r3inv = rinv * rinv * rinv;
				const T r5inv = rinv * r3inv;
				const T r7inv = rinv * r5inv;
				const T r9inv = rinv * r7inv;
				const T erfc = T(1.0) - erf(2.0 * r);
				const T exp0 = 4.0 * r * exp(-4.0 * r2) / sqrt(M_PI);
				const T d0 = -erfc * rinv;
				const T d1 = +(exp0 + erfc) * r3inv;
				const T d2 = -(exp0 * (T(3.0) + 8.0 * r2) + 3.0 * erfc) * r5inv;
				const T d3 = +(exp0 * (T(15.0) + 40.0 * r2 + 64.0 * r4) + 15.0 * erfc) * r7inv;
				const T d4 = -(exp0 * (T(105.0) + 8.0 * r2 * (T(35.0) + 56.0 * r2 + 64.0 * r4)) + 105.0 * erfc) * r9inv;
				D() += d0;												// 1
				for (int i = 0; i < NDIM; i++) {
					D(i) += dX[i] * d1;									// 6
					D(i, i) += d1;										// 3
					D(i, i, i) += dX[i] * d2;							// 6
					D(i, i, i, i) += dX[i] * dX[i] * d3;				// 9
					D(i, i, i, i) += 2.0 * d2;							// 6
					for (int j = 0; j <= i; j++) {
						D(i, j) += dX[i] * dX[j] * d2;					// 18
						D(i, i, j) += dX[j] * d2;						// 12
						D(i, j, j) += dX[i] * d2;						// 12
						D(i, i, i, j) += dX[i] * dX[j] * d3;			// 18
						D(i, j, j, j) += dX[i] * dX[j] * d3;			// 18
						D(i, i, j, j) += d2;							// 6
						for (int k = 0; k <= j; k++) {
							D(i, j, k) += dX[i] * dX[j] * dX[k] * d3;	// 40
							D(i, i, j, k) += dX[j] * dX[k] * d3;		// 30
							D(i, j, k, k) += dX[i] * dX[j] * d3;		// 30
							D(i, j, j, k) += dX[i] * dX[k] * d3;		// 30
							for (int l = 0; l <= k; l++) {
								D(i, j, k, l) += dX[i] * dX[j] * dX[k] * dX[l] * d4;	// 75
							}
						}
					}
				}
			}
		}
	}
	const T sw = min(X.dot(X) * T(huge), T(1.0));
	D() = 2.8372975 * (T(1.0) - sw) + D() * sw;
	for (int i = 1; i < LP; i++) {
		D[i] = sw * D[i];
	}

	return D;
}

extern expansion<ireal> expansion_factor;

template<class T>
inline void multipole_interaction(expansion<T> &L1, const multipole<T> &M2, vect<T> dX, bool ewald = false) { // 701

	expansion<T> D;
	if (ewald) {
		D = green_ewald(dX);
	} else {
		D = green_direct(dX);
	}

	L1() += M2() * D();
	for (int a = 0; a < 3; a++) {
		for (int b = a; b < 3; b++) {
			L1() += M2(a, b) * D(a, b) * expansion_factor(a, b);						// 18
		}
	}
	for (int a = 0; a < 3; a++) {
		L1(a) += M2() * D(a);
	}
	for (int a = 0; a < 3; a++) {
		for (int b = 0; b < 3; b++) {
			for (int c = b; c < 3; c++) {
				L1(a) += M2(c, b) * D(a, b, c) * expansion_factor(c, b);				// 54
			}
		}
	}

	for (int a = 0; a < 3; a++) {
		for (int b = a; b < 3; b++) {
			L1(a, b) += M2() * D(a, b);													// 12
		}
	}

	for (int a = 0; a < 3; a++) {
		for (int b = a; b < 3; b++) {
			for (int c = b; c < 3; c++) {
				L1(a, b, c) += M2() * D(a, b, c);										// 20
			}
		}
	}
	for (int a = 0; a < 3; a++) {
		for (int b = a; b < 3; b++) {
			for (int c = b; c < 3; c++) {
				L1() -= M2(a, b, c) * D(a, b, c) * expansion_factor(a, b, c);			// 30
			}
		}
	}
	for (int a = 0; a < 3; a++) {
		for (int b = 0; b < 3; b++) {
			for (int c = b; c < 3; c++) {
				for (int d = c; d < 3; d++) {
					L1(a) -= M2(b, c, d) * D(a, b, c, d) * expansion_factor(b, c, d);	// 90
				}
			}
		}
	}
	for (int a = 0; a < 3; a++) {
		for (int b = a; b < 3; b++) {
			for (int c = 0; c < 3; c++) {
				for (int d = c; d < 3; d++) {
					L1(a, b) += M2(c, d) * D(a, b, c, d) * expansion_factor(c, d);		// 108
				}
			}
		}
	}
	for (int a = 0; a < 3; a++) {
		for (int b = a; b < 3; b++) {
			for (int c = b; c < 3; c++) {
				for (int d = c; d < 3; d++) {
					L1(a, b, c, d) += M2() * D(a, b, c, d);								// 30
				}
			}
		}
	}
}

template<class T>
inline std::pair<T, vect<T>> multipole_interaction(const multipole<T> &M, vect<T> dX, bool ewald = false) {

	expansion<T> D;
	if (ewald) {
		D = green_ewald(dX);
	} else {
		D = green_direct(dX);
	}

	std::pair<T, vect<T>> f;
	f.first = 0.0;
	f.first += M() * D();
	for (int a = 0; a < 3; a++) {
		for (int b = a; b < 3; b++) {
			f.first += M(a, b) * D(a, b) * expansion_factor(a, b);
		}
	}
	for (int a = 0; a < 3; a++) {
		for (int b = 0; b < 3; b++) {
			for (int c = b; c < 3; c++) {
				f.first -= M(a, b, c) * D(a, b, c) * expansion_factor(a, b, c);
			}
		}
	}
	f.second = vect<float>(0);
	for (int a = 0; a < 3; a++) {
		f.second[a] -= M() * D(a);
	}
	for (int a = 0; a < 3; a++) {
		for (int b = 0; b < 3; b++) {
			for (int c = b; c < 3; c++) {
				f.second[a] -= M(c, b) * D(a, b, c) * expansion_factor(c, b);
			}
		}
	}
	for (int a = 0; a < 3; a++) {
		for (int b = 0; b < 3; b++) {
			for (int c = b; c < 3; c++) {
				for (int d = c; d < 3; d++) {
					f.second[a] += M(c, b, d) * D(a, b, c, d) * expansion_factor(b, c, d);
				}
			}
		}
	}
	return f;

}

/* namespace fmmx */
#endif /* expansion_H_ */
