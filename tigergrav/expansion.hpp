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

constexpr int LP = 20;


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
	expansion<T>& operator =(const expansion<T> &expansion);
	expansion<T>& operator =(T expansion);
	expansion<T> operator<<(const vect<T> &dX) const;
	void translate_to_particle(const vect<T> &dX, T &phi, vect<T> &g) const;
	T translate_to_particle(const vect<T> &dX) const;
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
			me() += me(a, b) * dX[a] * dX[b] * float(0.5);
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
				me(a) += me(a, b, c) * dX[b] * dX[c] * float(0.5);
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
	return me;
}

template<class T>
inline T expansion<T>::translate_to_particle(const vect<T> &dX) const {
	const auto &L = *this;
	T this_phi = L();
	for (int a = 0; a < 3; a++) {
		this_phi += L(a) * dX[a];
	}
	for (int a = 0; a < 3; a++) {
		for (int b = 0; b < 3; b++) {
			this_phi += L(a, b) * dX[a] * dX[b] * float(0.5);
		}
	}
	for (int a = 0; a < 3; a++) {
		for (int b = 0; b < 3; b++) {
			for (int c = 0; c < 3; c++) {
				this_phi += L(a, b, c) * dX[a] * dX[b] * dX[c] * (1.0 / 6.0);
			}
		}
	}
	return this_phi;
}

template<class T>
inline std::array<T, LP>& expansion<T>::operator +=(const std::array<T, LP> &vec) {
	for (int i = 0; i < LP; i++) {
		(*this)[i] += vec[i];
	}
	return *this;
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
inline void multipole_interaction(expansion<T> &L1, const multipole<T> &M2, vect<T> dX) {

	extern expansion<float> expansion_factor;

	T y0 = 0.0;
	for (int d = 0; d != NDIM; ++d) {
		y0 += dX[d] * dX[d];
	}
	expansion<T> D;
	const T r2inv = 1.0 / y0;
	const T d0 = -sqrt(r2inv);
	const T d1 = -d0 * r2inv;
	const T d2 = -3.0 * d1 * r2inv;
	const T d3 = -5.0 * d2 * r2inv;
	D() = 0.0;
	for (int a = 0; a < 3; a++) {
		D(a) = 0.0;
		for (int b = a; b < 3; b++) {
			D(a, b) = 0.0;
			for (int c = b; c < 3; c++) {
				D(a, b, c) = 0.0;
			}
		}
	}

	D() += d0;
	for (int a = 0; a < 3; a++) {
		D(a) += dX[a] * d1;
		D(a, a) += d1;
		D(a, a, a) += dX[a] * d2;
		for (int b = a; b < 3; b++) {
			D(a, b) += dX[a] * dX[b] * d2;
			D(a, a, b) += dX[b] * d2;
			D(a, b, b) += dX[a] * d2;
			for (int c = b; c < 3; c++) {
				D(a, b, c) += dX[a] * dX[b] * dX[c] * d3;
			}
		}
	}

	L1() += M2() * D();
	for (int a = 0; a < 3; a++) {
		for (int b = a; b < 3; b++) {
			L1() += M2(a, b) * D(a, b) * float(0.5) * expansion_factor(a, b);
		}
	}

	for (int a = 0; a < 3; a++) {
		L1(a) += M2() * D(a);
	}
	for (int a = 0; a < 3; a++) {
		for (int b = 0; b < 3; b++) {
			for (int c = b; c < 3; c++) {
				L1(a) += M2(c, b) * D(a, b, c) * float(0.5) * expansion_factor(c, b);
			}
		}
	}

	for (int a = 0; a < 3; a++) {
		for (int b = a; b < 3; b++) {
			L1(a, b) += M2() * D(a, b);
		}
	}

	for (int a = 0; a < 3; a++) {
		for (int b = a; b < 3; b++) {
			for (int c = b; c < 3; c++) {
				L1(a, b, c) += M2() * D(a, b, c);
			}
		}
	}
}


template<class T>
inline std::pair<T,vect<T>> multipole_interaction(const multipole<T> &M, vect<T> dX) {

	T y0 = 0.0;
	for (int d = 0; d != NDIM; ++d) {
		y0 += dX[d] * dX[d];
	}
	expansion<T> D;
	const T r2inv = 1.0 / y0;
	const T d0 = -sqrt(r2inv);
	const T d1 = -d0 * r2inv;
	const T d2 = -3.0 * d1 * r2inv;
	const T d3 = -5.0 * d2 * r2inv;
	D() = 0.0;
	for (int a = 0; a < 3; a++) {
		D(a) = 0.0;
		for (int b = a; b < 3; b++) {
			D(a, b) = 0.0;
			for (int c = b; c < 3; c++) {
				D(a, b, c) = 0.0;
			}
		}
	}

	D() += d0;
	for (int a = 0; a < 3; a++) {
		D(a) += dX[a] * d1;
		D(a, a) += d1;
		D(a, a, a) += dX[a] * d2;
		for (int b = a; b < 3; b++) {
			D(a, b) += dX[a] * dX[b] * d2;
			D(a, a, b) += dX[b] * d2;
			D(a, b, b) += dX[a] * d2;
			for (int c = b; c < 3; c++) {
				D(a, b, c) += dX[a] * dX[b] * dX[c] * d3;
			}
		}
	}

	extern expansion<float> expansion_factor;

	std::pair<T,vect<T>> f;
	f.first = 0.0;
	f.first += M() * D();
	for (int a = 0; a < 3; a++) {
		for (int b = a; b < 3; b++) {
			f.first += M(a, b) * D(a, b) * float(0.5) * expansion_factor(a, b);
		}
	}
	f.second = vect<float>(0);
	for (int a = 0; a < 3; a++) {
		f.second[a] -= M() * D(a);
	}
	for (int a = 0; a < 3; a++) {
		for (int b = 0; b < 3; b++) {
			for (int c = b; c < 3; c++) {
				f.second[a] -= M(c, b) * D(a, b, c) * float(0.5) * expansion_factor(c, b);
			}
		}
	}
	return f;

}

/* namespace fmmx */
#endif /* expansion_H_ */
