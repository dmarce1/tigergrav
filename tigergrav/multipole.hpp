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

#ifndef multipole_H_
#define multipole_H_

#include <tigergrav/vect.hpp>
#include <array>

#ifdef HEXAPOLE
constexpr int MP = 32;
#else
constexpr int MP = 17;
#endif

template<class T>
class multipole: public std::array<T, MP> {

public:
	multipole();
	T operator ()() const;
	T& operator ()();
	T operator ()(int i, int j) const;
	T& operator ()(int i, int j);
	T operator ()(int i, int j, int k) const;
	T& operator ()(int i, int j, int k);
	T operator ()(int i, int j, int k, int l) const;
	T& operator ()(int i, int j, int k, int l);
	multipole<T>& operator =(const multipole<T> &other);
	multipole<T>& operator =(T other);
	multipole operator>>(const vect<T> &dX) const;
	multipole<T>& operator>>=(const vect<T> &Y);
	multipole<T> operator +(const multipole<T> &vec) const;
};

template<class T>
inline multipole<T>::multipole() {
}

template<class T>
inline T multipole<T>::operator ()() const {
	return (*this)[0];
}

template<class T>
inline T& multipole<T>::operator ()() {
	return (*this)[0];
}

template<class T>
inline T multipole<T>::operator ()(int i, int j) const {
	static constexpr size_t map2[3][3] = { { 0, 1, 2 }, { 1, 3, 4 }, { 2, 4, 5 } };
	return (*this)[1 + map2[i][j]];
}

template<class T>
inline T& multipole<T>::operator ()(int i, int j) {
	static constexpr size_t map2[3][3] = { { 0, 1, 2 }, { 1, 3, 4 }, { 2, 4, 5 } };
	return (*this)[1 + map2[i][j]];
}

template<class T>
inline T multipole<T>::operator ()(int i, int j, int k) const {
	static constexpr size_t map3[3][3][3] = { { { 0, 1, 2 }, { 1, 3, 4 }, { 2, 4, 5 } }, { { 1, 3, 4 }, { 3, 6, 7 }, { 4, 7, 8 } }, { { 2, 4, 5 }, { 4, 7, 8 },
			{ 5, 8, 9 } } };

	return (*this)[7 + map3[i][j][k]];
}
template<class T>
inline T& multipole<T>::operator ()(int i, int j, int k) {
	static constexpr size_t map3[3][3][3] = { { { 0, 1, 2 }, { 1, 3, 4 }, { 2, 4, 5 } }, { { 1, 3, 4 }, { 3, 6, 7 }, { 4, 7, 8 } }, { { 2, 4, 5 }, { 4, 7, 8 },
			{ 5, 8, 9 } } };
	return (*this)[7 + map3[i][j][k]];
}

template<class T>
inline T& multipole<T>::operator ()(int i, int j, int k, int l) {
	static constexpr size_t map4[3][3][3][3] = { { { { 0, 1, 2 }, { 1, 3, 4 }, { 2, 4, 5 } }, { { 1, 3, 4 }, { 3, 6, 7 }, { 4, 7, 8 } }, { { 2, 4, 5 }, { 4, 7,
			8 }, { 5, 8, 9 } } }, { { { 1, 3, 4 }, { 3, 6, 7 }, { 4, 7, 8 } }, { { 3, 6, 7 }, { 6, 10, 11 }, { 7, 11, 12 } }, { { 4, 7, 8 }, { 7, 11, 12 }, { 8,
			12, 13 } } }, { { { 2, 4, 5 }, { 4, 7, 8 }, { 5, 8, 9 } }, { { 4, 7, 8 }, { 7, 11, 12 }, { 8, 12, 13 } }, { { 5, 8, 9 }, { 8, 12, 13 },
			{ 9, 13, 14 } } } };
	return (*this)[17 + map4[i][j][k][l]];
}

template<class T>
inline T multipole<T>::operator ()(int i, int j, int k, int l) const {
	static constexpr size_t map4[3][3][3][3] = { { { { 0, 1, 2 }, { 1, 3, 4 }, { 2, 4, 5 } }, { { 1, 3, 4 }, { 3, 6, 7 }, { 4, 7, 8 } }, { { 2, 4, 5 }, { 4, 7,
			8 }, { 5, 8, 9 } } }, { { { 1, 3, 4 }, { 3, 6, 7 }, { 4, 7, 8 } }, { { 3, 6, 7 }, { 6, 10, 11 }, { 7, 11, 12 } }, { { 4, 7, 8 }, { 7, 11, 12 }, { 8,
			12, 13 } } }, { { { 2, 4, 5 }, { 4, 7, 8 }, { 5, 8, 9 } }, { { 4, 7, 8 }, { 7, 11, 12 }, { 8, 12, 13 } }, { { 5, 8, 9 }, { 8, 12, 13 },
			{ 9, 13, 14 } } } };
	return (*this)[17 + map4[i][j][k][l]];
}

template<class T>
inline multipole<T>& multipole<T>::operator =(const multipole<T> &other) {
	for (int i = 0; i < MP; i++) {
		(*this)[i] = other[i];
	}
	return *this;
}

template<class T>
inline multipole<T>& multipole<T>::operator =(T other) {
	for (int i = 0; i < MP; i++) {
		(*this)[i] = other;
	}
	return *this;
}

template<class T>
inline multipole<T> multipole<T>::operator>>(const vect<T> &dX) const {
	multipole you = *this;
	you >>= dX;
	return you;
}

template<class T>
inline multipole<T> multipole<T>::operator +(const multipole<T> &vec) const {
	multipole<T> C;
	for (int i = 0; i < MP; i++) {
		C[i] = (*this)[i] + vec[i];
	}
	return C;
}

template<class T>
inline multipole<T>& multipole<T>::operator>>=(const vect<T> &Y) {
	multipole<T> &me = *this;
	multipole<T> A = *this;
	for (int i = 0; i < 3; i++) {
		for (int j = i; j < 3; j++) {
			for (int k = j; k < 3; k++) {
				me(i, j, k) += A() * Y[i] * Y[j] * Y[k];
				me(i, j, k) += A(i, j) * Y[k];
				me(i, j, k) += A(j, k) * Y[i];
				me(i, j, k) += A(k, i) * Y[j];
#ifdef HEXAPOLE
				for (int l = 0; l <= k; l++) {
					me(i, j, k, l) += A() * Y[i] * Y[j] * Y[k] * Y[l];
					me(i, j, k, l) += A(i, j) * Y[k] * Y[l];
					me(i, j, k, l) += A(i, k) * Y[j] * Y[l];
					me(i, j, k, l) += A(i, l) * Y[j] * Y[k];
					me(i, j, k, l) += A(j, k) * Y[i] * Y[l];
					me(i, j, k, l) += A(j, l) * Y[i] * Y[k];
					me(i, j, k, l) += A(k, l) * Y[i] * Y[j];
					me(i, j, k, l) += A(j, k, l) * Y[i];
					me(i, j, k, l) += A(i, k, l) * Y[j];
					me(i, j, k, l) += A(i, j, l) * Y[k];
					me(i, j, k, l) += A(i, j, k) * Y[l];
				}
#endif
			}
		}
	}
	for (int p = 0; p < 3; p++) {
		for (int q = p; q < 3; q++) {
			me(p, q) += A() * Y[p] * Y[q];
		}
	}
	return me;
}

#endif /* multipole_H_ */
