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

constexpr int MP = 7;


template<class T>
class multipole: public std::array<T, MP> {
private:
public:
	multipole();
	T operator ()() const;
	T& operator ()();
	T operator ()(int i, int j) const;
	T& operator ()(int i, int j);
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
	for (int p = 0; p < 3; p++) {
		for (int q = p; q < 3; q++) {
			me(p, q) += me() * Y[p] * Y[q];
		}
	}
	return me;
}


#endif /* multipole_H_ */
