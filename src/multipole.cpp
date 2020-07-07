/*
 * multipole.cpp
 *
 *  Created on: Jul 7, 2020
 *      Author: dmarce1
 */

#include <tigergrav/multipole.hpp>

multipole::multipole(float m, const vect<float> x) {
	(*this)() = m;
	for (int i = 0; i < NDIM; i++) {
		for (int j = 0; j <= NDIM; j++) {
			(*this)(i, j) = m * x[i] * x[j];
		}
	}
}

multipole multipole::translate(const vect<float> x) const {
	multipole A = (*this);
	float M = (*this)();
	for (int i = 0; i < NDIM; i++) {
		for (int j = 0; j <= i; j++) {
			A(i, j) += M * x[i] * x[j];
		}
	}
	return A;
}

multipole multipole::operator+(const multipole &other) const {
	multipole C;
	for (int i = 0; i < size; i++) {
		C.data[i] = data[i] + other.data[i];
	}
	return C;
}

constexpr std::size_t multipole::map2[NDIM][NDIM];
