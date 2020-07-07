#pragma once

#include <tigergrav/vect.hpp>

template<class T>
class multipole {
	static constexpr std::size_t size = 7;
	static constexpr std::size_t map2[NDIM][NDIM] = { {1, 2, 3}, {2, 4, 5}, {3, 5, 6} };

	std::array<T, size> data;
public:

	multipole() = default;
	multipole(const multipole&) = default;

	inline T operator()() const {
		return data[0];
	}

	inline T operator()(int i, int j) const {
		return data[map2[i][j]];
	}

	inline T& operator()() {
		return data[0];
	}

	inline T& operator()(int i, int j) {
		return data[map2[i][j]];
	}

	multipole(T m, const vect<T> x) {
		(*this)() = m;
		for (int i = 0; i < NDIM; i++) {
			for (int j = 0; j <= NDIM; j++) {
				(*this)(i, j) = m * x[i] * x[j];
			}
		}
	}

	multipole translate(const vect<T> x) const {
		multipole A = (*this);
		T M = (*this)();
		for (int i = 0; i < NDIM; i++) {
			for (int j = 0; j <= i; j++) {
				A(i, j) += M * x[i] * x[j];
			}
		}
		return A;
	}

	multipole operator+(const multipole &other) const {
		multipole C;
		for (int i = 0; i < size; i++) {
			C.data[i] = data[i] + other.data[i];
		}
		return C;
	}

};

template<class T>
constexpr std::size_t multipole<T>::map2[NDIM][NDIM];
