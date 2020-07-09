#pragma once

#include <tigergrav/vect.hpp>

template<class T>
class multipole {
	static constexpr std::size_t size = 17;
	static constexpr std::size_t map2[NDIM][NDIM] = { { 1, 2, 3 }, { 2, 4, 5 }, { 3, 5, 6 } };
	static constexpr std::size_t map3[NDIM][NDIM][NDIM] = { { { 7 + 0, 7 + 1, 7 + 2 }, { 7 + 1, 7 + 3, 7 + 4 }, { 7 + 2, 7 + 4, 7 + 5 } }, { { 7 + 1, 7 + 3, 7
			+ 4 }, { 7 + 3, 7 + 6, 7 + 7 }, { 7 + 4, 7 + 7, 7 + 8 } }, { { 7 + 2, 7 + 4, 7 + 5 }, { 7 + 4, 7 + 7, 7 + 8 }, { 7 + 5, 7 + 8, 7 + 9 } } };

	std::array<T, size> data;
public:

	multipole() = default;

	inline T operator()() const {
		return data[0];
	}

	inline T operator()(int i, int j) const {
		return data[map2[i][j]];
	}

	inline T operator()(int i, int j, int k) const {
		return data[map3[i][j][k]];
	}

	inline T& operator()() {
		return data[0];
	}

	inline T& operator()(int i, int j) {
		return data[map2[i][j]];
	}

	inline T& operator()(int i, int j, int k) {
		return data[map3[i][j][k]];
	}

	template<class V>
	multipole(const multipole<V> &other) {
		for (int i = 0; i < size; i++) {
			data[i] = T(other.data[i]);
		}
	}

	template<class V>
	friend class multipole;

	multipole(T m, const vect<T> x) {
		(*this)() = m;
		for (int i = 0; i < NDIM; i++) {
			for (int j = 0; j <= i; j++) {
				(*this)(i, j) = m * x[i] * x[j];
				for (int k = 0; k <= j; k++) {
					(*this)(i, j, k) = m * x[i] * x[j] * x[k];
				}
			}
		}
	}

	multipole translate(const vect<T> x) const {
		multipole A = (*this);
		T M = (*this)();
		for (int i = 0; i < NDIM; i++) {
			for (int j = 0; j <= i; j++) {
				A(i, j) += M * x[i] * x[j];
				for (int k = 0; k <= j; k++) {
					A(i, j, k) += (*this)(i, j) * x[k];
					A(i, j, k) += (*this)(j, k) * x[i];
					A(i, j, k) += (*this)(k, i) * x[j];
					A(i, j, k) += M * x[i] * x[j] * x[k];

				}
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

	inline multipole& zero() {
		for (int i = 0; i < size; i++) {
			data[i] = 0.0;
		}
		return *this;
	}

};

template<class T>
constexpr std::size_t multipole<T>::map2[NDIM][NDIM];

template<class T>
constexpr std::size_t multipole<T>::map3[NDIM][NDIM][NDIM];
