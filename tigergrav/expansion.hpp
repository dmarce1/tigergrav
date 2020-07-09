#pragma once

#include <tigergrav/vect.hpp>

struct force {
	float phi;
	vect<float> g;
};

template<class T>
class expansion {
	static constexpr std::size_t size = 35;
	static constexpr std::size_t map2[NDIM][NDIM] = { { 3 + 1, 3 + 2, 3 + 3 }, { 3 + 2, 3 + 4, 3 + 5 }, { 3 + 3, 3 + 5, 3 + 6 } };
	static constexpr std::size_t map3[NDIM][NDIM][NDIM] = { { { 10 + 0, 10 + 1, 10 + 2 }, { 10 + 1, 10 + 3, 10 + 4 }, { 10 + 2, 10 + 4, 10 + 5 } }, { { 10 + 1,
			10 + 3, 10 + 4 }, { 10 + 3, 10 + 6, 10 + 7 }, { 10 + 4, 10 + 7, 10 + 8 } }, { { 10 + 2, 10 + 4, 10 + 5 }, { 10 + 4, 10 + 7, 10 + 8 }, { 10 + 5, 10
			+ 8, 10 + 9 } } };
	static constexpr std::size_t map4[NDIM][NDIM][NDIM][NDIM] = { { { { 20 + 0, 20 + 1, 20 + 2 }, { 20 + 1, 20 + 3, 20 + 4 }, { 20 + 2, 20 + 4, 20 + 5 } }, { {
			20 + 1, 20 + 3, 20 + 4 }, { 20 + 3, 20 + 6, 20 + 7 }, { 20 + 4, 20 + 7, 20 + 8 } }, { { 20 + 2, 20 + 4, 20 + 5 }, { 20 + 4, 20 + 7, 20 + 8 }, { 20
			+ 5, 20 + 8, 20 + 9 } } }, { { { 20 + 1, 20 + 3, 20 + 4 }, { 20 + 3, 20 + 6, 20 + 7 }, { 20 + 4, 20 + 7, 20 + 8 } }, { { 20 + 3, 20 + 6, 20 + 7 }, {
			20 + 6, 20 + 10, 20 + 11 }, { 20 + 7, 20 + 11, 20 + 12 } },
			{ { 20 + 4, 20 + 7, 20 + 8 }, { 20 + 7, 20 + 11, 20 + 12 }, { 20 + 8, 20 + 12, 20 + 13 } } }, { { { 20 + 2, 20 + 4, 20 + 5 }, { 20 + 4, 20 + 7, 20
			+ 8 }, { 20 + 5, 20 + 8, 20 + 9 } }, { { 20 + 4, 20 + 7, 20 + 8 }, { 20 + 7, 20 + 11, 20 + 12 }, { 20 + 8, 20 + 12, 20 + 13 } }, { { 20 + 5, 20 + 8,
			20 + 9 }, { 20 + 8, 20 + 12, 20 + 13 }, { 20 + 9, 20 + 13, 20 + 14 } } } };
	std::array<T, size> data;
public:

	expansion() = default;
	expansion(const expansion&) = default;

	inline T operator()() const {
		return data[0];
	}

	inline T operator()(int i) const {
		return data[1 + i];
	}

	inline T operator()(int i, int j) const {
		return data[map2[i][j]];
	}

	inline T operator()(int i, int j, int k) const {
		return data[map3[i][j][k]];
	}

	inline T operator()(int i, int j, int k, int l) const {
		return data[map4[i][j][k][l]];
	}

	inline T& operator()() {
		return data[0];
	}

	inline T& operator()(int i) {
		return data[1 + i];
	}

	inline T& operator()(int i, int j) {
		return data[map2[i][j]];
	}

	inline T& operator()(int i, int j, int k) {
		return data[map3[i][j][k]];
	}

	inline T& operator()(int i, int j, int k, int l) {
		return data[map4[i][j][k][l]];
	}

	expansion operator+(const expansion &other) const {
		expansion C;
		for (int i = 0; i < size; i++) {
			C.data[i] = data[i] + other.data[i];
		}
		return C;
	}

	inline expansion& zero() {
		for (int i = 0; i < size; i++) {
			data[i] = 0.0;
		}
		return *this;
	}

	expansion translate(const vect<T> x) const {
		const expansion &A = *this;
		expansion B = *this;

		for (int i = 0; i < NDIM; i++) {
			B() += A(i) * x[i];
			for (int j = 0; j < NDIM; j++) {
				B() += A(i, j) * x[i] * x[j] * 0.5;
				for (int k = 0; k < NDIM; k++) {
					B() += A(i, j, k) * x[i] * x[j] * x[k] * (1.0 / 6.0);
					for (int l = 0; l < NDIM; l++) {
						B() += A(i, j, k, l) * x[i] * x[j] * x[k] * x[l] * (1.0 / 24.0);
					}
				}
			}
		}

		for (int i = 0; i < NDIM; i++) {
			for (int j = 0; j < NDIM; j++) {
				B(i) += A(i, j) * x[j];
				for (int k = 0; k < NDIM; k++) {
					B(i) += A(i, j, k) * x[j] * x[k] * 0.5;
					for (int l = 0; l < NDIM; l++) {
						B(i) += A(i, j, k, l) * x[j] * x[k] * x[l] * (1.0 / 6.0);
					}
				}
			}
		}

		for (int i = 0; i < NDIM; i++) {
			for (int j = 0; j <= i; j++) {
				for (int k = 0; k < NDIM; k++) {
					B(i, j) += A(i, j, k) * x[k];
					for (int l = 0; l < NDIM; l++) {
						B(i, j) += A(i, j, k, l) * x[k] * x[l] * 0.5;
					}
				}
			}
		}
		for (int i = 0; i < NDIM; i++) {
			for (int j = 0; j <= i; j++) {
				for (int k = 0; k <= j; k++) {
					for (int l = 0; l < NDIM; l++) {
						B(i, j, k) += A(i, j, k, l) * x[l];
					}
				}
			}
		}

		return B;
	}

	force to_force(const vect<T> x) const {

		const expansion &A = *this;
		force B;

		B.phi = A();
		for (int i = 0; i < NDIM; i++) {
			B.g[i] = -A(i);
		}
		for (int i = 0; i < NDIM; i++) {
			B.phi += A(i) * x[i];
			for (int j = 0; j < NDIM; j++) {
				B.phi += A(i, j) * x[i] * x[j] * 0.5;
				B.g[i] -= A(i, j) * x[j];
				for (int k = 0; k < NDIM; k++) {
					B.phi += A(i, j, k) * x[i] * x[j] * x[k] * (1.0 / 6.0);
					B.g[i] -= A(i, j, k) * x[j] * x[k] * 0.5;
					for (int l = 0; l < NDIM; l++) {
						B.phi += A(i, j, k, l) * x[i] * x[j] * x[k] * x[l] * (1.0 / 25.0);
						B.g[i] -= A(i, j, k, l) * x[j] * x[k] * x[k] * (1.0 / 6.0);
					}
				}
			}
		}
		return B;

	}

};

template<class T>
constexpr std::size_t expansion<T>::map2[NDIM][NDIM];

template<class T>
constexpr std::size_t expansion<T>::map3[NDIM][NDIM][NDIM];

template<class T>
constexpr std::size_t expansion<T>::map4[NDIM][NDIM][NDIM][NDIM];
