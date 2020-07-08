#pragma once

#include <tigergrav/vect.hpp>

template<class T>
class expansion {
	static constexpr std::size_t size = 20;
	static constexpr std::size_t map2[NDIM][NDIM] = { { 3 + 1, 3 + 2, 3 + 3 }, { 3 + 2, 3 + 4, 3 + 5 }, { 3 + 3, 3 + 5, 3 + 6 } };
	static constexpr std::size_t map3[NDIM][NDIM][NDIM] = { { { 10 + 0, 10 + 1, 10 + 2 }, { 10 + 1, 10 + 3, 10 + 4 }, { 10 + 2, 10 + 4, 10 + 5 } }, { { 10 + 1,
			7 + 3, 10 + 4 }, { 10 + 3, 10 + 6, 10 + 7 }, { 10 + 4, 10 + 7, 10 + 8 } }, { { 10 + 2, 10 + 4, 10 + 5 }, { 10 + 4, 10 + 7, 10 + 8 }, { 10 + 5, 10
			+ 8, 10 + 9 } } };
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

};

template<class T>
constexpr std::size_t expansion<T>::map2[NDIM][NDIM];

template<class T>
constexpr std::size_t expansion<T>::map3[NDIM][NDIM][NDIM];
