#pragma once

#include <tigergrav/vect.hpp>

class multipole {
	static constexpr std::size_t size = 7;
	static constexpr std::size_t map2[NDIM][NDIM] = { 1, 2, 3, 2, 4, 5, 3, 5, 6 };

	std::array<float, size> data;
public:

	multipole() = default;
	multipole(const multipole&) = default;
	multipole(float m, const vect<float> x);

	inline float operator()() const {
		return data[0];
	}

	inline float operator()(int i, int j) const {
		return data[map2[i][j]];
	}

	inline float& operator()() {
		return data[0];
	}

	inline float& operator()(int i, int j) {
		return data[map2[i][j]];
	}

	multipole operator+(const multipole &other) const;
	multipole translate(const vect<float> x) const;

};
