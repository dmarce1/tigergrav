/*
 * particle.hpp
 *
 *  Created on: Jun 2, 2020
 *      Author: dmarce1
 */

#ifndef TIGERGRAV_PARTICLE_HPP_
#define TIGERGRAV_PARTICLE_HPP_

#include <tigergrav/fixed_real.hpp>
#include <tigergrav/time.hpp>
#include <tigergrav/vect.hpp>

#include <vector>

struct monopole {
	float m;
	vect<float> x;
	float r;
};

struct source {
	float m;
	vect<float> x;
};

struct force {
	float phi;
	vect<float> g;
};

struct particle {
	vect<std::uint64_t> x;
	vect<float> v;
	rung_type rung;
	struct {
		std::uint8_t out : 1;
	} flags;
};

inline double pos_to_double(std::uint64_t x) {
	return ((double) x + (double) 0.5) / ((double) std::numeric_limits<std::uint64_t>::max() + (double) 1.0);
}

inline std::uint64_t double_to_pos(double x) {
	return x * ((double) std::numeric_limits<std::uint64_t>::max() + (double) 1.0);
}

inline vect<double> pos_to_double(vect<std::uint64_t> x) {
	vect<double> f;
	for (int dim = 0; dim < NDIM; dim++) {
		f[dim] = pos_to_double(x[dim]);
	}
	return f;
}

inline vect<std::uint64_t> double_to_pos(vect<double> d) {
	vect<std::uint64_t> x;
	for (int dim = 0; dim < NDIM; dim++) {
		x[dim] = double_to_pos(d[dim]);
	}
	return x;
}

using part_vect = std::vector<particle>;
using part_iter = part_vect::iterator;

template<class I, class F>
I bisect(I begin, I end, F &&below) {
	auto lo = begin;
	auto hi = end - 1;
	while (lo < hi) {
		if (!below(*lo)) {
			while (lo != hi) {
				if (below(*hi)) {
					auto tmp = *lo;
					*lo = *hi;
					*hi = tmp;
					break;
				}
				hi--;
			}

		}
		lo++;
	}
	return hi;
}

#endif /* TIGERGRAV_PARTICLE_HPP_ */
