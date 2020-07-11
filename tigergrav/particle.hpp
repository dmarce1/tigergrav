/*
 * particle.hpp
 *
 *  Created on: Jun 2, 2020
 *      Author: dmarce1
 */

#ifndef TIGERGRAV_PARTICLE_HPP_
#define TIGERGRAV_PARTICLE_HPP_

#include <tigergrav/time.hpp>
#include <tigergrav/vect.hpp>

#include <vector>

using pos_type = std::uint32_t;

struct monopole {
	float m;
	vect<float> x;
	float r;
};

struct source {
	float m;
	vect<float> x;
};


struct particle {
	vect<pos_type> x;
	vect<float> v;
	rung_type rung;
	struct {
		std::uint8_t out : 1;
	} flags;
};

inline double pos_to_double(pos_type x) {
	return ((double) x + (double) 0.5) / ((double) std::numeric_limits<pos_type>::max() + (double) 1.0);
}

inline pos_type double_to_pos(double x) {
	return x * ((double) std::numeric_limits<pos_type>::max() + (double) 1.0) + 0.5;
}

inline vect<double> pos_to_double(vect<pos_type> x) {
	vect<double> f;
	for (int dim = 0; dim < NDIM; dim++) {
		f[dim] = pos_to_double(x[dim]);
	}
	return f;
}

inline vect<pos_type> double_to_pos(vect<double> d) {
	vect<pos_type> x;
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
