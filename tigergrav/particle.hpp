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

#include <tigergrav/simd.hpp>

#include <vector>

using pos_type = std::int32_t;

struct particle {
	vect<pos_type> x;
	vect<float> v;
	rung_type rung;
	struct {
		std::uint8_t out :1;
	} flags;
};

inline simd_vector pos_to_simd_vector(simd_int_vector i) {
	return (simd_vector(i)) / simd_vector((long long) 1 << (long long) 31);

}

inline double pos_to_double(pos_type x) {
	return ((double) x) / ((long long) 1 << (long long) 31);
}

inline pos_type double_to_pos(double x) {
	return x * (((double) ((long long) 1 << (long long) 31)));
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
