/*
 * particle.hpp
 *
 *  Created on: Jun 2, 2020
 *      Author: dmarce1
 */

#ifndef TIGERGRAV_PARTICLE_HPP_
#define TIGERGRAV_PARTICLE_HPP_

#include <tigergrav/fixed_real.hpp>
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
#ifdef STORE_G
	vect<float> g;
	float phi;
#ifndef GLOBAL_DT
	std::int8_t rung;
#endif
#else
#ifdef GLOBAL_DT
	vect<float> g;
#else
	std::int8_t rung;
#endif
#endif
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

inline float rung_to_dt(std::int8_t rung) {
	return 1.0 / (1 << rung);
}

inline std::int8_t dt_to_rung(float dt) {
	int rung = 0;
	while (rung_to_dt(rung) > dt) {
		rung++;
		if (rung == std::numeric_limits<std::int8_t>::max()) {
			printf("logic error %s %i\n", __FILE__, __LINE__);
			abort();
		}
	}
	return rung;
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
