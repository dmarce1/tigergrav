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

#include <hpx/lcos/local/spinlock.hpp>

#include <mutex>
#include <stack>
#include <vector>

#include <forward_list>

struct particle {
	vect<fixed_real> x;
	vect<float> v;
	fixed_real dt;
	fixed_real t;
};

using part_vect = std::vector<particle>;

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
