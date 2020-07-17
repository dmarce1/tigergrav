/*
 * particle.hpp
 *
 *  Created on: Jun 2, 2020
 *      Author: dmarce1
 */

#ifndef TIGERGRAV_PARTICLE_HPP_
#define TIGERGRAV_PARTICLE_HPP_

#include <tigergrav/position.hpp>
#include <tigergrav/time.hpp>

#include <vector>



struct particle {
	vect<pos_type> x;
	vect<float> v;
	rung_type rung;
	struct {
		std::uint8_t out : 1;
	} flags;
};

using part_vect = std::vector<particle>;
using part_iter = part_vect::iterator;
using const_part_iter = part_vect::const_iterator;

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
