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


using part_iter = std::uint64_t;


#define DEFAULT_GROUP ((std::uint64_t) (0xFFFFFFFFFFFFFF))

struct particle {
	vect<pos_type> x;
	vect<float> v;
	struct {
		std::uint64_t out :1;
		std::uint64_t rung :7;
		std::uint64_t group :56;
	} flags;
	template<class A>
	void serialize(A &&arc, unsigned) {
		std::uint64_t tmp;
		arc & x;
		arc & v;
		tmp = flags.out;
		arc & tmp;
		flags.out = tmp;
		tmp = flags.rung;
		arc & tmp;
		flags.rung = tmp;
		tmp = flags.group;
		arc & tmp;
		flags.group = tmp;
	}
};

struct particle_group_info {
	vect<pos_type> x;
	std::uint64_t id;
	template<class A>
	void serialize(A &&arc, unsigned) {
		arc & x;
		arc & id;
	}
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


struct pair_hash {
	std::size_t operator()(std::pair<part_iter, part_iter> p) const {
		return p.first * p.second;
	}
};

#endif /* TIGERGRAV_PARTICLE_HPP_ */
