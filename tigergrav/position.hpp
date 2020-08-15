/*
 * position.hpp
 *
 *  Created on: Jul 11, 2020
 *      Author: dmarce1
 */

#ifndef TIGERGRAV_POSITION_HPP_
#define TIGERGRAV_POSITION_HPP_

#include <tigergrav/vect.hpp>

using pos_type = std::int32_t;

constexpr auto POS_MAX = ((double) std::numeric_limits<std::uint32_t>::max() + 1.0);
constexpr auto POS_INV = 1.0 / POS_MAX;

inline double pos_to_double(pos_type x) {
	return ((double) x) * POS_INV + 0.5;
}

inline vect<double> pos_to_double(vect<pos_type> x) {
	vect<double> f;
	for (int dim = 0; dim < NDIM; dim++) {
		f[dim] = pos_to_double(x[dim]);
	}
	return f;
}

inline pos_type double_to_pos(double x) {
	const auto y = x - 0.5;
	auto z = std::round(y * POS_MAX);
	if( z >= POS_MAX / 2.0 ) {
		z -= POS_MAX;
	}
	return z;
}

inline vect<pos_type> double_to_pos(vect<double> d) {
	vect<pos_type> x;
	for (int dim = 0; dim < NDIM; dim++) {
		x[dim] = double_to_pos(d[dim]);
	}
	return x;
}

#endif /* TIGERGRAV_POSITION_HPP_ */
