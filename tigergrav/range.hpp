/*
 * math.hpp
 *
 *  Created on: Nov 29, 2019
 *      Author: dmarce1
 */

#ifndef RANGE_HPP_
#define RANGE_HPP_

#include <tigergrav/vect.hpp>

#include <limits>

struct range {
	vect<float> min;
	vect<float> max;
	template<class Arc>
	void serialize(Arc& arc,unsigned) {
		arc & min;
		arc & max;
	}
};

range reflect_range(const range&, int dim, float axis);
vect<float> range_center(const range &r);
range shift_range(const range& r, const vect<float>&);
range scale_range(const range& , float);
vect<float> range_span(const range&);
bool in_range(const vect<float>&, const range&);
bool in_range(const range&, const range&);
bool ranges_intersect(const range&, const range&);
range range_around(const vect<float>&, float);
range expand_range(const range&, float);
float range_volume(const range&);
range null_range();
bool operator==(const range&, const range&);
bool operator!=(const range&, const range&);

#endif /* MATH_HPP_ */

