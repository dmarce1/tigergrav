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
	vect<double> min;
	vect<double> max;
	template<class Arc>
	void serialize(Arc &arc, unsigned) {
		arc & min;
		arc & max;
	}
};

using box_id_type = unsigned __int128;

range box_id_to_range(box_id_type id);

double range_max_span(const range &r);
range reflect_range(const range&, int dim, double axis);
vect<double> range_center(const range &r);
range shift_range(const range &r, const vect<double>&);
range scale_range(const range&, double);
vect<double> range_span(const range&);
bool in_range(const vect<double>&, const range&);
bool in_range(const range&, const range&);
bool ranges_intersect(const range&, const range&);
range range_around(const vect<double>&, double);
range range_expand(const range&, double);
double range_volume(const range&);
range null_range();
bool operator==(const range&, const range&);
bool operator!=(const range&, const range&);

#endif /* MATH_HPP_ */

