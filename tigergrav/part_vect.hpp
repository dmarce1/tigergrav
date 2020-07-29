#pragma once

#include <tigergrav/particle.hpp>
#include <tigergrav/options.hpp>
#include <tigergrav/range.hpp>


using part_iter = int;
using const_part_iter = int;

void part_vect_init();
std::vector<particle> part_vect_read(part_iter b, part_iter e);
void part_vect_write(part_iter b, part_iter e, std::vector<particle>);
part_iter part_vect_sort(part_iter b, part_iter e, double mid, int dim);
range part_vect_range(part_iter b, part_iter e);
