#pragma once

#include <tigergrav/particle.hpp>
#include <tigergrav/options.hpp>
#include <tigergrav/range.hpp>

#include <hpx/include/async.hpp>


using part_iter = std::uint64_t;
using const_part_iter = std::uint64_t;

void part_vect_init();
hpx::future<std::vector<particle>> part_vect_read(part_iter b, part_iter e);
hpx::future<std::vector<vect<pos_type>>> part_vect_read_position(part_iter b, part_iter e);
void part_vect_write(part_iter b, part_iter e, std::vector<particle>);
part_iter part_vect_sort(part_iter b, part_iter e, double mid, int dim);
range part_vect_range(part_iter b, part_iter e);
int part_vect_locality_id(part_iter);
void part_vect_cache_reset();
