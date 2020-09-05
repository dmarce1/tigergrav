#pragma once

#include <tigergrav/part_vect.hpp>

#define null_gwork_id (-1)

int gwork_assign_id();
void gwork_reset();
void gwork_checkin(int);
void gwork_show();
std::uint64_t gwork_pp_complete(int id, std::vector<force>* g, std::vector<vect<pos_type>>* x,
		const std::vector<std::pair<part_iter, part_iter>> &y, const std::vector<const multi_src*>& z, std::function<hpx::future<void>(void)>&&, bool do_phi);

