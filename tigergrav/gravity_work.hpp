#pragma once

#include <tigergrav/part_vect.hpp>

#define null_gwork_id (-1)

int gwork_assign_id();
void gwork_reset();
void gwork_checkin(int);
void gwork_show();
void gwork_pp_complete(int id, std::shared_ptr<std::vector<force>> g, std::shared_ptr<std::vector<vect<pos_type>>> x,
		const std::vector<std::pair<part_iter, part_iter>> &y, std::function<hpx::future<void>(void)>&&);
