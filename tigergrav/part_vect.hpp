#pragma once

#include <tigergrav/particle.hpp>
#include <tigergrav/output.hpp>
#include <tigergrav/options.hpp>
#include <tigergrav/range.hpp>
#include <tigergrav/gravity.hpp>

#ifdef HPX_LITE
#include <hpx/hpx_lite.hpp>
#else
#include <hpx/include/async.hpp>
#endif

using part_iter = std::uint64_t;
using const_part_iter = std::uint64_t;


struct statistics {
	vect<double> g;
	vect<double> p;
	double pot;
	double kin;

	template<class A>
	void serialize(A &&arc, unsigned) {
		arc & g;
		arc & p;
		arc & pot;
		arc & kin;
	}

	void zero() {
		pot = kin = 0.0;
		g = p = vect<double>(0);
	}
	statistics operator+(const statistics &other) const {
		statistics C;
		C.g = g + other.g;
		C.p = p + other.p;
		C.pot = pot + other.pot;
		C.kin = kin + other.kin;
		return C;
	}
};



struct kick_return {
	statistics stats;
	rung_type rung;
	std::vector<output> out;
	template<class A>
	void serialize(A &&arc, unsigned) {
		arc & stats;
		arc & rung;
		arc & out;
	}
};

kick_return part_vect_kick_return();
void part_vect_write_glass();
void part_vect_write(part_iter b, part_iter e, std::vector<particle> these_parts);
hpx::future<std::vector<particle>> part_vect_read(part_iter b, part_iter e);
void part_vect_init();
hpx::future<std::vector<vect<pos_type>>> part_vect_read_position(part_iter b, part_iter e);
std::vector<vect<pos_type>> part_vect_read_positions(const std::vector<std::pair<part_iter,part_iter>>&);
hpx::future<std::vector<particle_group_info>> part_vect_read_group(part_iter b, part_iter e);
void part_vect_init_groups();
bool part_vect_find_groups(part_iter b, part_iter e, std::vector<particle_group_info>);
part_iter part_vect_sort(part_iter b, part_iter e, double mid, int dim);
range part_vect_range(part_iter b, part_iter e);
int part_vect_locality_id(part_iter);
void part_vect_reset();
std::pair<double, vect<double>> part_vect_center_of_mass(part_iter b, part_iter e);
multipole_info part_vect_multipole_info(vect<double> com, rung_type mrung, part_iter b, part_iter e);
double part_vect_drift(double dt);
std::vector<vect<pos_type>> part_vect_read_active_positions(part_iter b, part_iter e, rung_type rung);
hpx::future<void> part_vect_kick(part_iter b, part_iter e, rung_type rung, bool do_out, std::vector<force>&& f);
void part_vect_find_groups2();

