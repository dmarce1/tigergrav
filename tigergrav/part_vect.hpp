#pragma once

#include <tigergrav/particle.hpp>
#include <tigergrav/output.hpp>
#include <tigergrav/options.hpp>
#include <tigergrav/range.hpp>
#include <tigergrav/gravity.hpp>
#include <tigergrav/time.hpp>
#include <tigergrav/pinned_vector.hpp>

#ifdef HPX_LITE
#include <hpx/hpx_lite.hpp>
#else
#include <hpx/include/async.hpp>
#endif

#include <unordered_map>

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

using part_iter_pair = std::pair<part_iter,part_iter>;
using ymap_type = std::unordered_map<part_iter_pair,part_iter_pair, pair_hash>;

hpx::future<void> part_vect_gather_positions(pinned_vector<vect<pos_type>>&, ymap_type&);
double part_vect_find_median(part_iter b, part_iter e, int dim);
kick_return part_vect_kick_return();
void part_vect_write_glass();
void part_vect_write(part_iter b, part_iter e, std::vector<particle> these_parts);
hpx::future<std::vector<particle>> part_vect_read(part_iter b, part_iter e);
void part_vect_init();
void part_vect_reset_sort();
part_iter part_vect_massvolume_sort(part_iter b, part_iter e, int dim, double& x);
hpx::future<std::vector<vect<pos_type>>> part_vect_read_position(part_iter b, part_iter e);
std::vector<vect<pos_type>> part_vect_read_positions(const std::vector<std::pair<part_iter,part_iter>>&);
hpx::future<std::vector<particle_group_info>> part_vect_read_group(part_iter b, part_iter e, range r, bool use_range);
void part_vect_init_groups();
bool part_vect_find_groups(part_iter b, part_iter e, std::vector<particle_group_info>);
part_iter part_vect_sort(part_iter b, part_iter e, double mid, int dim);
range part_vect_range(part_iter b, part_iter e);
bool part_vect_is_local(const std::pair<part_iter,part_iter>&);
particle& part_vect_read_local(part_iter);
int part_vect_locality_id(part_iter);
void part_vect_reset();
std::pair<std::uint64_t, vect<double>> part_vect_center_of_mass(part_iter b, part_iter e);
multipole_info part_vect_multipole_info(vect<double> com, rung_type mrung, part_iter b, part_iter e);
double part_vect_drift(double t, rung_type);
std::vector<vect<pos_type>> part_vect_read_active_positions(part_iter b, part_iter e, rung_type rung);
hpx::future<void> part_vect_kick(part_iter b, part_iter e, rung_type rung, bool do_out, std::vector<force>& f, bool local = true);
void part_vect_find_groups2();

