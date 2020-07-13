#pragma once

#include <tigergrav/gravity.hpp>
#include <tigergrav/multipole.hpp>
#include <tigergrav/output.hpp>
#include <tigergrav/options.hpp>
#include <tigergrav/particle.hpp>
#include <tigergrav/range.hpp>

#include <array>
#include <atomic>
#include <memory>

#include <hpx/synchronization/spinlock.hpp>

class tree;

using tree_ptr = std::shared_ptr<tree>;

using mutex_type = hpx::lcos::local::spinlock;

struct statistics {
	vect<double> g;
	vect<double> p;
	double pot;
	double kin;
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
};

struct check_item {
	bool opened;
	tree_ptr ptr;
};


struct multipole_info {
	multipole<ireal> m;
	vect<ireal> x;
	ireal r;
	bool has_active;
};


class tree {
	multipole_info multi;
	part_iter part_begin;
	part_iter part_end;
	bool leaf;
	std::array<tree_ptr, NCHILD> children;
	std::array<vect<double>, NCHILD> child_com;
	vect<double> coord_cent;

	static double theta_inv;
	static std::atomic<std::uint64_t> flop;

public:
	static void set_theta(double);
	static std::uint64_t get_flop();
	static void reset_flop();
	static tree_ptr new_(range, part_iter, part_iter);
	tree(range, part_iter, part_iter);
	multipole_info compute_multipoles(rung_type min_rung, bool do_out);
	multipole_info get_multipole() const;
	monopole get_monopole() const;
	bool is_leaf() const;
	std::array<tree_ptr, NCHILD> get_children() const;
	const_part_set get_positions() const;
	void drift(double);
//	void output(double,int) const;
	bool active_particles(int rung, bool do_out);
	kick_return kick_bh(std::vector<tree_ptr> dchecklist, std::vector<const_part_set> dsources, std::vector<multi_src> multi_srcs,
			std::vector<tree_ptr> echecklist, std::vector<source> esources, rung_type min_rung, bool do_output);
	kick_return kick_fmm(std::vector<check_item> dchecklist, std::vector<check_item> echecklist, expansion<ireal> L, rung_type min_rung, bool do_output);
	kick_return kick_direct(std::vector<const_part_set>&, rung_type min_rung, bool do_output);
	kick_return do_kick(const std::vector<force> &forces, rung_type min_rung, bool do_out);

};

