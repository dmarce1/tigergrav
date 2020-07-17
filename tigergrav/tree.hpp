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
	vect<float> g;
	vect<float> p;
	float pot;
	float kin;
	void zero() {
		pot = kin = 0.0;
		g = p = vect<float>(0);
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
	std::array<tree_ptr, NCHILD> children;
	static float theta_inv;
	static std::atomic<std::uint64_t> flop;

public:
	static void set_theta(float);
	static std::uint64_t get_flop();
	static void reset_flop();
	static tree_ptr new_(range, part_iter, part_iter);
	tree(range, part_iter, part_iter);
	std::pair<multipole_info, range> compute_multipoles(rung_type min_rung, bool do_out);
	multipole_info get_multipole() const;
	bool is_leaf() const;
	std::array<tree_ptr, NCHILD> get_children() const;
	std::pair<const_part_iter, const_part_iter> get_positions() const;
	void drift(float);
//	void output(float,int) const;
	bool active_particles(int rung, bool do_out);
	kick_return kick_bh(std::vector<tree_ptr> dchecklist, std::vector<vect<float>> dsources, std::vector<multi_src> multi_srcs,
			std::vector<tree_ptr> echecklist, std::vector<vect<float>> esources, std::vector<multi_src> emulti_srcs, rung_type min_rung, bool do_output);
	kick_return kick_fmm(std::vector<check_item> dchecklist, std::vector<check_item> echecklist, const vect<ireal> &Lcom, expansion<float> L,
			rung_type min_rung, bool do_output);
	kick_return kick_direct(std::vector<vect<float>> &sources, rung_type min_rung, bool do_output);
	kick_return do_kick(const std::vector<force> &forces, rung_type min_rung, bool do_out);

};

