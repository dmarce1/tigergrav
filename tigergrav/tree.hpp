#pragma once

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
		g = p = vect<float>(0);
	}
	statistics operator+( const statistics& other) const {
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

class tree {
	monopole mono;
	part_iter part_begin;
	part_iter part_end;
	bool leaf;
	float max_span;
	bool has_active;
	std::array<tree_ptr, NCHILD> children;

	static float theta_inv;
	static std::atomic<std::uint64_t> flop;

public:
	static void set_theta(float);
	static std::uint64_t get_flop();
	static tree_ptr new_(range, part_iter, part_iter);
	tree(range, part_iter, part_iter);
	monopole compute_monopoles();
	monopole get_monopole() const;
	bool is_leaf() const;
	std::array<tree_ptr, NCHILD> get_children() const;
	std::vector<vect<float>> get_positions() const;
	void drift(float);
//	void output(float,int) const;
	bool active_particles(int rung, bool do_out);
	kick_return kick_bh(std::vector<tree_ptr> dchecklist, std::vector<source> dsources, std::vector<tree_ptr> echecklist, std::vector<source> esources,
			rung_type min_rung, bool do_output);
	kick_return kick_direct(std::vector<source>&, rung_type min_rung, bool do_output);
	kick_return do_kick(const std::vector<force> &forces, rung_type min_rung, bool do_out);

};

