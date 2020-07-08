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
};

struct kick_return {
	statistics stats;
	rung_type rung;
};

struct multipole_attr {
	multipole<float> m;
	vect<float> x;
	float r;
};

class tree {
	multipole_attr multi;
	part_iter part_begin;
	part_iter part_end;
	vect<float> xc;
	bool leaf;
	bool has_active;
	std::array<tree_ptr, NCHILD> children;

	static std::atomic<std::uint64_t> flop;
	static int num_threads;
	static mutex_type thread_mtx;
	static mutex_type out_mtx;
	static bool inc_thread();
	static void dec_thread();

public:
	static std::uint64_t get_flop();
	static tree_ptr new_(range, part_iter, part_iter);
	tree(range, part_iter, part_iter);
	multipole_attr compute_multipoles();
	multipole_attr get_multipole() const;
	bool is_leaf() const;
	std::array<tree_ptr, NCHILD> get_children() const;
	std::vector<vect<float>> get_positions() const;
	void drift(float);
//	void output(float,int) const;
	bool active_particles(int rung, bool do_out);
	kick_return kick(std::vector<tree_ptr> dchecklist, std::vector<vect<float>>,  std::vector<multi_source>, std::vector<tree_ptr> echecklist, std::vector<mono_source> esources,
			rung_type min_rung, bool do_statistics, bool do_output);
};

