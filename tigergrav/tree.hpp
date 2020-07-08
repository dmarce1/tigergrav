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
	multipole<double> m;
	vect<double> x;
	double r;
};

struct check_item {
	tree_ptr ptr;
	bool use_parts;
};

class tree {
	multipole_attr multi;
	part_iter part_begin;
	part_iter part_end;
	vect<float> xc;
	bool leaf;
	bool has_active;
	std::array<tree_ptr, NCHILD> children;
	std::array<vect<float>, NCHILD> child_com;

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
	bool parts_separate_from(const vect<float>& x, const float r);
	bool parts_separate_from_far_ewald(const vect<float>& x, const float r);
	kick_return kick( expansion<double>, std::vector<check_item> dchecklist, std::vector<check_item> echecklist, rung_type min_rung,
			bool do_statistics, bool do_output);
};

