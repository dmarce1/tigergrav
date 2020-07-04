#pragma once

#include <tigergrav/options.hpp>
#include <tigergrav/particle.hpp>
#include <tigergrav/range.hpp>
#include <tigergrav/tree_id.hpp>

#include <array>
#include <atomic>
#include <memory>

#include <hpx/synchronization/spinlock.hpp>

class tree;

using tree_ptr = std::shared_ptr<tree>;

struct stats {
	vect<double> mom_tot;
#ifdef STORE_G
	vect<double> acc_tot;
#endif
	double kin_tot;
#ifdef STORE_G
	double pot_tot;
	double ene_tot;
	double virial_err;
#endif
	std::uint64_t flop;
};

using mutex_type = hpx::lcos::local::spinlock;

class tree {
	monopole mono;
	part_iter part_begin;
	part_iter part_end;
	bool leaf;
	std::array<tree_ptr, NCHILD> children;

	static std::atomic<std::uint64_t> flop;
	static int num_threads;
	static mutex_type mtx;
	static bool inc_thread();
	static void dec_thread();

public:
	static tree_ptr new_(range, part_iter, part_iter);
	tree(range, part_iter, part_iter);
	monopole compute_monopoles();
	monopole get_monopole() const;
	bool is_leaf() const;
	std::array<tree_ptr, NCHILD> get_children() const;
	std::vector<vect<float>> get_positions() const;
	void drift(float);
	void output(float,int) const;
	stats statistics() const;
#ifdef GLOBAL_DT
	void kick(float);
	float compute_gravity(std::vector<tree_ptr> checklist, std::vector<source> sources);
#else
	std::int8_t kick(std::vector<tree_ptr> checklist, std::vector<source> sources, std::int8_t min_rung);
#endif
};
