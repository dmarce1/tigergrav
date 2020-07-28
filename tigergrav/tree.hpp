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

#ifdef USE_HPX
#include <hpx/synchronization/spinlock.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/threads.hpp>
#else
#include <thread>
#include <mutex>
#include <future>
#endif

class tree;

using tree_ptr = std::shared_ptr<tree>;
using raw_tree_ptr = const tree*;

#ifdef USE_HPX

using mutex_type = hpx::lcos::local::spinlock;
template<class T>
using future_type = hpx::future<T>;

#else

using mutex_type = std::mutex;
template<class T>
using future_type = std::future<T>;

#endif

class node_attr;
class check_item;

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

struct multipole_info {
	multipole<ireal> m;
	vect<ireal> x;
	ireal r;
	bool has_active;
};
class tree_client {
	std::shared_ptr<tree> ptr;
public:
	tree_client() = default;
	tree_client(std::shared_ptr<tree> ptr_);
	tree* get_raw_ptr() const;
	std::pair<multipole_info, range> compute_multipoles(rung_type min_rung, bool do_out) const;
	void drift(float dt) const;
	kick_return kick_fmm(std::vector<check_item> dchecklist, std::vector<check_item> echecklist, const vect<ireal> &Lcom, expansion<float> L,
			rung_type min_rung, bool do_output) const;
	std::uint64_t get_flop() const;
};


class raw_tree_client {
	tree *ptr;
public:
	raw_tree_client() = default;
	raw_tree_client(tree *ptr_);
	node_attr get_node_attributes() const;
};


struct check_item {
	bool opened;
	raw_tree_client node;
};


struct node_attr {
	multipole_info multi;
	bool leaf;
	const_part_iter pbegin;
	const_part_iter pend;
	std::array<raw_tree_client, NCHILD> children;
};



class tree {
	multipole_info multi;
	part_iter part_begin;
	part_iter part_end;
	std::array<tree_client, NCHILD> children;
	int level;

	static float theta_inv;
	static std::atomic<std::uint64_t> flop;

public:
	static void set_theta(float);
	static std::uint64_t get_flop();
	static void reset_flop();
	static tree_client new_(range, part_iter, part_iter, int);
	tree(range, part_iter, part_iter, int level);
	bool is_leaf() const;
	std::pair<multipole_info, range> compute_multipoles(rung_type min_rung, bool do_out);
	node_attr get_node_attributes() const;
	void drift(float);
	kick_return kick_fmm(std::vector<check_item> dchecklist, std::vector<check_item> echecklist, const vect<ireal> &Lcom, expansion<float> L,
			rung_type min_rung, bool do_output);
	kick_return do_kick(const std::vector<force> &forces, rung_type min_rung, bool do_out);

};

inline tree_client::tree_client(std::shared_ptr<tree> ptr_) {
	ptr = ptr_;
}

inline tree* tree_client::get_raw_ptr() const {
	return &(*ptr);
}

inline std::pair<multipole_info, range> tree_client::compute_multipoles(rung_type min_rung, bool do_out) const {
	return ptr->compute_multipoles(min_rung, do_out);
}

inline void tree_client::drift(float dt) const {
	return ptr->drift(dt);
}

inline kick_return tree_client::kick_fmm(std::vector<check_item> dchecklist, std::vector<check_item> echecklist, const vect<ireal> &Lcom, expansion<float> L,
		rung_type min_rung, bool do_output) const {
	return ptr->kick_fmm(std::move(dchecklist), std::move(echecklist), Lcom, L, min_rung, do_output);
}

inline std::uint64_t tree_client::get_flop() const {
	return ptr->get_flop();
}


inline raw_tree_client::raw_tree_client(tree *ptr_) {
	ptr = ptr_;
}

inline node_attr raw_tree_client::get_node_attributes() const {
	return ptr->get_node_attributes();
}

