#pragma once

#include <tigergrav/gravity.hpp>
#include <tigergrav/multipole.hpp>
#include <tigergrav/output.hpp>
#include <tigergrav/options.hpp>
#include <tigergrav/part_vect.hpp>
#include <tigergrav/range.hpp>

#include <array>
#include <atomic>
#include <memory>

#ifndef HPX_LITE
#include <hpx/synchronization/spinlock.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/threads.hpp>
#include <hpx/include/components.hpp>
#else
#include <hpx/hpx_lite.hpp>
#endif

class tree;

using mutex_type = hpx::lcos::local::spinlock;
template<class T>
using future_type = hpx::future<T>;

class node_attr;
class check_item;

struct statistics {
	vect<float> g;
	vect<float> p;
	float pot;
	float kin;

	template<class A>
	void serialize(A &&arc, unsigned) {
		arc & g;
		arc & p;
		arc & pot;
		arc & kin;
	}

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
	template<class A>
	void serialize(A &&arc, unsigned) {
		arc & stats;
		arc & rung;
		arc & out;
	}
};


struct raw_id_type {
	int loc_id;
	std::uint64_t ptr;
	template<class A>
	void serialize(A &&arc, unsigned) {
		arc & loc_id;
		arc & ptr;
	}
	bool operator==(const raw_id_type &other) const {
		return ptr == other.ptr && loc_id == other.loc_id;
	}
};
using id_type = hpx::id_type;

class tree_client {
	id_type ptr;
public:
	template<class A>
	void serialize(A&& arc, unsigned) {
		arc & ptr;
	}
	tree_client() = default;
	tree_client(id_type ptr_);
	raw_id_type get_raw_ptr() const;
	std::pair<multipole_info, range> compute_multipoles(rung_type min_rung, bool do_out, int stack_cnt) const;
	void drift(float dt) const;
	kick_return kick_fmm(std::vector<check_item> dchecklist, std::vector<check_item> echecklist, const vect<ireal> &Lcom, expansion<float> L,
			rung_type min_rung, bool do_output, int stack_cnt) const;
	std::uint64_t get_flop() const;
};

class raw_tree_client {
	raw_id_type ptr;
public:
	raw_tree_client() = default;
	raw_tree_client(raw_id_type ptr_);
	hpx::future<node_attr> get_node_attributes() const;
	std::pair<multipole_info, range> compute_multipoles(rung_type min_rung, bool do_out, int stack_cnt) const;
	int get_locality() const {
		return ptr.loc_id;
	}
	template<class A>
	void serialize(A &&arc, unsigned) {
		arc & ptr;
	}
};

struct check_item {
	bool opened;
	raw_tree_client node;
	template<class A>
	void serialize(A &&arc, unsigned) {
		arc & opened;
		arc & node;
	}

};

struct node_attr {
	multipole_info multi;
	bool leaf;
	const_part_iter pbegin;
	const_part_iter pend;
	std::array<raw_tree_client, NCHILD> children;
	template<class A>
	void serialize(A &&arc, unsigned) {
		arc & multi;
		arc & leaf;
		arc & pbegin;
		arc & pend;
		arc & children;
	}
};

class tree: public hpx::components::managed_component_base<tree> {
	multipole_info multi;
	part_iter part_begin;
	part_iter part_end;
	std::array<tree_client, NCHILD> children;
	std::array<raw_id_type, NCHILD> raw_children;
	int level;

	static float theta_inv;
	static std::atomic<std::uint64_t> flop;

public:
	template<class A>
	void serialize(A&& arc, unsigned) {
		arc & multi;
		arc & part_begin;
		arc & part_end;
		arc & children;
		arc & raw_children;
		arc & level;
	}
	tree() = default;
	static void set_theta(float);
	std::uint64_t get_flop();
	static void reset_flop();
	static tree_client new_(range, part_iter, part_iter, int, int);
	tree(range, part_iter, part_iter, int level, int stack_cnt);
	bool is_leaf() const;
	std::pair<multipole_info, range> compute_multipoles(rung_type min_rung, bool do_out, int stack_cnt);
	node_attr get_node_attributes() const;
	void drift(float);
	kick_return kick_fmm(std::vector<check_item> dchecklist, std::vector<check_item> echecklist, const vect<ireal> &Lcom, expansion<float> L,
			rung_type min_rung, bool do_output, int stack_ccnt);
	kick_return do_kick(const std::vector<force> &forces, rung_type min_rung, bool do_out);
	raw_id_type get_raw_ptr() const;//
	HPX_DEFINE_COMPONENT_DIRECT_ACTION(tree,get_flop); 				//
	HPX_DEFINE_COMPONENT_DIRECT_ACTION(tree,compute_multipoles);//
	HPX_DEFINE_COMPONENT_DIRECT_ACTION(tree,kick_fmm);//
	HPX_DEFINE_COMPONENT_DIRECT_ACTION(tree,drift);//
	HPX_DEFINE_COMPONENT_DIRECT_ACTION(tree,get_raw_ptr);
	//
};

inline tree_client::tree_client(id_type ptr_) {
	ptr = ptr_;
}

inline raw_tree_client::raw_tree_client(raw_id_type ptr_) {
	ptr = ptr_;
}

inline raw_id_type tree_client::get_raw_ptr() const {
	return tree::get_raw_ptr_action()(ptr);
}

inline std::pair<multipole_info, range> tree_client::compute_multipoles(rung_type min_rung, bool do_out, int stack_cnt) const {
	return tree::compute_multipoles_action()(ptr, min_rung, do_out, stack_cnt);
}

inline void tree_client::drift(float dt) const {
	return tree::drift_action()(ptr, dt);
}

inline kick_return tree_client::kick_fmm(std::vector<check_item> dchecklist, std::vector<check_item> echecklist, const vect<ireal> &Lcom, expansion<float> L,
		rung_type min_rung, bool do_output, int stack_cnt) const {
	return tree::kick_fmm_action()(ptr, std::move(dchecklist), std::move(echecklist), Lcom, L, min_rung, do_output, stack_cnt);
}

inline std::uint64_t tree_client::get_flop() const {
	return tree::get_flop_action()(ptr);
}
