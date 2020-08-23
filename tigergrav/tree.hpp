#pragma once

#include <tigergrav/gravity.hpp>
#include <tigergrav/multipole.hpp>
#include <tigergrav/output.hpp>
#include <tigergrav/options.hpp>
#include <tigergrav/part_vect.hpp>
#include <tigergrav/range.hpp>
#include <tigergrav/gravity_work.hpp>

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

template<class T>
struct future_data {
	hpx::future<T> fut;
	T data;
	T get() {
		if (fut.valid()) {
			return fut.get();
		} else {
			return data;
		}
	}
};

using mutex_type = hpx::lcos::local::spinlock;
template<class T>
using future_type = hpx::future<T>;

class node_attr;
class multi_src;
class check_item;
class multipole_return;

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
	void serialize(A &&arc, unsigned) {
		arc & ptr;
	}
	tree_client() = default;
	tree_client(id_type ptr_);
	check_item get_check_item() const;
	multipole_return compute_multipoles(rung_type min_rung, bool do_out, int wid, int stack_cnt) const;
	double drift(double dt) const;
	int kick_fmm(std::vector<check_item> dchecklist, std::vector<check_item> echecklist, const vect<double> &Lcom, expansion<double> L, rung_type min_rung,
			bool do_output, int stack_cnt) const;
	bool find_groups(std::vector<check_item> dchecklist, int stack_cnt) const;
	bool refine(int) const;
};

class raw_tree_client {
	raw_id_type ptr;
public:
	raw_tree_client() = default;
	raw_tree_client(raw_id_type ptr_);
	future_data<node_attr> get_node_attributes() const;
	future_data<multi_src> get_multi_srcs() const;
	int get_locality() const {
		return ptr.loc_id;
	}
	template<class A>
	void serialize(A &&arc, unsigned) {
		arc & ptr;
	}
};

struct check_item {
	raw_tree_client node;
	vect<pos_type> x;
	part_iter pbegin;
	part_iter pend;
	float r;
	struct {
		std::uint8_t opened :1;
		std::uint8_t is_leaf :1;
		std::uint8_t group_active :1;
	} flags;
	template<class A>
	void serialize(A &&arc, unsigned) {
		bool tmp = flags.opened;
		arc & tmp;
		flags.opened = tmp;
		tmp = flags.is_leaf;
		arc & tmp;
		flags.is_leaf = tmp;
		tmp = flags.group_active;
		arc & tmp;
		flags.group_active = tmp;
		arc & node;
		arc & r;
		arc & x;
		arc & pbegin;
		arc & pend;
	}

};

struct node_attr {
	std::array<check_item, NCHILD> children;
	template<class A>
	void serialize(A &&arc, unsigned) {
		arc & children;
	}
};

struct multipole_return {
	multipole_info m;
	range r;
	check_item c;
	template<class A>
	void serialize(A &&arc, unsigned) {
		arc & m;
		arc & r;
		arc & c;
	}
};

class tree: public hpx::components::managed_component_base<tree> {
	multipole_info multi;
	std::array<tree_client, NCHILD> children;
	std::array<check_item, NCHILD> child_check;
	box_id_type boxid;
	part_iter part_begin;
	part_iter part_end;
	int gwork_id;
	struct {
		std::uint8_t level        :6;
		std::uint8_t leaf 		  :1;
		std::uint8_t group_active :1;
	} flags;

	static double theta_inv;
	static std::atomic<std::uint64_t> flop;
	static double pct_active;


public:
	template<class A>
	void serialize(A &&arc, unsigned) {
		std::uint32_t tmp;
		tmp = flags.group_active;
		arc & tmp;
		flags.group_active = tmp;
		tmp = flags.leaf;
		arc & tmp;
		flags.leaf = tmp;
		tmp = flags.level;
		arc & tmp;
		flags.level = tmp;
		arc & gwork_id;
		arc & multi;
		arc & part_begin;
		arc & part_end;
		arc & children;
		arc & child_check;
		arc & boxid;
	}
	static std::uint64_t get_flop();
	static double get_pct_active() {
		return pct_active;
	}
	tree() = default;
	static void set_theta(double);
	static void reset_flop();
	tree(box_id_type, part_iter, part_iter, int level);
	bool refine(int);
	bool is_leaf() const;
	multipole_return compute_multipoles(rung_type min_rung, bool do_out, int workid, int stack_cnt);
	node_attr get_node_attributes() const;
	multi_src get_multi_srcs() const;
	double drift(double);
	int kick_fmm(std::vector<check_item> dchecklist, std::vector<check_item> echecklist, const vect<double> &Lcom, expansion<double> L, rung_type min_rung,
			bool do_output, int stack_ccnt);

	bool find_groups(std::vector<check_item> checklist, int stack_ccnt);

	check_item get_check_item() const; //
	HPX_DEFINE_COMPONENT_DIRECT_ACTION(tree,refine); 				//
	HPX_DEFINE_COMPONENT_DIRECT_ACTION(tree,compute_multipoles); 				//
	HPX_DEFINE_COMPONENT_DIRECT_ACTION(tree,find_groups); 				//
	HPX_DEFINE_COMPONENT_DIRECT_ACTION(tree,kick_fmm); 				//
	HPX_DEFINE_COMPONENT_DIRECT_ACTION(tree,drift); 				//
	HPX_DEFINE_COMPONENT_DIRECT_ACTION(tree,get_check_item);
	//
};

inline tree_client::tree_client(id_type ptr_) {
	ptr = ptr_;
}

inline raw_tree_client::raw_tree_client(raw_id_type ptr_) {
	ptr = ptr_;
}

inline check_item tree_client::get_check_item() const {
	return tree::get_check_item_action()(ptr);
}

inline multipole_return tree_client::compute_multipoles(rung_type min_rung, bool do_out, int wid, int stack_cnt) const {
	return tree::compute_multipoles_action()(ptr, min_rung, do_out, wid, stack_cnt);
}

inline double tree_client::drift(double dt) const {
	return tree::drift_action()(ptr, dt);
}

inline bool tree_client::refine(int stack_cnt) const {
	return tree::refine_action()(ptr, stack_cnt);
}

inline int tree_client::kick_fmm(std::vector<check_item> dchecklist, std::vector<check_item> echecklist, const vect<double> &Lcom, expansion<double> L,
		rung_type min_rung, bool do_output, int stack_cnt) const {
	return tree::kick_fmm_action()(ptr, std::move(dchecklist), std::move(echecklist), Lcom, L, min_rung, do_output, stack_cnt);
}

inline bool tree_client::find_groups(std::vector<check_item> dchecklist, int stack_cnt) const {
	return tree::find_groups_action()(ptr, std::move(dchecklist), stack_cnt);
}
