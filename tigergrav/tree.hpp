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
	template<class A>
	void serialize(A &&arc, unsigned) {
		arc & data;
		arc & fut;
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

struct refine_return {
	bool rc;
	int max_depth;
	int min_depth;
	std::uint64_t leaves;
	std::uint64_t nodes;
	template<class A>
	void serialize(A&& arc, unsigned) {
		arc & max_depth;
		arc & min_depth;
		arc & leaves;
		arc & nodes;
	}
};



struct interaction_stats {
	std::uint64_t PP_direct;
	std::uint64_t PP_ewald;
	std::uint64_t CC_direct;
	std::uint64_t CC_ewald;
	std::uint64_t CP_direct;
	std::uint64_t CP_ewald;
	interaction_stats() {
		PP_direct = PP_ewald = 0;
		CP_direct = CP_ewald = 0;
		CC_direct = CC_ewald = 0;
	}
	interaction_stats& operator+=( interaction_stats other) {
		PP_direct += other.PP_direct;
		PP_ewald += other.PP_ewald;
		CC_direct += other.CC_direct;
		CC_ewald += other.CC_ewald;
		CP_direct += other.CP_direct;
		CP_ewald += other.CP_ewald;
		return *this;
	}
	template<class A>
	void serialize(A&& arc, unsigned) {
		arc & PP_direct;
		arc & PP_ewald;
		arc & CP_direct;
		arc & CP_ewald;
		arc & CC_direct;
		arc & CC_ewald;
	}
};

class tree_client {
	id_type ptr;
public:
	template<class A>
	void serialize(A &&arc, unsigned) {
		arc & ptr;
	}
	bool local() const {
		return hpx::naming::get_locality_id_from_gid(ptr) == hpx::get_locality_id();
	}
	tree_client() = default;
	tree_client(id_type ptr_);
	check_item get_check_item() const;
	multipole_return compute_multipoles(rung_type min_rung, bool do_out, int wid, int stack_cnt) const;
	double drift(double t, rung_type r) const;
	interaction_stats kick_fmm(std::vector<check_item> dchecklist, std::vector<check_item> echecklist, const vect<double> &Lcom, expansion<double> L, rung_type min_rung,
			bool do_output, int stack_cnt) const;
	int find_groups(std::vector<check_item> dchecklist, int stack_cnt) const;
	refine_return refine(int) const;
};

class raw_tree_client {
	raw_id_type ptr;
public:
	raw_tree_client() = default;
	raw_tree_client(raw_id_type ptr_);
	future_data<node_attr> get_node_attributes() const;
	future_data<const multi_src*> get_multi_srcs() const;
	int get_locality() const {
		return ptr.loc_id;
	}
	template<class A>
	void serialize(A &&arc, unsigned) {
		arc & ptr;
	}
};

struct check_flags {
	std::uint8_t opened :1;
	std::uint8_t is_leaf :1;
	check_flags& operator=(check_flags f) {
		opened = f.opened;
		is_leaf = f.is_leaf;
		return *this;
	}
};

struct check_item {
	raw_tree_client node;
	vect<pos_type> x;
	box_id_type boxid;
	part_iter pbegin;
	part_iter pend;
	float r;
	check_flags flags;
	template<class A>
	void serialize(A &&arc, unsigned) {
		bool tmp = flags.opened;
		arc & tmp;
		flags.opened = tmp;
		tmp = flags.is_leaf;
		arc & tmp;
		flags.is_leaf = tmp;
		arc & node;
		arc & r;
		arc & x;
		arc & pbegin;
		arc & pend;
		arc & boxid;
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
	std::uint64_t N;
	template<class A>
	void serialize(A &&arc, unsigned) {
		arc & m;
		arc & r;
		arc & c;
		arc & N;
	}
};

class tree: public hpx::components::managed_component_base<tree> {
	multi_src multi;
	std::array<vect<pos_type>, NCHILD> cx;
	std::array<tree_client, NCHILD> children;
	std::array<raw_tree_client, NCHILD> cnode;
	std::array<part_iter, NCHILD> cpbegin;
	std::array<part_iter, NCHILD> cpend;
	box_id_type boxid;
	std::vector<std::pair<part_iter,part_iter>> group_ranges;
	part_iter part_begin;
	part_iter part_end;
	std::array<float, NCHILD> cr;
	std::array<check_flags, NCHILD> cflags;
	std::uint64_t num_active;
	int gwork_id;
	float r;
	struct {
		std::uint32_t level :6;
		std::uint32_t depth :6;
		std::uint32_t rdepth :6;
		std::uint32_t ldepth :6;
		std::uint32_t max_iter : 7;
		std::uint32_t leaf :1;
	} flags;

	static double theta_inv;
	static std::atomic<std::uint64_t> flop;
	static double pct_active;

public:
	template<class A>
	void serialize(A &&arc, unsigned) {
		std::uint32_t tmp;
		tmp = flags.leaf;
		arc & tmp;
		flags.leaf = tmp;
		tmp = flags.level;
		arc & tmp;
		flags.level = tmp;
		tmp = flags.depth;
		arc & tmp;
		flags.depth = tmp;
		tmp = flags.rdepth;
		arc & tmp;
		flags.rdepth = tmp;
		tmp = flags.ldepth;
		arc & tmp;
		flags.ldepth = tmp;
		arc & gwork_id;
		arc & multi;
		arc & r;
		arc & num_active;
		arc & part_begin;
		arc & part_end;
		arc & children;
		arc & boxid;
		arc & cnode;
		arc & cx;
		arc & cpbegin;
		arc & cpend;
		arc & cr;
		for (int i = 0; i < NCHILD; i++) {
			tmp = cflags[i].opened;
			arc & tmp;
			cflags[i].opened = tmp;
			tmp = cflags[i].is_leaf;
			arc & tmp;
			cflags[i].is_leaf = tmp;
		}
	}
	static std::uint64_t get_flop();
	static double get_pct_active() {
		return pct_active;
	}
	tree() = default;
	static void set_theta(double);
	static void reset_flop();
	tree(box_id_type, part_iter, part_iter, int level);
	refine_return refine(int);
	bool is_leaf() const;
	multipole_return compute_multipoles(rung_type min_rung, bool do_out, int workid, int stack_cnt);
	node_attr get_node_attributes() const;
	const multi_src* get_multi_srcs() const;
	double drift(double,rung_type);
	interaction_stats kick_fmm(std::vector<check_item> dchecklist, std::vector<check_item> echecklist, const vect<double> &Lcom, expansion<double> L, rung_type min_rung,
			bool do_output, int stack_ccnt);

	void find_groups(std::vector<check_item> checklist, int stack_ccnt);

	check_item get_check_item() const; //
	HPX_DEFINE_COMPONENT_DIRECT_ACTION(tree,refine);//
	HPX_DEFINE_COMPONENT_DIRECT_ACTION(tree,compute_multipoles);//
	HPX_DEFINE_COMPONENT_DIRECT_ACTION(tree,find_groups);//
	HPX_DEFINE_COMPONENT_DIRECT_ACTION(tree,kick_fmm);//
	HPX_DEFINE_COMPONENT_DIRECT_ACTION(tree,drift);//
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

inline double tree_client::drift(double t, rung_type r) const {
	return tree::drift_action()(ptr, t, r);
}

inline refine_return tree_client::refine(int stack_cnt) const {
	return tree::refine_action()(ptr, stack_cnt);
}

inline interaction_stats tree_client::kick_fmm(std::vector<check_item> dchecklist, std::vector<check_item> echecklist, const vect<double> &Lcom, expansion<double> L,
		rung_type min_rung, bool do_output, int stack_cnt) const {
	return tree::kick_fmm_action()(ptr, std::move(dchecklist), std::move(echecklist), Lcom, L, min_rung, do_output, stack_cnt);
}

inline int tree_client::find_groups(std::vector<check_item> dchecklist, int stack_cnt) const {
	tree::find_groups_action()(ptr, std::move(dchecklist), stack_cnt);
	return 0;
}
