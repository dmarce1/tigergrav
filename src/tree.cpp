#include <tigergrav/options.hpp>
#include <tigergrav/time.hpp>
#include <tigergrav/tree.hpp>
#include <tigergrav/groups.hpp>
#include <tigergrav/memory.hpp>
#include <tigergrav/gravity_cuda.hpp>
#include <tigergrav/timer.hpp>

#include <atomic>
#include <algorithm>
#include <stack>
#include <thread>
#include <unordered_map>

#ifdef HPX_LITE
HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(hpx::components::managed_component<tree>, tree);

using compute_multipoles_action_type = tree::compute_multipoles_action;
using kick_fmm_action_type = tree::kick_fmm_action;
using find_groups_action_type = tree::find_groups_action;
using drift_action_type = tree:: drift_action;
using get_check_item_action_type = tree::get_check_item_action;
using refine_action_type = tree::refine_action;

HPX_REGISTER_ACTION(refine_action_type);
HPX_REGISTER_ACTION(compute_multipoles_action_type);
HPX_REGISTER_ACTION(find_groups_action_type);
HPX_REGISTER_ACTION(kick_fmm_action_type);
HPX_REGISTER_ACTION(drift_action_type);
HPX_REGISTER_ACTION(get_check_item_action_type);

#else
#include <hpx/runtime/actions/plain_action.hpp>
HPX_REGISTER_COMPONENT(hpx::components::managed_component<tree>, tree);
#endif

#define MAX_STACK 18

std::atomic<std::uint64_t> tree::flop(0);
float tree::theta_inv;
double tree::pct_active;
std::uint64_t workgroup_size;

#include <unordered_set>

void reset_node_cache();

static std::atomic<int> num_threads(1);
static bool inc_thread();
static void dec_thread();

static std::vector<hpx::id_type> localities;
static int myid;
static int hardware_concurrency = std::thread::hardware_concurrency();
static int target_nthreads = hardware_concurrency;
static std::unordered_set<check_item*> check_ptrs;
static std::unordered_set<multi_src*> multi_src_ptrs;
static mutex_type check_ptr_mtx;
static mutex_type multi_src_ptr_mtx;

struct raw_id_type_hash {
	std::size_t operator()(raw_id_type id) const {
		return id.ptr ^ id.loc_id;
	}
};

//hpx::future<std::vector<check_item>> get_child_checks(std::vector<raw_id_type> &&ids);

//HPX_PLAIN_ACTION (get_child_checks);
//
//hpx::future<std::vector<check_item>> get_child_checks(std::vector<raw_id_type> &&ids) {
//	std::vector<check_item> checks;
//	std::unordered_map<int, std::vector<raw_id_type>> map;
//	for (int i = 0; i < ids.size(); i++) {
//		if (ids[i].loc_id != myid) {
//			map[ids[i].loc_id].push_back(ids[i]);
//		}
//	}
//	std::vector<hpx::future<std::vector<check_item>>> futs;
//	for (const auto& i : map) {
//		futs.push_back(hpx::async < get_child_checks_action > (localities[i.first], i.second));
//	}
//	for (int i = 0; i < ids.size(); i++) {
//		if (ids[i].loc_id == myid) {
//			auto tmp = reinterpret_cast<tree*>(ids[i].ptr)->get_node_attributes();
//			checks.push_back(tmp->children[1]);
//			checks.push_back(tmp->children[0]);
//		}
//	}
//	int size = 2 * ids.size();
//	return hpx::async([size](decltype(futs)&& futs, decltype(checks)&& checks) {
//		for( auto& f : futs) {
//			auto tmp = f.get();
//			for( int i = 0; i < tmp.size(); i++) {
//				checks.push_back(std::move(tmp[i]));
//			}
//		}
//		return checks;
//	},std::move(futs),std::move(checks));
//}

void yield_to_hpx() {
	num_threads--;
	hpx::this_thread::yield();
	num_threads++;
}

void manage_checkptr(check_item *ptr) {
	std::lock_guard<mutex_type> lock(check_ptr_mtx);
	check_ptrs.insert(ptr);
}

HPX_PLAIN_ACTION (trash_checkptrs);

void trash_checkptrs() {
	std::vector<hpx::future<void>> futs;
	if (myid == 0) {
		for (int i = 1; i < futs.size(); i++) {
			futs.push_back(hpx::async < trash_checkptrs_action > (localities[i]));
		}
	}
	for (auto ptr : check_ptrs) {
		delete ptr;
	}
	check_ptrs.clear();
	hpx::wait_all(futs.begin(), futs.end());
}

void manage_multi_src(multi_src *ptr) {
	std::lock_guard<mutex_type> lock(multi_src_ptr_mtx);
	multi_src_ptrs.insert(ptr);
}

HPX_PLAIN_ACTION (trash_multi_srcs);

void trash_multi_srcs() {
	std::vector<hpx::future<void>> futs;
	if (myid == 0) {
		for (int i = 1; i < futs.size(); i++) {
			futs.push_back(hpx::async < trash_multi_srcs_action > (localities[i]));
		}
	}
	for (auto ptr : multi_src_ptrs) {
		delete ptr;
	}
	multi_src_ptrs.clear();
	hpx::wait_all(futs.begin(), futs.end());
}

template<class F>
inline auto thread_if_avail(F &&f, bool left, int nparts, bool force, bool remote, int stack_cnt) {
	const auto opts = options::get();
	bool thread;
	bool thread_bcof_stack = false;
	if (remote) {
		thread = true;
		thread_bcof_stack = false;
	} else if (force) {
		thread = true;
		thread_bcof_stack = true;
	} else {
		if (nparts > workgroup_size) {
			if (left && num_threads < opts.oversubscription * hardware_concurrency) {
				thread = true;
			} else {
				thread = (stack_cnt == MAX_STACK - 1);
				thread_bcof_stack = true;
				if (thread)
					printf("Stack thread\n");
			}
		} else {
			thread = (stack_cnt == MAX_STACK - 1);
			thread_bcof_stack = true;
			if (thread)
				printf("Stack thread\n");
		}
	}
	if (thread) {
		if (!thread_bcof_stack) {
			num_threads++;
			auto rc = hpx::async([](F &&f) {
				auto rc = f(0);
				num_threads--;
				return rc;
			},std::forward<F>(f));
			return rc;
		} else {
			return hpx::make_ready_future(hpx::async([](F &&f) {
				return f(0);
			},std::forward<F>(f)).get());
		}
	} else {
		return hpx::async(hpx::launch::deferred, [stack_cnt](F &&f) {
			return f(stack_cnt + 1);
		}, std::move(f));
	}
}

HPX_PLAIN_ACTION(tree::set_theta, set_theta_action);

void tree::set_theta(double t) {
	set_theta_action action;

	localities = hpx::find_all_localities();
	myid = hpx::get_locality_id();
	theta_inv = 1.0 / t;
	const auto opts = options::get();
	const int sz1 = opts.workgroup_size * opts.parts_per_node;
	const int sz2 = opts.problem_size / localities.size() / std::thread::hardware_concurrency() / opts.oversubscription;
	workgroup_size = std::min(sz1, sz2);

	if (myid == 0) {
		std::vector<hpx::future<void>> futs;
		for (int i = 1; i < localities.size(); i++) {
			futs.push_back(hpx::async < set_theta_action > (localities[i], t));
		}
		hpx::wait_all(futs.begin(), futs.end());
	}
}

tree::tree(box_id_type id_, part_iter b, part_iter e, int level_) {
	const auto &opts = options::get();
	part_begin = b;
	part_end = e;
	boxid = id_;
	flags.level = level_;
	flags.leaf = true;
}

refine_return tree::refine(int stack_cnt) {
//	printf("Forming %i %i\n", b, e);
//	if( level_ == 1 ) {
//		sleep(100);
//	}
	bool force_fork;
	bool remote_left, remote_right;
	if (!is_leaf()) {
		force_fork = (flags.depth >= MAX_STACK && (flags.ldepth < MAX_STACK || flags.rdepth < MAX_STACK));
		remote_left = !children[0].local();
		remote_right = !children[1].local();
	}
	const auto &opts = options::get();

	refine_return rc;
	if (((part_end - part_begin > opts.parts_per_node) || (flags.level < opts.min_level)) && is_leaf()) {
		double max_span = 0.0;
		range box = box_id_to_range(boxid);
		int max_dim;
		for (int dim = 0; dim < NDIM; dim++) {
			const auto this_span = box.max[dim] - box.min[dim];
			if (this_span > max_span) {
				max_span = this_span;
				max_dim = dim;
			}
		}
		range boxl = box;
		range boxr = box;
		part_iter mid_iter;
		if (opts.balanced_tree) {
			const auto mid_x = part_vect_find_median(part_begin, part_end, max_dim);
			mid_iter = (part_begin + part_end) / 2;
			boxl.max[max_dim] = boxr.min[max_dim] = mid_x;
		} else {
			double mid = (box.max[max_dim] + box.min[max_dim]) * 0.5;
			boxl.max[max_dim] = boxr.min[max_dim] = mid;
			mid_iter = part_vect_sort(part_begin, part_end, mid, max_dim);
		}
		auto rcl = hpx::new_ < tree > (localities[part_vect_locality_id(part_begin)], (boxid << 1), part_begin, mid_iter, flags.level + 1);
		auto rcr = hpx::new_ < tree > (localities[part_vect_locality_id(mid_iter)], (boxid << 1) + 1, mid_iter, part_end, flags.level + 1);

		children[1] = rcr.get();
		children[0] = rcl.get();
		flags.leaf = false;
		flags.depth = 1;
		flags.rdepth = 0;
		flags.ldepth = 0;
		rc.max_depth = 1;
		rc.min_depth = 1;
		rc.leaves = 2;
		rc.nodes = 3;
		rc.rc = true;
	} else if (!is_leaf()) {
		auto rcl = thread_if_avail([=](int stack_cnt) {
			return children[0].refine(stack_cnt);
		}, true, part_end - part_begin, force_fork, remote_left, stack_cnt);
		auto rcr = thread_if_avail([=](int stack_cnt) {
			return children[1].refine(stack_cnt);
		}, false, part_end - part_begin, force_fork, remote_right, stack_cnt);
		auto rc1 = rcr.get();
		auto rc2 = rcl.get();
		flags.ldepth = rc1.max_depth;
		flags.rdepth = rc2.max_depth;
		flags.depth = 1 + std::max(rc1.max_depth, rc2.max_depth);
		rc.max_depth = flags.depth;
		rc.min_depth = 1 + std::min(rc1.min_depth, rc2.min_depth);
		rc.leaves = rc1.leaves + rc2.leaves;
		rc.nodes = rc1.nodes + rc2.nodes + 1;
		rc.rc = rc1.rc || rc2.rc;
	} else {
		flags.depth = 0;
		flags.rdepth = 0;
		flags.ldepth = 0;
		rc.max_depth = 0;
		rc.min_depth = 0;
		rc.leaves = 1;
		rc.nodes = 1;
		rc.rc = false;
	}
	return rc;
}

multipole_return tree::compute_multipoles(rung_type mrung, bool do_out, int workid, int stack_cnt) {
	if (flags.level == 0) {
		gwork_reset();
		reset_node_cache();
		part_vect_reset();
		trash_checkptrs();
		trash_multi_srcs();
	}
	bool force_fork;
	bool remote_left, remote_right;
	if (!is_leaf()) {
		force_fork = (flags.depth >= MAX_STACK && (flags.ldepth < MAX_STACK || flags.rdepth < MAX_STACK));
		remote_left = !children[0].local();
		remote_right = !children[1].local();
	}
	const auto &opts = options::get();
	range prange;
	gwork_id = workid;
	if ((gwork_id == null_gwork_id) && ((part_end - part_begin <= workgroup_size) || is_leaf())) {
		gwork_id = gwork_assign_id();
	}

	multipole_return rc;
	if (part_end - part_begin == 0 && is_leaf()) {
		range box = box_id_to_range(boxid);
		rc.N = 0;
		multi.m = 0.0;
		for (int dim = 0; dim < NDIM; dim++) {
			multi.x[dim] = double_to_pos(0.5 * (box.max[dim] - box.min[dim]));
		}
		num_active = 0;
		r = 0.0;
		rc.m.m = multi.m;
		rc.m.x = multi.x;
		rc.m.r = r;
		rc.m.num_active = num_active;
		rc.r = box;
		rc.c = get_check_item();
		return rc;
	}
	if (is_leaf()) {
		const auto com = part_vect_center_of_mass(part_begin, part_end);
		rc.N = com.first;
		multi.x = double_to_pos(com.second);
		auto tmp = part_vect_multipole_info(pos_to_double(multi.x), do_out ? 0 : mrung, part_begin, part_end);
		multi.m = tmp.m;
		multi.x = tmp.x;
		r = tmp.r;
		num_active = tmp.num_active;
		prange = part_vect_range(part_begin, part_end);
		flop += (part_end - part_begin) * 64;
	} else {
		multipole_return ml, mr;
		auto rcl = thread_if_avail([=](int stack_cnt) {
			return children[0].compute_multipoles(mrung, do_out, gwork_id, stack_cnt);
		}, true, part_end - part_begin, force_fork, remote_left, stack_cnt);
		auto rcr = thread_if_avail([=](int stack_cnt) {
			return children[1].compute_multipoles(mrung, do_out, gwork_id, stack_cnt);
		}, false, part_end - part_begin, force_fork, remote_right, stack_cnt);
		mr = rcr.get();
		ml = rcl.get();
		rc.N = ml.N + mr.N;
		if (rc.N != 0) {
			multi.x = double_to_pos((pos_to_double(ml.m.x) * ml.N + pos_to_double(mr.m.x) * mr.N) / (double) rc.N);
		} else {
			range box = box_id_to_range(boxid);
			for (int dim = 0; dim < NDIM; dim++) {
				multi.x[dim] = double_to_pos(0.5 * (box.max[dim] - box.min[dim]));
			}
		}
		const auto multixdouble = pos_to_double(multi.x);
		const auto mlmxdouble = pos_to_double(ml.m.x);
		const auto mrmxdouble = pos_to_double(mr.m.x);
		multi.m = (ml.m.m >> (mlmxdouble - multixdouble)) + (mr.m.m >> (mrmxdouble - multixdouble));
		num_active = ml.m.num_active + mr.m.num_active;
		if( ml.N == 0 ) {
			if( mr.N == 0 ) {
				r = 0.0;
			} else {
				r = mr.m.r;
			}
		} if( mr.N == 0 ) {
			r = ml.m.r;
		} else {
			r = std::max(abs(mlmxdouble - multixdouble) + ml.m.r, abs(mrmxdouble - multixdouble) + mr.m.r);
		}
		if (mr.N || ml.N) {
			for (int dim = 0; dim < NDIM; dim++) {
				prange.max[dim] = std::max(ml.r.max[dim], mr.r.max[dim]);
				prange.min[dim] = std::min(ml.r.min[dim], mr.r.min[dim]);
			}
			double rmax = abs(multixdouble - vect<double>( { prange.min[0], prange.min[1], prange.min[2] }));
			rmax = std::max(rmax, abs(multixdouble - vect<double>( { prange.max[0], prange.min[1], prange.min[2] })));
			rmax = std::max(rmax, abs(multixdouble - vect<double>( { prange.min[0], prange.max[1], prange.min[2] })));
			rmax = std::max(rmax, abs(multixdouble - vect<double>( { prange.max[0], prange.max[1], prange.min[2] })));
			rmax = std::max(rmax, abs(multixdouble - vect<double>( { prange.min[0], prange.min[1], prange.max[2] })));
			rmax = std::max(rmax, abs(multixdouble - vect<double>( { prange.max[0], prange.min[1], prange.max[2] })));
			rmax = std::max(rmax, abs(multixdouble - vect<double>( { prange.min[0], prange.max[1], prange.max[2] })));
			rmax = std::max(rmax, abs(multixdouble - vect<double>( { prange.max[0], prange.max[1], prange.max[2] })));
			r = std::min(r, (float) rmax);
		}
		child_check.children[0] = ml.c;
		child_check.children[1] = mr.c;
	}
	if (num_active && is_leaf()) {
		gwork_checkin(gwork_id);
	}
	rc.m.m = multi.m;
	rc.m.x = multi.x;
	rc.m.r = r;
	rc.m.num_active = num_active;
	rc.r = prange;
	rc.c = get_check_item();
	if (flags.level == 0) {
		pct_active = (double) num_active / (double) opts.problem_size;
	}
	return rc;
}

const node_attr* tree::get_node_attributes() const {
	return &child_check;
}

check_item tree::get_check_item() const {
	const auto loc_id = myid;
	raw_id_type id;
	id.loc_id = loc_id;
	id.ptr = reinterpret_cast<std::uint64_t>(this);
	check_item check;
	check.is_leaf = flags.leaf;
	check.pbegin = part_begin;
	check.pend = part_end;
	check.r = r;
	check.node = raw_tree_client(id);
	check.multi = &multi;
	return check;
}

bool tree::is_leaf() const {
	return flags.leaf;
}

struct workspace {
	std::vector<future_data<const node_attr*>> dfuts;
	std::vector<future_data<const node_attr*>> efuts;
	std::vector<std::pair<part_iter, part_iter>> dsource_iters;
	std::vector<std::pair<part_iter, part_iter>> esource_iters;
	std::vector<const multi_src*> emulti_srcs;
	std::vector<const multi_src*> dmulti_srcs;
	workspace& operator=(const workspace&) = delete;
	workspace& operator=(workspace&&) = default;
	workspace(const workspace&) = delete;
	workspace(workspace&&) = default;
	workspace() = default;

};

std::stack<workspace> workspaces;
mutex_type workspace_mutex;

workspace get_workspace() {
	workspace this_space;
	{
		std::lock_guard<mutex_type> lock(workspace_mutex);
		if (workspaces.size() != 0) {
			this_space = std::move(workspaces.top());
			workspaces.pop();
		}
	}
	this_space.emulti_srcs.resize(0);
	this_space.dmulti_srcs.resize(0);
	this_space.dfuts.resize(0);
	this_space.efuts.resize(0);
	this_space.dsource_iters.resize(0);
	this_space.esource_iters.resize(0);
	return std::move(this_space);
}

void trash_workspace(workspace &&w) {
	std::lock_guard<mutex_type> lock(workspace_mutex);
	workspaces.push(std::move(w));
}

interaction_stats tree::kick_fmm(std::vector<check_pair> dchecklist, std::vector<check_pair> echecklist, const vect<double> &Lcom, expansion<float> L,
		rung_type min_rung, bool do_out, int stack_cnt) {
	static const auto opts = options::get();
	static const float h = opts.soft_len;
	decltype(group_ranges)().swap(group_ranges);
	interaction_stats istats;
	bool force_fork;
	bool remote_left, remote_right;
	if (!is_leaf()) {
		force_fork = (flags.depth >= MAX_STACK && (flags.ldepth < MAX_STACK || flags.rdepth < MAX_STACK));
		remote_left = !children[0].local();
		remote_right = !children[1].local();
	}

	if ((part_end - part_begin == 0) || (!num_active && !do_out)) {
		return istats;
	}

	const auto multixdouble = pos_to_double(multi.x);
	L = L << (multixdouble - Lcom);

	constexpr double ewald_toler = 1.0e-3;
	static const float r_ewald = 2.0 * opts.theta * std::pow(ewald_toler / 8.0, 1.0 / 3.0);

	const float rmax = range_max_span(part_vect_range(part_begin, part_end));

	std::vector<check_pair> next_dchecklist;
	std::vector<check_pair> next_echecklist;
	auto space = get_workspace();
	auto &emulti_srcs = space.emulti_srcs;
	auto &dmulti_srcs = space.dmulti_srcs;
	auto &dfuts = space.dfuts;
	auto &efuts = space.efuts;
	auto &dsource_iters = space.dsource_iters;
	auto &esource_iters = space.esource_iters;

	next_dchecklist.reserve(NCHILD * dchecklist.size());
	next_echecklist.reserve(NCHILD * echecklist.size());

	std::uint64_t dsource_count = 0;
	std::uint64_t esource_count = 0;

	vect<simd_int> X;
	simd_float RpH = r + h;
	for (int dim = 0; dim < NDIM; dim++) {
		X[dim] = multi.x[dim];
	}

	int max_ci = dchecklist.size() - 1;
	dmulti_srcs.resize(0);
	emulti_srcs.resize(0);
	for (int ci = 0; ci < dchecklist.size(); ci += simd_float::size()) {
		vect<simd_int> Y;
		simd_float CR;
		for (int dim = 0; dim < NDIM; dim++) {
			for (int k = 0; k < simd_float::size(); k++) {
				const int index = std::min((ci + k), max_ci);
				Y[dim][k] = dchecklist[index].chk->multi->x[dim];
			}
		}
		for (int k = 0; k < simd_float::size(); k++) {
			const int index = std::min((ci + k), max_ci);
			CR[k] = dchecklist[index].chk->r;
		}
		vect<simd_float> dX;
		for (int dim = 0; dim < NDIM; dim++) {
			if (opts.ewald) {
				dX[dim] = simd_float(X[dim] - Y[dim]) * POS_INV;
			} else {
				dX[dim] = (simd_float(X[dim]) - simd_float(Y[dim])) * POS_INV;
			}
		}
		const simd_float dX2 = dX.dot(dX);
		const simd_float radius = (RpH + CR) * simd_float(theta_inv);
		const simd_float far = dX2 > (radius * radius);
		for (int this_ci = ci; this_ci < std::min(ci + (int) simd_float::size(), max_ci + 1); this_ci++) {
			auto &c = dchecklist[this_ci];
			if (c.chk->pend == c.chk->pbegin) {
				continue;
			}
			if (far[this_ci - ci] == 1.0) {
				if (c.opened) {
					dsource_iters.push_back(std::make_pair(c.chk->pbegin, c.chk->pend));
					dsource_count += c.chk->pend - c.chk->pbegin;
				} else {
					dmulti_srcs.push_back(c.chk->multi);
				}
			} else {
				if (c.chk->is_leaf) {
					c.opened = true;
					next_dchecklist.push_back(c);
				} else {
					dfuts.push_back(c.chk->node.get_node_attributes());
				}
			}
		}
	}
	if (opts.ewald) {
		for (auto c : echecklist) {
			if (c.chk->pend == c.chk->pbegin) {
				continue;
			}
			const float dx2 = ewald_far_separation2(multi.x, c.chk->multi->x);
			const float radius = (r + c.chk->r + h) * theta_inv;
			const bool far = dx2 > radius * radius;
			if (far) {
				if (c.opened) {
					esource_iters.push_back(std::make_pair(c.chk->pbegin, c.chk->pend));
					esource_count += c.chk->pend - c.chk->pbegin;
				} else {
					emulti_srcs.push_back(c.chk->multi);
				}
			} else {
				if (c.chk->is_leaf) {
					c.opened = true;
					next_echecklist.push_back(c);
				} else {
					efuts.push_back(c.chk->node.get_node_attributes());
				}
			}
		}
	}
	flop += gravity_CC_direct(L, multi.x, dmulti_srcs, do_out);
	flop += gravity_CP_direct(L, multi.x, part_vect_read_positions(dsource_iters), do_out);
	istats.CC_direct += dmulti_srcs.size();
	istats.CP_direct += dsource_count;
	if (opts.ewald) {
		flop += gravity_CC_ewald(L, multi.x, emulti_srcs, do_out);
		flop += gravity_CP_ewald(L, multi.x, part_vect_read_positions(esource_iters), do_out);
		istats.CC_ewald += emulti_srcs.size();
		istats.CP_ewald += esource_count;
	}
	for (auto &f : dfuts) {
		auto c = f.get();
		next_dchecklist.push_back(c->children[0]);
		next_dchecklist.push_back(c->children[1]);
	}
	for (auto &f : efuts) {
		auto c = f.get();
		next_echecklist.push_back(c->children[0]);
		next_echecklist.push_back(c->children[1]);
	}
	std::swap(dchecklist, next_dchecklist);
	std::swap(echecklist, next_echecklist);
	next_dchecklist.resize(0);
	next_echecklist.resize(0);
	if (!is_leaf()) {
		auto rc_l_fut = thread_if_avail([=](int stack_cnt) {
			return children[0].kick_fmm(std::move(dchecklist), std::move(echecklist), multixdouble, L, min_rung, do_out, stack_cnt);
		}, true, part_end - part_begin, force_fork, remote_left, stack_cnt);
		auto rc_r_fut = thread_if_avail([&](int stack_cnt) {
			return children[1].kick_fmm(std::move(dchecklist), std::move(echecklist), multixdouble, L, min_rung, do_out, stack_cnt);
		}, false, part_end - part_begin, force_fork, remote_right, stack_cnt);
		const auto rc_r = rc_r_fut.get();
		const auto rc_l = rc_l_fut.get();
		istats += rc_r;
		istats += rc_l;
	} else {
		const memory<force> fmem;
		auto fptr = fmem.get_vector();
		auto xptr = std::make_shared<std::vector<vect<pos_type>>>();
		dsource_iters.resize(0);
		esource_iters.resize(0);
		emulti_srcs.resize(0);
		dmulti_srcs.resize(0);
		esource_count = 0;
		dsource_count = 0;
		while (!dchecklist.empty()) {
			dfuts.resize(0);
			for (auto c : dchecklist) {
				if (c.chk->pend == c.chk->pbegin) {
					continue;
				}
				if (c.opened) {
					dsource_iters.push_back(std::make_pair(c.chk->pbegin, c.chk->pend));
					dsource_count += c.chk->pend - c.chk->pbegin;
				} else {
					const float dx2 = separation2(multi.x, c.chk->multi->x);
					const float radius = (r + c.chk->r + h) * theta_inv;
					const bool far = dx2 > radius * radius;
					if (far) {
						dmulti_srcs.push_back(c.chk->multi);
					} else {
						if (c.chk->is_leaf) {
							c.opened = true;
							next_dchecklist.push_back(c);
						} else {
							dfuts.push_back(c.chk->node.get_node_attributes());
						}
					}
				}
			}
			for (auto &f : dfuts) {
				auto c = f.get();
				next_dchecklist.push_back(c->children[0]);
				next_dchecklist.push_back(c->children[1]);
			}
			std::swap(dchecklist, next_dchecklist);
			next_dchecklist.resize(0);
		}
		if (opts.ewald) {
			while (!echecklist.empty()) {
				efuts.resize(0);
				for (auto c : echecklist) {
					if (c.chk->pend == c.chk->pbegin) {
						continue;
					}
					const float dx2 = ewald_far_separation2(multi.x, c.chk->multi->x);
					const float radius = (r + c.chk->r + h) * theta_inv;
					const bool far = dx2 > radius * radius;
					if (c.opened) {
						esource_iters.push_back(std::make_pair(c.chk->pbegin, c.chk->pend));
						esource_count += c.chk->pend - c.chk->pbegin;
					} else {
						if (far) {
							emulti_srcs.push_back(c.chk->multi);
						} else {
							if (c.chk->is_leaf) {
								c.opened = true;
								next_echecklist.push_back(c);
							} else {
								efuts.push_back(c.chk->node.get_node_attributes());
							}
						}
					}
				}
				for (auto &f : efuts) {
					auto c = f.get();
					next_echecklist.push_back(c->children[0]);
					next_echecklist.push_back(c->children[1]);
				}
				std::swap(echecklist, next_echecklist);
				next_echecklist.resize(0);
			}
		}
		*xptr = part_vect_read_active_positions(part_begin, part_end, do_out ? rung_type(0) : min_rung);
		fptr->resize(xptr->size());
		if (xptr->size() > 4) {
			expansion<simd_float> Lvec;
			for (int i = 0; i < LP; i++) {
				Lvec[i] = L[i];
			}
			vect<pos_type> Y = multi.x;
			for (int i = 0; i < xptr->size(); i += simd_float::size()) {
				vect<simd_int> X;
				simd_float phi;
				vect<simd_float> g;
				for (int k = 0; k < simd_float::size(); k++) {
					const int j = std::min(i + k, (int) xptr->size() - 1);
					for (int dim = 0; dim < NDIM; dim++) {
						X[dim][k] = (*xptr)[j][dim];
					}
				}
				vect<simd_float> dX;
				for (int dim = 0; dim < NDIM; dim++) {
					dX[dim] = simd_float(X[dim] - Y[dim]) * POS_INV;
				}
				Lvec.translate_L2(g, phi, dX);
				for (int k = 0; k < simd_float::size(); k++) {
					const int j = i + k;
					if (j == xptr->size()) {
						break;
					}
					for (int dim = 0; dim < NDIM; dim++) {
						(*fptr)[j].g[dim] = g[dim][k];
					}
					(*fptr)[j].phi = phi[k];
				}
			}
		} else {
			int j = 0;
			for (auto i = xptr->begin(); i != xptr->end(); i++) {
				force this_f;
				L.translate_L2(this_f.g, this_f.phi, vect<float>(pos_to_double((*xptr)[j]) - multixdouble));
				(*fptr)[j].phi = this_f.phi;
				(*fptr)[j].g = this_f.g;
				j++;
			}
		}

		if (!opts.cuda || dmulti_srcs.size() < 32) {
			flop += gravity_PC_direct(*fptr, *xptr, dmulti_srcs, do_out);
			dmulti_srcs.resize(0);
		}
//		printf( "%i\n", multi_srcs.size());
//		flop += gravity_PP_direct(*fptr, *xptr, part_vect_read_positions(dsource_iters), do_out);
		istats.CP_direct += xptr->size() * dmulti_srcs.size();
		istats.PP_direct += xptr->size() * dsource_count;
		if (opts.ewald) {
			flop += gravity_PC_ewald(*fptr, *xptr, emulti_srcs, do_out);
//			printf( "%i\n", esource_iters.size());
			flop += gravity_PP_ewald(*fptr, *xptr, part_vect_read_positions(esource_iters), do_out);
			istats.CP_ewald += xptr->size() * emulti_srcs.size();
			istats.PP_ewald += xptr->size() * esource_count;
		}
		flop += gwork_pp_complete(gwork_id, &(*fptr), &(*xptr), dsource_iters, dmulti_srcs, [this, min_rung, do_out, fptr, xptr]() {
			static const auto opts = options::get();
			static const auto m = opts.m_tot / opts.problem_size;
			static const auto h = opts.soft_len;
			static const auto phi_self = (-2.8) * m / h;
			xptr->size();
			for (int i = 0; i < fptr->size(); i++) {
				(*fptr)[i].phi -= phi_self;
			}
			auto tmp = part_vect_kick(part_begin, part_end, min_rung, do_out, *fptr);
			return tmp;
		}, do_out);

	}
	trash_workspace(std::move(space));
	return istats;
}

void tree::find_groups(std::vector<check_pair> checklist, int stack_cnt) {
	static const auto opts = options::get();
	static const float L = 1.001 * std::pow(opts.problem_size, -1.0 / 3.0) * opts.link_len;
	if (flags.level == 0) {
		reset_node_cache();
		part_vect_reset();
		trash_checkptrs();
	}
	bool force_fork;
	bool remote_left, remote_right;
	if (!is_leaf()) {
		force_fork = (flags.depth >= MAX_STACK && (flags.ldepth < MAX_STACK || flags.rdepth < MAX_STACK));
		remote_left = !children[0].local();
		remote_right = !children[1].local();
	}

	if (part_end - part_begin == 0) {
		return;
	}

	std::vector<check_pair> next_checklist;
	std::vector<future_data<const node_attr*>> futs;

	next_checklist.reserve(NCHILD * checklist.size());

	range myrange = range_expand(box_id_to_range(boxid), L);

	const auto multixdouble = pos_to_double(multi.x);
	for (auto c : checklist) {
		if (c.chk->pend == c.chk->pbegin) {
			continue;
		}
		const float dx2 = separation2(multi.x, c.chk->multi->x);
		const float radius = (r + c.chk->r + L);
		const bool far = dx2 > radius * radius;
		if (!far) {
			if (c.chk->is_leaf) {
				c.opened = true;
				next_checklist.push_back(c);
			} else {
				futs.push_back(c.chk->node.get_node_attributes());
			}
		}
	}
	for (auto &f : futs) {
		auto c = f.get();
		next_checklist.push_back(c->children[0]);
		next_checklist.push_back(c->children[1]);
	}
	std::swap(checklist, next_checklist);
	next_checklist.resize(0);
	if (!is_leaf()) {
		auto rc_l_fut = thread_if_avail([=](int stack_cnt) {
			return children[0].find_groups(std::move(checklist), stack_cnt);
		}, true, part_end - part_begin, force_fork, remote_left, stack_cnt);
		auto rc_r_fut = thread_if_avail([&](int stack_cnt) {
			return children[1].find_groups(std::move(checklist), stack_cnt);
		}, false, part_end - part_begin, force_fork, remote_right, stack_cnt);
		const auto rc_r = rc_r_fut.get();
		const auto rc_l = rc_l_fut.get();
	} else {
		while (!checklist.empty()) {
			futs.resize(0);
			for (auto c : checklist) {
				if (c.chk->pend == c.chk->pbegin) {
					continue;
				}
				const float dx2 = separation2(multi.x, c.chk->multi->x);
				const float radius = (r + c.chk->r + L);
				const bool far = dx2 > radius * radius;
				if (c.opened) {
					const auto other_range = part_vect_range(c.chk->pbegin, c.chk->pend);
					if (ranges_intersect(myrange, other_range)) {
						group_ranges.push_back(std::make_pair(c.chk->pbegin, c.chk->pend));
					}
				} else if (!far) {
					if (c.chk->is_leaf) {
						c.opened = true;
						next_checklist.push_back(c);
					} else {
						futs.push_back(c.chk->node.get_node_attributes());
					}
				}

			}
			for (auto &f : futs) {
				auto c = f.get();
				next_checklist.push_back(c->children[0]);
				next_checklist.push_back(c->children[1]);
			}
			std::swap(checklist, next_checklist);
			next_checklist.resize(0);
		}
		flags.max_iter = 0;
		while (part_vect_find_groups(part_begin, part_end, part_vect_read_group(part_begin, part_end, range(), false).get())) {
			flags.max_iter++;
		}
		flags.max_iter = std::max(1, (int) flags.max_iter);
		groups_add_finder([=]() {
			bool rc;
			if (group_ranges.size()) {
				std::vector<particle_group_info> sources;
				for (auto &iter : group_ranges) {
					auto others = part_vect_read_group(iter.first, iter.second, myrange, true).get();
					for (int j = 0; j < others.size(); j++) {
						sources.push_back(others[j]);
					}
				}
				rc = part_vect_find_groups(part_begin, part_end, std::move(sources));
				int iter = 1;
				bool this_rc;
				if (rc && iter < flags.max_iter) {
					do {
						sources = part_vect_read_group(part_begin, part_end, range(), false).get();
						iter++;
						this_rc = part_vect_find_groups(part_begin, part_end, std::move(sources));
					} while (this_rc && iter < flags.max_iter);
				}
			} else {
				rc = false;
			}
			return rc;
		});
	}
}

double tree::drift(double t, rung_type r) {
	return part_vect_drift(t, r);
}

HPX_PLAIN_ACTION(tree::get_flop, get_flop_action);

std::uint64_t tree::get_flop() {
	std::vector<hpx::future<std::uint64_t>> futs;
	if (myid == 0) {
		for (int i = 1; i < localities.size(); i++) {
			futs.push_back(hpx::async < get_flop_action > (localities[i]));
		}
	}
	if (options::get().cuda) {
		flop += cuda_reset_flop();
	}
	auto total = (std::uint64_t) flop;
	for (auto &f : futs) {
		total += f.get();
	}
	return total;
}

void tree::reset_flop() {
	flop = 0;
}

#define NODE_CACHE_SIZE 1024
std::unordered_map<raw_id_type, hpx::shared_future<node_attr>, raw_id_type_hash> node_cache[NODE_CACHE_SIZE];
mutex_type node_cache_mtx[NODE_CACHE_SIZE];

HPX_PLAIN_ACTION (reset_node_cache);

void reset_node_cache() {
	std::vector<hpx::future<void>> futs;
	if (myid == 0) {
		for (int i = 1; i < localities.size(); i++) {
			futs.push_back(hpx::async < reset_node_cache_action > (localities[i]));
		}
	}
	for (int i = 0; i < NODE_CACHE_SIZE; i++) {
		node_cache[i].clear();
	}
	hpx::wait_all(futs.begin(), futs.end());
}

hpx::future<node_attr> get_node_attributes_(raw_id_type id);

HPX_PLAIN_ACTION(get_node_attributes_, get_node_attributes_action);

hpx::future<node_attr> get_node_attributes_(raw_id_type id) {
	if (myid == id.loc_id) {
		return hpx::make_ready_future(*reinterpret_cast<tree*>(id.ptr)->get_node_attributes());
	} else {
		return hpx::async < get_node_attributes_action > (localities[id.loc_id], id);
	}
}

hpx::future<const node_attr*> read_node_cache(raw_id_type id) {
	const int index = raw_id_type_hash()(id) % NODE_CACHE_SIZE;
	std::unique_lock<mutex_type> lock(node_cache_mtx[index]);
	auto iter = node_cache[index].find(id);
	if (iter == node_cache[index].end()) {
		hpx::lcos::local::promise < hpx::future < node_attr >> promise;
		auto fut = promise.get_future();
		node_cache[index][id] = fut.then([](decltype(fut) f) {
			return f.get().get();
		});
		lock.unlock();
		promise.set_value(get_node_attributes_(id));
	}
	return hpx::async(hpx::launch::deferred, [id, index] {
		std::unique_lock<mutex_type> lock(node_cache_mtx[index]);
		auto future = node_cache[index][id];
		lock.unlock();
		return (const node_attr*)&future.get();
	});
}

future_data<const node_attr*> raw_tree_client::get_node_attributes() const {
	future_data<const node_attr*> data;
	if (myid == ptr.loc_id) {
		tree *tree_ptr = reinterpret_cast<tree*>(ptr.ptr);
		data.data = tree_ptr->get_node_attributes();
	} else {
		data.fut = read_node_cache(ptr);
	}
	return data;
}
