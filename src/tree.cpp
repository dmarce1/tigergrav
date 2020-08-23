#include <tigergrav/options.hpp>
#include <tigergrav/time.hpp>
#include <tigergrav/tree.hpp>

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

#define MAX_STACK 8

std::atomic<std::uint64_t> tree::flop(0);
double tree::theta_inv;
double tree::pct_active;


void reset_node_cache();

static std::atomic<int> num_threads(1);
static bool inc_thread();
static void dec_thread();

static std::vector<hpx::id_type> localities;
static int myid;
static int hardware_concurrency = std::thread::hardware_concurrency();
static int target_nthreads = hardware_concurrency;

struct raw_id_type_hash {
	std::size_t operator()(raw_id_type id) const {
		return id.ptr ^ id.loc_id;
	}
};

template<class F>
inline auto thread_if_avail(F &&f, bool left, int nparts, int stack_cnt) {
//	const auto static N = options::get().problem_size / localities.size() / (4 * hardware_concurrency);
	bool thread;
//	printf( "%i\n", (int) num_threads);
//	int count = num_threads++;
	if (left && num_threads < 4 * hardware_concurrency) {
		thread = true;
	} else {
//		num_threads--;
		if (stack_cnt == MAX_STACK - 1) {
//			num_threads++;
			thread = true;
		} else {
			thread = false;
		}
	}
	if (thread) {
		num_threads++;
//		printf("Threading %i %i\n", (int) num_threads, 4 * hardware_concurrency);
		auto rc = hpx::async([](F &&f) {
			auto rc = f(0);
			num_threads--;
			return rc;
		},std::forward<F>(f));
		return rc;
	} else {
		return hpx::async(hpx::launch::deferred, [stack_cnt](F &&f) {
			return f(stack_cnt + 1);
		}, std::move(f));
	}
}

HPX_PLAIN_ACTION(tree::set_theta, set_theta_action);

void tree::set_theta(double t) {
	set_theta_action action;
	theta_inv = 1.0 / t;
	localities = hpx::find_all_localities();
	myid = hpx::get_locality_id();
	if (myid == 0) {
		std::vector<hpx::future<void>> futs;
		for (int i = 1; i < localities.size(); i++) {
			futs.push_back(hpx::async<set_theta_action>(localities[i], t));
		}
		hpx::wait_all(futs.begin(), futs.end());
	}
}

tree::tree(range box_, part_iter b, part_iter e, int level_) {
	const auto &opts = options::get();
	part_begin = b;
	part_end = e;
	box = box_;
	flags.level = level_;
	flags.leaf = true;
}

bool tree::refine(int stack_cnt) {
//	printf("Forming %i %i\n", b, e);
//	if( level_ == 1 ) {
//		sleep(100);
//	}
	const auto &opts = options::get();
//	auto myparts = part_vect_read(part_begin, part_end).get();
//	bool abortme = false;
//	for (const auto &p : myparts) {
//		const auto x = pos_to_double(p.x);
//		if (!in_range(x, box)) {
//			printf("Found particle out of range!\n");
//			printf("%e %e %e\n", x[0], x[1], x[2]);
//			for (int dim = 0; dim < NDIM; dim++) {
//				printf("%e %e\n", box.min[dim], box.max[dim]);
//			}
//			abortme = true;
//		}
//	}
//	if (abortme) {
//		abort();
//	}

	if (part_end - part_begin > opts.parts_per_node && is_leaf()) {
		double max_span = 0.0;
		range prange;
//		if (part_end - part_begin > 512 * opts.parts_per_node) {
		prange = box;
//		} else {
//			prange = part_vect_range(part_begin, part_end);
//		}
		int max_dim;
		for (int dim = 0; dim < NDIM; dim++) {
			const auto this_span = prange.max[dim] - prange.min[dim];
//			const auto this_span = box.max[dim] - box.min[dim];
			if (this_span > max_span) {
				max_span = this_span;
				max_dim = dim;
			}
		}
		range boxl = box;
		range boxr = box;
		double mid = (box.max[max_dim] + box.min[max_dim]) * 0.5;
		boxl.max[max_dim] = boxr.min[max_dim] = mid;
		part_iter mid_iter;
		if (part_end - part_begin == 0) {
			mid_iter = part_end;
		} else {
			mid_iter = part_vect_sort(part_begin, part_end, mid, max_dim);
		}
//		}
		auto rcl = hpx::new_<tree>(localities[part_vect_locality_id(part_begin)], boxl, part_begin, mid_iter, flags.level + 1);
		auto rcr = hpx::new_<tree>(localities[part_vect_locality_id(mid_iter)], boxr, mid_iter, part_end, flags.level + 1);

		children[1] = rcr.get();
		children[0] = rcl.get();
		flags.leaf = false;
		return true;
	} else if (!is_leaf()) {
		auto rcl = thread_if_avail([=](int stack_cnt) {
			return children[0].refine(stack_cnt);
		}, true, part_end - part_begin, stack_cnt);
		auto rcr = thread_if_avail([=](int stack_cnt) {
			return children[1].refine(stack_cnt);
		}, false, part_end - part_begin, stack_cnt);
		bool rc1 = rcr.get();
		bool rc2 = rcl.get();
		return rc1 || rc2;
	} else {
		return false;
	}
}

multipole_return tree::compute_multipoles(rung_type mrung, bool do_out, int workid, int stack_cnt) {
	if (flags.level == 0) {
		gwork_reset();
	}
	flags.group_active = true;
	const auto &opts = options::get();
	range prange;
	gwork_id = workid;
	if ((gwork_id == null_gwork_id) && (part_end - part_begin <= 128 * opts.parts_per_node)) {
		gwork_id = gwork_assign_id();
	}

	if (part_end - part_begin == 0) {
		multipole_return rc;
		multi.m = 0.0;
		for (int dim = 0; dim < NDIM; dim++) {
			multi.x[dim] = 0.5 * (box.max[dim] - box.min[dim]);
		}
		multi.num_active = 0;
		multi.r = 0.0;
		rc.m = multi;
		rc.r = box;
		rc.c = get_check_item();
		return rc;
	}
	if (is_leaf()) {
		multi.x = part_vect_center_of_mass(part_begin, part_end).second;
		multi = part_vect_multipole_info(multi.x, do_out ? 0 : mrung, part_begin, part_end);
		prange = part_vect_range(part_begin, part_end);
		flop += (part_end - part_begin) * 64;
	} else {
		multipole_return ml, mr;
		auto rcl = thread_if_avail([=](int stack_cnt) {
			return children[0].compute_multipoles(mrung, do_out, gwork_id, stack_cnt);
		}, true, part_end - part_begin, stack_cnt);
		auto rcr = thread_if_avail([=](int stack_cnt) {
			return children[1].compute_multipoles(mrung, do_out, gwork_id, stack_cnt);
		}, false, part_end - part_begin, stack_cnt);
		mr = rcr.get();
		ml = rcl.get();
		multi.m() = ml.m.m() + mr.m.m();
		if (multi.m() != 0.0) {
			multi.x = (ml.m.x * ml.m.m() + mr.m.x * mr.m.m()) / multi.m();
		} else {
			ERROR();
		}
		multi.m = (ml.m.m >> (ml.m.x - multi.x)) + (mr.m.m >> (mr.m.x - multi.x));
		multi.num_active = ml.m.num_active + mr.m.num_active;
		if( ml.m.m() == 0.0) {
			multi.r = mr.m.r;
		} else if( mr.m.m() ==0.0) {
			multi.r = ml.m.r;
		} else {
			multi.r = std::max(abs(ml.m.x - multi.x) + ml.m.r, abs(mr.m.x - multi.x) + mr.m.r);
		}
		for (int dim = 0; dim < NDIM; dim++) {
			prange.max[dim] = std::max(ml.r.max[dim], mr.r.max[dim]);
			prange.min[dim] = std::min(ml.r.min[dim], mr.r.min[dim]);
		}
		double rmax = abs(multi.x - vect<double>( { prange.min[0], prange.min[1], prange.min[2] }));
		rmax = std::max(rmax, abs(multi.x - vect<double>( { prange.max[0], prange.min[1], prange.min[2] })));
		rmax = std::max(rmax, abs(multi.x - vect<double>( { prange.min[0], prange.max[1], prange.min[2] })));
		rmax = std::max(rmax, abs(multi.x - vect<double>( { prange.max[0], prange.max[1], prange.min[2] })));
		rmax = std::max(rmax, abs(multi.x - vect<double>( { prange.min[0], prange.min[1], prange.max[2] })));
		rmax = std::max(rmax, abs(multi.x - vect<double>( { prange.max[0], prange.min[1], prange.max[2] })));
		rmax = std::max(rmax, abs(multi.x - vect<double>( { prange.min[0], prange.max[1], prange.max[2] })));
		rmax = std::max(rmax, abs(multi.x - vect<double>( { prange.max[0], prange.max[1], prange.max[2] })));
		multi.r = std::min(multi.r, (float) rmax);
		child_check[0] = ml.c;
		child_check[1] = mr.c;
	}
	if (multi.num_active && is_leaf()) {
		gwork_checkin(gwork_id);
	}
	auto rc = multipole_return( { multi, prange, get_check_item() });
	if( flags.level == 0 ) {
		pct_active = (double) multi.num_active / (double) opts.problem_size;
	}
	return rc;
}

node_attr tree::get_node_attributes() const {
	node_attr attr;
	attr.children[0] = child_check[0];
	attr.children[1] = child_check[1];
	return attr;
}

multi_src tree::get_multi_srcs() const {
	multi_src attr;
	attr.m = multi.m;
	attr.x = multi.x;
	return attr;
}

check_item tree::get_check_item() const {
	const auto loc_id = myid;
	raw_id_type id;
	id.loc_id = loc_id;
	id.ptr = reinterpret_cast<std::uint64_t>(this);
	check_item check;
	check.flags.opened = false;
	check.flags.is_leaf = flags.leaf;
	check.pbegin = part_begin;
	check.pend = part_end;
	check.x = multi.x;
	check.r = multi.r;
	check.node = raw_tree_client(id);
	return check;
}

bool tree::is_leaf() const {
	return flags.leaf;
}

struct workspace {
	std::vector<future_data<node_attr>> dfuts;
	std::vector<future_data<node_attr>> efuts;
	std::vector<future_data<multi_src>> dmulti_futs;
	std::vector<future_data<multi_src>> emulti_futs;
	std::vector<std::pair<part_iter, part_iter>> dsource_iters;
	std::vector<std::pair<part_iter, part_iter>> esource_iters;
	std::vector<multi_src> multi_srcs;

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
	this_space.emulti_futs.resize(0);
	this_space.dmulti_futs.resize(0);
	this_space.multi_srcs.resize(0);
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

int tree::kick_fmm(std::vector<check_item> dchecklist, std::vector<check_item> echecklist, const vect<double> &Lcom, expansion<double> L, rung_type min_rung,
		bool do_out, int stack_cnt) {
	static const auto opts = options::get();
	static const auto h = opts.soft_len;
	static const double m = opts.m_tot / opts.problem_size;
	if (flags.level == 0) {
		reset_node_cache();
		part_vect_reset();
	}

	if ((part_end - part_begin == 0) || (!multi.num_active && !do_out)) {
		return 0;
	}

	L = L << (multi.x - Lcom);

	std::vector<check_item> next_dchecklist;
	std::vector<check_item> next_echecklist;
	auto space = get_workspace();
	auto &multi_srcs = space.multi_srcs;
	auto &dfuts = space.dfuts;
	auto &efuts = space.efuts;
	auto &dsource_iters = space.dsource_iters;
	auto &esource_iters = space.esource_iters;
	auto &dmulti_futs = space.dmulti_futs;
	auto &emulti_futs = space.emulti_futs;

	next_dchecklist.reserve(NCHILD * dchecklist.size());
	next_echecklist.reserve(NCHILD * echecklist.size());

	for (auto c : dchecklist) {
		if (c.pend == c.pbegin) {
			continue;
		}
		const auto dx = opts.ewald ? ewald_near_separation(multi.x - c.x) : abs(multi.x - c.x);
		const bool far = dx > (multi.r + c.r + 2 * h) * theta_inv;
		if (far) {
			if (c.flags.opened) {
				dsource_iters.push_back(std::make_pair(c.pbegin, c.pend));
			} else {
				dmulti_futs.push_back(c.node.get_multi_srcs());
			}
		} else {
			if (c.flags.is_leaf) {
				c.flags.opened = true;
				next_dchecklist.push_back(c);
			} else {
				dfuts.push_back(c.node.get_node_attributes());
			}
		}
	}

	if (opts.ewald) {
		for (auto c : echecklist) {
			if (c.pend == c.pbegin) {
				continue;
			}
			const auto dx = ewald_far_separation(multi.x - c.x, multi.r);
			const bool far = dx > (multi.r + c.r + 2 * h) * theta_inv;
			if (far) {
				if (c.flags.opened) {
					esource_iters.push_back(std::make_pair(c.pbegin, c.pend));
				} else {
					emulti_futs.push_back(c.node.get_multi_srcs());
				}
			} else {
				if (c.flags.is_leaf) {
					c.flags.opened = true;
					next_echecklist.push_back(c);
				} else {
					efuts.push_back(c.node.get_node_attributes());
				}
			}
		}
	}
	multi_srcs.resize(0);
	for (auto &v : dmulti_futs) {
		multi_srcs.push_back(v.get());
	}
	flop += gravity_CC_direct(L, multi.x, multi_srcs);
	flop += gravity_CP_direct(L, multi.x, part_vect_read_positions(dsource_iters));
	if (opts.ewald) {
		multi_srcs.resize(0);
		for (auto &v : emulti_futs) {
			multi_srcs.push_back(v.get());
		}
		flop += gravity_CC_ewald(L, multi.x, multi_srcs);
		flop += gravity_CP_ewald(L, multi.x, part_vect_read_positions(esource_iters));
	}
	for (auto &f : dfuts) {
		auto c = f.get();
		next_dchecklist.push_back(c.children[0]);
		next_dchecklist.push_back(c.children[1]);
	}
	for (auto &f : efuts) {
		auto c = f.get();
		next_echecklist.push_back(c.children[0]);
		next_echecklist.push_back(c.children[1]);
	}
	std::swap(dchecklist, next_dchecklist);
	std::swap(echecklist, next_echecklist);
	next_dchecklist.resize(0);
	next_echecklist.resize(0);
	if (!is_leaf()) {
		auto rc_l_fut = thread_if_avail([=](int stack_cnt) {
			return children[0].kick_fmm(std::move(dchecklist), std::move(echecklist), multi.x, L, min_rung, do_out, stack_cnt);
		}, true, part_end - part_begin, stack_cnt);
		auto rc_r_fut = thread_if_avail([&](int stack_cnt) {
			return children[1].kick_fmm(std::move(dchecklist), std::move(echecklist), multi.x, L, min_rung, do_out, stack_cnt);
		}, false, part_end - part_begin, stack_cnt);
		const auto rc_r = rc_r_fut.get();
		const auto rc_l = rc_l_fut.get();
	} else {

		auto fptr = std::make_shared<std::vector<force>>();
		auto xptr = std::make_shared<std::vector<vect<pos_type>>>();
		dsource_iters.resize(0);
		esource_iters.resize(0);
		dmulti_futs.resize(0);
		emulti_futs.resize(0);
		while (!dchecklist.empty()) {
			dfuts.resize(0);
			for (auto c : dchecklist) {
				if (c.pend == c.pbegin) {
					continue;
				}
				const auto dx = opts.ewald ? ewald_near_separation(multi.x - c.x) : abs(multi.x - c.x);
				const bool far = dx > (multi.r + c.r + 2 * h) * theta_inv;
				if (c.flags.opened) {
					dsource_iters.push_back(std::make_pair(c.pbegin, c.pend));
				} else {
					if (far) {
						dmulti_futs.push_back(c.node.get_multi_srcs());
					} else {
						if (c.flags.is_leaf) {
							c.flags.opened = true;
							next_dchecklist.push_back(c);
						} else {
							dfuts.push_back(c.node.get_node_attributes());
						}
					}
				}
			}
			for (auto &f : dfuts) {
				auto c = f.get();
				next_dchecklist.push_back(c.children[0]);
				next_dchecklist.push_back(c.children[1]);
			}
			std::swap(dchecklist, next_dchecklist);
			next_dchecklist.resize(0);
		}
		if (opts.ewald) {
			while (!echecklist.empty()) {
				efuts.resize(0);
				for (auto c : echecklist) {
					if (c.pend == c.pbegin) {
						continue;
					}
					const auto dx = ewald_far_separation(multi.x - c.x, multi.r);
					const bool far = dx > (multi.r + c.r + 2 * h) * theta_inv;
					if (c.flags.opened) {
						esource_iters.push_back(std::make_pair(c.pbegin, c.pend));
					} else {
						if (far) {
							emulti_futs.push_back(c.node.get_multi_srcs());
						} else {
							if (c.flags.is_leaf) {
								c.flags.opened = true;
								next_echecklist.push_back(c);
							} else {
								efuts.push_back(c.node.get_node_attributes());
							}
						}
					}
				}
				for (auto &f : efuts) {
					auto c = f.get();
					next_echecklist.push_back(c.children[0]);
					next_echecklist.push_back(c.children[1]);
				}
				std::swap(echecklist, next_echecklist);
				next_echecklist.resize(0);
			}
		}
		*xptr = part_vect_read_active_positions(part_begin, part_end, do_out ? rung_type(0) : min_rung);
		fptr->resize(xptr->size());
		int j = 0;
		for (auto i = xptr->begin(); i != xptr->end(); i++) {
			force this_f = L.translate_L2(vect<float>(pos_to_double((*xptr)[j]) - multi.x));
			(*fptr)[j].phi = this_f.phi;
			(*fptr)[j].g = this_f.g;
			j++;
		}
		multi_srcs.resize(0);
		for (auto &v : dmulti_futs) {
			multi_srcs.push_back(v.get());
		}
		flop += gravity_PC_direct(*fptr, *xptr, multi_srcs);
//		flop += gravity_PP_direct(*fptr, *xptr, part_vect_read_positions(dsource_iters), do_out);
		if (opts.ewald) {
			multi_srcs.resize(0);
			for (auto &v : emulti_futs) {
				multi_srcs.push_back(v.get());
			}
			flop += gravity_PC_ewald(*fptr, *xptr, multi_srcs);
			flop += gravity_PP_ewald(*fptr, *xptr, part_vect_read_positions(esource_iters));
//			if (esource_iters.size())
//				printf("%i %i %e %e %e\n", gwork_id, esource_iters.size(), multi.r, h, 2.0 * (multi.r + h));
		}
		flop += gwork_pp_complete(gwork_id, &(*fptr), &(*xptr), dsource_iters, [this, min_rung, do_out, fptr, xptr]() {
			static const auto opts = options::get();
			static const auto m = opts.m_tot / opts.problem_size;
			static const auto h = opts.soft_len;
			static const auto phi_self = (-315.0 / 128.0) * m / h;
			xptr->size();
			for (int i = 0; i < fptr->size(); i++) {
				(*fptr)[i].phi -= phi_self;
			}
			return part_vect_kick(part_begin, part_end, min_rung, do_out, std::move(*fptr));
		}, do_out);

	}
	trash_workspace(std::move(space));
	return 0;
}

bool tree::find_groups(std::vector<check_item> checklist, int stack_cnt) {
	static const auto opts = options::get();
	static const auto L = std::pow(opts.problem_size, -1.0 / 3.0) * opts.link_len;
	if (flags.level == 0) {
		reset_node_cache();
		part_vect_reset();
	}

	if (part_end - part_begin == 0) {
		return false;
	}

	std::vector<check_item> next_checklist;
	std::vector<future_data<node_attr>> futs;
	std::vector<particle_group_info> sources;
	std::vector<hpx::future<std::vector<particle_group_info>>> source_futs;

	next_checklist.reserve(NCHILD * checklist.size());

	for (auto c : checklist) {
		if (c.pend == c.pbegin) {
			continue;
		}
		const auto dx = opts.ewald ? ewald_near_separation(multi.x - c.x) : abs(multi.x - c.x);
		const bool far = dx > (multi.r + c.r + 2.0 * L) * theta_inv;
		if (!far) {
			if (c.flags.is_leaf) {
				c.flags.opened = true;
				next_checklist.push_back(c);
			} else {
				futs.push_back(c.node.get_node_attributes());
			}
			flags.group_active = flags.group_active || c.flags.group_active;
		}
	}
	if (!flags.group_active) {
		return false;
	}
	for (auto &f : futs) {
		auto c = f.get();
		next_checklist.push_back(c.children[0]);
		next_checklist.push_back(c.children[1]);
	}
	std::swap(checklist, next_checklist);
	next_checklist.resize(0);
	if (!is_leaf()) {
		auto rc_l_fut = thread_if_avail([=](int stack_cnt) {
			return children[0].find_groups(std::move(checklist), stack_cnt);
		}, true, part_end - part_begin, stack_cnt);
		auto rc_r_fut = thread_if_avail([&](int stack_cnt) {
			return children[1].find_groups(std::move(checklist), stack_cnt);
		}, false, part_end - part_begin, stack_cnt);
		const auto rc_r = rc_r_fut.get();
		const auto rc_l = rc_l_fut.get();
		child_check[0].flags.group_active = rc_l;
		child_check[1].flags.group_active = rc_r;
		flags.group_active = rc_r || rc_l;
		return flags.group_active;
	} else {
		source_futs.resize(0);
		while (!checklist.empty()) {
			futs.resize(0);
			for (auto c : checklist) {
				if (c.pend == c.pbegin) {
					continue;
				}
				const auto dx = opts.ewald ? ewald_near_separation(multi.x - c.x) : abs(multi.x - c.x);
				const bool far = dx > (multi.r + c.r + 2.0 * L) * theta_inv;
				if (c.flags.opened) {
					source_futs.push_back(part_vect_read_group(c.pbegin, c.pend));
				} else if (!far) {
					if (c.flags.is_leaf) {
						c.flags.opened = true;
						next_checklist.push_back(c);
					} else {
						futs.push_back(c.node.get_node_attributes());
					}
				}

			}
			for (auto &f : futs) {
				auto c = f.get();
				next_checklist.push_back(c.children[0]);
				next_checklist.push_back(c.children[1]);
			}
			std::swap(checklist, next_checklist);
			next_checklist.resize(0);
		}
		range prange = part_vect_range(part_begin, part_end);
		prange = expand_range(prange, 1.00001 * L);
		for (auto &f : source_futs) {
			auto v = f.get();
			for (auto &s : v) {
				if (in_range(pos_to_double(s.x), prange)) {
					sources.push_back(s);
				}
			}
		}
//		printf( "%i\n", sources.size());
		flags.group_active = part_vect_find_groups(part_begin, part_end, std::move(sources));
		return flags.group_active;
	}
}

double tree::drift(double dt) {
	return part_vect_drift(dt);
}

HPX_PLAIN_ACTION(tree::get_flop, get_flop_action);

std::uint64_t tree::get_flop() {
	std::vector<hpx::future<std::uint64_t>> futs;
	if (myid == 0) {
		for (int i = 1; i < localities.size(); i++) {
			futs.push_back(hpx::async<get_flop_action>(localities[i]));
		}
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

std::unordered_map<raw_id_type, hpx::shared_future<multi_src>, raw_id_type_hash> multipole_cache[NODE_CACHE_SIZE];
mutex_type multipole_cache_mtx[NODE_CACHE_SIZE];

HPX_PLAIN_ACTION (reset_node_cache);

void reset_node_cache() {
	std::vector<hpx::future<void>> futs;
	if (myid == 0) {
		for (int i = 1; i < localities.size(); i++) {
			futs.push_back(hpx::async<reset_node_cache_action>(localities[i]));
		}
	}
	for (int i = 0; i < NODE_CACHE_SIZE; i++) {
		node_cache[i].clear();
		multipole_cache[i].clear();
	}
	hpx::wait_all(futs.begin(), futs.end());
}

hpx::future<node_attr> get_node_attributes_(raw_id_type id);

HPX_PLAIN_ACTION(get_node_attributes_, get_node_attributes_action);

hpx::future<node_attr> get_node_attributes_(raw_id_type id) {
	if (myid == id.loc_id) {
		return hpx::make_ready_future(reinterpret_cast<tree*>(id.ptr)->get_node_attributes());
	} else {
		return hpx::async<get_node_attributes_action>(localities[id.loc_id], id);
	}
}

hpx::future<node_attr> read_node_cache(raw_id_type id) {
	const int index = raw_id_type_hash()(id) % NODE_CACHE_SIZE;
	std::unique_lock<mutex_type> lock(node_cache_mtx[index]);
	auto iter = node_cache[index].find(id);
	if (iter == node_cache[index].end()) {
		hpx::lcos::local::promise<hpx::future<node_attr>> promise;
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
		return future.get();
	});
}

future_data<node_attr> raw_tree_client::get_node_attributes() const {
	future_data<node_attr> data;
	if (myid == ptr.loc_id) {
		tree *tree_ptr = reinterpret_cast<tree*>(ptr.ptr);
		data.data = tree_ptr->get_node_attributes();
	} else {
		data.fut = read_node_cache(ptr);
	}
	return data;
}

hpx::future<multi_src> get_multi_srcs_(raw_id_type id);

HPX_PLAIN_ACTION(get_multi_srcs_, get_multi_srcs_action);

hpx::future<multi_src> get_multi_srcs_(raw_id_type id) {
	if (myid == id.loc_id) {
		return hpx::make_ready_future(reinterpret_cast<tree*>(id.ptr)->get_multi_srcs());
	} else {
		return hpx::async<get_multi_srcs_action>(localities[id.loc_id], id);
	}
}

hpx::future<multi_src> read_multipole_cache(raw_id_type id) {
	const int index = raw_id_type_hash()(id) % NODE_CACHE_SIZE;
	std::unique_lock<mutex_type> lock(multipole_cache_mtx[index]);
	auto iter = multipole_cache[index].find(id);
	if (iter == multipole_cache[index].end()) {
		hpx::lcos::local::promise<hpx::future<multi_src>> promise;
		auto fut = promise.get_future();
		multipole_cache[index][id] = fut.then([](decltype(fut) f) {
			return f.get().get();
		});
		lock.unlock();
		promise.set_value(get_multi_srcs_(id));
	}
	return hpx::async(hpx::launch::deferred, [id, index] {
		std::unique_lock<mutex_type> lock(multipole_cache_mtx[index]);
		auto future = multipole_cache[index][id];
		lock.unlock();
		return future.get();
	});
}

future_data<multi_src> raw_tree_client::get_multi_srcs() const {
	future_data<multi_src> data;
	if (myid == ptr.loc_id) {
		tree *tree_ptr = reinterpret_cast<tree*>(ptr.ptr);
		data.data = tree_ptr->get_multi_srcs();
	} else {
		data.fut = read_multipole_cache(ptr);
	}
	return data;
}

