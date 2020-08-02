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

using get_flop_action_type = tree::get_flop_action;
using compute_multipoles_action_type = tree::compute_multipoles_action;
using kick_fmm_action_type = tree::kick_fmm_action;
using drift_action_type = tree:: drift_action;
using get_check_item_action_type = tree::get_check_item_action;
using refine_action_type = tree::refine_action;

HPX_REGISTER_ACTION(refine_action_type);
HPX_REGISTER_ACTION(get_flop_action_type);
HPX_REGISTER_ACTION(compute_multipoles_action_type);
HPX_REGISTER_ACTION(kick_fmm_action_type);
HPX_REGISTER_ACTION(drift_action_type);
HPX_REGISTER_ACTION(get_check_item_action_type);

#else
#include <hpx/runtime/actions/plain_action.hpp>
HPX_REGISTER_COMPONENT(hpx::components::managed_component<tree>, tree);
#endif

std::atomic<std::uint64_t> tree::flop(0);
float tree::theta_inv;

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
auto thread_if_avail(F &&f, bool left, int stack_cnt) {
//	const auto static N = options::get().problem_size / localities.size() / (8 * hardware_concurrency);
	bool thread;
//	printf( "%i\n", (int) num_threads);
	int count = num_threads++;
	if (count < 4 * hardware_concurrency && left) {
		thread = true;
	} else {
		num_threads--;
		if (stack_cnt % 8 == 7) {
			num_threads++;
			thread = true;
		} else {
			thread = false;
		}
	}
	if (thread) {
		auto rc = hpx::async([](F &&f) {
			auto rc = f(0);
			num_threads--;
			return rc;
		},std::forward<F>(f));
		return rc;
	} else {
		return hpx::make_ready_future(f(stack_cnt + 1));
	}
}

HPX_PLAIN_ACTION(tree::set_theta, set_theta_action);

void tree::set_theta(float t) {
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
	level = level_;
	leaf = true;
}

bool tree::refine(int stack_cnt) {
//	printf("Forming %i %i\n", b, e);
//	if( level_ == 1 ) {
//		sleep(100);
//	}
	const auto &opts = options::get();
//	auto myparts = part_vect_read(b, e).get();
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
//	if( abortme ) {
//		abort();
//	}
	if (part_end - part_begin > opts.parts_per_node && is_leaf()) {
		float max_span = 0.0;
		range prange;
		// Don't bother choosing a particular bisection plane when lots of particles
		if (part_end - part_begin > 512 * opts.parts_per_node) {
			prange = box;
		} else {
			prange = part_vect_range(part_begin, part_end);
		}
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
		float mid = (box.max[max_dim] + box.min[max_dim]) * 0.5;
		boxl.max[max_dim] = boxr.min[max_dim] = mid;
		part_iter mid_iter;
		if (part_end - part_begin == 0) {
			mid_iter = part_end;
		} else {
			mid_iter = part_vect_sort(part_begin, part_end, mid, max_dim);
		}
//		}
		auto rcl = hpx::new_ < tree > (localities[part_vect_locality_id(part_begin)], boxl, part_begin, mid_iter, level + 1);
		auto rcr = hpx::new_ < tree > (localities[part_vect_locality_id(mid_iter)], boxr, mid_iter, part_end, level + 1);

		children[1] = rcr.get();
		children[0] = rcl.get();
		leaf = false;
		return true;
	} else if (!is_leaf()) {
		auto rcl = thread_if_avail([=](int stack_cnt) {
			return children[0].refine(stack_cnt);
		}, true, stack_cnt);
		auto rcr = thread_if_avail([=](int stack_cnt) {
			return children[1].refine(stack_cnt);
		}, false, stack_cnt);
		bool rc1 = rcr.get();
		bool rc2 = rcl.get();
		return rc1 || rc2;
	} else {
		return false;
	}
}

multipole_return tree::compute_multipoles(rung_type mrung, bool do_out, int stack_cnt) {
	if (level == 0) {
		reset_node_cache();
		part_vect_cache_reset();
		printf("compute_multipoles\n");
	}
	const auto &opts = options::get();
	range prange;
	if (is_leaf()) {
		multi.x = part_vect_center_of_mass(part_begin, part_end).second;
		multi = part_vect_multipole_info(multi.x, do_out ? 0 : mrung, part_begin, part_end);
		prange = part_vect_range(part_begin, part_end);
	} else {
		multipole_return ml, mr;
		auto rcl = thread_if_avail([=](int stack_cnt) {
			return children[0].compute_multipoles(mrung, do_out, stack_cnt);
		}, true, stack_cnt);
		auto rcr = thread_if_avail([=](int stack_cnt) {
			return children[1].compute_multipoles(mrung, do_out, stack_cnt);
		}, false, stack_cnt);
		mr = rcr.get();
		ml = rcl.get();
		multi.m() = ml.m.m() + mr.m.m();
		if (multi.m() != 0.0) {
			multi.x = (ml.m.x * ml.m.m() + mr.m.x * mr.m.m()) / multi.m();
		} else {
			ERROR();
		}
		multi.m = (ml.m.m >> (ml.m.x - multi.x)) + (mr.m.m >> (mr.m.x - multi.x));
		multi.has_active = ml.m.has_active || mr.m.has_active;
		multi.r = std::max(abs(ml.m.x - multi.x) + ml.m.r, abs(mr.m.x - multi.x) + mr.m.r);
		for (int dim = 0; dim < NDIM; dim++) {
			prange.max[dim] = std::max(ml.r.max[dim], mr.r.max[dim]);
			prange.min[dim] = std::min(ml.r.min[dim], mr.r.min[dim]);
		}
		ireal rmax = abs(multi.x - vect<ireal>( { (ireal) prange.min[0], (ireal) prange.min[1], (ireal) prange.min[2] }));
		rmax = std::max(rmax, abs(multi.x - vect<ireal>( { (ireal) prange.max[0], (ireal) prange.min[1], (ireal) prange.min[2] })));
		rmax = std::max(rmax, abs(multi.x - vect<ireal>( { (ireal) prange.min[0], (ireal) prange.max[1], (ireal) prange.min[2] })));
		rmax = std::max(rmax, abs(multi.x - vect<ireal>( { (ireal) prange.max[0], (ireal) prange.max[1], (ireal) prange.min[2] })));
		rmax = std::max(rmax, abs(multi.x - vect<ireal>( { (ireal) prange.min[0], (ireal) prange.min[1], (ireal) prange.max[2] })));
		rmax = std::max(rmax, abs(multi.x - vect<ireal>( { (ireal) prange.max[0], (ireal) prange.min[1], (ireal) prange.max[2] })));
		rmax = std::max(rmax, abs(multi.x - vect<ireal>( { (ireal) prange.min[0], (ireal) prange.max[1], (ireal) prange.max[2] })));
		rmax = std::max(rmax, abs(multi.x - vect<ireal>( { (ireal) prange.max[0], (ireal) prange.max[1], (ireal) prange.max[2] })));
		multi.r = std::min(multi.r, rmax);
		child_check[0] = ml.c;
		child_check[1] = mr.c;
	}
	return multipole_return({multi, prange, get_check_item()});
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
	check.opened = false;
	check.is_leaf = leaf;
	check.pbegin = part_begin;
	check.pend = part_end;
	check.x = multi.x;
	check.r = multi.r;
	check.node = raw_tree_client(id);
	return check;
}

bool tree::is_leaf() const {
	return leaf;
}

struct workspace {
	std::vector<hpx::future<node_attr>> dfuts;
	std::vector<hpx::future<node_attr>> efuts;
	std::vector<hpx::future<multi_src>> dmulti_futs;
	std::vector<hpx::future<multi_src>> emulti_futs;
	std::vector<hpx::future<std::vector<vect<pos_type>>>> dsource_futs;
	std::vector<hpx::future<std::vector<vect<pos_type>>>> esource_futs;
	std::vector<multi_src> multi_srcs;
	std::vector<vect<float>> sources;
	std::vector<vect<float>> x;
	std::vector<force> f;

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
	this_space.dsource_futs.resize(0);
	this_space.esource_futs.resize(0);
	this_space.sources.resize(0);
	this_space.x.resize(0);
	this_space.f.resize(0);
	return std::move(this_space);
}

void trash_workspace(workspace &&w) {
	std::lock_guard<mutex_type> lock(workspace_mutex);
	workspaces.push(std::move(w));
}

kick_return tree::kick_fmm(std::vector<check_item> dchecklist, std::vector<check_item> echecklist, const vect<ireal> &Lcom, expansion<float> L,
		rung_type min_rung, bool do_out, int stack_cnt) {
	if (level == 0) {
		printf("kick_fmm\n");
	}

	kick_return rc;
	if (!multi.has_active && !do_out) {
		rc.rung = 0;
		return rc;
	}

	L = L << (multi.x - Lcom);

	const auto opts = options::get();
	const float m = 1.0 / opts.problem_size;
	std::vector<check_item> next_dchecklist;
	std::vector<check_item> next_echecklist;
	auto space = get_workspace();
	auto &multi_srcs = space.multi_srcs;
	auto &dfuts = space.dfuts;
	auto &efuts = space.efuts;
	auto &dsource_futs = space.dsource_futs;
	auto &esource_futs = space.esource_futs;
	auto &dmulti_futs = space.dmulti_futs;
	auto &emulti_futs = space.emulti_futs;
	auto &sources = space.sources;
	auto &x = space.x;
	auto &f = space.f;

	next_dchecklist.reserve(NCHILD * dchecklist.size());
	next_echecklist.reserve(NCHILD * echecklist.size());

	for (auto c : dchecklist) {
		const auto dx = opts.ewald ? ewald_near_separation(multi.x - c.x) : abs(multi.x - c.x);
		const bool far = dx > (multi.r + c.r) * theta_inv;
		if (far) {
			if (c.opened) {
				dsource_futs.push_back(part_vect_read_position(c.pbegin, c.pend));
			} else {
				dmulti_futs.push_back(c.node.get_multi_srcs());
			}
		} else {
			if (c.is_leaf) {
				c.opened = true;
				next_dchecklist.push_back(c);
			} else {
				dfuts.push_back(c.node.get_node_attributes());
			}
		}
	}

	if (opts.ewald) {
		for (auto c : echecklist) {
			const auto dx = ewald_far_separation(multi.x - c.x, multi.r + c.r);
			const bool far = dx > (multi.r + c.r) * theta_inv;
			if (far) {
				if (c.opened) {
					esource_futs.push_back(part_vect_read_position(c.pbegin, c.pend));
				} else {
					emulti_futs.push_back(c.node.get_multi_srcs());
				}
			} else {
				if (c.is_leaf) {
					c.opened = true;
					next_echecklist.push_back(c);
				} else {
					efuts.push_back(c.node.get_node_attributes());
				}
			}
		}
	}
	multi_srcs.resize(0);
	sources.resize(0);
	for (auto &v : dmulti_futs) {
		multi_srcs.push_back(v.get());
	}
	for (auto &v : dsource_futs) {
		auto s = v.get();
		for (auto x : s) {
			sources.push_back(pos_to_double(x));
		}
	}
	flop += gravity_CC_direct(L, multi.x, multi_srcs);
	flop += gravity_CP_direct(L, multi.x, sources);
	if (opts.ewald) {
		multi_srcs.resize(0);
		sources.resize(0);
		for (auto &v : emulti_futs) {
			multi_srcs.push_back(v.get());
		}
		for (auto &v : esource_futs) {
			auto s = v.get();
			for (auto x : s) {
				sources.push_back(pos_to_double(x));
			}
		}
		flop += gravity_CC_ewald(L, multi.x, multi_srcs);
		flop += gravity_CP_ewald(L, multi.x, sources);
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
		}, true, stack_cnt);
		auto rc_r_fut = thread_if_avail([&](int stack_cnt) {
			return children[1].kick_fmm(std::move(dchecklist), std::move(echecklist), multi.x, L, min_rung, do_out, stack_cnt);
		}, false, stack_cnt);
		const auto rc_r = rc_r_fut.get();
		const auto rc_l = rc_l_fut.get();
		rc.rung = std::max(rc_r.rung, rc_l.rung);
		if (do_out) {
			rc.out = std::move(rc_l.out);
			rc.out.insert(rc.out.end(), rc_r.out.begin(), rc_r.out.end());
		}
		if (do_out) {
			rc.stats = rc_r.stats + rc_l.stats;
		}
	} else {
		dsource_futs.resize(0);
		esource_futs.resize(0);
		dmulti_futs.resize(0);
		emulti_futs.resize(0);
		while (!dchecklist.empty()) {
			dfuts.resize(0);
			for (auto c : dchecklist) {
				const auto dx = opts.ewald ? ewald_near_separation(multi.x - c.x) : abs(multi.x - c.x);
				const bool far = dx > (multi.r + c.r) * theta_inv;
				if (c.opened) {
					dsource_futs.push_back(part_vect_read_position(c.pbegin, c.pend));
				} else {
					if (far) {
						dmulti_futs.push_back(c.node.get_multi_srcs());
					} else {
						if (c.is_leaf) {
							c.opened = true;
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
					const auto dx = ewald_far_separation(multi.x - c.x, multi.r + c.r);
					const bool far = dx > (multi.r + c.r) * theta_inv;
					if (c.opened) {
						esource_futs.push_back(part_vect_read_position(c.pbegin, c.pend));
					} else {
						if (far) {
							emulti_futs.push_back(c.node.get_multi_srcs());
						} else {
							if (c.is_leaf) {
								c.opened = true;
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
		const auto parts = part_vect_read(part_begin, part_end).get();
		for (auto i = parts.begin(); i != parts.end(); i++) {
			if (i->rung >= min_rung || do_out) {
				x.push_back(pos_to_double(i->x));
			}
		}
		f.resize(x.size());
		int j = 0;
		for (auto i = parts.begin(); i != parts.end(); i++) {
			if (i->rung >= min_rung || do_out) {
				force this_f = L.translate_L2(x[j] - multi.x);
				f[j].phi = this_f.phi;
				f[j].g = this_f.g;
				j++;
			}
		}
		multi_srcs.resize(0);
		for (auto &v : dmulti_futs) {
			multi_srcs.push_back(v.get());
		}
		sources.resize(0);
		for (auto &v : dsource_futs) {
			auto s = v.get();
			for (auto x : s) {
				sources.push_back(pos_to_double(x));
			}
		}
		flop += gravity_PC_direct(f, x, multi_srcs);
		flop += gravity_PP_direct(f, x, sources);
		if (opts.ewald) {
			multi_srcs.resize(0);
			for (auto &v : emulti_futs) {
				multi_srcs.push_back(v.get());
			}
			sources.resize(0);
			for (auto &v : esource_futs) {
				auto s = v.get();
				for (auto x : s) {
					sources.push_back(pos_to_double(x));
				}
			}
			flop += gravity_PC_ewald(f, x, multi_srcs);
			flop += gravity_PP_ewald(f, x, sources);
		}
		rc = do_kick(f, min_rung, do_out);
	}
	trash_workspace(std::move(space));
	return rc;
}

kick_return tree::do_kick(const std::vector<force> &f, rung_type min_rung, bool do_out) {
	const auto opts = options::get();
	const float eps = 10.0 * std::numeric_limits<float>::min();
	const float m = 1.0 / opts.problem_size;
	kick_return rc;
	rc.rung = 0;
	int j = 0;
	if (do_out) {
		rc.stats.zero();
	}
	auto parts = part_vect_read(part_begin, part_end).get();
	for (auto i = parts.begin(); i != parts.end(); i++) {
		if (i->rung >= min_rung || do_out) {
			if (i->rung >= min_rung) {
				if (i->rung != 0) {
					const float dt = rung_to_dt(i->rung);
					i->v = i->v + f[j].g * (0.5 * dt);
				}
				const float a = abs(f[j].g);
				float dt = std::min(opts.dt_max, opts.eta * std::sqrt(opts.soft_len / (a + eps)));
				rung_type rung = dt_to_rung(dt);
				rung = std::max(rung, min_rung);
				rc.rung = std::max(rc.rung, rung);
				dt = rung_to_dt(rung);
				i->rung = std::max(std::max(rung, rung_type(i->rung - 1)), (rung_type) 1);
				i->v = i->v + f[j].g * (0.5 * dt);
			}
			if (do_out) {
				rc.stats.g = rc.stats.g + f[j].g * m;
				rc.stats.p = rc.stats.p + i->v * m;
				rc.stats.pot += 0.5 * m * f[j].phi;
				rc.stats.kin += 0.5 * m * i->v.dot(i->v);
				if (i->flags.out) {
					output out;
					out.x = pos_to_double(i->x);
					out.v = i->v;
					out.g = f[j].g;
					out.phi = f[j].phi;
					out.rung = i->rung;
					rc.out.push_back(out);
				}
			}
			j++;
		}
	}
	part_vect_write(part_begin, part_end, std::move(parts));
	return rc;
}

void tree::drift(float dt) {
	if (level == 0) {
		part_vect_drift(dt);
	}
}

std::uint64_t tree::get_flop() {
	return flop;
}

void tree::reset_flop() {
	flop = 0;
}

#define NODE_CACHE_SIZE 1024
std::unordered_map<raw_id_type, hpx::shared_future<node_attr>, raw_id_type_hash> node_cache[NODE_CACHE_SIZE];
mutex_type node_cache_mtx[NODE_CACHE_SIZE];

std::unordered_map<raw_id_type, hpx::shared_future<multi_src>, raw_id_type_hash> multipole_cache[NODE_CACHE_SIZE];
mutex_type multipole_cache_mtx[NODE_CACHE_SIZE];

HPX_PLAIN_ACTION(reset_node_cache);

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

hpx::future<node_attr> raw_tree_client::get_node_attributes() const {
	if (myid == ptr.loc_id) {
		tree *tree_ptr = reinterpret_cast<tree*>(ptr.ptr);
		return hpx::async(hpx::launch::deferred, [tree_ptr]() {
			return tree_ptr->get_node_attributes();
		});
	} else {
		return read_node_cache(ptr);
	}
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

hpx::future<multi_src> raw_tree_client::get_multi_srcs() const {
	if (myid == ptr.loc_id) {
		tree *tree_ptr = reinterpret_cast<tree*>(ptr.ptr);
		return hpx::async(hpx::launch::deferred, [tree_ptr]() {
			return tree_ptr->get_multi_srcs();
		});
	} else {
		return read_multipole_cache(ptr);
	}
}

