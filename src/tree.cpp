#include <tigergrav/options.hpp>
#include <tigergrav/time.hpp>
#include <tigergrav/tree.hpp>

#include <hpx/runtime/actions/plain_action.hpp>

#include <atomic>
#include <algorithm>
#include <stack>
#include <thread>
#include <unordered_map>

HPX_REGISTER_COMPONENT(hpx::components::component<tree>, tree);

std::atomic<std::uint64_t> tree::flop(0);
float tree::theta_inv;

void reset_node_cache();

static std::atomic<int> num_threads(1);
static bool inc_thread();
static void dec_thread();

static std::vector<hpx::id_type> localities;
static int myid;
static int hardware_concurrency = std::thread::hardware_concurrency();

struct raw_id_type_hash {
	std::size_t operator()(raw_id_type id) const {
		return id.ptr | id.loc_id;
	}
};

bool inc_thread() {
	const int nmax = 4 * hardware_concurrency;
	if (num_threads++ < nmax) {
		return true;
	} else {
		num_threads--;
		return false;
	}
}

void dec_thread() {
	num_threads--;
}

template<class F>
auto thread_if_avail(F &&f, int level, bool left = false) {
	bool thread;
	if (level % 8 == 0) {
		thread = true;
		num_threads++;
	} else if (left) {
		thread = inc_thread();
	}
	if (thread) {
		auto rc = hpx::async([](F &&f) {
			auto rc = f();
			dec_thread();
			return rc;
		},std::forward<F>(f));
		return rc;
	} else {
		return hpx::make_ready_future(f());
	}
}

HPX_PLAIN_ACTION(tree::set_theta,set_theta_action);

void tree::set_theta(float t) {
	set_theta_action action;
	theta_inv = 1.0 / t;
	localities = hpx::find_all_localities();
	myid = hpx::get_locality_id();
	if (myid == 0) {
		std::vector<hpx::future<void>> futs;
		for (int i = 1; i < localities.size(); i++) {
			futs.push_back(hpx::async < set_theta_action > (localities[i], t));
		}
		hpx::wait_all(futs);
	}
}

tree_client tree::new_(range r, part_iter b, part_iter e, int level) {
	return tree_client(hpx::new_ < tree > (localities[part_vect_locality_id(b)], r, b, e, level).get());
}

tree::tree(range box, part_iter b, part_iter e, int level_) {
//	printf("Forming %i %i\n", b, e);
	level = level_;
	const auto &opts = options::get();
	part_begin = b;
	part_end = e;
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
	if (e - b > opts.parts_per_node) {
		float max_span = 0.0;
		const range prange = part_vect_range(b, e);
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
		decltype(b) mid_iter;
		if (e - b == 0) {
			mid_iter = e;
		} else {
			mid_iter = part_vect_sort(b, e, mid, max_dim);
		}
//		}
		auto rcl = thread_if_avail([=]() {
			return new_(boxl, b, mid_iter, level + 1);
		}, level, true);
		auto rcr = thread_if_avail([=]() {
			return new_(boxr, mid_iter, e, level + 1);
		}, level);
		children[1] = rcr.get();
		children[0] = rcl.get();
		raw_children[0] = children[0].get_raw_ptr();
		raw_children[1] = children[1].get_raw_ptr();
	}
}

std::pair<multipole_info, range> tree::compute_multipoles(rung_type mrung, bool do_out) {
	if (level == 0) {
		printf("compute_multipoles\n");
	}
	const auto &opts = options::get();
	const auto m = 1.0 / opts.problem_size;
	multi.m = 0.0;
	range prange;
	if (is_leaf()) {
		multi.x = vect<float>(0.0);
		const auto parts = part_vect_read(part_begin, part_end).get();
		for (const auto &p : parts) {
			multi.m() += m;
			multi.x += pos_to_double(p.x) * m;
		}
		if (multi.m() != 0.0) {
			multi.x = multi.x / multi.m();
		} else {
			ERROR();
		}
		for (const auto &p : parts) {
			const auto X = pos_to_double(p.x) - multi.x;
			for (int j = 0; j < NDIM; j++) {
				for (int k = 0; k <= j; k++) {
					multi.m(j, k) += m * X[j] * X[k];
					for (int l = 0; l <= k; l++) {
						multi.m(j, k, l) += m * X[j] * X[k] * X[l];
					}
				}
			}
		}
		multi.r = 0.0;
		for (const auto &p : parts) {
			multi.r = std::max(multi.r, (ireal) abs(pos_to_double(p.x) - multi.x));
		}
		bool rc = do_out;
		if (!do_out) {
			for (const auto &p : parts) {
				if (p.rung >= mrung) {
					rc = true;
					break;
				}
			}
		}
		multi.has_active = rc;
		for (int dim = 0; dim < NDIM; dim++) {
			prange.max[dim] = 0.0;
			prange.min[dim] = 1.0;
		}
		for (const auto &p : parts) {
			const auto X = pos_to_double(p.x);
			for (int dim = 0; dim < NDIM; dim++) {
				prange.max[dim] = std::max(prange.max[dim], X[dim]);
				prange.min[dim] = std::min(prange.min[dim], X[dim]);
			}
		}
	} else {
		std::pair<multipole_info, range> ml, mr;
		auto rcl = thread_if_avail([=]() {
			return children[0].compute_multipoles(mrung, do_out);
		}, level, true);
		auto rcr = thread_if_avail([=]() {
			return children[1].compute_multipoles(mrung, do_out);
		}, level);
		mr = rcr.get();
		ml = rcl.get();
		multi.m() = ml.first.m() + mr.first.m();
		if (multi.m() != 0.0) {
			multi.x = (ml.first.x * ml.first.m() + mr.first.x * mr.first.m()) / multi.m();
		} else {
			ERROR();
		}
		multi.m = (ml.first.m >> (ml.first.x - multi.x)) + (mr.first.m >> (mr.first.x - multi.x));
		multi.has_active = ml.first.has_active || mr.first.has_active;
		multi.r = std::max(abs(ml.first.x - multi.x) + ml.first.r, abs(mr.first.x - multi.x) + mr.first.r);
		for (int dim = 0; dim < NDIM; dim++) {
			prange.max[dim] = std::max(ml.second.max[dim], mr.second.max[dim]);
			prange.min[dim] = std::min(ml.second.min[dim], mr.second.min[dim]);
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
	}
	return std::make_pair(multi, prange);
}

node_attr tree::get_node_attributes() const {
	node_attr attr;
	attr.multi = multi;
	attr.leaf = is_leaf();
	attr.children[0] = raw_children[0];
	attr.children[1] = raw_children[1];
	attr.pbegin = part_begin;
	attr.pend = part_end;
	return attr;
}

raw_id_type tree::get_raw_ptr() const {
	const auto loc_id = myid;
	raw_id_type id;
	id.loc_id = loc_id;
	id.ptr = reinterpret_cast<std::uint64_t>(this);
	return id;
}

bool tree::is_leaf() const {
	const auto opts = options::get();
	return (part_end - part_begin) <= opts.parts_per_node;
}

struct workspace {
	std::vector<multi_src> dmulti_srcs;
	std::vector<multi_src> emulti_srcs;
	std::vector<hpx::future<node_attr>> dfuts;
	std::vector<hpx::future<node_attr>> efuts;
	std::vector<hpx::future<std::vector<vect<pos_type>>>> dsource_futs;
	std::vector<hpx::future<std::vector<vect<pos_type>>>> esource_futs;
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
	this_space.dmulti_srcs.resize(0);
	this_space.emulti_srcs.resize(0);
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
		rung_type min_rung, bool do_out) {
	if (level == 0) {
		reset_node_cache();
		part_vect_cache_reset();
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
	auto &dmulti_srcs = space.dmulti_srcs;
	auto &emulti_srcs = space.emulti_srcs;
	auto &dfuts = space.dfuts;
	auto &efuts = space.efuts;
	auto &dsource_futs = space.dsource_futs;
	auto &esource_futs = space.esource_futs;
	auto &sources = space.sources;
	auto &x = space.x;
	auto &f = space.f;

	next_dchecklist.reserve(NCHILD * dchecklist.size());
	next_echecklist.reserve(NCHILD * echecklist.size());

	for (auto c : dchecklist) {
		dfuts.push_back(c.node.get_node_attributes());
	}
	if (opts.ewald) {
		for (auto c : echecklist) {
			efuts.push_back(c.node.get_node_attributes());
		}
	}
	int futi = 0;
	for (auto c : dchecklist) {
		const auto other = dfuts[futi++].get();
		const auto dx = opts.ewald ? ewald_near_separation(multi.x - other.multi.x) : abs(multi.x - other.multi.x);
		const bool far = dx > (multi.r + other.multi.r) * theta_inv;
		if (far) {
			if (c.opened) {
				dsource_futs.push_back(part_vect_read_position(other.pbegin, other.pend));
			} else {
				dmulti_srcs.push_back( { other.multi.m, other.multi.x });
			}
		} else {
			if (other.leaf) {
				next_dchecklist.push_back( { true, c.node });
			} else {
				next_dchecklist.push_back( { false, other.children[0] });
				next_dchecklist.push_back( { false, other.children[1] });
			}
		}
	}

	if (opts.ewald) {
		futi = 0;
		for (auto c : echecklist) {
			const auto other = efuts[futi++].get();
			const auto dx = ewald_far_separation(multi.x - other.multi.x, multi.r + other.multi.r);
			const bool far = dx > (multi.r + other.multi.r) * theta_inv;
			if (far) {
				if (c.opened) {
					esource_futs.push_back(part_vect_read_position(other.pbegin, other.pend));
				} else {
					emulti_srcs.push_back( { other.multi.m, other.multi.x });
				}
			} else {
				if (other.leaf) {
					next_echecklist.push_back( { true, c.node });
				} else {
					next_echecklist.push_back( { false, other.children[0] });
					next_echecklist.push_back( { false, other.children[1] });
				}
			}
		}
	}
	flop += gravity_CC_direct(L, multi.x, dmulti_srcs);
	sources.resize(0);
	for (auto &v : dsource_futs) {
		auto s = v.get();
		for (auto x : s) {
			sources.push_back(pos_to_double(x));
		}
	}
	flop += gravity_CP_direct(L, multi.x, sources);
	if (opts.ewald) {
		flop += gravity_CC_ewald(L, multi.x, emulti_srcs);
		sources.resize(0);
		for (auto &v : esource_futs) {
			auto s = v.get();
			for (auto x : s) {
				sources.push_back(pos_to_double(x));
			}
		}
		flop += gravity_CP_ewald(L, multi.x, sources);
	}
	std::swap(dchecklist, next_dchecklist);
	std::swap(echecklist, next_echecklist);
	next_dchecklist.resize(0);
	next_echecklist.resize(0);
	dmulti_srcs.resize(0);
	emulti_srcs.resize(0);
	if (!is_leaf()) {
		auto rc_l_fut = thread_if_avail([=]() {
			return children[0].kick_fmm(std::move(dchecklist), std::move(echecklist), multi.x, L, min_rung, do_out);
		}, level, true);
		auto rc_r_fut = thread_if_avail([&]() {
			return children[1].kick_fmm(std::move(dchecklist), std::move(echecklist), multi.x, L, min_rung, do_out);
		}, level);
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
		while (!dchecklist.empty()) {
			dfuts.resize(0);
			for (auto c : dchecklist) {
				dfuts.push_back(c.node.get_node_attributes());
			}
			int futi = 0;
			for (auto c : dchecklist) {
				const auto other = dfuts[futi++].get();
				const auto dx = opts.ewald ? ewald_near_separation(multi.x - other.multi.x) : abs(multi.x - other.multi.x);
				const bool far = dx > (multi.r + other.multi.r) * theta_inv;
				if (c.opened) {
					dsource_futs.push_back(part_vect_read_position(other.pbegin, other.pend));
				} else {
					if (far) {
						dmulti_srcs.push_back( { other.multi.m, other.multi.x });
					} else {
						if (other.leaf) {
							next_dchecklist.push_back( { true, c.node });
						} else {
							next_dchecklist.push_back( { false, other.children[0] });
							next_dchecklist.push_back( { false, other.children[1] });
						}
					}
				}
			}
			std::swap(dchecklist, next_dchecklist);
			next_dchecklist.resize(0);
		}
		if (opts.ewald) {
			while (!echecklist.empty()) {
				efuts.resize(0);
				for (auto c : echecklist) {
					efuts.push_back(c.node.get_node_attributes());
				}
				int futi = 0;
				for (auto c : echecklist) {
					const auto other = efuts[futi++].get();
					const auto dx = ewald_far_separation(multi.x - other.multi.x, multi.r + other.multi.r);
					const bool far = dx > (multi.r + other.multi.r) * theta_inv;
					if (c.opened) {
						esource_futs.push_back(part_vect_read_position(other.pbegin, other.pend));
					} else {
						if (far) {
							emulti_srcs.push_back( { other.multi.m, other.multi.x });
						} else {
							if (other.leaf) {
								next_echecklist.push_back( { true, c.node });
							} else {
								next_echecklist.push_back( { false, other.children[0] });
								next_echecklist.push_back( { false, other.children[1] });
							}
						}
					}
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
		flop += gravity_PC_direct(f, x, dmulti_srcs);
		sources.resize(0);
		for (auto &v : dsource_futs) {
			auto s = v.get();
			for (auto x : s) {
				sources.push_back(pos_to_double(x));
			}
		}
		flop += gravity_PP_direct(f, x, sources);
		if (opts.ewald) {
			flop += gravity_PC_ewald(f, x, emulti_srcs);
			sources.resize(0);
			for (auto &v : esource_futs) {
				auto s = v.get();
				for (auto x : s) {
					sources.push_back(pos_to_double(x));
				}
			}
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
		printf("drift\n");
	}

	if (is_leaf()) {
		auto parts = part_vect_read(part_begin, part_end).get();
		for (auto i = parts.begin(); i != parts.end(); i++) {
			const vect<double> dx = i->v * dt;
			vect<double> x = pos_to_double(i->x);
			x += dx;
			for (int dim = 0; dim < NDIM; dim++) {
				while (x[dim] >= 1.0) {
					x[dim] -= 1.0;
				}
				while (x[dim] < 0.0) {
					x[dim] += 1.0;
				}
			}
			i->x = double_to_pos(x);
		}
		part_vect_write(part_begin, part_end, std::move(parts));
	} else {
		auto rcl = thread_if_avail([=]() {
			children[0].drift(dt);
			return 1;
		}, level, true);
		auto rcr = thread_if_avail([=]() {
			children[1].drift(dt);
			return 1;
		}, level);
		rcl.get();
		rcr.get();
	}
}

std::uint64_t tree::get_flop() {
	return flop;
}

void tree::reset_flop() {
	flop = 0;
}

std::unordered_map<raw_id_type, hpx::shared_future<node_attr>, raw_id_type_hash> node_cache;
mutex_type node_cache_mtx;

HPX_PLAIN_ACTION (reset_node_cache);

void reset_node_cache() {
	std::vector<hpx::future<void>> futs;
	if (myid == 0) {
		for (int i = 1; i < localities.size(); i++) {
			futs.push_back(hpx::async < reset_node_cache_action > (localities[i]));
		}
	}
	node_cache.clear();
	hpx::wait_all(futs);
}

hpx::future<node_attr> get_node_attributes_(raw_id_type id);

HPX_PLAIN_ACTION (get_node_attributes_, get_node_attributes_action);

hpx::future<node_attr> get_node_attributes_(raw_id_type id) {
	if (myid == id.loc_id) {
		return hpx::make_ready_future(reinterpret_cast<tree*>(id.ptr)->get_node_attributes());
	} else {
		return hpx::async < get_node_attributes_action > (localities[id.loc_id], id);
	}
}

hpx::future<node_attr> read_node_cache(raw_id_type id) {
	std::lock_guard<mutex_type> lock(node_cache_mtx);
	auto iter = node_cache.find(id);
	if (iter == node_cache.end()) {
		node_cache[id] = get_node_attributes_(id);
	}
	return hpx::async(hpx::launch::deferred, [id] {
		std::unique_lock<mutex_type> lock(node_cache_mtx);
		auto future = node_cache[id];
		lock.unlock();
		return future.get();
	});
}

hpx::future<node_attr> raw_tree_client::get_node_attributes() const {
	if (myid == ptr.loc_id) {
		tree* tree_ptr = reinterpret_cast<tree*>(ptr.ptr);
		return hpx::async(hpx::launch::deferred, [tree_ptr]() {
			return tree_ptr->get_node_attributes();
		});
	} else {
		return read_node_cache(ptr);
	}
}

