#include <tigergrav/options.hpp>
#include <tigergrav/time.hpp>
#include <tigergrav/tree.hpp>

#include <atomic>
#include <algorithm>

std::atomic<std::uint64_t> tree::flop(0);
float tree::theta_inv;

static std::atomic<int> num_threads(1);
static bool inc_thread();
static void dec_thread();

bool inc_thread() {
#ifdef USE_HPX
	static const int nmax = 4 * hpx::threads::hardware_concurrency();
#else
	static const int nmax = 4 * std::thread::hardware_concurrency();
#endif
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
auto thread_if_avail(F &&f, int level, bool left=false) {
	bool thread;
	if (level % 8 == 0) {
		thread = true;
		num_threads++;
	} else if (left){
		thread = inc_thread();
	}
	if (thread) {
#ifdef USE_HPX
		auto rc = hpx::async([](F &&f) {
#else
		auto rc = std::async(std::launch::async, [](F &&f) {
#endif
			auto rc = f();
			dec_thread();
			return rc;
		},std::forward<F>(f));
		return rc;
	} else {
#ifdef USE_HPX
		return hpx::make_ready_future(f());
#else
		auto rc = f();
		using T = decltype(rc);
		std::promise<T> prms;
		prms.set_value(std::move(rc));
		return prms.get_future();
#endif
	}
}

void tree::set_theta(float t) {
	theta_inv = 1.0 / t;
}

tree_ptr tree::new_(range r, part_iter b, part_iter e, int level) {
	return std::make_shared<tree>(r, b, e, level);
}

tree::tree(range box, part_iter b, part_iter e, int level_) {
	level = level_;
	const auto &opts = options::get();
	part_begin = b;
	part_end = e;
	if (e - b > opts.parts_per_node) {
		float max_span = 0.0;
		range prange;
		for (int dim = 0; dim < NDIM; dim++) {
			prange.max[dim] = 0.0;
			prange.min[dim] = +1.0;
		}
		for (auto i = b; i != e; i++) {
			for (int dim = 0; dim < NDIM; dim++) {
				const auto x = pos_to_double(i->x[dim]);
				prange.max[dim] = std::max(prange.max[dim], x);
				prange.min[dim] = std::min(prange.min[dim], x);
			}
		}
		int max_dim;
		for (int dim = 0; dim < NDIM; dim++) {
			const auto this_span = prange.max[dim] - prange.min[dim];
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
		if (e - b < 64 * opts.parts_per_node) {
			std::sort(b, e, [max_dim](const particle &p1, const particle &p2) {
				return p1.x[max_dim] < p2.x[max_dim];
			});
			mid_iter = b + (e - b) / 2;
		} else {
			if (e - b == 0) {
				mid_iter = e;
			} else {
				mid_iter = bisect(b, e, [max_dim, mid](const particle &p) {
					return pos_to_double(p.x[max_dim]) < mid;
				});
			}
		}
		auto rcl = thread_if_avail([=]() {
			return new_(boxl, b, mid_iter, level + 1);
		}, level, true);
		auto rcr = thread_if_avail([=]() {
			return new_(boxr, mid_iter, e, level + 1);
		}, level);
		children[1] = rcr.get();
		children[0] = rcl.get();
	}
}

std::pair<multipole_info, range> tree::compute_multipoles(rung_type mrung, bool do_out) {
	const auto &opts = options::get();
	const auto m = 1.0 / opts.problem_size;
	multi.m = 0.0;
	range prange;
	if (is_leaf()) {
		multi.x = vect<float>(0.0);
		for (auto i = part_begin; i != part_end; i++) {
			multi.m() += m;
			multi.x += pos_to_double(i->x) * m;
		}
		if (multi.m() != 0.0) {
			multi.x = multi.x / multi.m();
		} else {
			ERROR();
		}
		for (auto i = part_begin; i != part_end; i++) {
			const auto X = pos_to_double(i->x) - multi.x;
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
		for (auto i = part_begin; i != part_end; i++) {
			multi.r = std::max(multi.r, (ireal) abs(pos_to_double(i->x) - multi.x));
		}
		bool rc = do_out;
		if (!do_out) {
			for (auto i = part_begin; i != part_end; i++) {
				if (i->rung >= mrung) {
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
		for (auto i = part_begin; i != part_end; i++) {
			const auto X = pos_to_double(i->x);
			for (int dim = 0; dim < NDIM; dim++) {
				prange.max[dim] = std::max(prange.max[dim], X[dim]);
				prange.min[dim] = std::min(prange.min[dim], X[dim]);
			}
		}
	} else {
		std::pair<multipole_info, range> ml, mr;
		auto rcl = thread_if_avail([=]() {
			return children[0]->compute_multipoles(mrung, do_out);
		}, level,true);
		auto rcr = thread_if_avail([=]() {
			return children[1]->compute_multipoles(mrung, do_out);
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

multipole_info tree::get_multipole() const {
	return multi;
}

monopole tree::get_monopole() const {
	return {(float) multi.m(),multi.x,multi.r};
}

bool tree::is_leaf() const {
	static const auto opts = options::get();
	return (part_end - part_begin) <= opts.parts_per_node;
}

std::array<tree_ptr, NCHILD> tree::get_children() const {
	return children;
}

std::pair<const_part_iter, const_part_iter> tree::get_positions() const {
	std::pair<const_part_iter, const_part_iter> iters;
	iters.first = part_begin;
	iters.second = part_end;
	return iters;
}

kick_return tree::kick_fmm(std::vector<check_item> dchecklist, std::vector<check_item> echecklist, const vect<ireal> &Lcom, expansion<float> L,
		rung_type min_rung, bool do_out) {

	kick_return rc;
	if (!multi.has_active && !do_out) {
		rc.rung = 0;
		return rc;
	}

	L = L << (multi.x - Lcom);

	static thread_local std::vector<check_item> next_dchecklist;
	static thread_local std::vector<check_item> next_echecklist;
	static const auto opts = options::get();
	static const float m = 1.0 / opts.problem_size;

	static thread_local std::vector<multi_src> dmulti_srcs;
	static thread_local std::vector<vect<float>> dsources;
	static thread_local std::vector<multi_src> emulti_srcs;
	static thread_local std::vector<vect<float>> esources;
	dmulti_srcs.resize(0);
	emulti_srcs.resize(0);
	esources.resize(0);
	dsources.resize(0);
	next_dchecklist.resize(0);
	next_echecklist.resize(0);
	next_dchecklist.reserve(NCHILD * dchecklist.size());
	next_echecklist.reserve(NCHILD * echecklist.size());

	for (auto c : dchecklist) {
		auto other = c.ptr->get_multipole();
		const auto dx = opts.ewald ? ewald_near_separation(multi.x - other.x) : abs(multi.x - other.x);
		const bool far = dx > (multi.r + other.r) * theta_inv;
		if (far) {
			if (c.opened) {
				const auto pos = c.ptr->get_positions();
				for (auto i = pos.first; i != pos.second; i++) {
					dsources.push_back(pos_to_double(i->x));
				}
			} else {
				dmulti_srcs.push_back( { other.m, other.x });
			}
		} else {
			if (c.ptr->is_leaf()) {
				next_dchecklist.push_back( { true, c.ptr });
			} else {
				auto next = c.ptr->get_children();
				next_dchecklist.push_back( { false, next[0] });
				next_dchecklist.push_back( { false, next[1] });
			}
		}
	}

	if (opts.ewald) {
		for (auto c : echecklist) {
			auto other = c.ptr->get_multipole();
			const auto dx = ewald_far_separation(multi.x - other.x, multi.r + other.r);
			const bool far = dx > (multi.r + other.r) * theta_inv;
			if (far) {
				if (c.opened) {
					const auto pos = c.ptr->get_positions();
					for (auto i = pos.first; i != pos.second; i++) {
						esources.push_back(pos_to_double(i->x));
					}
				} else {
					emulti_srcs.push_back( { other.m, other.x });
				}
			} else {
				if (c.ptr->is_leaf()) {
					next_echecklist.push_back( { true, c.ptr });
				} else {
					auto next = c.ptr->get_children();
					next_echecklist.push_back( { false, next[0] });
					next_echecklist.push_back( { false, next[1] });
				}
			}
		}
	}
	flop += gravity_CC_direct(L, multi.x, dmulti_srcs);
	flop += gravity_CP_direct(L, multi.x, dsources);
	if (opts.ewald) {
		flop += gravity_CC_ewald(L, multi.x, emulti_srcs);
		flop += gravity_CP_ewald(L, multi.x, esources);
	}
	std::swap(dchecklist, next_dchecklist);
	std::swap(echecklist, next_echecklist);
	next_dchecklist.resize(0);
	next_echecklist.resize(0);
	dmulti_srcs.resize(0);
	emulti_srcs.resize(0);
	dsources.resize(0);
	esources.resize(0);
	if (!is_leaf()) {
		auto rc_l_fut = thread_if_avail([=]() {
			return children[0]->kick_fmm(std::move(dchecklist), std::move(echecklist), multi.x, L, min_rung, do_out);
		}, level, true);
		auto rc_r_fut = thread_if_avail([=]() {
			return children[1]->kick_fmm(std::move(dchecklist), std::move(echecklist), multi.x, L, min_rung, do_out);
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
		while (!dchecklist.empty()) {
			for (auto c : dchecklist) {
				auto other = c.ptr->get_multipole();
				const auto dx = opts.ewald ? ewald_near_separation(multi.x - other.x) : abs(multi.x - other.x);
				const bool far = dx > (multi.r + other.r) * theta_inv;
				if (c.opened) {
					const auto pos = c.ptr->get_positions();
					for (auto i = pos.first; i != pos.second; i++) {
						dsources.push_back(pos_to_double(i->x));
					}
				} else {
					if (far) {
						dmulti_srcs.push_back( { other.m, other.x });
					} else {
						if (c.ptr->is_leaf()) {
							next_dchecklist.push_back( { true, c.ptr });
						} else {
							auto next = c.ptr->get_children();
							next_dchecklist.push_back( { false, next[0] });
							next_dchecklist.push_back( { false, next[1] });
						}
					}
				}
			}
			std::swap(dchecklist, next_dchecklist);
			next_dchecklist.resize(0);
		}
		if (opts.ewald) {
			while (!echecklist.empty()) {
				for (auto c : echecklist) {
					auto other = c.ptr->get_multipole();
					const auto dx = ewald_far_separation(multi.x - other.x, multi.r + other.r);
					const bool far = dx > (multi.r + other.r) * theta_inv;
					if (c.opened) {
						const auto pos = c.ptr->get_positions();
						for (auto i = pos.first; i != pos.second; i++) {
							esources.push_back(pos_to_double(i->x));
						}
					} else {
						if (far) {
							emulti_srcs.push_back( { other.m, other.x });
						} else {
							if (c.ptr->is_leaf()) {
								next_echecklist.push_back( { true, c.ptr });
							} else {
								auto next = c.ptr->get_children();
								next_echecklist.push_back( { false, next[0] });
								next_echecklist.push_back( { false, next[1] });
							}
						}
					}
				}
				std::swap(echecklist, next_echecklist);
				next_echecklist.resize(0);
			}
		}
		static thread_local std::vector<vect<float>> x;
		x.resize(0);
		for (auto i = part_begin; i != part_end; i++) {
			if (i->rung >= min_rung || do_out) {
				x.push_back(pos_to_double(i->x));
			}
		}
		std::vector<force> f(x.size());
		int j = 0;
		for (auto i = part_begin; i != part_end; i++) {
			if (i->rung >= min_rung || do_out) {
				force this_f = L.translate_L2(x[j] - multi.x);
				f[j].phi = this_f.phi;
				f[j].g = this_f.g;
				j++;
			}
		}

		flop += gravity_PP_direct(f, x, dsources);
		flop += gravity_PC_direct(f, x, dmulti_srcs);
		if (opts.ewald) {
			flop += gravity_PP_ewald(f, x, esources);
			flop += gravity_PC_ewald(f, x, emulti_srcs);
		}
		rc = do_kick(f, min_rung, do_out);
	}
	return rc;
}

kick_return tree::kick_bh(std::vector<tree_ptr> dchecklist, std::vector<vect<float>> dsources, std::vector<multi_src> multi_srcs,
		std::vector<tree_ptr> echecklist, std::vector<vect<float>> esources, std::vector<multi_src> emulti_srcs, rung_type min_rung, bool do_out) {

	kick_return rc;
	if (!multi.has_active && !do_out) {
		rc.rung = 0;
		return rc;
	}

	std::vector<tree_ptr> next_dchecklist;
	std::vector<tree_ptr> next_echecklist;
	static const auto opts = options::get();
	static const float m = 1.0 / opts.problem_size;
	for (auto c : dchecklist) {
		auto other = c->get_multipole();
		const auto dx = opts.ewald ? ewald_near_separation(multi.x - other.x) : abs(multi.x - other.x);
		if (dx > (multi.r + other.r) * theta_inv) {
			multi_srcs.push_back( { other.m, other.x });
		} else {
			if (c->is_leaf()) {
				const auto pos = c->get_positions();
				for (auto i = pos.first; i != pos.second; i++) {
					dsources.push_back(pos_to_double(i->x));
				}
			} else {
				auto next = c->get_children();
				next_dchecklist.push_back(next[0]);
				next_dchecklist.push_back(next[1]);
			}
		}
	}
	dchecklist = std::move(next_dchecklist);
	if (opts.ewald) {
		for (auto c : echecklist) {
			auto other = c->get_multipole();
			const auto dx = ewald_far_separation(multi.x - other.x, multi.r + other.r);
			if (dx > (multi.r + other.r) * theta_inv) {
				emulti_srcs.push_back( { other.m, other.x });
			} else {
				if (c->is_leaf()) {
					const auto pos = c->get_positions();
					for (auto i = pos.first; i != pos.second; i++) {
						esources.push_back(pos_to_double(i->x));
					}
				} else {
					auto next = c->get_children();
					next_echecklist.push_back(next[0]);
					next_echecklist.push_back(next[1]);
				}
			}
		}
	}
	echecklist = std::move(next_echecklist);
	if (!is_leaf()) {
		auto rc_l_fut = thread_if_avail(
				[=]() {
					return children[0]->kick_bh(std::move(dchecklist), std::move(dsources), std::move(multi_srcs), std::move(echecklist), std::move(esources),
							std::move(emulti_srcs), min_rung, do_out);
				}, level,true);
		auto rc_r_fut = thread_if_avail(
				[=]() {
					return children[1]->kick_bh(std::move(dchecklist), std::move(dsources), std::move(multi_srcs), std::move(echecklist), std::move(esources),
							std::move(emulti_srcs), min_rung, do_out);
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
		if (!dchecklist.empty() || !echecklist.empty()) {
			rc = kick_bh(std::move(dchecklist), std::move(dsources), std::move(multi_srcs), std::move(echecklist), std::move(esources), std::move(emulti_srcs),
					min_rung, do_out);
		} else {
			std::vector<vect<float>> x;
			for (auto i = part_begin; i != part_end; i++) {
				if (i->rung >= min_rung || do_out) {
					x.push_back(pos_to_double(i->x));
				}
			}
			std::vector<force> f(x.size(), { 0, vect<float>(0) });
			flop += gravity_PC_direct(f, x, multi_srcs);
			flop += gravity_PP_direct(f, x, dsources);
			if (opts.ewald) {
				flop += gravity_PP_ewald(f, x, esources);
				flop += gravity_PC_ewald(f, x, emulti_srcs);
			}
			rc = do_kick(f, min_rung, do_out);
		}
	}
	return rc;
}

kick_return tree::kick_direct(std::vector<vect<float>> &sources, rung_type min_rung, bool do_out) {

	static const auto opts = options::get();
	static const float m = 1.0 / opts.problem_size;
	if (sources.size() == 0) {
		for (auto i = part_begin; i != part_end; i++) {
			sources.push_back(pos_to_double(i->x));
		}
	}

	kick_return rc;
	if (!multi.has_active && !do_out) {
		rc.rung = 0;
		return rc;
	}

	if (!is_leaf()) {
		auto rc_l_fut = thread_if_avail([&]() {
			return children[0]->kick_direct(sources, min_rung, do_out);
		}, level,true);
		auto rc_r_fut = thread_if_avail([&]() {
			return children[1]->kick_direct(sources, min_rung, do_out);
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
		std::vector<vect<float>> x;
		for (auto i = part_begin; i != part_end; i++) {
			if (i->rung >= min_rung || do_out) {
				x.push_back(pos_to_double(i->x));
			}
		}
		std::vector<force> f(x.size(), { 0, vect<float>(0) });
		auto esources = sources;
		flop += gravity_PP_direct(f, x, sources);
		if (opts.ewald) {
			flop += gravity_PP_ewald(f, x, esources);
		}
		rc = do_kick(f, min_rung, do_out);
	}
	return rc;
}

kick_return tree::do_kick(const std::vector<force> &f, rung_type min_rung, bool do_out) {
	static const auto opts = options::get();
	static const float eps = 10.0 * std::numeric_limits<float>::min();
	static const float m = 1.0 / opts.problem_size;
	kick_return rc;
	rc.rung = 0;
	int j = 0;
	if (do_out) {
		rc.stats.zero();
	}
	for (auto i = part_begin; i != part_end; i++) {
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
	return rc;
}

void tree::drift(float dt) {
	if (is_leaf()) {
		for (auto i = part_begin; i != part_end; i++) {
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
	} else {
		auto rcl = thread_if_avail([=]() {
			children[0]->drift(dt);
			return 1;
		}, level, true);
		auto rcr = thread_if_avail([=]() {
			children[1]->drift(dt);
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

bool tree::active_particles(int rung, bool do_out) {
	if (do_out) {
		return true;
	}
	bool rc;
	if (is_leaf()) {
		rc = false;
		for (auto i = part_begin; i != part_end; i++) {
			if (i->rung >= rung) {
				rc = true;
				break;
			}
		}
	} else {
		const auto rc1 = children[0]->active_particles(rung, do_out);
		const auto rc2 = children[1]->active_particles(rung, do_out);
		rc = rc1 || rc2;
	}
	multi.has_active = rc;
	return rc;
}
