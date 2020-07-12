#include <tigergrav/options.hpp>
#include <tigergrav/time.hpp>
#include <tigergrav/tree.hpp>

#include <hpx/include/async.hpp>
#include <hpx/include/threads.hpp>

#include <atomic>

std::atomic<std::uint64_t> tree::flop(0);
double tree::theta_inv;

static std::atomic<int> num_threads(1);
static bool inc_thread();
static void dec_thread();

bool inc_thread() {
	static const int nmax = 2 * hpx::threads::hardware_concurrency();
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
auto thread_if_avail(F &&f) {
	if (inc_thread()) {
		auto rc = hpx::async(f);
		dec_thread();
		return rc;
	} else {
		return hpx::make_ready_future(f());
	}
}

void tree::set_theta(double t) {
	theta_inv = 1.0 / t;
}

tree_ptr tree::new_(range r, part_iter b, part_iter e) {
	return std::make_shared<tree>(r, b, e);
}

tree::tree(range box, part_iter b, part_iter e) {
	const auto &opts = options::get();
	part_begin = b;
	part_end = e;
	max_span = box.max[0] - box.min[0];
	for (int dim = 1; dim < NDIM; dim++) {
		max_span = std::max(max_span, box.max[dim] - box.min[dim]);
	}
	for (int dim = 0; dim < NDIM; dim++) {
		coord_cent[dim] = 0.5 * (box.max[dim] + box.min[dim]);
	}
	if (e - b > opts.parts_per_node) {
		leaf = false;
		double max_span = 0.0;
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
		double mid = (box.max[max_dim] + box.min[max_dim]) * 0.5;
		boxl.max[max_dim] = boxr.min[max_dim] = mid;
		decltype(b) mid_iter;
		if (e - b == 0) {
			mid_iter = e;
		} else {
			mid_iter = bisect(b, e, [max_dim, mid](const particle &p) {
				return pos_to_double(p.x[max_dim]) < mid;
			});
		}
		children[0] = new_(boxl, b, mid_iter);
		children[1] = new_(boxr, mid_iter, e);
	} else {
		leaf = true;
	}
}

multipole_info tree::compute_multipoles() {
	const auto &opts = options::get();
	const auto m = 1.0 / opts.problem_size;
	multi.m = 0.0;
	if (leaf) {
		multi.x = vect<double>(0.0);
		for (auto i = part_begin; i != part_end; i++) {
			multi.m() += m;
			multi.x += pos_to_double(i->x) * m;
		}
		if (multi.m() != 0.0) {
			multi.x = multi.x / multi.m();
		} else {
			multi.x = coord_cent;
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
	} else {
		multipole_info ml, mr;
		ml = children[0]->compute_multipoles();
		mr = children[1]->compute_multipoles();
		multi.m() = ml.m() + mr.m();
		if (multi.m() != 0.0) {
			multi.x = (ml.x * ml.m() + mr.x * mr.m()) / multi.m();
		} else {
			multi.x = coord_cent;
		}
		multi.m = (ml.m >> (ml.x - multi.x)) + (mr.m >> (mr.x - multi.x));
		multi.r = std::max(abs(ml.x - multi.x) + ml.r, abs(mr.x - multi.x) + mr.r);
		child_com[0] = ml.x;
		child_com[1] = mr.x;
		if (part_end - part_begin > 0) {
			range corners;
			for (int dim = 0; dim < NDIM; dim++) {
				corners.max[dim] = -std::numeric_limits<ireal>::max();
				corners.min[dim] = +std::numeric_limits<ireal>::max();
			}
			for (auto i = part_begin; i != part_end; i++) {
				const auto X = pos_to_double(i->x);
				for (int dim = 0; dim < NDIM; dim++) {
					corners.max[dim] = std::max(corners.max[dim], X[dim]);
					corners.min[dim] = std::min(corners.min[dim], X[dim]);
				}
			}
			ireal rmax = abs(multi.x - vect<ireal>( { (ireal) corners.min[0], (ireal) corners.min[1], (ireal) corners.min[2] }));
			rmax = std::max(rmax, abs(multi.x - vect<ireal>( { (ireal) corners.max[0], (ireal) corners.min[1], (ireal) corners.min[2] })));
			rmax = std::max(rmax, abs(multi.x - vect<ireal>( { (ireal) corners.min[0], (ireal) corners.max[1], (ireal) corners.min[2] })));
			rmax = std::max(rmax, abs(multi.x - vect<ireal>( { (ireal) corners.max[0], (ireal) corners.max[1], (ireal) corners.min[2] })));
			rmax = std::max(rmax, abs(multi.x - vect<ireal>( { (ireal) corners.min[0], (ireal) corners.min[1], (ireal) corners.max[2] })));
			rmax = std::max(rmax, abs(multi.x - vect<ireal>( { (ireal) corners.max[0], (ireal) corners.min[1], (ireal) corners.max[2] })));
			rmax = std::max(rmax, abs(multi.x - vect<ireal>( { (ireal) corners.min[0], (ireal) corners.max[1], (ireal) corners.max[2] })));
			rmax = std::max(rmax, abs(multi.x - vect<ireal>( { (ireal) corners.max[0], (ireal) corners.max[1], (ireal) corners.max[2] })));
			multi.r = std::min(multi.r, rmax);
		}
	}
	return multi;
}

multipole_info tree::get_multipole() const {
	return multi;
}

monopole tree::get_monopole() const {
	return {(float) multi.m(),multi.x,multi.r};
}

bool tree::is_leaf() const {
	return leaf;
}

std::array<tree_ptr, NCHILD> tree::get_children() const {
	return children;
}

std::vector<vect<float>> tree::get_positions() const {
	std::vector<vect<float>> pos;
	pos.reserve(part_end - part_begin);
	for (auto i = part_begin; i != part_end; i++) {
		vect<float> x;
		x = pos_to_float(i->x);
		pos.push_back(x);
	}
	return pos;
}

kick_return tree::kick_fmm(std::vector<tree_ptr> dchecklist, std::vector<vect<float>> dsources, std::vector<tree_ptr> echecklist, std::vector<source> esources,
		expansion<ireal> L, rung_type min_rung, bool do_out) {

	kick_return rc;
	if (!has_active && !do_out) {
		rc.rung = 0;
		return rc;
	}

	std::vector<tree_ptr> next_dchecklist;
	std::vector<tree_ptr> next_echecklist;
	static const auto opts = options::get();
	static const float m = 1.0 / opts.problem_size;

	static thread_local std::vector<multi_src> dmulti_srcs;
	static thread_local std::vector<source> emulti_srcs;
	dmulti_srcs.resize(0);
	emulti_srcs.resize(0);
	next_dchecklist.reserve(NCHILD * dchecklist.size());
	next_echecklist.reserve(NCHILD * echecklist.size());

	for (auto c : dchecklist) {
		auto other = c->get_multipole();
		const auto dx = opts.ewald ? ewald_near_separation(multi.x - other.x) : abs(multi.x - other.x);
		if (dx > (multi.r + other.r) * theta_inv) {
			dmulti_srcs.push_back( { other.m, other.x });
		} else {
			if (c->is_leaf()) {
				const auto pos = c->get_positions();
				for (auto x : pos) {
					dsources.push_back(x);
				}
			} else {
				auto next = c->get_children();
				next_dchecklist.push_back(next[0]);
				next_dchecklist.push_back(next[1]);
			}
		}
	}

	if (opts.ewald) {
		for (auto c : echecklist) {
			auto other = c->get_monopole();
			const auto dx = ewald_far_separation(multi.x - other.x);
			if (dx > (multi.r + other.r) * theta_inv) {
				emulti_srcs.push_back( { other.m, other.x });
			} else {
				if (c->is_leaf()) {
					const auto pos = c->get_positions();
					for (auto x : pos) {
						esources.push_back( { m, x });
					}
				} else {
					auto next = c->get_children();
					next_echecklist.push_back(next[0]);
					next_echecklist.push_back(next[1]);
				}
			}
		}
	}
	flop += gravity_indirect_multipole(L, multi.x, dmulti_srcs);
	if (opts.ewald) {
		flop += gravity_indirect_ewald(L, multi.x, emulti_srcs);
	}
	dchecklist = std::move(next_dchecklist);
	echecklist = std::move(next_echecklist);
	if (!is_leaf()) {
		auto rc_l_fut = thread_if_avail(
				[=]() {
					return children[0]->kick_fmm(std::move(dchecklist), std::move(dsources), std::move(echecklist), std::move(esources),
							L << (child_com[0] - multi.x), min_rung, do_out);
				});
		const auto rc_r = children[1]->kick_fmm(std::move(dchecklist), std::move(dsources), std::move(echecklist), std::move(esources),
				L << (child_com[1] - multi.x), min_rung, do_out);
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
				if (c->is_leaf()) {
					const auto pos = c->get_positions();
					for (auto x : pos) {
						dsources.push_back(x);
					}
				} else {
					auto next = c->get_children();
					next_dchecklist.push_back(next[0]);
					next_dchecklist.push_back(next[1]);
				}
			}
			dchecklist = std::move(next_dchecklist);
		}
		while (!echecklist.empty()) {
			for (auto c : echecklist) {
				if (c->is_leaf()) {
					const auto pos = c->get_positions();
					for (auto x : pos) {
						esources.push_back( { m, x });
					}
				} else {
					auto next = c->get_children();
					next_echecklist.push_back(next[0]);
					next_echecklist.push_back(next[1]);
				}
			}
			echecklist = std::move(next_echecklist);
		}
		std::vector<vect<float>> x;
		for (auto i = part_begin; i != part_end; i++) {
			if (i->rung >= min_rung || i->rung == null_rung || do_out) {
				x.push_back(pos_to_double(i->x));
			}
		}
		std::vector<force> f(x.size(), { 0, vect<double>(0) });
		int j = 0;
		for (auto i = part_begin; i != part_end; i++) {
			if (i->rung >= min_rung || i->rung == null_rung || do_out) {
				force this_f = L.translate_L2((pos_to_double(i->x) - multi.x));
				f[j].phi += this_f.phi;
				f[j].g = f[j].g + this_f.g;
				j++;
			}
		}

		flop += gravity_direct(f, x, dsources);
		if (opts.ewald) {
			flop += gravity_ewald(f, x, esources);
		}
//		printf( "%i %i\n", dsources.size(), esources.size());
		rc = do_kick(f, min_rung, do_out);
	}
	return rc;
}

kick_return tree::kick_bh(std::vector<tree_ptr> dchecklist, std::vector<vect<float>> dsources, std::vector<multi_src> multi_srcs,
		std::vector<tree_ptr> echecklist, std::vector<source> esources, rung_type min_rung, bool do_out) {

	kick_return rc;
	if (!has_active && !do_out) {
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
				for (auto x : pos) {
					dsources.push_back(x);
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
			auto other = c->get_monopole();
			const auto dx = ewald_far_separation(multi.x - other.x);
			if (dx > (multi.r + other.r) * theta_inv) {
				esources.push_back( { other.m, other.x });
			} else {
				if (c->is_leaf()) {
					const auto pos = c->get_positions();
					for (auto x : pos) {
						esources.push_back( { m, x });
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
							min_rung, do_out);
				});
		const auto rc_r = children[1]->kick_bh(std::move(dchecklist), std::move(dsources), std::move(multi_srcs), std::move(echecklist), std::move(esources),
				min_rung, do_out);
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
			rc = kick_bh(std::move(dchecklist), std::move(dsources), std::move(multi_srcs), std::move(echecklist), std::move(esources), min_rung, do_out);
		} else {
			std::vector<vect<float>> x;
			for (auto i = part_begin; i != part_end; i++) {
				if (i->rung >= min_rung || i->rung == null_rung || do_out) {
					x.push_back(pos_to_double(i->x));
				}
			}
			std::vector<force> f(x.size(), { 0, vect<double>(0) });
			flop += gravity_direct_multipole(f, x, multi_srcs);
			flop += gravity_direct(f, x, dsources);
			if (opts.ewald) {
				flop += gravity_ewald(f, x, esources);
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
	if (!has_active && !do_out) {
		rc.rung = 0;
		return rc;
	}

	if (!is_leaf()) {
		auto rc_l_fut = thread_if_avail([&]() {
			return children[0]->kick_direct(sources, min_rung, do_out);
		});
		const auto rc_r = children[1]->kick_direct(sources, min_rung, do_out);
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
			if (i->rung >= min_rung || i->rung == null_rung || do_out) {
				x.push_back(pos_to_double(i->x));
			}
		}
		std::vector<force> f(x.size(), { 0, vect<double>(0) });
		flop += gravity_direct(f, x, sources);
		if (opts.ewald) {
			std::vector<source> esources;
			for (auto s : sources) {
				esources.push_back( { m, s });
			}
			flop += gravity_ewald(f, x, esources);
		}
		rc = do_kick(f, min_rung, do_out);
	}
	return rc;
}

kick_return tree::do_kick(const std::vector<force> &f, rung_type min_rung, bool do_out) {
	static const auto opts = options::get();
	static const double eps = 10.0 * std::numeric_limits<double>::min();
	static const double m = 1.0 / opts.problem_size;
	kick_return rc;
	rc.rung = 0;
	int j = 0;
	if (do_out) {
		rc.stats.zero();
	}
	for (auto i = part_begin; i != part_end; i++) {
		if (i->rung >= min_rung || i->rung == null_rung || do_out) {
			if (i->rung >= min_rung || i->rung == null_rung) {
				if (i->rung != -1) {
					const double dt = rung_to_dt(i->rung);
					i->v = i->v + f[j].g * (0.5 * dt);
				}
				const double a = abs(f[j].g);
				double dt = std::min(opts.dt_max, opts.eta * std::sqrt(opts.soft_len / (a + eps)));
				rung_type rung = dt_to_rung(dt);
				rung = std::max(rung, min_rung);
				rc.rung = std::max(rc.rung, rung);
				dt = rung_to_dt(rung);
				i->rung = rung;
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

void tree::drift(double dt) {
	for (auto i = part_begin; i != part_end; i++) {
		const vect<double> dx = i->v * dt;
		const vect<pos_type> dxi = double_to_pos(dx);
		i->x = i->x + dxi;
	}
}

std::uint64_t tree::get_flop() {
	return flop;
}

void tree::reset_flop() {
	flop = 0;
}

bool tree::active_particles(int rung, bool do_out) {
	bool rc;
	if (is_leaf()) {
		rc = false;
		for (auto i = part_begin; i != part_end; i++) {
			if (i->rung >= rung || i->rung == null_rung || (do_out && i->flags.out)) {
				rc = true;
				break;
			}
		}
	} else {
		const auto rc1 = children[0]->active_particles(rung, do_out);
		const auto rc2 = children[1]->active_particles(rung, do_out);
		rc = rc1 || rc2;
	}
	has_active = rc;
	return rc;
}
