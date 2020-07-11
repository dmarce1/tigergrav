#include <tigergrav/options.hpp>
#include <tigergrav/time.hpp>
#include <tigergrav/tree.hpp>

#include <hpx/include/async.hpp>
#include <hpx/include/threads.hpp>

std::atomic<std::uint64_t> tree::flop(0);
float tree::theta_inv;

static int num_threads = 1;
static mutex_type thread_mtx;
static bool inc_thread();
static void dec_thread();

bool inc_thread() {
	std::lock_guard<mutex_type> lock(thread_mtx);
	static const int nmax = 2 * hpx::threads::hardware_concurrency();
	if (num_threads < nmax) {
		num_threads++;
		return true;
	} else {
		return false;
	}
}

void dec_thread() {
	std::lock_guard<mutex_type> lock(thread_mtx);
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

void tree::set_theta(float t) {
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
	if (e - b > opts.parts_per_node) {
		leaf = false;
		float max_span = 0.0;
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
		float mid = (box.max[max_dim] + box.min[max_dim]) * 0.5;
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
		multi.x = vect<float>(0.0);
		for (auto i = part_begin; i != part_end; i++) {
			multi.m() += m;
			multi.x += pos_to_double(i->x) * m;
		}
		if (multi.m() != 0.0) {
			multi.x = multi.x / multi.m();
		}
		for (auto i = part_begin; i != part_end; i++) {
			for (int j = 0; j < NDIM; j++) {
				for (int k = 0; k <= j; k++) {
					const auto Xj = pos_to_double(i->x[j]);
					const auto Xk = pos_to_double(i->x[k]);
					multi.m(j, k) += m * (Xj - multi.x[j]) * (Xk - multi.x[k]);
				}
			}
		}
		multi.r = 0.0;
		for (auto i = part_begin; i != part_end; i++) {
			multi.r = std::max(multi.r, (float) abs(pos_to_double(i->x) - multi.x));
		}
	} else {
		multipole_info ml, mr;
		ml = children[0]->compute_multipoles();
		mr = children[1]->compute_multipoles();
		multi.m() = ml.m() + mr.m();
		if (multi.m() != 0.0) {
			multi.x = (ml.x * ml.m() + mr.x * mr.m()) / multi.m();
		} else {
			multi.x = vect<float>(0.0);
		}
		multi.m = (ml.m >> (ml.x - multi.x)) + (mr.m >> (mr.x - multi.x));
		multi.r = std::max(abs(ml.x - multi.x) + ml.r, abs(mr.x - multi.x) + mr.r);
		child_com[0] = ml.x;
		child_com[1] = mr.x;
	}
	return multi;
}

multipole_info tree::get_multipole() const {
	return multi;
}

monopole tree::get_monopole() const {
	return {multi.m(),multi.x,multi.r};
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
		x = pos_to_double(i->x);
		pos.push_back(x);
	}
	return pos;
}


kick_return kick_fmm(std::vector<tree_ptr> dchecklist, std::vector<source> dsources, expansion<float> L, rung_type min_rung, bool do_output) {




}


kick_return tree::kick_bh(std::vector<tree_ptr> dchecklist, std::vector<source> dsources, std::vector<multi_src> multi_srcs, std::vector<tree_ptr> echecklist,
		std::vector<source> esources, rung_type min_rung, bool do_out) {

	kick_return rc;
	if (!has_active && !do_out) {
		rc.rung = 0;
		return rc;
	}

	std::vector<tree_ptr> next_dchecklist;
	std::vector<tree_ptr> next_echecklist;
	static const auto opts = options::get();
	static const float m = 1.0 / opts.problem_size;
	std::function<float(const vect<float>)> separation;
	if (opts.ewald) {
		separation = ewald_near_separation;
	} else {
		separation = [](const vect<float> x) {
			return abs(x);
		};
	}
	for (auto c : dchecklist) {
		auto other = c->get_multipole();
		const auto dx = separation(multi.x - other.x);
		if (dx > (multi.r + other.r) * theta_inv) {
			multi_srcs.push_back( { other.m, other.x });
		} else {
			if (c->is_leaf()) {
				const auto pos = c->get_positions();
				for (auto x : pos) {
					dsources.push_back( { m, x });
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
			std::vector<force> f(x.size(), { 0, vect<float>(0) });
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

kick_return tree::kick_direct(std::vector<source> &sources, rung_type min_rung, bool do_out) {

	static const auto opts = options::get();
	static const float m = 1.0 / opts.problem_size;
	if (sources.size() == 0) {
		for (auto i = part_begin; i != part_end; i++) {
			sources.push_back( { m, pos_to_double(i->x) });
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
		std::vector<force> f(x.size(), { 0, vect<float>(0) });
		flop += gravity_direct(f, x, sources);
		if (opts.ewald) {
			flop += gravity_ewald(f, x, sources);
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
		if (i->rung >= min_rung || i->rung == null_rung || do_out) {
			if (i->rung >= min_rung || i->rung == null_rung) {
				if (i->rung != -1) {
					const float dt = rung_to_dt(i->rung);
					i->v = i->v + f[j].g * (0.5 * dt);
				}
				const float a = abs(f[j].g);
				float dt = std::min(opts.dt_max, opts.eta * std::sqrt(opts.soft_len / (a + eps)));
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

void tree::drift(float dt) {
	for (auto i = part_begin; i != part_end; i++) {
		const vect<double> dx = i->v * dt;
		const vect<std::uint64_t> dxi = double_to_pos(dx);
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
