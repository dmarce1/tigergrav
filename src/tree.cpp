#include <tigergrav/gravity.hpp>
#include <tigergrav/options.hpp>
#include <tigergrav/time.hpp>
#include <tigergrav/tree.hpp>

#include <hpx/include/async.hpp>
#include <hpx/include/threads.hpp>

#include <silo.h>

#define EWALD_CRIT 0.25

std::atomic<std::uint64_t> tree::flop(0);
int tree::num_threads = 1;
mutex_type tree::mtx;

bool tree::inc_thread() {
	std::lock_guard<mutex_type> lock(mtx);
	static const int nmax = 2 * hpx::threads::hardware_concurrency();
	if (num_threads < nmax) {
		num_threads++;
		return true;
	} else {
		return false;
	}
}

void tree::dec_thread() {
	std::lock_guard<mutex_type> lock(mtx);
	num_threads--;
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
	if (e - b > opts.parts_per_node || (opts.ewald && max_span >= opts.theta * EWALD_CRIT)) {
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

monopole tree::compute_monopoles() {
	const auto &opts = options::get();
	const auto m = 1.0 / opts.problem_size;
	if (leaf) {
		mono.m = 0.0;
		mono.x = vect<float>(0.0);
		for (auto i = part_begin; i != part_end; i++) {
			mono.m += m;
			mono.x += pos_to_double(i->x) * m;
		}
		if (mono.m != 0.0) {
			mono.x = mono.x / mono.m;
		}
		mono.r = 0.0;
		for (auto i = part_begin; i != part_end; i++) {
			mono.r = std::max(mono.r, (float) abs(pos_to_double(i->x) - mono.x));
		}
	} else {
		monopole ml, mr;
		ml = children[0]->compute_monopoles();
		mr = children[1]->compute_monopoles();
		mono.m = ml.m + mr.m;
		if (mono.m != 0.0) {
			mono.x = (ml.x * ml.m + mr.x * mr.m) / mono.m;
		} else {
			mono.x = vect<float>(0.0);
		}
		mono.r = std::max(abs(ml.x - mono.x) + ml.r, abs(mr.x - mono.x) + mr.r);
	}
	return mono;
}

monopole tree::get_monopole() const {
	return mono;
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

#ifdef GLOBAL_DT
float tree::compute_gravity(std::vector<tree_ptr> dchecklist, std::vector<source> dsources, std::vector<tree_ptr> echecklist, std::vector<source> esources) {
#else
rung_type tree::kick(std::vector<tree_ptr> dchecklist, std::vector<source> dsources, std::vector<tree_ptr> echecklist, std::vector<source> esources,
		rung_type min_rung) {
#endif

#ifndef GLOBAL_DT
	if (!has_active) {
		return 0;
	}
#endif
	std::vector<tree_ptr> next_dchecklist;
	std::vector<tree_ptr> next_echecklist;
	static const auto opts = options::get();
	static const float m = 1.0 / opts.problem_size;
	static const float eps = 10.0 * std::numeric_limits<float>::min();
	std::function<float(const vect<float>)> separation;
	if (opts.ewald) {
		separation = ewald_near_separation;
	} else {
		separation = [](const vect<float> x) {
			return abs(x);
		};
	}
	for (auto c : dchecklist) {
		auto other = c->get_monopole();
		const auto dx = separation(mono.x - other.x);
		if (dx > (mono.r + other.r) / opts.theta) {
			dsources.push_back( { other.m, other.x });
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
			const auto dx = ewald_far_separation(mono.x - other.x);
			if (dx > (mono.r + other.r) / opts.theta) {
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
#ifdef GLOBAL_DT
	float min_dt;
	if (!is_leaf()) {
		hpx::future<float> dt_l_fut;
		if (inc_thread()) {
			dt_l_fut = hpx::async([=]() {
				const auto rc = children[0]->compute_gravity(std::move(dchecklist), std::move(dsources), std::move(echecklist), std::move(esources));
				dec_thread();
				return rc;
			});
		} else {
			dt_l_fut = hpx::make_ready_future(children[0]->compute_gravity(dchecklist, dsources, echecklist, esources));
		}
		const auto dt_r = children[1]->compute_gravity(std::move(dchecklist), std::move(dsources), std::move(echecklist), std::move(esources));
		min_dt = std::min(dt_l_fut.get(), dt_r);
	} else {
		if (!dchecklist.empty() || !echecklist.empty()) {
			min_dt = compute_gravity(std::move(dchecklist), std::move(dsources), std::move(echecklist), std::move(esources));
		} else {
			std::vector<vect<float>> x;
			for (auto i = part_begin; i != part_end; i++) {
				x.push_back(pos_to_double(i->x));
			}
			std::vector<force> f(x.size());
			flop += gravity_direct(f, x, dsources);
			if (opts.ewald) {
				flop += gravity_ewald(f, x, esources);
			}
			int j = 0;
			min_dt = std::numeric_limits<float>::max();
			for (auto i = part_begin; i != part_end; i++) {
				const float a = abs(f[j].g);
				float dt = std::min(opts.dt_max, opts.eta * std::sqrt(opts.soft_len / (a + eps)));
				min_dt = std::min(min_dt, dt);
#ifdef STORE_G
				i->phi = f[j].phi;
#endif
				i->g = f[j].g;
				j++;
			}
		}
	}
	return min_dt;
#else
	rung_type max_rung;
	if (!is_leaf()) {
		hpx::future<rung_type> rung_l_fut;
		if (inc_thread()) {
			rung_l_fut = hpx::async([=]() {
				const auto rc = children[0]->kick(std::move(dchecklist), std::move(dsources), std::move(echecklist), std::move(esources), min_rung);
				dec_thread();
				return rc;
			});
		} else {
			rung_l_fut = hpx::make_ready_future(children[0]->kick(dchecklist, dsources, echecklist, esources, min_rung));
		}
		const auto rung_r = children[1]->kick(std::move(dchecklist), std::move(dsources), std::move(echecklist), std::move(esources), min_rung);
		max_rung = std::max(rung_l_fut.get(), rung_r);
	} else {
		if (!dchecklist.empty() || !echecklist.empty()) {
			max_rung = kick(std::move(dchecklist), std::move(dsources), std::move(echecklist), std::move(esources), min_rung);
		} else {
			std::vector<vect<float>> x;
			for (auto i = part_begin; i != part_end; i++) {
				if (i->rung >= min_rung || i->rung == null_rung) {
					x.push_back(pos_to_double(i->x));
				}
			}
			std::vector<force> f(x.size());
			flop += gravity_direct(f, x, dsources);
			if (opts.ewald) {
				flop += gravity_ewald(f, x, esources);
			}
			int j = 0;
			max_rung = 0;
			for (auto i = part_begin; i != part_end; i++) {
				if (i->rung >= min_rung || i->rung == null_rung) {
					if (i->rung != -1) {
						const float dt = rung_to_dt(i->rung);
						i->v = i->v + f[j].g * (0.5 * dt);
					}
					const float a = abs(f[j].g);
					float dt = std::min(opts.dt_max, opts.eta * std::sqrt(opts.soft_len / (a + eps)));
					rung_type rung = dt_to_rung(dt);
					rung = std::max(rung, min_rung);
					max_rung = std::max(max_rung, rung);
					dt = rung_to_dt(rung);
					i->rung = rung;
					i->v = i->v + f[j].g * (0.5 * dt);
#ifdef STORE_G
					i->phi = f[j].phi;
					i->g = f[j].g;
#endif
					j++;
				}
			}
		}
	}
	return max_rung;
#endif

}

void tree::drift(float dt) {
	for (auto i = part_begin; i != part_end; i++) {
		const vect<double> dx = i->v * dt;
		const vect<std::uint64_t> dxi = double_to_pos(dx);
		i->x = i->x + dxi;
	}
}

#ifdef GLOBAL_DT
void tree::kick(float dt) {
	for (auto i = part_begin; i != part_end; i++) {
		i->v = i->v + i->g * dt;
	}
}
#endif

stats tree::statistics() const {
	static const auto &opts = options::get();
	static const auto m = 1.0 / opts.problem_size;
	static const auto h = opts.soft_len;
	stats s;
	s.kin_tot = 0.0;
	s.mom_tot = vect<double>(0.0);
	for (auto i = part_begin; i != part_end; i++) {
		const auto &v = i->v;
		s.kin_tot += 0.5 * m * v.dot(v);
		s.mom_tot += v * m;
	}
#ifdef STORE_G
	s.pot_tot = 0.0;
	s.acc_tot = vect<double>(0.0);
	for (auto i = part_begin; i != part_end; i++) {
		const auto &g = i->g;
		s.pot_tot += 0.5 * m * i->phi;
		s.acc_tot += g * m;
	}
	s.ene_tot = s.pot_tot + s.kin_tot;
	const auto a = 2.0 * s.kin_tot;
	const auto b = s.pot_tot;
	s.virial_err = (a + b) / (std::abs(a) + std::abs(b));
	s.flop = flop;
#endif
	return s;
}

#ifndef GLOBAL_DT
bool tree::active_particles(int rung) {
	bool rc;
	if (is_leaf()) {
		rc = false;
		for (auto i = part_begin; i != part_end; i++) {
			if (i->rung >= rung || i->rung == null_rung) {
				rc = true;
				break;
			}
		}
	} else {
		const auto rc1 = children[0]->active_particles(rung);
		const auto rc2 = children[1]->active_particles(rung);
		rc = rc1 || rc2;
	}
	has_active = rc;
	return rc;
}
#endif

void tree::output(float t, int num) const {
	std::string filename = std::string("parts.") + std::to_string(num) + std::string(".silo");
	DBfile *db = DBCreateReal(filename.c_str(), DB_CLOBBER, DB_LOCAL, "Meshless", DB_HDF5);
	auto optlist = DBMakeOptlist(1);
	DBAddOption(optlist, DBOPT_TIME, &t);

	const int nnodes = std::distance(part_begin, part_end);

	{
		double *coords[NDIM];
		for (int dim = 0; dim < NDIM; dim++) {
			coords[dim] = new double[nnodes];
			int j = 0;
			for (auto i = part_begin; i != part_end; i++, j++) {
				coords[dim][j] = pos_to_double(i->x[dim]);
			}
		}
		DBPutPointmesh(db, "points", NDIM, coords, nnodes, DB_DOUBLE, optlist);
		for (int dim = 0; dim < NDIM; dim++) {
			delete[] coords[dim];
		}
	}

	{
		std::array<std::vector<float>, NDIM> v;
		for (int dim = 0; dim < NDIM; dim++) {
			v[dim].reserve(nnodes);
		}
		for (auto i = part_begin; i != part_end; i++) {
			for (int dim = 0; dim < NDIM; dim++) {
				v[dim].push_back(i->v[dim]);
			}
		}
		for (int dim = 0; dim < NDIM; dim++) {
			std::string nm = std::string() + "v_" + char('x' + char(dim));
			DBPutPointvar1(db, nm.c_str(), "points", v[dim].data(), nnodes, DB_FLOAT, optlist);
		}
	}
#ifdef STORE_G
	{
		std::vector<float> phi;
		phi.reserve(nnodes);
		for (auto i = part_begin; i != part_end; i++) {
			phi.push_back(i->phi);
		}
		DBPutPointvar1(db, "phi", "points", phi.data(), nnodes, DB_FLOAT, optlist);
	}
#endif

#ifdef GLOBAL_DT
#define OUTPUT_ACCEL
#endif
#ifdef STORE_G
#define OUTPUT_ACCEL
#endif

#ifdef OUTPUT_ACCEL
	{
		std::array<std::vector<float>, NDIM> g;
		for (int dim = 0; dim < NDIM; dim++) {
			g[dim].reserve(nnodes);
		}
		for (auto i = part_begin; i != part_end; i++) {
			for (int dim = 0; dim < NDIM; dim++) {
				g[dim].push_back(i->g[dim]);
			}
		}
		for (int dim = 0; dim < NDIM; dim++) {
			std::string nm = std::string() + "g_" + char('x' + char(dim));
			DBPutPointvar1(db, nm.c_str(), "points", g[dim].data(), nnodes, DB_FLOAT, optlist);
		}
	}
#endif

#ifndef GLOBAL_DT
	{
		std::vector<int> rung;
		rung.reserve(nnodes);
		for (auto i = part_begin; i != part_end; i++) {
			rung.push_back(i->rung);
		}
		DBPutPointvar1(db, "rung", "points", rung.data(), nnodes, DB_INT, optlist);
	}
#endif

	DBClose(db);
	DBFreeOptlist(optlist);

}
