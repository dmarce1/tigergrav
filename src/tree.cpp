#include <tigergrav/options.hpp>
#include <tigergrav/tree.hpp>

#include <silo.h>

tree_ptr tree::new_(range r, part_iter b, part_iter e) {
	return std::make_shared<tree>(r, b, e);
}

tree::tree(range box, part_iter b, part_iter e) {
	const auto &opts = options::get();
	part_begin = b;
	part_end = e;
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
		const auto mid_iter = bisect(b, e, [max_dim, mid](const particle &p) {
			return pos_to_double(p.x[max_dim]) < mid;
		});
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

std::vector<force> gravity(const std::vector<vect<float>> &x, const std::vector<source> &y) {
	const auto h = options::get().soft_len;
	const auto h3inv = 1.0 / (h * h * h);
	const auto h2 = h * h;
	std::vector<force> f(x.size());
	for (int i = 0; i < x.size(); i++) {
		auto &g = f[i].g;
		auto &phi = f[i].phi;
		phi = 0.0;
		g = vect<float>(0.0);
		for (int j = 0; j < y.size(); j++) {
			const vect<float> dx = x[i] - y[j].x;
			const float m = y[j].m;
			const float r = abs(dx);
			const float rinv = 1.0 / r;
			const float rinv3 = rinv * rinv * rinv;
			if (r > h) {
				g -= dx * (m * rinv3);
				phi -= m * rinv;
			} else if (r > 0.0) {
				const auto mh3inv = m * h3inv;
				g -= dx * mh3inv;
				phi -= (1.5 * h2 - 0.5 * r * r) * mh3inv;
			}
		}
	}
	return f;
}

#ifdef GLOBAL_DT
float tree::compute_gravity(std::vector<tree_ptr> checklist, std::vector<source> sources) {
#else
std::int8_t tree::kick(std::vector<tree_ptr> checklist, std::vector<source> sources, std::int8_t min_rung) {
#endif
	std::vector<tree_ptr> next_checklist;
	static const auto opts = options::get();
	static const float m = 1.0 / opts.problem_size;
	static const float eps = 10.0 * std::numeric_limits<float>::min();
	for (auto c : checklist) {
		const auto other = c->get_monopole();
		const auto dx = abs(mono.x - other.x);
//		printf( "%e %e\n", dx,  (mono.r + other.r) / opts.theta);
		if (dx > (mono.r + other.r) / opts.theta) {
			sources.push_back( { other.m, other.x });
		} else {
			if (c->is_leaf()) {
				const auto pos = c->get_positions();
				for (auto x : pos) {
					sources.push_back( { m, x });
				}
			} else {
				auto next = c->get_children();
				next_checklist.push_back(next[0]);
				next_checklist.push_back(next[1]);
			}
		}
	}
	checklist = std::move(next_checklist);
#ifdef GLOBAL_DT
	float min_dt;
	if (!is_leaf()) {
		const auto dt_l = children[0]->compute_gravity(checklist, sources);
		const auto dt_r = children[1]->compute_gravity(std::move(checklist), std::move(sources));
		min_dt = std::min(dt_l, dt_r);
	} else {
		if (!checklist.empty()) {
			min_dt = compute_gravity(std::move(checklist), std::move(sources));
		} else {
			std::vector<vect<float>> x;
			for (auto i = part_begin; i != part_end; i++) {
				x.push_back(pos_to_double(i->x));
			}
			const auto f = gravity(x, sources);
			int j = 0;
			min_dt = std::numeric_limits<float>::max();
			for (auto i = part_begin; i != part_end; i++, j++) {
				const float a = abs(f[j].g);
				float dt = std::min(opts.dt_max, opts.eta * std::sqrt(opts.soft_len / (a + eps)));
				min_dt = std::min(min_dt, dt);
#ifdef STORE_G
				i->phi = f[j].phi;
#endif
				i->g = f[j].g;
			}
		}
	}
	return min_dt;
#else
	std::int8_t max_rung;
	if (!is_leaf()) {
		const auto rung_l = children[0]->kick(checklist, sources, min_rung);
		const auto rung_r = children[1]->kick(std::move(checklist), std::move(sources), min_rung);
		max_rung = std::max(rung_l, rung_r);
	} else {
		if (!checklist.empty()) {
			max_rung = kick(std::move(checklist), std::move(sources), min_rung);
		} else {
			std::vector<vect<float>> x;
			for (auto i = part_begin; i != part_end; i++) {
				if (i->rung >= min_rung || i->rung == -1) {
					x.push_back(pos_to_double(i->x));
				}
			}
			const auto f = gravity(x, sources);
			int j = 0;
			max_rung = 0;
			for (auto i = part_begin; i != part_end; i++, j++) {
				if (i->rung >= min_rung || i->rung == -1) {
					if (i->rung != -1) {
						const float dt = rung_to_dt(i->rung);
						i->v = i->v + f[j].g * (0.5 * dt);
					}
					const float a = abs(f[j].g);
					float dt = std::min(opts.dt_max, opts.eta * std::sqrt(opts.soft_len / (a + eps)));
					std::int8_t rung = dt_to_rung(dt);
					rung = std::max(rung, min_rung);
					max_rung = std::max(max_rung, rung);
					dt = rung_to_dt(rung);
					i->rung = rung;
					i->v = i->v + f[j].g * (0.5 * dt);
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
	const auto &opts = options::get();
	const auto m = 1.0 / opts.problem_size;
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
#endif
	return s;
}

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
