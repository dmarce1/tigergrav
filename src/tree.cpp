#include <tigergrav/gravity.hpp>
#include <tigergrav/options.hpp>
#include <tigergrav/time.hpp>
#include <tigergrav/tree.hpp>

#include <hpx/include/async.hpp>
#include <hpx/include/threads.hpp>

#define EWALD_CRIT 0.25

std::atomic<std::uint64_t> tree::flop(0);
int tree::num_threads = 1;
mutex_type tree::thread_mtx;
mutex_type tree::out_mtx;

bool tree::inc_thread() {
	std::lock_guard<mutex_type> lock(thread_mtx);
	static const int nmax = 2 * hpx::threads::hardware_concurrency();
	if (num_threads < nmax) {
		num_threads++;
		return true;
	} else {
		return false;
	}
}

void tree::dec_thread() {
	std::lock_guard<mutex_type> lock(thread_mtx);
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

kick_return tree::kick(std::vector<tree_ptr> dchecklist, std::vector<source> dsources, std::vector<tree_ptr> echecklist, std::vector<source> esources,
		rung_type min_rung, bool do_stats, bool do_out) {

	kick_return rc;
	if (!has_active && !do_stats) {
		rc.rung = 0;
		return rc;
	}

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
	if (!is_leaf()) {
		hpx::future<kick_return> rc_l_fut;
		if (inc_thread()) {
			rc_l_fut = hpx::async(
					[=]() {
						const auto rc = children[0]->kick(std::move(dchecklist), std::move(dsources), std::move(echecklist), std::move(esources), min_rung,
								do_stats, do_out);
						dec_thread();
						return rc;
					});
		} else {
			rc_l_fut = hpx::make_ready_future(children[0]->kick(dchecklist, dsources, echecklist, esources, min_rung, do_stats, do_out));
		}
		const auto rc_r = children[1]->kick(std::move(dchecklist), std::move(dsources), std::move(echecklist), std::move(esources), min_rung, do_stats, do_out);
		const auto rc_l = rc_l_fut.get();
		rc.rung = std::max(rc_r.rung, rc_l.rung);
		if (do_stats) {
			for (int dim = 0; dim < NDIM; dim++) {
				rc.stats.g[dim] = rc_r.stats.g[dim] + rc_l.stats.g[dim];
				rc.stats.p[dim] = rc_r.stats.p[dim] + rc_l.stats.p[dim];
			}
			rc.stats.pot = rc_r.stats.pot + rc_l.stats.pot;
			rc.stats.kin = rc_r.stats.kin + rc_l.stats.kin;
		}
	} else {
		if (do_stats) {
			for (int dim = 0; dim < NDIM; dim++) {
				rc.stats.g[dim] = 0.0;
				rc.stats.p[dim] = 0.0;
			}
			rc.stats.pot = 0.0;
			rc.stats.kin = 0.0;
		}
		if (!dchecklist.empty() || !echecklist.empty()) {
			rc = kick(std::move(dchecklist), std::move(dsources), std::move(echecklist), std::move(esources), min_rung, do_stats, do_out);
		} else {
			std::vector<vect<float>> x;
			for (auto i = part_begin; i != part_end; i++) {
				if (i->rung >= min_rung || i->rung == null_rung || do_stats || (i->flags.out && do_out)) {
					x.push_back(pos_to_double(i->x));
				}
			}
			std::vector<force> f(x.size());
			const bool do_phi = do_stats || do_out;
			flop += gravity_direct(f, x, dsources, do_phi);
			if (opts.ewald) {
				flop += gravity_ewald(f, x, esources, do_phi);
			}
			int j = 0;
			rc.rung = 0;
			for (auto i = part_begin; i != part_end; i++) {
				if (i->rung >= min_rung || i->rung == null_rung || do_stats || (i->flags.out && do_out)) {
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
					if (do_stats) {
						rc.stats.g = rc.stats.g + f[j].g * m;
						rc.stats.p = rc.stats.p + i->v * m;
						rc.stats.pot += 0.5 * m * f[j].phi;
						rc.stats.kin += 0.5 * m * i->v.dot(i->v);
					}
					if (do_out && i->flags.out) {
						output out;
						out.x = pos_to_double(i->x);
						out.v = i->v;
						out.g = f[j].g;
						out.phi = f[j].phi;
						out.rung = i->rung;
						std::lock_guard<mutex_type> lock(out_mtx);
						FILE *fp = fopen("output.bin", "ab");
						fwrite(&out, sizeof(output), 1, fp);
						fclose(fp);
					}
					j++;
				}
			}
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

//void tree::output(float t, int num) const {
//	std::string filename = std::string("parts.") + std::to_string(num) + std::string(".silo");
//	DBfile *db = DBCreateReal(filename.c_str(), DB_CLOBBER, DB_LOCAL, "Meshless", DB_HDF5);
//	auto optlist = DBMakeOptlist(1);
//	DBAddOption(optlist, DBOPT_TIME, &t);
//
//	const int nnodes = std::distance(part_begin, part_end);
//
//	{
//		double *coords[NDIM];
//		for (int dim = 0; dim < NDIM; dim++) {
//			coords[dim] = new double[nnodes];
//			int j = 0;
//			for (auto i = part_begin; i != part_end; i++, j++) {
//				coords[dim][j] = pos_to_double(i->x[dim]);
//			}
//		}
//		DBPutPointmesh(db, "points", NDIM, coords, nnodes, DB_DOUBLE, optlist);
//		for (int dim = 0; dim < NDIM; dim++) {
//			delete[] coords[dim];
//		}
//	}
//
//	{
//		std::array<std::vector<float>, NDIM> v;
//		for (int dim = 0; dim < NDIM; dim++) {
//			v[dim].reserve(nnodes);
//		}
//		for (auto i = part_begin; i != part_end; i++) {
//			for (int dim = 0; dim < NDIM; dim++) {
//				v[dim].push_back(i->v[dim]);
//			}
//		}
//		for (int dim = 0; dim < NDIM; dim++) {
//			std::string nm = std::string() + "v_" + char('x' + char(dim));
//			DBPutPointvar1(db, nm.c_str(), "points", v[dim].data(), nnodes, DB_FLOAT, optlist);
//		}
//	}
//#ifdef STORE_G
//	{
//		std::vector<float> phi;
//		phi.reserve(nnodes);
//		for (auto i = part_begin; i != part_end; i++) {
//			phi.push_back(i->phi);
//		}
//		DBPutPointvar1(db, "phi", "points", phi.data(), nnodes, DB_FLOAT, optlist);
//	}
//#endif
//
//#ifdef STORE_G
//	{
//		std::array<std::vector<float>, NDIM> g;
//		for (int dim = 0; dim < NDIM; dim++) {
//			g[dim].reserve(nnodes);
//		}
//		for (auto i = part_begin; i != part_end; i++) {
//			for (int dim = 0; dim < NDIM; dim++) {
//				g[dim].push_back(i->g[dim]);
//			}
//		}
//		for (int dim = 0; dim < NDIM; dim++) {
//			std::string nm = std::string() + "g_" + char('x' + char(dim));
//			DBPutPointvar1(db, nm.c_str(), "points", g[dim].data(), nnodes, DB_FLOAT, optlist);
//		}
//	}
//#endif
//
//	{
//		std::vector<int> rung;
//		rung.reserve(nnodes);
//		for (auto i = part_begin; i != part_end; i++) {
//			rung.push_back(i->rung);
//		}
//		DBPutPointvar1(db, "rung", "points", rung.data(), nnodes, DB_INT, optlist);
//	}
//
//	DBClose(db);
//	DBFreeOptlist(optlist);
//
//}
