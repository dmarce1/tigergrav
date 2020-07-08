#include <tigergrav/gravity.hpp>
#include <tigergrav/options.hpp>
#include <tigergrav/time.hpp>
#include <tigergrav/tree.hpp>

#include <hpx/include/async.hpp>
#include <hpx/include/threads.hpp>

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
	for (int dim = 0; dim < NDIM; dim++) {
		xc[dim] = (box.min[dim] + box.max[dim]) * float(0.5);
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

multipole_attr tree::compute_multipoles() {
	const auto &opts = options::get();
	const auto m = 1.0 / opts.problem_size;
	if (leaf) {
		multi.x = vect<float>(0.0);
		const float mtot = m * (part_end - part_begin);
		for (auto i = part_begin; i != part_end; i++) {
			multi.x += pos_to_double(i->x) * m;
		}
		if (mtot != 0.0) {
			multi.x = multi.x / mtot;
		} else {
			multi.x = xc;
		}
		multi.m.zero();
		for (auto i = part_begin; i != part_end; i++) {
			multi.m = multi.m + multipole<double>(m, pos_to_double(i->x) - multi.x);
		}

		multi.r = 0.0;
		for (auto i = part_begin; i != part_end; i++) {
			multi.r = std::max(multi.r, (double) abs(pos_to_double(i->x) - multi.x));
		}
	} else {
		multipole_attr ml, mr;
		ml = children[0]->compute_multipoles();
		mr = children[1]->compute_multipoles();
		child_com[0] = ml.x;
		child_com[1] = mr.x;

		const float mtot = ml.m() + mr.m();
		if (mtot != 0.0) {
			multi.x = (ml.x * ml.m() + mr.x * mr.m()) / mtot;
		} else {
			multi.x = xc;
		}
		multi.m = ml.m.translate(ml.x - multi.x) + mr.m.translate(mr.x - multi.x);
		multi.r = std::max(abs(ml.x - multi.x) + ml.r, abs(mr.x - multi.x) + mr.r);
	}
	return multi;
}

multipole_attr tree::get_multipole() const {
	return multi;
}

bool tree::is_leaf() const {
	return leaf;
}

std::array<tree_ptr, NCHILD> tree::get_children() const {
	return children;
}

std::vector<vect<pos_type>> tree::get_positions() const {
	std::vector<vect<pos_type>> pos;
	pos.reserve(part_end - part_begin);
	for (auto i = part_begin; i != part_end; i++) {
		pos.push_back(i->x);
	}
	return pos;
}

bool tree::parts_separated_from(const vect<float> &x, float r) {
	static const auto opts = options::get();
	static const float theta_inv = 1.0 / opts.theta;
	bool sep = true;
	for (auto i = part_begin; i != part_end; i++) {
		const auto dxv = x - pos_to_double(i->x);
		const auto dx = opts.ewald ? ewald_near_separation(dxv) : abs(dxv);
		if (dx <= r * theta_inv) {
			sep = false;
			break;
		}
	}
	return sep;
}

kick_return tree::kick(std::vector<check_item> dchecklist, expansion<double> L, std::vector<tree_ptr> echecklist, std::vector<mono_source> esources,
		rung_type min_rung, bool do_stats, bool do_out) {

	static thread_local std::size_t checklist_size = 0;

	kick_return rc;
	if (!has_active && !do_stats) {
		rc.rung = 0;
		return rc;
	}

	static const auto opts = options::get();
	static const float m = 1.0 / opts.problem_size;
	static const float eps = 10.0 * std::numeric_limits<float>::min();
	static const float theta_inv = 1.0 / opts.theta;
	std::function<float(const vect<float>)> separation;
	static thread_local std::vector<vect<pos_type>> mono_srcs;
	static thread_local std::vector<multi_source> multi_srcs;
	mono_srcs.resize(0);
	multi_srcs.resize(0);

	if (opts.ewald) {
		do {
			std::vector<tree_ptr> next_echecklist;
			for (auto c : echecklist) {
				auto other = c->get_multipole();
				const auto dx = ewald_far_separation(multi.x - other.x);
				if (dx > (multi.r + other.r) / opts.theta) {
					esources.push_back( { (float) other.m(), other.x });
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
			echecklist = std::move(next_echecklist);
		} while (is_leaf() && !echecklist.empty());
	}

	std::vector<check_item> next_dchecklist;
	next_dchecklist.reserve(checklist_size);
	for (auto c : dchecklist) {
		auto other = c.ptr->get_multipole();
		const auto dx = opts.ewald ? ewald_near_separation(multi.x - other.x) : abs(multi.x - other.x);
		if (dx > (multi.r + other.r) * theta_inv && !c.use_parts) {
			multi_srcs.push_back( { other.m, other.x });
		} else {
			if (c.ptr->is_leaf()) {
				if (c.use_parts) {
					if (c.ptr->parts_separated_from(multi.x, multi.r)) {
						const auto pos = c.ptr->get_positions();
						for (auto x : pos) {
							mono_srcs.push_back(x);
						}
					} else {
						next_dchecklist.push_back(c);
					}
				} else {
					c.use_parts = true;
					next_dchecklist.push_back(c);
				}
			} else {
				auto next = c.ptr->get_children();
				next_dchecklist.push_back( { next[0], false });
				next_dchecklist.push_back( { next[1], false });
			}
		}
	}
	checklist_size = std::max(checklist_size, next_dchecklist.size());
	dchecklist = std::move(next_dchecklist);
	flop += gravity_multi_multi(L, multi.x, multi_srcs);
	flop += gravity_multi_mono(L, multi.x, mono_srcs);

	mono_srcs.resize(0);
	multi_srcs.resize(0);

	if (!is_leaf()) {

		const expansion<double> Ll = L.translate(child_com[0] - multi.x);
		const expansion<double> Lr = L.translate(child_com[1] - multi.x);

		dchecklist.reserve(checklist_size);
		echecklist.reserve(checklist_size);

		hpx::future<kick_return> rc_l_fut;
		if (inc_thread()) {
			rc_l_fut = hpx::async([=]() {
				const auto rc = children[0]->kick(std::move(dchecklist), Ll, std::move(echecklist), std::move(esources), min_rung, do_stats, do_out);
				dec_thread();
				return rc;
			});
		} else {
			rc_l_fut = hpx::make_ready_future(children[0]->kick(dchecklist, Ll, echecklist, esources, min_rung, do_stats, do_out));
		}
		const auto rc_r = children[1]->kick(std::move(dchecklist), Lr, std::move(echecklist), std::move(esources), min_rung, do_stats, do_out);
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

		do {
			for (auto c : dchecklist) {
				auto other = c.ptr->get_multipole();
				if (c.ptr->is_leaf()) {
					const auto pos = c.ptr->get_positions();
					for (auto x : pos) {
						mono_srcs.push_back(x);
					}
				} else {
					if (parts_separated_from(other.x, other.r)) {
						multi_srcs.push_back( { other.m, other.x });
					} else {
						auto next = c.ptr->get_children();
						next_dchecklist.push_back( { next[0], false });
						next_dchecklist.push_back( { next[1], false });
					}
				}
			}
			checklist_size = std::max(checklist_size, next_dchecklist.size());
			dchecklist = std::move(next_dchecklist);
		} while (!dchecklist.empty());

		if (do_stats) {
			for (int dim = 0; dim < NDIM; dim++) {
				rc.stats.g[dim] = 0.0;
				rc.stats.p[dim] = 0.0;
			}
			rc.stats.pot = 0.0;
			rc.stats.kin = 0.0;
		}
		std::vector<vect<pos_type>> x;
		std::vector<force> f;
		for (auto i = part_begin; i != part_end; i++) {
			if (i->rung >= min_rung || i->rung == null_rung || do_stats || (i->flags.out && do_out)) {
				x.push_back(i->x);
				f.push_back(L.to_force(pos_to_double(i->x) - multi.x));
			}
		}
		flop += gravity_mono_mono(f, x, mono_srcs, do_stats || do_out);
		flop += gravity_mono_multi(f, x, multi_srcs, do_stats || do_out);
		if (opts.ewald) {
			flop += gravity_ewald(f, x, esources, do_stats || do_out);
		}
		printf( "%li\n", esources.size());
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
