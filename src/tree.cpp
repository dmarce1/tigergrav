#include <tigergrav/options.hpp>
#include <tigergrav/time.hpp>
#include <tigergrav/tree.hpp>

#include <hpx/runtime/actions/plain_action.hpp>

#include <atomic>
#include <algorithm>

HPX_REGISTER_COMPONENT(hpx::components::component<tree>, tree);

std::atomic<std::uint64_t> tree::flop(0);
float tree::theta_inv;

static std::atomic<int> num_threads(1);
static bool inc_thread();
static void dec_thread();

bool inc_thread() {
	static const int nmax = 4 * hpx::threads::hardware_concurrency();
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

void tree::set_theta(float t) {
	theta_inv = 1.0 / t;
}

tree_client tree::new_(range r, part_iter b, part_iter e, int level) {
	const static auto localities = hpx::find_all_localities();
	return tree_client(hpx::new_ < tree > (localities[part_vect_locality_id(b)], r, b, e, level).get());
}

tree::tree(range box, part_iter b, part_iter e, int level_) {
	level = level_;
	const auto &opts = options::get();
	part_begin = b;
	part_end = e;
	if (e - b > opts.parts_per_node) {
		float max_span = 0.0;
		const range prange = part_vect_range(b, e);
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
	const auto &opts = options::get();
	const auto m = 1.0 / opts.problem_size;
	multi.m = 0.0;
	range prange;
	if (is_leaf()) {
		multi.x = vect<float>(0.0);
		const auto parts = part_vect_read(part_begin, part_end);
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
	static const auto loc_id = hpx::get_locality_id();
	raw_id_type id;
	id.loc_id = loc_id;
	id.ptr = reinterpret_cast<std::uint64_t>(this);
	return id;
}

bool tree::is_leaf() const {
	static const auto opts = options::get();
	return (part_end - part_begin) <= opts.parts_per_node;
}

kick_return tree::kick_fmm(std::vector<check_item> dchecklist, std::vector<check_item> echecklist, const vect<ireal> &Lcom, expansion<float> L,
		rung_type min_rung, bool do_out) {

	kick_return rc;
	if (!multi.has_active && !do_out) {
		rc.rung = 0;
		return rc;
	}

	L = L << (multi.x - Lcom);

	std::vector<check_item> next_dchecklist;
	std::vector<check_item> next_echecklist;
	static const auto opts = options::get();
	static const float m = 1.0 / opts.problem_size;

	std::vector<multi_src> dmulti_srcs;
	std::vector<vect<float>> dsources;
	std::vector<multi_src> emulti_srcs;
	std::vector<vect<float>> esources;
	next_dchecklist.reserve(NCHILD * dchecklist.size());
	next_echecklist.reserve(NCHILD * echecklist.size());

	for (auto c : dchecklist) {
		const auto other = c.node.get_node_attributes();
		const auto dx = opts.ewald ? ewald_near_separation(multi.x - other.multi.x) : abs(multi.x - other.multi.x);
		const bool far = dx > (multi.r + other.multi.r) * theta_inv;
		if (far) {
			if (c.opened) {
				const auto parts = part_vect_read(other.pbegin, other.pend);
				for (auto i = parts.begin(); i != parts.end(); i++) {
					dsources.push_back(pos_to_double(i->x));
				}
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
		for (auto c : echecklist) {
			const auto other = c.node.get_node_attributes();
			const auto dx = ewald_far_separation(multi.x - other.multi.x, multi.r + other.multi.r);
			const bool far = dx > (multi.r + other.multi.r) * theta_inv;
			if (far) {
				if (c.opened) {
					const auto parts = part_vect_read(other.pbegin, other.pend);
					for (auto i = parts.begin(); i != parts.end(); i++) {
						esources.push_back(pos_to_double(i->x));
					}
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
		while (!dchecklist.empty()) {
			for (auto c : dchecklist) {
				const auto other = c.node.get_node_attributes();
				const auto dx = opts.ewald ? ewald_near_separation(multi.x - other.multi.x) : abs(multi.x - other.multi.x);
				const bool far = dx > (multi.r + other.multi.r) * theta_inv;
				if (c.opened) {
					const auto parts = part_vect_read(other.pbegin, other.pend);
					for (auto i = parts.begin(); i != parts.end(); i++) {
						dsources.push_back(pos_to_double(i->x));
					}
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
				for (auto c : echecklist) {
					const auto other = c.node.get_node_attributes();
					const auto dx = ewald_far_separation(multi.x - other.multi.x, multi.r + other.multi.r);
					const bool far = dx > (multi.r + other.multi.r) * theta_inv;
					if (c.opened) {
						const auto parts = part_vect_read(other.pbegin, other.pend);
						for (auto i = parts.begin(); i != parts.end(); i++) {
							esources.push_back(pos_to_double(i->x));
						}
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
		static thread_local std::vector<vect<float>> x;
		static thread_local std::vector<force> f;
		x.resize(0);
		const auto parts = part_vect_read(part_begin, part_end);
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
	auto parts = part_vect_read(part_begin, part_end);
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
	if (is_leaf()) {
		auto parts = part_vect_read(part_begin, part_end);
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

node_attr get_node_attributes_(raw_id_type id);

HPX_PLAIN_ACTION (get_node_attributes_, get_node_attributes_action);

node_attr get_node_attributes_(raw_id_type id) {
	static const auto here = hpx::get_locality_id();
	static const auto localities = hpx::find_all_localities();
	if (here == id.loc_id) {
		return reinterpret_cast<tree*>(id.ptr)->get_node_attributes();
	} else {
		return get_node_attributes_action()(localities[id.loc_id], id);
	}
}

node_attr raw_tree_client::get_node_attributes() const {
	return get_node_attributes_(ptr);
}

