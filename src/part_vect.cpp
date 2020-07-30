#include <tigergrav/part_vect.hpp>
#include <tigergrav/initialize.hpp>

#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/serialization.hpp>

static std::vector<particle> parts;

static int part_begin;
static int part_end;

HPX_PLAIN_ACTION (part_vect_init);
HPX_PLAIN_ACTION (part_vect_read);
HPX_PLAIN_ACTION (part_vect_write);

void part_vect_init() {
	static const auto opts = options::get();
	static const auto localities = hpx::find_all_localities();
	static const auto N = localities.size();
	static const auto n = hpx::get_locality_id();
	static const auto M = opts.problem_size;
	std::vector<hpx::future<void>> futs;
	if (n == 0) {
		for (int i = 1; i < N; i++) {
			futs.push_back(hpx::async < part_vect_init_action > (localities[i]));
		}
	}
	part_begin = n * M / N;
	part_end = (n + 1) * M / N;
	parts = initial_particle_set(opts.problem, part_end - part_begin, opts.out_parts);
	hpx::wait_all(futs);
}

std::vector<particle> part_vect_read(part_iter b, part_iter e) {
	static const auto myid = hpx::get_locality_id();
	static const auto localities = hpx::find_all_localities();
	const auto id = part_vect_locality_id(b);
	std::vector<particle> these_parts;
	if (id == myid) {
		these_parts.reserve(e - b);
		const auto this_e = std::min(e, part_end);
		for (int i = b; i < this_e; i++) {
			these_parts.push_back(parts[i]);
		}
		if (these_parts.size() != e - b) {
			auto next_parts = part_vect_read_action()(localities[myid + 1], b + these_parts.size(), e);
			for (const auto &p : next_parts) {
				these_parts.push_back(p);
			}
		}
	} else {
		these_parts = part_vect_read_action()(localities[id], b, e);
	}
	return these_parts;
}

void part_vect_write(part_iter b, part_iter e, std::vector<particle> these_parts) {
	static const auto myid = hpx::get_locality_id();
	static const auto localities = hpx::find_all_localities();
	const auto id = part_vect_locality_id(b);
	if (id == myid) {
		int i = b;
		for (auto this_part : these_parts) {
			parts[i] = this_part;
			i++;
			if (i == part_end) {
				break;
			}
		}
		std::vector<particle> next_parts;
		while (i < e) {
			next_parts.push_back(these_parts[i - b]);
			i++;
		}
		if (next_parts.size()) {
			part_vect_write_action()(localities[myid + 1], e - next_parts.size(), e, std::move(next_parts));
		}
	} else {
		part_vect_write_action()(localities[id], b, e, std::move(these_parts));
	}
}

//template<class I, class F>
//I bisect(I begin, I end, F &&below) {
//	auto lo = begin;
//	auto hi = end - 1;
//	while (lo < hi) {
//		if (!below(*lo)) {
//			while (lo != hi) {
//				if (below(*hi)) {
//					auto tmp = *lo;
//					*lo = *hi;
//					*hi = tmp;
//					break;
//				}
//				hi--;
//			}
//
//		}
//		lo++;
//	}
//	return hi;
//}

part_iter part_vect_sort_lo(part_iter, part_iter, double xmid, int dim);
std::pair<particle, part_iter> part_vect_sort_hi(part_iter, part_iter, double xmid, int dim, particle lo_part);

HPX_PLAIN_ACTION (part_vect_sort_lo);
HPX_PLAIN_ACTION (part_vect_sort_hi);

part_iter part_vect_sort_lo(part_iter lo, part_iter hi, double xmid, int dim) {
	static const auto localities = hpx::find_all_localities();
	static const auto N = localities.size();
	static const auto n = hpx::get_locality_id();
	static const auto M = options::get().problem_size;
	const auto this_hi = std::min((int) ((n + 1) * M / N - 1), (int) hi);
	const auto hi0 = this_hi;
	while (lo < std::min(this_hi, hi)) {
		if (pos_to_double(parts[lo].x[dim]) >= xmid) {
			const auto hiid = part_vect_locality_id(hi);
			auto tmp = part_vect_sort_hi_action()(localities[hiid], lo, hi, xmid, dim, parts[lo]);
			parts[lo] = tmp.first;
			hi = tmp.second;
		}
		lo++;
	}
	if (hi0 != this_hi && lo != hi) {
		return part_vect_sort_lo_action()(localities[n + 1], this_hi, hi, xmid, dim);
	} else {
		return hi;
	}

}

std::pair<particle, part_iter> part_vect_sort_hi(part_iter lo, part_iter hi, double xmid, int dim, particle lo_part) {
	static const auto localities = hpx::find_all_localities();
	static const auto N = localities.size();
	static const auto n = hpx::get_locality_id();
	static const auto M = options::get().problem_size;
	const auto this_lo = std::max((int) (n * M / N), lo);
	bool found = false;
	std::pair<particle, part_iter> rc;
	while (this_lo != hi) {
		if (pos_to_double(parts[hi].x[dim]) < xmid) {
			found = true;
			break;
		}
		hi--;
	}
	if (!found && this_lo != lo) {
		rc = part_vect_sort_hi_action()(localities[n - 1], lo, this_lo - 1, xmid, dim, lo_part);
	} else {
		rc.first = parts[hi];
		rc.second = hi;
		parts[hi] = lo_part;
	}
	return rc;
}

part_iter part_vect_sort(part_iter b, part_iter e, double xmid, int dim) {
	if (e == b) {
		return e;
	} else {
		auto rc = part_vect_sort_lo(b, e - 1, xmid, dim);
		return rc;
	}
}

range part_vect_range(part_iter b, part_iter e) {
	range r;
	for (int dim = 0; dim < NDIM; dim++) {
		r.max[dim] = 0.0;
		r.min[dim] = 1.0;
	}
	for (int i = b; i < e; i++) {
		const auto &p = parts[i];
		for (int dim = 0; dim < NDIM; dim++) {
			const auto x = pos_to_double(p.x[dim]);
			r.max[dim] = std::max(r.max[dim], x);
			r.min[dim] = std::min(r.min[dim], x);
		}
	}
	return r;
}

int part_vect_locality_id(part_iter i) {
	static const auto N = hpx::find_all_localities().size();
	static const auto M = options::get().problem_size;
	int n = i * N / M;
	while (i < n * M / N) {
		n--;
	}
	while (i >= (n + 1) * M / N) {
		n++;
	}
	return n;
}
