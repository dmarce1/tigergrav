#include <tigergrav/part_vect.hpp>
#include <tigergrav/initialize.hpp>

#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/serialization.hpp>
#include <hpx/synchronization/spinlock.hpp>

#include <unordered_map>

using mutex_type = hpx::lcos::local::spinlock;

static std::vector<particle> particles;
static part_iter part_begin;
static part_iter part_end;
static std::vector<hpx::id_type> localities;
static int myid;

static std::unordered_map<part_iter, hpx::shared_future<std::vector<vect<pos_type>>>> pos_cache;
static mutex_type pos_cache_mtx;

part_iter part_vect_sort_lo(part_iter, part_iter, double xmid, int dim);
std::pair<particle, part_iter> part_vect_sort_hi(part_iter, part_iter, double xmid, int dim, particle lo_part);

HPX_PLAIN_ACTION (part_vect_init);
HPX_PLAIN_ACTION (part_vect_read);
HPX_PLAIN_ACTION (part_vect_read_position);
HPX_PLAIN_ACTION (part_vect_write);
HPX_PLAIN_ACTION (part_vect_range);
HPX_PLAIN_ACTION (part_vect_sort_lo);
HPX_PLAIN_ACTION (part_vect_sort_hi);
HPX_PLAIN_ACTION (part_vect_cache_reset);

void part_vect_cache_reset() {
	std::vector<hpx::future<void>> futs;
	if (myid == 0) {
		for (int i = 1; i < localities.size(); i++) {
			futs.push_back(hpx::async < part_vect_cache_reset_action > (localities[i]));
		}
	}
	pos_cache.clear();
	hpx::wait_all(futs);
}

inline particle& parts(part_iter i) {
	int j = i - part_begin;
//	if (j < 0 || j >= particles.size()) {
//		printf("Index out of bounds! %i should be between 0 and %i\n", j, particles.size());
//		abort();
//	}
	return particles[j];
}

void part_vect_init() {
	localities = hpx::find_all_localities();
	myid = hpx::get_locality_id();

	const auto opts = options::get();

	const std::uint64_t N = localities.size();
	const std::uint64_t n = myid;
	const std::uint64_t M = opts.problem_size;
	std::vector<hpx::future<void>> futs;
	if (n == 0) {
		for (int i = 1; i < N; i++) {
			futs.push_back(hpx::async < part_vect_init_action > (localities[i]));
		}
	}
	part_begin = n * M / N;
	part_end = (n + 1) * M / N;
	particles = initial_particle_set(opts.problem, part_end - part_begin, (n + 1) * opts.out_parts / N - n * opts.out_parts / N);
	hpx::wait_all(futs);
}

hpx::future<std::vector<particle>> part_vect_read(part_iter b, part_iter e) {
	const auto id = part_vect_locality_id(b);
	std::vector<particle> these_parts;
	if (id == myid) {
		these_parts.reserve(e - b);
		const auto this_e = std::min(e, part_end);
		for (int i = b; i < this_e; i++) {
			these_parts.push_back(parts(i));
		}
		if (these_parts.size() != e - b) {
			auto fut = part_vect_read_action()(localities[myid + 1], b + these_parts.size(), e);
			auto next_parts = fut.get();
			for (const auto &p : next_parts) {
				these_parts.push_back(p);
			}
			if (these_parts.size() != e - b) {
				printf("Error in part_vect_read\n");
				abort();
			}
		}
	} else {
		auto fut = part_vect_read_action()(localities[id], b, e);
		these_parts = fut.get();
	}
	return hpx::make_ready_future(these_parts);
}

inline hpx::future<std::vector<vect<pos_type>>> part_vect_read_pos_cache(part_iter b, part_iter e) {
	std::lock_guard<mutex_type> lock(pos_cache_mtx);
	auto iter = pos_cache.find(b);
	if (iter == pos_cache.end()) {
		auto fut = hpx::async < part_vect_read_position_action > (localities[part_vect_locality_id(b)], b, e);
		pos_cache[b] = fut.then([b](decltype(fut) f) {
			return f.get().get();
		});
	}
	return hpx::async(hpx::launch::deferred, [b]() {
		std::unique_lock < mutex_type > lock(pos_cache_mtx);
		auto future = pos_cache[b];
		lock.unlock();
		return future.get();
	});
}

hpx::future<std::vector<vect<pos_type>>> part_vect_read_position(part_iter b, part_iter e) {
	if (b >= part_begin && b < part_end) {
		if (e <= part_end) {
			return hpx::async(hpx::launch::deferred, [=]() {
				std::vector<vect<pos_type>> these_parts;
				these_parts.reserve(e - b);
				for (int i = b; i < e; i++) {
					these_parts.push_back(parts(i).x);
				}
				return these_parts;
			});
		} else {
			auto fut = part_vect_read_pos_cache(part_end, e);
			return hpx::async(hpx::launch::deferred, [=](decltype(fut) f) {
				std::vector<vect<pos_type>> these_parts;
				these_parts.reserve(e - b);
				for (int i = b; i < part_end; i++) {
					these_parts.push_back(parts(i).x);
				}
				auto next_parts = f.get();
				for (const auto &p : next_parts) {
					these_parts.push_back(p);
				}
				return these_parts;
			}, std::move(fut));

		}
	} else {
		return part_vect_read_pos_cache(b, e);
	}
}

void part_vect_write(part_iter b, part_iter e, std::vector<particle> these_parts) {
	const auto id = part_vect_locality_id(b);
	if (id == myid) {
		int i = b;
		for (auto this_part : these_parts) {
			parts(i) = this_part;
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

inline std::pair<particle, part_iter> part_vect_sort_hi(part_iter lo, part_iter hi, double xmid, int dim, particle lo_part) {

	const std::uint64_t N = localities.size();
	const std::uint64_t n = myid;
	const std::uint64_t M = options::get().problem_size;
	bool complete = true;
	std::pair<particle, part_iter> rc;
	while (lo != hi) {
		if (pos_to_double(parts(hi).x[dim]) < xmid) {
			break;
		}
		hi--;
		if (hi == part_begin - 1) {
			complete = false;
			break;
		}
	}
	if (!complete) {
//		printf("hi_jump\n");
		rc = part_vect_sort_hi_action()(localities[n - 1], lo, part_begin - 1, xmid, dim, lo_part);
	} else {
		rc.first = parts(hi);
		rc.second = hi;
		parts(hi) = lo_part;
	}
	return rc;
}

part_iter part_vect_sort_lo(part_iter lo, part_iter hi, double xmid, int dim) {

	const std::uint64_t N = localities.size();
	const std::uint64_t n = myid;
	const std::uint64_t M = options::get().problem_size;
	bool complete = true;
	while (lo < hi) {
		if (pos_to_double(parts(lo).x[dim]) >= xmid) {
			const auto hiid = part_vect_locality_id(hi);
			std::pair<particle, part_iter> tmp;
			if (hiid == n) {
				tmp = part_vect_sort_hi(lo, hi, xmid, dim, parts(lo));
			} else {
				tmp = part_vect_sort_hi_action()(localities[hiid], lo, hi, xmid, dim, parts(lo));
			}
			parts(lo) = tmp.first;
			hi = tmp.second;
		}
		lo++;
		if (lo == part_end) {
			complete = false;
			break;
		}
	}
	if (!complete) {
		//	printf("lo_jump\n");
		return part_vect_sort_lo_action()(localities[n + 1], part_end, hi, xmid, dim);
	} else {
		return hi;
	}

}

part_iter part_vect_sort(part_iter b, part_iter e, double xmid, int dim) {

	if (e == b) {
		return e;
	} else {
		auto rc = part_vect_sort_lo_action()(localities[part_vect_locality_id(b)], b, e - 1, xmid, dim);
		return rc;
	}
}

range part_vect_range(part_iter b, part_iter e) {
	const auto id = part_vect_locality_id(b);
	range r;
	if (id == myid) {
		for (int dim = 0; dim < NDIM; dim++) {
			r.max[dim] = 0.0;
			r.min[dim] = 1.0;
		}
		const auto this_e = std::min(e, part_end);
		for (int i = b; i < this_e; i++) {
			const auto &p = parts(i);
			for (int dim = 0; dim < NDIM; dim++) {
				const auto x = pos_to_double(p.x[dim]);
				r.max[dim] = std::max(r.max[dim], x);
				r.min[dim] = std::min(r.min[dim], x);
			}
		}
		if (this_e != e) {
			const auto tmp = part_vect_range_action()(localities[myid + 1], this_e, e);
			for (int dim = 0; dim < NDIM; dim++) {
				r.max[dim] = std::max(r.max[dim], tmp.max[dim]);
				r.min[dim] = std::min(r.min[dim], tmp.max[dim]);
			}
		}
	} else {
		r = part_vect_range_action()(localities[id], b, e);
	}
	return r;
}

int part_vect_locality_id(part_iter i) {
	const std::uint64_t N = localities.size();
	const std::uint64_t M = options::get().problem_size;
	std::uint64_t n = i * N / M;
	while (i < n * M / N) {
		n--;
	}
	while (i >= (n + 1) * M / N) {
		n++;
	}
	return n;
}
