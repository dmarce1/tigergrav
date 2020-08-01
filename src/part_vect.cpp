#include <tigergrav/part_vect.hpp>
#include <tigergrav/initialize.hpp>

#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/serialization.hpp>
#include <hpx/synchronization/spinlock.hpp>

#include <unistd.h>
#include <unordered_map>

using mutex_type = hpx::lcos::local::spinlock;

static std::vector<particle> particles;
static part_iter part_begin;
static part_iter part_end;
static std::vector<hpx::id_type> localities;
static int myid;

#define POS_CACHE_SIZE 1024
static std::unordered_map<part_iter, hpx::shared_future<std::vector<vect<pos_type>>>> pos_cache[POS_CACHE_SIZE];
static mutex_type pos_cache_mtx[POS_CACHE_SIZE];

HPX_PLAIN_ACTION (part_vect_init);
HPX_PLAIN_ACTION (part_vect_read);
HPX_PLAIN_ACTION (part_vect_read_position);
HPX_PLAIN_ACTION (part_vect_write);
HPX_PLAIN_ACTION (part_vect_range);
HPX_PLAIN_ACTION (part_vect_cache_reset);

void part_vect_cache_reset() {
	std::vector<hpx::future<void>> futs;
	if (myid == 0) {
		for (int i = 1; i < localities.size(); i++) {
			futs.push_back(hpx::async < part_vect_cache_reset_action > (localities[i]));
		}
	}
	for (int i = 0; i < POS_CACHE_SIZE; i++) {
		pos_cache[i].clear();
	}
	hpx::wait_all(futs);
}

inline particle& parts(part_iter i) {
	const int j = i - part_begin;
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

	const part_iter N = localities.size();
	const part_iter n = myid;
	const part_iter M = opts.problem_size;
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
	const int index = (b / sizeof(particle)) % POS_CACHE_SIZE;
	std::lock_guard<mutex_type> lock(pos_cache_mtx[index]);
	auto iter = pos_cache[index].find(b);
	if (iter == pos_cache[index].end()) {
		auto fut = hpx::async < part_vect_read_position_action > (localities[part_vect_locality_id(b)], b, e);
		pos_cache[index][b] = fut.then([b](decltype(fut) f) {
			return f.get().get();
		});
	}
	return hpx::async(hpx::launch::deferred, [b, index]() {
		std::unique_lock < mutex_type > lock(pos_cache_mtx[index]);
		auto future = pos_cache[index][b];
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

part_iter part_vect_count_lo(part_iter, part_iter, double xmid, int dim);
void part_vect_sort_lo(part_iter, part_iter, part_iter, double xmid, int dim);
std::pair<std::vector<particle>, part_iter> part_vect_sort_hi(part_iter, double xmid, int dim, std::vector<particle> lo_parts);

HPX_PLAIN_ACTION (part_vect_sort_lo);
HPX_PLAIN_ACTION (part_vect_sort_hi);
HPX_PLAIN_ACTION (part_vect_count_lo);

part_iter part_vect_count_lo(part_iter b, part_iter e, double xmid, int dim) {
//	printf( "part_vect_count_lo\n");
	const part_iter N = localities.size();
	const part_iter M = options::get().problem_size;
	part_iter count = 0;
	std::vector<hpx::future<part_iter>> futs;
	if (part_end < e) {
		int n = part_vect_locality_id(part_end);
		for (int n = part_vect_locality_id(part_end); n * M / N < e; n++) {
			auto fut = hpx::async < part_vect_count_lo_action > (localities[n], n * M / N, std::min(e, (n + 1) * M / N), xmid, dim);
			futs.push_back(std::move(fut));

		}
	}
	auto this_begin = std::max(b, part_begin);
	auto this_end = std::min(e, part_end);
	constexpr part_iter chunk_size = 65536;
	for (part_iter i = this_begin; i < this_end; i += chunk_size) {
		auto func = [i, this_end, dim, xmid]() {
			const auto end = std::min(i + chunk_size, this_end);
			part_iter count = 0;
			for (part_iter j = i; j < end; j++) {
				if (pos_to_double(parts(j).x[dim]) < xmid) {
					count++;
				}
			}
			return count;
		};
		if (chunk_size >= this_end - this_begin) {
			count += func();
		} else {
			futs.push_back(hpx::async(func));
		}
	}
	for (auto &fut : futs) {
		count += fut.get();
	}
//	printf( "Counted %i in low partition on locality %i\n", count, myid);
	return count;
}

inline std::pair<std::vector<particle>, part_iter> part_vect_sort_hi(part_iter hi, double xmid, int dim, std::vector<particle> lo_parts) {
//	printf( "part_vect_sort_hi\n");
	part_iter current = 0;
	const part_iter n = myid;
	const part_iter N = localities.size();
	const part_iter M = options::get().problem_size;
	bool complete = true;
	std::pair<std::vector<particle>, part_iter> rc;
	while (hi != part_begin - 1 && current != lo_parts.size()) {
		if (pos_to_double(parts(hi).x[dim]) < xmid) {
			auto tmp = lo_parts[current];
			lo_parts[current++] = parts(hi);
			parts(hi) = tmp;
		}
		hi--;
	}
	if (current != lo_parts.size()) {
		std::vector<particle> new_lo_parts;
		new_lo_parts.reserve(lo_parts.size() - current);
		for (part_iter i = current; i < lo_parts.size(); i++) {
			new_lo_parts.push_back(lo_parts[i]);
		}
		auto fut = hpx::async < part_vect_sort_hi_action > (localities[n - 1], part_begin - 1, xmid, dim, std::move(new_lo_parts));
		rc = fut.get();
		for (part_iter i = current; i < lo_parts.size(); i++) {
			lo_parts[i] = rc.first[i - current];
		}
		rc.first = std::move(lo_parts);
	} else {
		rc.first = std::move(lo_parts);
		rc.second = hi;
	}
	return rc;
}

void part_vect_sort_lo(part_iter lo, part_iter hi, part_iter mid, double xmid, int dim) {
//	printf( "part_vect_sort_lo\n");
	const part_iter N = localities.size();
	const part_iter n = myid;
	const part_iter M = options::get().problem_size;
	bool complete = true;
	std::vector<particle> hi_parts;
	hi_parts.reserve(std::min(mid - lo, part_end - lo));
	const auto lo0 = lo;
	while (lo < mid && lo != part_end) {
		if (pos_to_double(parts(lo).x[dim]) >= xmid) {
			hi_parts.push_back(parts(lo));
		}
		lo++;
	}

	if (hi_parts.size()) {
		const auto hiid = part_vect_locality_id(hi);
		std::pair<std::vector<particle>, part_iter> rc;
		if (hiid != myid) {
			auto fut = hpx::async < part_vect_sort_hi_action > (localities[hiid], hi, xmid, dim, std::move(hi_parts));
			rc = fut.get();
		} else {
			rc = part_vect_sort_hi(hi, xmid, dim, std::move(hi_parts));
		}
		hi = rc.second;
		part_iter current = 0;
		for (lo = lo0; lo < mid && lo != part_end; lo++) {
			if (pos_to_double(parts(lo).x[dim]) >= xmid) {
				parts(lo) = rc.first[current++];
			}
		}
		if (current > rc.first.size()) {
			printf("Size error %i %i\n", current, rc.first.size());
			abort();
		}
	}
	if (lo != mid) {
		auto fut = hpx::async < part_vect_sort_lo_action > (localities[n + 1], part_end, hi, mid, xmid, dim);
		fut.get();
	}
}

part_iter part_vect_sort(part_iter b, part_iter e, double xmid, int dim) {

	if (e == b) {
		return e;
	} else {
		auto count = part_vect_count_lo(b, e, xmid, dim);
//		sleep(1);
		part_vect_sort_lo_action()(localities[part_vect_locality_id(b)], b, e - 1, b + count, xmid, dim);
		return b + count;
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
		hpx::future<range> fut;
		if (this_e != e) {
			fut = hpx::async < part_vect_range_action > (localities[myid + 1], this_e, e);
		}
		for (part_iter i = b; i < this_e; i++) {
			const auto &p = parts(i);
			for (int dim = 0; dim < NDIM; dim++) {
				const auto x = pos_to_double(p.x[dim]);
				r.max[dim] = std::max(r.max[dim], x);
				r.min[dim] = std::min(r.min[dim], x);
			}
		}
		if (this_e != e) {
			const auto tmp = fut.get();
			for (int dim = 0; dim < NDIM; dim++) {
				r.max[dim] = std::max(r.max[dim], tmp.max[dim]);
				r.min[dim] = std::min(r.min[dim], tmp.max[dim]);
			}
		}
	} else {
		auto fut = hpx::async < part_vect_range_action > (localities[id], b, e);
		r = fut.get();
	}
	return r;
}

int part_vect_locality_id(part_iter i) {
	const part_iter N = localities.size();
	const part_iter M = options::get().problem_size;
	part_iter n = i * N / M;
	while (i < n * M / N) {
		n--;
	}
	while (i >= (n + 1) * M / N) {
		n++;
	}
	return n;
}
