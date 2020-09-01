#include <tigergrav/part_vect.hpp>
#include <tigergrav/initialize.hpp>
#include <tigergrav/simd.hpp>
#include <tigergrav/load.hpp>
#include <tigergrav/cosmo.hpp>
#include <tigergrav/groups.hpp>
#include <tigergrav/map.hpp>

#ifdef HPX_LITE
#include <hpx/hpx_lite.hpp>
#else
#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/serialization.hpp>
#include <hpx/synchronization/spinlock.hpp>
#include <hpx/local_lcos/promise.hpp>
#endif

#include <unistd.h>
#include <thread>
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

static std::unordered_map<part_iter, hpx::shared_future<std::vector<particle_group_info>>> group_cache[POS_CACHE_SIZE];
static mutex_type group_cache_mtx[POS_CACHE_SIZE];

#define GROUP_MTX_SIZE ((std::uint64_t) 1024)
static mutex_type group_mtx[GROUP_MTX_SIZE];

kick_return kick_rc;
mutex_type kick_rc_mtx;

HPX_PLAIN_ACTION (part_vect_init);
HPX_PLAIN_ACTION (part_vect_read_position);
HPX_PLAIN_ACTION (part_vect_read_group);
HPX_PLAIN_ACTION (part_vect_range);
HPX_PLAIN_ACTION (part_vect_reset);
HPX_PLAIN_ACTION (part_vect_center_of_mass);
HPX_PLAIN_ACTION (part_vect_multipole_info);
HPX_PLAIN_ACTION (part_vect_drift);
HPX_PLAIN_ACTION (part_vect_read_active_positions);
HPX_PLAIN_ACTION (part_vect_kick);
HPX_PLAIN_ACTION (part_vect_init_groups);
HPX_PLAIN_ACTION (part_vect_find_groups);
HPX_PLAIN_ACTION (part_vect_kick_return);

inline particle& parts(part_iter i) {
	const int j = i - part_begin;
#ifndef NDEBUG
	if (j < 0 || j >= particles.size()) {
		printf("Index out of bounds! %i should be between 0 and %i\n", j, particles.size());
		abort();
	}
#endif
	return particles[j];
}

kick_return part_vect_kick_return() {
	std::vector<hpx::future<kick_return>> futs;
	if (myid == 0) {
		for (int i = 1; i < localities.size(); i++) {
			futs.push_back(hpx::async < part_vect_kick_return_action > (localities[i]));
		}
	}
	for (auto &f : futs) {
		auto rc = f.get();
		kick_rc.stats = kick_rc.stats + rc.stats;
		kick_rc.rung = std::max(kick_rc.rung, rc.rung);
		kick_rc.out.insert(kick_rc.out.end(), rc.out.begin(), rc.out.end());
	}
	return std::move(kick_rc);
}

bool part_vect_find_groups(part_iter b, part_iter e, std::vector<particle_group_info> others) {
	static const auto opts = options::get();
	static const std::uint64_t L = (std::pow(opts.problem_size, -1.0 / 3.0) * opts.link_len) * std::numeric_limits<std::uint32_t>::max();
	static const std::uint64_t L2 = L * L;
	const auto this_end = std::min(e, part_end);
	bool rc = false;
	hpx::future<bool> fut;
	if (this_end != e) {
		fut = hpx::async < part_vect_find_groups_action > (localities[myid + 1], this_end, e, others);
	}
	int mtx_index = b % GROUP_MTX_SIZE;
	std::lock_guard<mutex_type> lock(group_mtx[mtx_index]);
	for (auto i = b; i != this_end; i++) {
		for (const auto &other : others) {
			vect<pos_type> dx;
			vect<std::uint64_t> dxl;
			dx = parts(i).x - other.x;
			dxl = dx;
			const auto dx2 = dxl.dot(dxl);
			if (dx2 < L2 && dx2 != 0.0) {
				auto this_id = parts(i).flags.group;
				if (this_id == DEFAULT_GROUP) {
					this_id = i - part_begin;
					parts(i).flags.group = this_id;
					rc = true;
				}
				if (this_id != other.id) {
					const auto g = std::min(this_id, other.id);
					rc = rc || (parts(i).flags.group != g);
					parts(i).flags.group = g;
				}
			}
		}
	}
	if (this_end != e) {
		const bool other_rc = fut.get();
		rc = rc || other_rc;
	}
	return rc;
}

void part_vect_init_groups() {
	std::vector<hpx::future<void>> futs;
	if (myid == 0) {
		for (int i = 1; i < localities.size(); i++) {
			futs.push_back(hpx::async < part_vect_init_groups_action > (localities[i]));
		}
	}
	for (auto i = part_begin; i != part_end; i++) {
		parts(i).flags.group = DEFAULT_GROUP;
	}
	hpx::wait_all(futs.begin(), futs.end());
}

int round_robin(int me, int round, int N) {
	int nrounds = N % 2 == 0 ? N : N + 1;
	if (N % 2 == 0) {
		return (me + round % nrounds) % N;
	} else {
		int tmp = (me + round % nrounds) % (N + 1);
		if (tmp == N) {
			return -1;
		} else {
			return tmp;
		}
	}
}

void part_vect_sort_begin(part_iter b, part_iter e, part_iter mid, double xmid, int dim);
std::vector<particle> part_vect_sort_end(part_iter b, part_iter e, part_iter mid, double xmid, int dim, std::vector<particle>);

HPX_PLAIN_ACTION (part_vect_sort_begin);
HPX_PLAIN_ACTION (part_vect_sort_end);

mutex_type sort_mutex;

void part_vect_write_glass() {
	if (localities.size() > 1) {
		printf("Error: Cannot write glass file with multipole nodes!\n");
		abort();
	}
	const auto N = part_end - part_begin;
	const auto Nmax = std::numeric_limits<int>::max();
	if (N > Nmax) {
		printf("Unable to write glass file larger than %i", Nmax);
		abort();
	}
	int dummy;
	io_header_1 header;
	for (int i = 0; i < 6; i++) {
		header.npart[i] = header.npartTotal[i] = 0.0;
	}
	header.npartTotal[1] = header.npart[1] = N;
	header.BoxSize = 1.0;
	FILE *fp = fopen("glass.bin", "wb");
	if (fp == NULL) {
		printf("Unable to write glass.bin\n");
		abort();
	}
	dummy = sizeof(header);
	fwrite(&dummy, sizeof(dummy), 1, fp);
	fwrite(&header, sizeof(header), 1, fp);
	fwrite(&dummy, sizeof(dummy), 1, fp);
	dummy = N * NDIM * sizeof(double);
	fwrite(&dummy, sizeof(dummy), 1, fp);
	for (int i = 0; i < N; i++) {
		vect<double> x = pos_to_double(parts(i).x);
		for (int d = 0; d < NDIM; d++) {
			fwrite(&(x[d]), sizeof(double), 1, fp);
		}
	}
	fwrite(&dummy, sizeof(dummy), 1, fp);

	fclose(fp);

}

void part_vect_sort_begin(part_iter b, part_iter e, part_iter mid, double xmid, int dim) {
	static const auto opts = options::get();
	if (b == mid || e == mid) {
//		printf("%i %i %i\n", b, mid, e);
		return;
	}
	std::vector<hpx::future<void>> futs;
	if (part_vect_locality_id(b) == myid) {
		for (int n = myid + 1; n <= part_vect_locality_id(mid); n++) {
			futs.push_back(hpx::async < part_vect_sort_begin_action > (localities[n], b, e, mid, xmid, dim));
		}
	}
	int nproc = part_vect_locality_id(e - 1) - part_vect_locality_id(b) + 1;
	int low_proc = part_vect_locality_id(b);
	int chunk_size = opts.problem_size / localities.size() / 10 + 1;
	bool done = false;
	part_iter begin = std::max(b, part_begin);
	for (int round = 0; !done; round++) {
		int other = round_robin(myid - low_proc, round, nproc) + low_proc;
//		printf("%i %i %i %i %i %i\n", myid, b, mid, e, round % nproc, other);
		if (other >= 0 && other >= part_vect_locality_id(mid)) {
			std::vector<particle> send;
			send.reserve(chunk_size);
			done = true;
			std::unique_lock<mutex_type> lock(sort_mutex);
//			printf( "--%i %i %i\n",myid, begin,std::min(part_end, mid) );
			for (part_iter i = begin; i < std::min(part_end, mid); i++) {
				if (pos_to_double(parts(i).x[dim]) >= xmid) {
					send.push_back(parts(i));
					done = false;
					if (send.size() >= chunk_size) {
						break;
					}
				}
			}
			if (!done) {
				//			printf("%i Sending %i to %i\n", myid, send.size(), other);
				lock.unlock();
				auto recv = part_vect_sort_end_action()(localities[other], b, e, mid, xmid, dim, std::move(send));
				lock.lock();
				int j = 0;
				if (recv.size()) {
					for (part_iter i = begin; i < std::min(part_end, mid); i++) {
						begin = i;
						if (pos_to_double(parts(i).x[dim]) >= xmid) {
							parts(i) = recv[j++];
							if (recv.size() == j) {
								break;
							}
						}
					}
				}
			}
		}
	}
	hpx::wait_all(futs.begin(), futs.end());
}

std::vector<particle> part_vect_sort_end(part_iter b, part_iter e, part_iter mid, double xmid, int dim, std::vector<particle> low) {
	std::lock_guard<mutex_type> lock(sort_mutex);
	int j = 0;
	for (part_iter i = std::max(mid, part_begin); i < std::min(part_end, e); i++) {
		if (pos_to_double(parts(i).x[dim]) < xmid) {
			auto tmp = parts(i);
			parts(i) = low[j];
			low[j++] = tmp;
			if (j == low.size()) {
				break;
			}
		}
	}
	low.resize(j);
//	if(myid==1)
//	printf("%i used %i\n", myid, j);
	return low;
}

HPX_PLAIN_ACTION (part_vect_read);
HPX_PLAIN_ACTION (part_vect_write);

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

void part_vect_group_proc1(std::vector<particle> ps) {
	for (const auto &p : ps) {
		groups_add_particle1(p);
	}
}

HPX_PLAIN_ACTION (part_vect_group_proc1);

hpx::future<void> part_vect_kick(part_iter b, part_iter e, rung_type min_rung, bool do_out, std::vector<force> &&f) {
	kick_return rc;
	const auto opts = options::get();
	const double eps = 10.0 * std::numeric_limits<double>::min();
	const double scale = opts.cosmic ? cosmo_scale().first : 1.0;
	const double ainv = 1.0 / (scale);
	const double a3inv = 1.0 / (scale * scale * scale);
	const double m = opts.m_tot / opts.problem_size;
	const double sgn = opts.glass ? -1.0 : 1.0;
	std::unordered_map<int, std::vector<particle>> group_proc;
	rc.rung = 0;
	part_iter j = 0;
	rc.stats.zero();
	const double glass_drag = 1.0;
	for (auto i = b; i != std::min(e, part_end); i++) {
		rc.stats.p = rc.stats.p + parts(i).v / opts.problem_size / (scale * scale);
		rc.stats.kin += 0.5 * m * parts(i).v.dot(parts(i).v) / (scale * scale);
		if (parts(i).flags.rung >= min_rung || do_out) {
			if (parts(i).flags.rung >= min_rung) {
				if (parts(i).flags.rung != 0) {
					const double dt = rung_to_dt(parts(i).flags.rung);
					const auto dt1 = opts.cosmic ? cosmo_kick_dt1(parts(i).flags.rung) : 0.5 * dt;
					auto &v = parts(i).v;
					v = v + f[j].g * (dt1 * sgn * opts.G);
					if (opts.glass) {
						v = v / (1.0 + glass_drag * dt1);
					}
				}
				const double a = abs(f[j].g * opts.G) * a3inv;
				double dt = std::min(opts.dt_max, opts.eta * std::sqrt(opts.soft_len * (1.0 / 2.8) / (a + eps)));
				rung_type rung = dt_to_rung(dt);
				rung = std::max(rung, min_rung);
				rc.rung = std::max(rc.rung, rung);
				dt = rung_to_dt(rung);
				parts(i).flags.rung = std::max(std::max(rung, rung_type(parts(i).flags.rung - 1)), (rung_type) 1);
				const auto dt2 = opts.ewald ? cosmo_kick_dt2(parts(i).flags.rung) : 0.5 * dt;
				auto &v = parts(i).v;
				v = v + f[j].g * (dt2 * sgn * opts.G);
				if (opts.glass) {
					v = v / (1.0 + glass_drag * dt2);
				}
			}
			if (do_out) {
				rc.stats.g = rc.stats.g + f[j].g * opts.G / opts.problem_size / (scale * scale * scale);
				rc.stats.pot += 0.5 * m * f[j].phi * opts.G / scale;
			}
			if (do_out && parts(i).flags.out) {
				output out;
				out.x = pos_to_double(parts(i).x);
				out.v = parts(i).v;
				out.g = f[j].g;
				out.phi = f[j].phi;
				out.id = parts(i).flags.group;
				out.rung = parts(i).flags.rung;
				rc.out.push_back(out);
			}
			if (do_out && opts.groups) {
				particle p = parts(i);
				if (p.flags.group != DEFAULT_GROUP) {
					int proc = part_vect_locality_id(p.flags.group);
					if (proc == myid) {
						groups_add_particle1(p);
					} else {
						group_proc[proc].push_back(p);
					}
				}
			}
			j++;
		}
	}
	{
		std::lock_guard<mutex_type> lock(kick_rc_mtx);
		kick_rc.stats = kick_rc.stats + rc.stats;
		kick_rc.rung = std::max(kick_rc.rung, rc.rung);
		kick_rc.out.insert(kick_rc.out.end(), rc.out.begin(), rc.out.end());
	}
	const auto k = j;
	for (; j < f.size(); j++) {
		f[j - k] = f[j];
	}
	f.resize(f.size() - k);
	if (f.size()) {
		part_vect_kick_action()(localities[myid + 1], part_end, e, min_rung, do_out, std::move(f));
	}
	if (do_out && group_proc.size()) {
		return hpx::async([](std::unordered_map<int, std::vector<particle>> &&group_proc) {
			std::vector<hpx::future<void>> futs;
			for (auto &other : group_proc) {
				futs.push_back(hpx::async < part_vect_group_proc1_action > (localities[other.first], std::move(other.second)));
			}
			hpx::wait_all(futs.begin(), futs.end());
		}, std::move(group_proc));
	} else {
		return hpx::make_ready_future();
	}
}

void part_vect_group_proc2(std::vector<particle> ps) {
	for (const auto &p : ps) {
		groups_add_particle2(p);
	}
}

HPX_PLAIN_ACTION (part_vect_group_proc2);
HPX_PLAIN_ACTION (part_vect_find_groups2);

void part_vect_find_groups2() {
	std::vector<hpx::future<void>> futs;
	if (myid == 0) {
		for (int i = 1; i < localities.size(); i++) {
			futs.push_back(hpx::async < part_vect_find_groups2_action > (localities[i]));
		}
	}
	std::unordered_map<int, std::vector<particle>> group_proc;
	for (auto i = part_begin; i != part_end; i++) {
		particle p = parts(i);
		if (p.flags.group != DEFAULT_GROUP) {
			int proc = part_vect_locality_id(p.flags.group);
			if (proc == myid) {
				groups_add_particle2(p);
			} else {
				group_proc[proc].push_back(p);
			}
		}
	}
	for (auto &other : group_proc) {
		futs.push_back(hpx::async < part_vect_group_proc2_action > (localities[other.first], std::move(other.second)));
	}
	hpx::wait_all(futs.begin(), futs.end());
}

std::vector<vect<pos_type>> part_vect_read_active_positions(part_iter b, part_iter e, rung_type rung) {
	std::vector<vect<pos_type>> x;
	hpx::future<std::vector<vect<pos_type>>> fut;
	x.reserve(e - b);
	if (e > part_end) {
		fut = hpx::async < part_vect_read_active_positions_action > (localities[myid + 1], part_end, e, rung);
	}
	for (part_iter i = b; i < std::min(e, part_end); i++) {
		if (parts(i).flags.rung >= rung) {
			x.push_back(parts(i).x);
		}
	}
	if (e > part_end) {
		auto other = fut.get();
		for (const auto &o : other) {
			x.push_back(o);
		}
	}
	return x;
}

double part_vect_drift(double t, rung_type mrung) {
	static const auto opts = options::get();
	static const auto m = opts.m_tot / opts.problem_size;
	std::vector<hpx::future<double>> futs;
	if (myid == 0) {
		for (int i = 1; i < localities.size(); i++) {
			futs.push_back(hpx::async < part_vect_drift_action > (localities[i], t, mrung));
		}
	}
	double ekin = 0.0;
	const auto a1 = opts.cosmic ? cosmo_scale().second : 1.0;
	const auto a1inv2 = 1.0 / (a1 * a1);
	const auto dt = rung_to_dt(mrung);
	const auto drift_dt = opts.cosmic ? cosmo_drift_dt() : dt;
	const auto kick_dt = opts.cosmic ? cosmo_kick_dt2(mrung) : dt;
	const part_iter chunk_size = std::max(part_iter(1), (part_end - part_begin) / std::thread::hardware_concurrency());
	for (part_iter i = part_begin; i < part_end; i += chunk_size) {
		auto func = [i, chunk_size, t, kick_dt, drift_dt, a1inv2]() {
			double this_ekin = 0.0;
			const auto end = std::min(part_end, i + chunk_size);
			for (int j = i; j < end; j++) {
				const auto v = parts(j).v;
				this_ekin += 0.5 * m * v.dot(v) * a1inv2;
				const vect<double> dx = v * drift_dt;
				vect<double> x = pos_to_double(parts(j).x);
				if (opts.map) {
					map_add_particle(x, t, 2.0 * kick_dt);
				}
				x += dx;
				for (int dim = 0; dim < NDIM; dim++) {
					while (x[dim] >= 1.0) {
						x[dim] -= 1.0;
					}
					while (x[dim] < 0.0) {
						x[dim] += 1.0;
					}
				}
				parts(j).x = double_to_pos(x);
			}
			return this_ekin;
		};
		futs.push_back(hpx::async(std::move(func)));
	}
	for (auto &f : futs) {
		ekin += f.get();
	}
	return ekin;
}

std::pair<std::uint64_t, vect<double>> part_vect_center_of_mass(part_iter b, part_iter e) {
	static const auto m = options::get().m_tot / options::get().problem_size;
	std::pair<std::uint64_t, vect<double>> rc;
	hpx::future<std::pair<std::uint64_t, vect<double>>> fut;
	if (e > part_end) {
		fut = hpx::async < part_vect_center_of_mass_action > (localities[myid + 1], part_end, e);
	}
	const auto this_end = std::min(part_end, e);
	rc.first = 0;
	rc.second = vect<double>(0.0);
	for (part_iter i = b; i < this_end; i++) {
		rc.first++;
		rc.second += pos_to_double(parts(i).x);
	}
	if (e > part_end) {
		auto tmp = fut.get();
		rc.first += tmp.first;
		rc.second = rc.second + tmp.second * tmp.first;
	}
	if (rc.first > 0) {
		rc.second = rc.second / double(rc.first);
	}
	return rc;
}

multipole_info part_vect_multipole_info(vect<double> com, rung_type mrung, part_iter b, part_iter e) {
	static const auto opts = options::get();
	static const auto m = opts.m_tot / opts.problem_size;
	multipole_info rc;
	hpx::future<multipole_info> fut;
	if (e > part_end) {
		fut = hpx::async < part_vect_multipole_info_action > (localities[myid + 1], com, mrung, part_end, e);
	}
	const auto this_end = std::min(part_end, e);
	rc.m = 0.0;
	rc.x = double_to_pos(com);
	multipole<simd_float> M;
	vect<simd_double> Xcom;
	M = simd_float(0.0);
	for (int dim = 0; dim < NDIM; dim++) {
		Xcom[dim] = com[dim];
	}
	for (part_iter i = b; i < this_end; i += simd_float::size()) {
		vect<simd_double> X;
		simd_float mass;
		for (int k = 0; k < simd_float::size(); k++) {
			if (i + k < this_end) {
				for (int dim = 0; dim < NDIM; dim++) {
					X[dim][k] = pos_to_double(parts(i + k).x[dim]);		// 3 OP
				}
				mass[k] = m;
			} else {
				for (int dim = 0; dim < NDIM; dim++) {
					X[dim][k] = 0.0;
				}
				mass[k] = 0.0;
			}
		}
		vect<simd_float> dx;
		for (int dim = 0; dim < NDIM; dim++) {
			dx[dim] = simd_float(X[dim] - Xcom[dim]);
		}
		M() += mass;													// 1 OP
		for (int j = 0; j < NDIM; j++) {
			for (int k = 0; k <= j; k++) {
				const auto Xjk = dx[j] * dx[k];							// 6 OP
				M(j, k) += mass * Xjk;									// 12 OP
				for (int l = 0; l <= k; l++) {
					M(j, k, l) += mass * Xjk * dx[l];					// 30 OP
				}
			}
		}
	}
	for (int i = 0; i < MP; i++) {
		rc.m[i] = M[i].sum();
	}
	rc.r = 0.0;
	rc.num_active = 0;
	for (part_iter i = b; i < this_end; i++) {
		rc.r = std::max(rc.r, (float) abs(pos_to_double(parts(i).x) - pos_to_double(rc.x))); // 12 OP
		if (parts(i).flags.rung >= mrung) {
			rc.num_active++;
		}
	}
	if (e > part_end) {
		auto tmp = fut.get();
		for (int i = 0; i < MP; i++) {
			rc.m[i] += tmp.m[i];
		}
		rc.r = std::max(rc.r, tmp.r);
		rc.num_active = rc.num_active + tmp.num_active;
	}
	return rc;
}

void part_vect_reset() {
	std::vector<hpx::future<void>> futs;
	if (myid == 0) {
		for (int i = 1; i < localities.size(); i++) {
			futs.push_back(hpx::async < part_vect_reset_action > (localities[i]));
		}
	}
	for (int i = 0; i < POS_CACHE_SIZE; i++) {
		pos_cache[i].clear();
		group_cache[i].clear();
	}
	kick_rc.stats.zero();
	kick_rc.rung = 0;
	kick_rc.out.clear();
	hpx::wait_all(futs.begin(), futs.end());
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
	hpx::wait_all(futs.begin(), futs.end());
}

inline hpx::future<std::vector<vect<pos_type>>> part_vect_read_pos_cache(part_iter b, part_iter e) {
	const int index = (b / sizeof(particle)) % POS_CACHE_SIZE;
	std::unique_lock<mutex_type> lock(pos_cache_mtx[index]);
	auto iter = pos_cache[index].find(b);
	if (iter == pos_cache[index].end()) {
		hpx::lcos::local::promise<hpx::future<std::vector<vect<pos_type>>>> promise;
		auto fut = promise.get_future();
		pos_cache[index][b] = fut.then([b](decltype(fut) f) {
			return f.get().get();
		});
		lock.unlock();
		promise.set_value(hpx::async < part_vect_read_position_action > (localities[part_vect_locality_id(b)], b, e));
	}
	return hpx::async(hpx::launch::deferred, [b, index]() {
		std::unique_lock < mutex_type > lock(pos_cache_mtx[index]);
		auto future = pos_cache[index][b];
		lock.unlock();
		return future.get();
	});
}

inline hpx::future<std::vector<particle_group_info>> part_vect_read_group_cache(part_iter b, part_iter e) {
	const int index = (b / sizeof(particle)) % POS_CACHE_SIZE;
	std::unique_lock<mutex_type> lock(group_cache_mtx[index]);
	auto iter = group_cache[index].find(b);
	if (iter == group_cache[index].end()) {
		hpx::lcos::local::promise < hpx::future<std::vector<particle_group_info>> > promise;
		auto fut = promise.get_future();
		group_cache[index][b] = fut.then([b](decltype(fut) f) {
			return f.get().get();
		});
		lock.unlock();
		promise.set_value(hpx::async < part_vect_read_group_action > (localities[part_vect_locality_id(b)], b, e, range(), false));
	}
	return hpx::async(hpx::launch::deferred, [b, index]() {
		std::unique_lock < mutex_type > lock(group_cache_mtx[index]);
		auto future = group_cache[index][b];
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

std::vector<vect<pos_type>> part_vect_read_positions(const std::vector<std::pair<part_iter, part_iter>> &iters) {
	std::vector<std::pair<part_iter, part_iter>> local;
	std::vector<std::pair<part_iter, part_iter>> nonlocal;
	std::vector<vect<pos_type>> pos;
	local.reserve(iters.size());
	std::size_t size = 0;
	for (int i = 0; i < iters.size(); i++) {
		size += iters[i].second - iters[i].first;
		if (part_vect_locality_id(iters[i].first) == myid) {
			if (iters[i].second <= part_end) {
				local.push_back(iters[i]);
			} else {
				local.push_back(std::make_pair(iters[i].first, part_end));
				nonlocal.push_back(std::make_pair(part_end, iters[i].second));
			}
		} else {
			nonlocal.push_back(iters[i]);
//			printf( "-%i %i %i\n", size, local.size(), nonlocal.size());
		}
	}
	std::vector<hpx::future<std::vector<vect<pos_type>>>> futs;
	for (auto &iter : nonlocal) {
		futs.push_back(part_vect_read_position(iter.first, iter.second));
	}
	size = ((size - 1 + simd_float::size()) / simd_float::size()) * simd_float::size();
	pos.reserve(size);
	for (auto &iter : local) {
		for (part_iter i = iter.first; i < iter.second; i++) {
			pos.push_back(parts(i).x);
		}
	}
	for (auto &f : futs) {
		auto v = f.get();
		for (const auto &x : v) {
			pos.push_back(x);
		}
	}
	return pos;
}

hpx::future<std::vector<particle_group_info>> part_vect_read_group(part_iter b, part_iter e, range r, bool use_range) {
	if (b >= part_begin && b < part_end) {
		const int mtx_index = b % GROUP_MTX_SIZE;
		if (e <= part_end) {
			return hpx::async(hpx::launch::deferred, [=]() {
				std::lock_guard<mutex_type> lock(group_mtx[mtx_index]);
				std::vector<particle_group_info> these_parts;
				these_parts.reserve(e - b);
				for (int i = b; i < e; i++) {
					particle_group_info p;
					p.x = parts(i).x;
					if (!use_range || in_range(pos_to_double(p.x), r)) {
						p.id = parts(i).flags.group;
						these_parts.push_back(p);
					}
				}
				return these_parts;
			});
		} else {
			auto fut = part_vect_read_group_cache(part_end, e);
			return hpx::async(hpx::launch::deferred, [=](decltype(fut) f) {
				std::lock_guard<mutex_type> lock(group_mtx[mtx_index]);
				std::vector<particle_group_info> these_parts;
				these_parts.reserve(e - b);
				for (int i = b; i < part_end; i++) {
					particle_group_info p;
					const int mtx_index = parts(i).flags.group % GROUP_MTX_SIZE;
					p.x = parts(i).x;
					if (!use_range || in_range(pos_to_double(p.x), r)) {
						p.id = parts(i).flags.group;
						these_parts.push_back(p);
					}
				}
				auto next_parts = f.get();
				for (const auto &p : next_parts) {
					if (!use_range || in_range(pos_to_double(p.x), r)) {
						these_parts.push_back(p);
					}
				}
				return these_parts;
			}, std::move(fut));

		}
	} else {
		auto fut = part_vect_read_group_cache(b, e);
		return hpx::async(hpx::launch::deferred, [r, use_range](decltype(fut) fut) {
			auto these_parts = fut.get();
			if (use_range) {
				for (int i = 0; i < these_parts.size(); i++) {
					if (!in_range(pos_to_double(these_parts[i].x), r)) {
						these_parts[i] = these_parts[these_parts.size() - 1];
						these_parts.resize(these_parts.size() - 1);
						i--;
					}
				}
			}
			return these_parts;
		},std::move(fut));
	}
}

part_iter part_vect_count_lo(part_iter, part_iter, double xmid, int dim);

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
	const part_iter chunk_size = std::max(part_iter(1), (part_end - part_begin) / std::thread::hardware_concurrency());
	for (part_iter i = this_begin; i < this_end; i += chunk_size) {
		auto func = [i, this_end, dim, xmid, chunk_size]() {
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

part_iter part_vect_sort(part_iter b, part_iter e, double xmid, int dim) {

//	printf("Sorting %i %i on %i %i %i\n", b, e, myid, part_begin, part_end);
	if (e == b) {
		return e;
	} else {
		for (auto i = part_begin; i < part_end; i++) {
//			printf("%e\n", pos_to_double(parts(i).x[dim]));
		}
		//	printf("----\n");
		if (e <= part_end) {
			auto lo = b;
			auto hi = e;
			while (lo < hi) {
				if (pos_to_double(parts(lo).x[dim]) >= xmid) {
					while (lo != hi) {
						hi--;
						if (pos_to_double(parts(hi).x[dim]) < xmid) {
							auto tmp = parts(lo);
							parts(lo) = parts(hi);
							parts(hi) = tmp;
							break;
						}
					}
				}
				lo++;
			}
			for (auto i = part_begin; i < part_end; i++) {
				//			printf("%e\n", pos_to_double(parts(i).x[dim]));
			}
			//		printf("****\n");
			return hi;
		} else {
			auto count = part_vect_count_lo(b, e, xmid, dim);
			part_vect_sort_begin_action()(localities[part_vect_locality_id(b)], b, e, b + count, xmid, dim);
			return b + count;
		}
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
	return std::min(n, N - 1);
}

struct avg_pos_return {
	double avg;
	double max;
	template<class A>
	void serialize(A &&arc, unsigned) {
		arc & avg;
		arc & max;
	}
};

double part_vect_find_median_helper(part_iter b, part_iter e, part_iter median, double xmid, int dim);
avg_pos_return part_vect_avg_pos(part_iter b, part_iter e, int dim);

HPX_PLAIN_ACTION (part_vect_find_median_helper);
HPX_PLAIN_ACTION (part_vect_avg_pos);

avg_pos_return part_vect_avg_pos(part_iter b, part_iter e, int dim) {
	static const auto opts = options::get();
	std::vector<hpx::future<avg_pos_return>> futs;
	const part_iter N = localities.size();
	const part_iter M = opts.problem_size;
	double max = -std::numeric_limits<double>::max();
	if (part_end < e) {
		for (int n = part_vect_locality_id(part_end); n * M / N < e; n++) {
			auto fut = hpx::async < part_vect_avg_pos_action > (localities[n], n * M / N, std::min(e, (n + 1) * M / N), dim);
			futs.push_back(std::move(fut));
		}
	}
	const auto this_e = std::min(part_end, e);
	double avg = 0.0;
	std::uint64_t count = this_e - b;
	for (auto i = b; i < this_e; i++) {
		const auto x = pos_to_double(parts(i).x[dim]);
		avg += x;
		max = std::max(max, x);
	}
	int j = 0;
	if (part_end < e) {
		for (int n = part_vect_locality_id(part_end); n * M / N < e; n++) {
			const auto rc = futs[j++].get();
			const auto this_avg = rc.avg;
			const auto this_count = std::min(e, (n + 1) * M / N) - n * M / N;
			count += this_count;
			avg += this_avg * this_count;
			max = std::max(max, rc.max);
		}
	}
	avg_pos_return rc;
	avg /= count;
	rc.avg = avg;
	rc.max = max;
	return rc;
}

inline bool almost_equal(double a, double b) {
	const auto avg = (a + b) / 2.0;
	return (a == avg) || (b == avg);
}

double part_vect_find_median_helper(part_iter b, part_iter e, part_iter median, double xmid, int dim) {
	double r;
//	printf("%i %i %i %.8e\n", b, e, median, xmid);
//	if (e - b == 2) {
//		printf( "!\n");
//		for (auto i = b; i < std::min(part_end, e); i++) {
//			printf("%.8e\n", parts(i).x[dim]);
//		}
//
//	}
	if (e - b <= 1) {
		r = xmid;
	} else {
		const auto m = part_vect_sort(b, e, xmid, dim);
		if (median > m && median < e) {
			const auto newid = part_vect_locality_id(m);
			avg_pos_return rc;
			if (newid == myid) {
				rc = part_vect_avg_pos(m, e, dim);
			} else {
				const auto rc = hpx::async < part_vect_avg_pos_action > (localities[newid], m, e, dim).get();
			}
			if (almost_equal(rc.max, rc.avg)) {
				r = rc.max;
			} else {
				const auto new_xmid = rc.avg;
				if (newid == myid) {
					r = part_vect_find_median_helper(m, e, median, new_xmid, dim);
				} else {
					r = hpx::async < part_vect_find_median_helper_action > (localities[newid], m, e, median, new_xmid, dim).get();
				}
			}
		} else if (median >= b && median < m) {
			const auto rc = part_vect_avg_pos(b, m, dim);
			if (almost_equal(rc.max, rc.avg)) {
				r = rc.max;
			} else {
				const auto new_xmid = rc.avg;
				r = part_vect_find_median_helper(b, m, median, new_xmid, dim);
			}
		} else if (median == m) {
			return xmid;
		} else {
			assert(false);
		}
	}
	return r;
}

double part_vect_find_median(part_iter b, part_iter e, int dim) {
	const auto xmid = part_vect_avg_pos(b, e, dim).avg;
	return part_vect_find_median_helper(b, e, (b + e) / 2, xmid, dim);
}

