#include <tigergrav/cosmo.hpp>
#include <tigergrav/groups.hpp>
#include <tigergrav/options.hpp>

#include <cmath>

#ifdef HPX_LITE
#include <hpx/hpx_lite.hpp>
#else
#include <hpx/include/async.hpp>
#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/synchronization/spinlock.hpp>
#endif

#include <unordered_map>

using mutex_type = hpx::lcos::local::spinlock;

#define NMAP ((std::uint64_t)1024)
std::unordered_map<std::uint64_t, group> map[NMAP];
static mutex_type mtx[NMAP];

static std::vector<hpx::id_type> localities;
static int myid = -1;
static double a;
static double ainv;
static double ainv2;

HPX_PLAIN_ACTION (groups_reset);
HPX_PLAIN_ACTION (groups_finish1);
HPX_PLAIN_ACTION (groups_finish2);
HPX_PLAIN_ACTION (groups_output);
HPX_PLAIN_ACTION (groups_execute_finders);

std::vector<std::function<bool(void)>> group_finders;
mutex_type group_finder_mtx;

void groups_add_finder(std::function<bool(void)> f) {
	std::lock_guard<mutex_type> lock(group_finder_mtx);
	group_finders.push_back(f);
}

bool groups_execute_finders() {
	std::vector<hpx::future<bool>> futs;
	bool rc = false;
	if (myid == 0) {
		for (int i = 1; i < localities.size(); i++) {
			futs.push_back(hpx::async < groups_execute_finders_action > (localities[i]));
		}
	}
	constexpr int chunk_size = 1024;
	for (int i = 0; i < group_finders.size(); i += chunk_size) {
		futs.push_back(hpx::async([i]() {
			bool rc = false;
			for (int j = i; j < std::min((int) group_finders.size(), i + chunk_size); j++) {
				bool this_rc = group_finders[j]();
				rc = rc || this_rc;
			}
			return rc;
		}));
	}

	for (int i = 0; i < futs.size(); i++) {
		bool this_rc = futs[i].get();
		rc = rc || this_rc;
	}
	return rc;
}

void groups_output(int oi) {
	if (myid == -1) {
		myid = hpx::get_locality_id();
		localities = hpx::find_all_localities();
	}
	std::vector<hpx::future<void>> futs;
	if (myid == 0) {
		for (int i = 1; i < localities.size(); i++) {
			futs.push_back(hpx::async < groups_output_action > (localities[i], oi));
		}
	}
	std::string filename = std::string("groups.") + std::to_string(oi) + "." + std::to_string(myid) + ".dat";
	FILE *fp = fopen(filename.c_str(), "wt");
	if (fp == NULL) {
		printf("Unable to open output file for groups\n");
		abort();
	}
	bool found_some = false;
	for (int m = 0; m < NMAP; m++) {
		for (const auto &entry : map[m]) {
			const auto &g = entry.second;
			if (g.N >= 10) {
				const auto vdisp = abs(g.dv);
				const auto v = abs(g.v);
				fprintf(fp, "%16li %5i %12e %12e %12e %12e\n", entry.first, g.N, g.rmax, g.rc, v, vdisp);
				found_some = true;
			}
		}
	}
	if (!found_some) {
		fprintf(fp, "No groups\n");
	}
	fclose(fp);
	hpx::wait_all(futs.begin(), futs.end());
}

void groups_reset() {
	if (myid == -1) {
		myid = hpx::get_locality_id();
		localities = hpx::find_all_localities();
	}
	std::vector<hpx::future<void>> futs;
	if (myid == 0) {
		for (int i = 1; i < localities.size(); i++) {
			futs.push_back(hpx::async < groups_reset_action > (localities[i]));
		}
	}
	for (int m = 0; m < NMAP; m++) {
		map[m].clear();
	}
	group_finders.resize(0);
	a = cosmo_scale().second;
	ainv = 1.0 / a;
	ainv2 = ainv * ainv;
	hpx::wait_all(futs.begin(), futs.end());
}

void groups_add_particle1(particle p) {
	static const auto opts = options::get();
	static const auto m = opts.m_tot / opts.problem_size;
	const int map_index = p.flags.group % NMAP;
	std::lock_guard<mutex_type> lock(mtx[map_index]);
	auto &g = map[map_index][p.flags.group];
	if (g.N == 0) {
		g.x = pos_to_double(p.x);
		g.N++;
	} else {
		auto dx = pos_to_double(p.x) - g.x / g.N;
		for (int dim = 0; dim < NDIM; dim++) {
			const double absdx = std::abs(dx[dim]);
			dx[dim] = std::copysign(std::min(absdx, (double) 1.0 - absdx), dx[dim] * ((double) 0.5 - absdx));
		}
		g.x += dx + g.x / g.N;
		g.N++;
		for (int dim = 0; dim < NDIM; dim++) {
			while (g.x[dim] < 0.0) {
				g.x[dim] += g.N;
			}
			while (g.x[dim] > g.N) {
				g.x[dim] -= g.N;
			}
		}
	}
	g.v += p.v * ainv;

}

void groups_finish1() {
	if (myid == -1) {
		myid = hpx::get_locality_id();
		localities = hpx::find_all_localities();
	}
	std::vector<hpx::future<void>> futs;
	if (myid == 0) {
		for (int i = 1; i < localities.size(); i++) {
			futs.push_back(hpx::async < groups_finish1_action > (localities[i]));
		}
	}
	for (int m = 0; m < NMAP; m++) {
		for (auto &entry : map[m]) {
			const auto N = entry.second.N;
			entry.second.x /= N;
			entry.second.v /= N;
		}
	}

	hpx::wait_all(futs.begin(), futs.end());
}

void groups_add_particle2(particle p) {
	static const auto opts = options::get();
	static const auto m = opts.m_tot / opts.problem_size;
	const int map_index = p.flags.group % NMAP;
	std::lock_guard<mutex_type> lock(mtx[map_index]);
	auto &g = map[map_index][p.flags.group];
	const auto dv = (p.v * ainv - g.v);
	for (int dim = 0; dim < NDIM; dim++) {
		g.dv[dim] += dv[dim] * dv[dim];
	}
	auto dx = pos_to_double(p.x) - g.x;
	for (int dim = 0; dim < NDIM; dim++) {
		dx[dim] = std::min(std::abs(dx[dim]), 1.0 - std::abs(dx[dim]));
	}
	const double r = a * abs(dx);
	g.rmax = std::max(g.rmax, (float) r);
	g.radii.push_back(r);

}

void groups_finish2() {
	if (myid == -1) {
		myid = hpx::get_locality_id();
		localities = hpx::find_all_localities();
	}
	std::vector<hpx::future<void>> futs;
	if (myid == 0) {
		for (int i = 1; i < localities.size(); i++) {
			futs.push_back(hpx::async < groups_finish2_action > (localities[i]));
		}
	}

	for (int m = 0; m < NMAP; m++) {
		for (auto &entry : map[m]) {
			const auto N = entry.second.N;
			entry.second.dv /= N;
			for (int dim = 0; dim < NDIM; dim++) {
				entry.second.dv[dim] = std::sqrt(entry.second.dv[dim]);
			}
			auto &radii = entry.second.radii;
			std::sort(radii.begin(), radii.end());
			const int mid = radii.size() / 2 + radii.size() % 1;
			entry.second.rc = radii[mid];
			radii = std::vector<float>();
		}
	}

	hpx::wait_all(futs.begin(), futs.end());
}

