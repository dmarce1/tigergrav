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

std::unordered_map<std::uint64_t, group> map;
mutex_type mtx;

static std::vector<hpx::id_type> localities;
static int myid = -1;
static double a;
static double ainv;
static double ainv2;

HPX_PLAIN_ACTION(groups_reset);
HPX_PLAIN_ACTION(groups_finish1);
HPX_PLAIN_ACTION(groups_finish2);
HPX_PLAIN_ACTION(groups_output);

void groups_output(int oi) {
	if (myid == -1) {
		myid = hpx::get_locality_id();
		localities = hpx::find_all_localities();
	}
	std::vector<hpx::future<void>> futs;
	if (myid == 0) {
		for (int i = 1; i < localities.size(); i++) {
			futs.push_back(hpx::async<groups_output_action>(localities[i], oi));
		}
	}
	std::string filename = std::string("groups.") + std::to_string(oi) + "." + std::to_string(myid) + ".dat";
	FILE *fp = fopen(filename.c_str(), "wt");
	if (fp == NULL) {
		printf("Unable to open output file for groups\n");
		abort();
	}
	bool found_some = false;
	for (const auto &entry : map) {
		const auto &g = entry.second;
		if (g.N >= 10 && g.T + g.W < 0.0) {
			const auto verr = (2.0 * g.T + g.W) / (g.T + std::abs(g.W));
			const auto vdisp = abs(g.dv);
			const auto v = abs(g.v);
			fprintf(fp, "%16li %5i %12e %12e %12e %12e %12e %12e %12e\n", entry.first, g.N, g.rmax, g.rc, g.T, g.W, verr, v, vdisp);
			found_some = true;
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
			futs.push_back(hpx::async<groups_reset_action>(localities[i]));
		}
	}

	map.clear();
	a = cosmo_scale().second;
	ainv = 1.0 / a;
	ainv2 = ainv * ainv;
	hpx::wait_all(futs.begin(), futs.end());
}

void groups_add_particle1(gmember p) {
	static const auto opts = options::get();
	static const auto m = opts.m_tot / opts.problem_size;
	std::lock_guard<mutex_type> lock(mtx);
	auto &g = map[p.id];
	if (g.N == 0) {
		g.x = p.x;
	} else {
		auto x0 = p.x / g.N;
		auto dx = g.x - x0;
		for (int dim = 0; dim < NDIM; dim++) {
			const float absdx = std::abs(dx[dim]);
			dx[dim] = std::copysign(std::min(absdx, (float)1.0 - absdx), dx[dim] * ((float)0.5 - absdx));
		}
		x0 += dx;
		g.x += x0;
	}
	g.N++;
	g.W += 0.5 * m * p.phi * ainv * opts.G;
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
			futs.push_back(hpx::async<groups_finish1_action>(localities[i]));
		}
	}

	for (auto &entry : map) {
		const auto N = entry.second.N;
		entry.second.x /= N;
		entry.second.v /= N;
		entry.second.T /= N;
		entry.second.W /= N;

	}

	hpx::wait_all(futs.begin(), futs.end());
}

void groups_add_particle2(particle p) {
	static const auto opts = options::get();
	static const auto m = opts.m_tot / opts.problem_size;
	std::lock_guard<mutex_type> lock(mtx);
	auto &g = map[p.flags.group];
	const auto dv = (p.v * ainv - g.v);
	g.T += 0.5 * m * dv.dot(dv);
	for (int dim = 0; dim < NDIM; dim++) {
		g.dv[dim] += dv[dim] * dv[dim];
	}
	auto dx = pos_to_double(p.x) - g.x;
	for (int dim = 0; dim < NDIM; dim++) {
		dx[dim] = std::min(std::abs(dx[dim]), 1.0 - std::abs(dx[dim]));
	}
	const float r = a * dx.dot(dx);
	g.rmax = std::max(g.rmax, r);
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
			futs.push_back(hpx::async<groups_finish2_action>(localities[i]));
		}
	}

	for (auto &entry : map) {
		const auto N = entry.second.N;
		entry.second.T /= N;
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

	hpx::wait_all(futs.begin(), futs.end());
}

