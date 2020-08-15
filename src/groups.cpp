#include <tigergrav/cosmo.hpp>
#include <tigergrav/groups.hpp>
#include <tigergrav/options.hpp>

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
static double ainv;
static double ainv2;

HPX_PLAIN_ACTION(groups_reset);
HPX_PLAIN_ACTION(groups_finish1);
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
		if( g.N >= 10 && g.T + g.W < 0.0) {
			fprintf(fp, "%16li %5i %12e %12e %12e %12e %12e %12e %12e %12e\n", entry.first, g.N, g.T, g.W, g.x[0], g.x[1], g.x[2], g.v[0], g.v[1], g.v[2]);
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
	ainv = 1.0 / cosmo_scale().second;
	ainv2 = ainv * ainv;
	hpx::wait_all(futs.begin(), futs.end());
}

void groups_add_particle1(gmember p) {
	static const auto opts = options::get();
	static const auto m = opts.m_tot / opts.problem_size;
	std::lock_guard<mutex_type> lock(mtx);
	auto &g = map[p.id];
	g.N++;
	g.W += 0.5 * m * p.phi * ainv * opts.G;
	g.T += 0.5 * m * p.v.dot(p.v) * ainv2;
	g.x += p.x;
	g.v += p.v;

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
		entry.second.x /= entry.second.N;
		entry.second.v /= entry.second.N;
	}

	hpx::wait_all(futs.begin(), futs.end());
}

