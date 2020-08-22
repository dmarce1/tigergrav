#include <tigergrav/gravity_work.hpp>
#include <tigergrav/part_vect.hpp>

#ifdef HPX_LITE
#include <hpx/hpx_lite.hpp>
#else
#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/synchronization/spinlock.hpp>
#include <hpx/async.hpp>
#endif

#include <unordered_map>

static std::vector<hpx::id_type> localities;
static int myid = -1;

using mutex_type = hpx::lcos::local::spinlock;

std::atomic<int> next_id_base(0);

HPX_PLAIN_ACTION(gwork_reset);

struct gwork_unit {
	part_iter yb;
	part_iter ye;
};

struct gwork_group {
	std::unordered_map<part_iter, gwork_unit> units;
	int mcount;
	gwork_group() {
		mcount = 0;
	}
};

mutex_type groups_mtx;
std::unordered_map<int, gwork_group> groups;

void gwork_show() {
	for (const auto &group : groups) {
		printf("%i %i\n", group.first, group.second.mcount);
	}
}

void gwork_checkin(int id) {
	std::lock_guard<mutex_type> lock(groups_mtx);
	groups[id].mcount++;
}

void gwork_reset() {
	if (myid == -1) {
		localities = hpx::find_all_localities();
		myid = hpx::get_locality_id();
	}
	std::vector<hpx::future<void>> futs;
	if (myid == 0) {
		for (int i = 1; i < localities.size(); i++) {
			futs.push_back(hpx::async<gwork_reset_action>(localities[i]));
		}
	}
	groups.clear();
	hpx::wait_all(futs.begin(), futs.end());
}

int gwork_assign_id() {
	int id;
	do {
		id = next_id_base++ * localities.size() + myid;
	} while (id == null_gwork_id);
	return id;
}
