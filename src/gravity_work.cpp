#include <tigergrav/gravity_work.hpp>

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
	std::shared_ptr<std::vector<force>> fptr;
	std::shared_ptr<std::vector<vect<pos_type>>> xptr;
};

struct gwork_group {
	std::vector<gwork_unit> units;
	std::vector<std::function<hpx::future<void>(void)>> complete;
	mutex_type mtx;
	int workadded;
	int mcount;
	gwork_group() {
		mcount = 0;
		workadded = 0;
	}
};

mutex_type groups_mtx;
std::unordered_map<int, gwork_group> groups;

void gwork_pp_complete(int id, std::shared_ptr<std::vector<force>> g, std::shared_ptr<std::vector<vect<pos_type>>> x,
		const std::vector<std::pair<part_iter, part_iter>> &y, std::function<hpx::future<void>(void)> &&complete) {
	bool do_work;
	gwork_unit unit;
	unit.fptr = g;
	unit.xptr = x;
	auto &entry = groups[id];
	{
		std::lock_guard<mutex_type> lock(entry.mtx);
		for (auto &j : y) {
			unit.yb = j.first;
			unit.ye = j.second;
			entry.units.push_back(unit);
		}
		entry.complete.push_back(std::move(complete));
		entry.workadded++;
		do_work = entry.workadded == entry.mcount;
	}
	if (entry.workadded > entry.mcount) {
		printf("Error too much work added\n");
		abort();
	}

	if (do_work) {
//		printf("Checkin complete starting work on group %i\n", id);
		std::vector<hpx::future<void>> futs;
		for (auto &cfunc : entry.complete) {
			futs.push_back(cfunc());
		}
		hpx::wait_all(futs.begin(), futs.end());
		groups.erase(id);
	}

}

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
