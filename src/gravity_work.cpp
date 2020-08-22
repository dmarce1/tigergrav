#include <tigergrav/gravity_work.hpp>
#include <tigergrav/options.hpp>
#include <tigergrav/simd.hpp>

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
	static const auto opts = options::get();
	static const auto m = opts.m_tot / opts.problem_size;
	static const auto h = opts.soft_len;
	static const simd_float H3inv(1.0 / h / h / h);
	static const simd_float Hinv(1.0 / h);
	static const simd_float H(h);
	static const auto phi_self = (-315.0 / 128.0) * m / h;
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

		std::sort(entry.units.begin(), entry.units.end(), [](const gwork_unit &a, const gwork_unit &b) {
			return a.yb < b.yb;
		});

		part_iter last_begin = 0;
		bool first_round = true;
		std::vector<vect<simd_int>> Y;
		std::vector<simd_float> M;
		std::vector<vect<simd_double>> G;
		std::vector<simd_double> Phi;
		for (const auto &unit : entry.units) {
			if (unit.yb != last_begin || first_round) {
				auto y = part_vect_read_position(unit.yb, unit.ye).get();
				vect<simd_int> this_y;
				simd_float this_m;
				Y.resize(0);
				M.resize(0);
				int k = 0;
				for (int j = 0; j < y.size(); j++) {
					for (int dim = 0; dim < NDIM; dim++) {
						this_y[dim][k] = y[j][dim];
					}
					this_m[k] = m;
					k++;
					if (k == simd_float::size()) {
						Y.push_back(this_y);
						M.push_back(this_m);
						k = 0;
					}
				}
				if (k != 0) {
					for (; k < simd_float::size(); k++) {
						for (int dim = 0; dim < NDIM; dim++) {
							this_y[dim][k] = 0.0;
						}
						this_m[k] = 0.0;
					}
					Y.push_back(this_y);
					M.push_back(this_m);
				}
			}
			last_begin = unit.yb;

			const auto xsize = unit.xptr->size();
			const auto ysize = Y.size();
			Phi.resize(xsize);
			G.resize(xsize);
			for (int i = 0; i < xsize; i++) {
				G[i] = vect < simd_double > (0.0);
				Phi[i] = 0.0;
			}
			for (int j = 0; j < ysize; j++) {
				for (int i = 0; i < xsize; i++) {
					vect<simd_int> X;
					for (int dim = 0; dim < NDIM; dim++) {
						X[dim] = simd_int((*unit.xptr)[i][dim]);
					}

					vect<simd_float> dX;
					if (opts.ewald) {
						for (int dim = 0; dim < NDIM; dim++) {
							dX[dim] = simd_float(simd_double(X[dim] - Y[j][dim]) * simd_double(POS_INV));
							// 0 / 9
						}
					} else {
						for (int dim = 0; dim < NDIM; dim++) {
							dX[dim] = simd_float(simd_double(X[dim]) * simd_double(POS_INV) - simd_double(Y[j][dim]) * simd_double(POS_INV));
						}
					}
					const simd_float r2 = dX.dot(dX);								   // 5 / 0
					const simd_float r = sqrt(r2);									   // 7 / 0
					const simd_float rinv = simd_float(1) / max(r, H);                 //36 / 0
					const simd_float rinv3 = rinv * rinv * rinv;                       // 2 / 0
					simd_float sw1 = r > H;                                            // 1 / 0
					simd_float sw2 = (simd_float(1.0) - sw1);                          // 1 / 0
					const simd_float roh = min(r * Hinv, 1);                           // 2 / 0
					const simd_float roh2 = roh * roh;                                 // 1 / 0

					const simd_float f1 = rinv3;

					simd_float f2 = simd_float(-35.0 / 16.0);
					f2 = fmadd(f2, roh2, simd_float(+135.0 / 16.0));                   // 2 / 0
					f2 = fmadd(f2, roh2, simd_float(-189.0 / 16.0));                   // 2 / 0
					f2 = fmadd(f2, roh2, simd_float(105.0 / 16.0));                    // 2 / 0
					f2 *= H3inv;                                                       // 1 / 0

					const auto dXM = dX * M[j];
					for (int dim = 0; dim < NDIM; dim++) {
						G[i][dim] -= simd_double(dXM[dim] * (sw1 * f1 + sw2 * f2));    //12 / 6
					}

					// 13S + 2D = 15
					const simd_float p1 = rinv;

					simd_float p2 = simd_float(35.0 / 128.0);
					p2 = fmadd(p2, roh2, simd_float(-45.0 / 32.0));				   // 2 / 0
					p2 = fmadd(p2, roh2, simd_float(+189.0 / 64.0));               // 2 / 0
					p2 = fmadd(p2, roh2, simd_float(-105.0 / 32.0));               // 2 / 0
					p2 = fmadd(p2, roh2, simd_float(+315.0 / 128.0));              // 2 / 0
					p2 *= Hinv;                                                    // 1 / 0

					Phi[i] -= simd_double((sw1 * p1 + sw2 * p2) * M[j]);              // 4 / 2
				}
			}
			auto &f = *unit.fptr;
			for (int i = 0; i < xsize; i++) {
				for (int dim = 0; dim < NDIM; dim++) {
					f[i].g[dim] += G[i][dim].sum();
				}
				f[i].phi += Phi[i].sum();
				f[i].phi -= phi_self;
			}
			first_round = false;
		}

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
