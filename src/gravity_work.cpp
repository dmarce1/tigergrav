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

HPX_PLAIN_ACTION (gwork_reset);

struct gwork_unit {
	part_iter yb;
	part_iter ye;
	std::vector<force> *fptr;
	std::vector<vect<pos_type>> *xptr;
};

struct gwork_super_unit {
	part_iter yb;
	part_iter ye;
	std::vector<std::vector<force>*> fs;
	std::vector<std::vector<vect<pos_type>>*> xs;

};

struct gwork_group {
	std::vector<gwork_unit> units;
	std::vector<std::function<hpx::future<void>(void)>> complete;
	mutex_type mtx;
	int workadded;
	int mcount;
	bool first_call;
	gwork_group() {
		mcount = 0;
		workadded = 0;
		first_call = true;
	}
};

mutex_type groups_mtx;
std::unordered_map<int, gwork_group> groups;

std::uint64_t gwork_pp_complete(int id, std::vector<force> *g, std::vector<vect<pos_type>> *x, const std::vector<std::pair<part_iter, part_iter>> &y,
		std::function<hpx::future<void>(void)> &&complete) {
	static const auto opts = options::get();
	static const auto m = opts.m_tot / opts.problem_size;
	static const auto h = opts.soft_len;
	static const simd_float H3inv(1.0 / h / h / h);
	static const simd_float Hinv(1.0 / h);
	static const simd_float H(h);
	static const simd_float n35o16 = simd_float(-35.0 / 16.0);
	static const simd_float p135o16 = simd_float(+135.0 / 16.0);
	static const simd_float n189o16 = simd_float(-189.0 / 16.0);
	static const simd_float p105o16 = simd_float(105.0 / 16.0);
	static const simd_float p35o128 = simd_float(35.0 / 128.0);
	static const simd_float n45o32 = simd_float(-45.0 / 32.0);
	static const simd_float p189o64 = simd_float(+189.0 / 64.0);
	static const simd_float n105o32 = simd_float(-105.0 / 32.0);
	static const simd_float p315o128 = simd_float(+315.0 / 128.0);
	bool do_work;
	gwork_unit unit;
	unit.fptr = g;
	unit.xptr = x;
	auto iter = groups.find(id);
	if( iter == groups.end()) {
		printf( "Error - work group %i not found\n", id);
		abort();
	}
	auto &entry = iter->second;
	{
		std::lock_guard<mutex_type> lock(entry.mtx);
		if (entry.first_call) {
			entry.first_call = false;
			entry.units.reserve(entry.mcount);
			entry.complete.reserve(entry.mcount);
		}
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
		printf("Error too much work added %i %i %i\n", id, entry.workadded, entry.mcount);
		abort();
	}

	std::uint64_t flop = 0;
	if (do_work) {
//		printf("Checkin complete starting work on group %i\n", id);

		std::sort(entry.units.begin(), entry.units.end(), [](const gwork_unit &a, const gwork_unit &b) {
			return a.yb < b.yb;
		});

		std::vector<gwork_super_unit> sunits;
		int sunit_count = 1;
		int max_ucount = 0;
		int this_ucount = 1;
		for (int i = 1; i < entry.units.size(); i++) {
			if (entry.units[i].yb != entry.units[i - 1].yb) {
				max_ucount = std::max(max_ucount, this_ucount);
				this_ucount = 1;
				sunit_count++;
			} else {
				this_ucount++;
			}
		}
		max_ucount = std::max(max_ucount, this_ucount);
		sunits.reserve(sunit_count);
		gwork_super_unit this_sunit;
		this_sunit.yb = entry.units[0].yb;
		this_sunit.ye = entry.units[0].ye;
		this_sunit.fs.reserve(max_ucount);
		this_sunit.xs.reserve(max_ucount);
		for (int i = 0; i < entry.units.size(); i++) {
			if (this_sunit.yb != entry.units[i].yb) {
				sunits.push_back(std::move(this_sunit));
				this_sunit.fs.reserve(max_ucount);
				this_sunit.xs.reserve(max_ucount);
				this_sunit.yb = entry.units[i].yb;
				this_sunit.ye = entry.units[i].ye;
			}
			this_sunit.fs.push_back(entry.units[i].fptr);
			this_sunit.xs.push_back(entry.units[i].xptr);
		}
		sunits.push_back(std::move(this_sunit));
		decltype(entry.units)().swap(entry.units);

		static thread_local std::vector<vect<simd_int>> X;
		static thread_local std::vector<vect<simd_double>> G;
		static thread_local std::vector<simd_double> Phi;
		for (auto &sunit : sunits) {
			const auto y = part_vect_read_position(sunit.yb, sunit.ye).get();
			int xcount = 0;
			int k = 0;
			vect<simd_int> this_x;
			X.resize(0);
			for (int i = 0; i < sunit.xs.size(); i++) {
				for (int j = 0; j < sunit.xs[i]->size(); j++) {
					for (int dim = 0; dim < NDIM; dim++) {
						this_x[dim][k] = (*sunit.xs[i])[j][dim];
					}
					k++;
					xcount++;
					if (k == simd_float::size()) {
						X.push_back(this_x);
						k = 0;
					}
				}
			}
			if (k > 0) {
				for (; k < simd_float::size(); k++) {
					for (int dim = 0; dim < NDIM; dim++) {
						this_x[dim][k] = 0;
					}
				}
				X.push_back(this_x);
			}
			G.resize(X.size());
			Phi.resize(X.size());
			for (int i = 0; i < X.size(); i++) {
				for (int dim = 0; dim < NDIM; dim++) {
					G[i][dim] = 0.0;
				}
				Phi[i] = 0.0;
			}

			flop += y.size() * xcount * 121;
			for (int i = 0; i < X.size(); i++) {
				for (int j = 0; j < y.size(); j++) {
					vect<simd_int> Y;
					vect<simd_float> dX;
					for (int dim = 0; dim < NDIM; dim++) {
						Y[dim] = simd_int(y[j][dim]);
					}
					if (opts.ewald) {
						for (int dim = 0; dim < NDIM; dim++) {
							dX[dim] = simd_float(simd_double(X[i][dim] - Y[dim]) * simd_double(POS_INV));
							// 0 / 9
						}
					} else {
						for (int dim = 0; dim < NDIM; dim++) {
							dX[dim] = simd_float(simd_double(X[i][dim]) * simd_double(POS_INV) - simd_double(Y[dim]) * simd_double(POS_INV));
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
					simd_float f2 = n35o16;
					f2 = fmadd(f2, roh2, p135o16);                   // 2 / 0
					f2 = fmadd(f2, roh2, n189o16);                   // 2 / 0
					f2 = fmadd(f2, roh2, p105o16);                    // 2 / 0
					f2 *= H3inv;                                                       // 1 / 0

					const auto dXM = dX * m;
					for (int dim = 0; dim < NDIM; dim++) {
						G[i][dim] -= simd_double(dXM[dim] * (sw1 * f1 + sw2 * f2));    //12 / 6
					}

					// 13S + 2D = 15
					const simd_float p1 = rinv;

					simd_float p2 = p35o128;
					p2 = fmadd(p2, roh2, n45o32);				   // 2 / 0
					p2 = fmadd(p2, roh2, p189o64);               // 2 / 0
					p2 = fmadd(p2, roh2, n105o32);               // 2 / 0
					p2 = fmadd(p2, roh2, p315o128);              // 2 / 0
					p2 *= Hinv;                                                    // 1 / 0

					Phi[i] -= simd_double((sw1 * p1 + sw2 * p2) * m);              // 4 / 2

				}
			}
			k = 0;
			for (int i = 0; i < sunit.fs.size(); i++) {
				for (int j = 0; j < sunit.fs[i]->size(); j++) {
					const int p = k / simd_float::size();
					const int q = k % simd_float::size();
					for (int dim = 0; dim < NDIM; dim++) {
						(*sunit.fs[i])[j].g[dim] += G[p][dim][q];
					}
					(*sunit.fs[i])[j].phi += Phi[p][q];
					k++;
				}
			}
		}

		std::vector<hpx::future<void>> futs;
		futs.reserve(entry.complete.size());
		for (auto &cfunc : entry.complete) {
			futs.push_back(cfunc());
		}
		hpx::wait_all(futs.begin(), futs.end());
		std::lock_guard<mutex_type> lock(groups_mtx);
		groups.erase(id);
	}

	return flop;
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
