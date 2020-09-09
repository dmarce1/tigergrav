#include <tigergrav/gravity_cuda.hpp>
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
std::atomic<int> thread_cnt(0);

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
	std::vector<cuda_work_unit> cunits;
	std::vector<std::function<hpx::future<void>(void)>> complete;
	mutex_type mtx;
	int workadded;
	int mcount;
	gwork_group() {
		mcount = 0;
		workadded = 0;
		static const auto opts = options::get();
		if (opts.cuda) {
			cunits.reserve(opts.workgroup_size);
		} else {
			units.reserve(opts.workgroup_size * 1024);
		}
		complete.reserve(opts.workgroup_size);
	}
	void free() {
		decltype(units)().swap(units);
		decltype(cunits)().swap(cunits);
		decltype(complete)().swap(complete);
	}
};

#define GROUP_TABLE_SIZE 999
mutex_type groups_mtx[GROUP_TABLE_SIZE];
std::unordered_map<int, gwork_group> groups[GROUP_TABLE_SIZE];

std::uint64_t gwork_pp_complete(int id, std::vector<force> *g, std::vector<vect<pos_type>> *x, const std::vector<std::pair<part_iter, part_iter>> &y,
		const std::vector<const multi_src*> &z, std::function<hpx::future<void>(void)> &&complete, bool do_phi) {
	static const auto opts = options::get();
	static const auto m = opts.m_tot / opts.problem_size;
	static const auto h = opts.soft_len;
	static const simd_float H3inv(1.0 / h / h / h);
	static const simd_float Hinv(1.0 / h);
	static const simd_float H(h);
	static const simd_float halfH(0.5 * h);
	bool do_work;
	gwork_unit unit;
	unit.fptr = g;
	unit.xptr = x;
	const int gi = id % GROUP_TABLE_SIZE;
	auto iter = groups[gi].find(id);
	if (iter == groups[gi].end()) {
		printf("Error - work group %i not found\n", id);
		abort();
	}
	auto &entry = iter->second;
	if (opts.cuda) {
		cuda_work_unit cu;
		cu.yiters = y;
		cu.z = z;
		cu.xptr = x;
		cu.fptr = g;
		entry.cunits.push_back(cu);
	} else {
		for (auto &j : y) {
			unit.yb = j.first;
			unit.ye = j.second;
			entry.units.push_back(unit);
		}
	}
	entry.complete.push_back(std::move(complete));
	entry.workadded++;
	do_work = entry.workadded == entry.mcount;
	if (entry.workadded > entry.mcount) {
		printf("Error too much work added %i %i %i\n", id, entry.workadded, entry.mcount);
		abort();
	}

	std::uint64_t flop = 0;
	if (do_work) {
		if (opts.cuda) {
			for (auto &unit : entry.cunits) {
				if (unit.yiters.size()) {
					static thread_local std::vector<std::pair<part_iter, part_iter>> tmp;
					tmp.resize(0);
					std::sort(unit.yiters.begin(), unit.yiters.end(), [](const std::pair<part_iter, part_iter> &a, const std::pair<part_iter, part_iter> &b) {
						return a.first < b.first;
					});
					std::pair<part_iter, part_iter> iter;
					int group_size = SYNCRATE;
					tmp.push_back(unit.yiters[0]);
					for (int i = 1; i < unit.yiters.size(); i++) {
						if (tmp.back().second == unit.yiters[i].first) {
							tmp.back().second = unit.yiters[i].second;
						} else {
							tmp.push_back(unit.yiters[i]);
						}
					}
					unit.yiters.resize(0);
					for (auto &this_iter : tmp) {
						const int this_size = this_iter.second - this_iter.first;
						const int ngroups = (this_size - 1) / group_size + 1;
						const int this_group_size = (this_size - 1) / ngroups + 1;
						for (int j = this_iter.first; j < this_iter.second; j += group_size) {
							iter.first = j;
							iter.second = std::min(this_iter.second, (part_iter) (j + group_size));
							unit.yiters.push_back(iter);
						}
					}
				}
			}

			flop += gravity_PP_direct_cuda(std::move(entry.cunits));

		} else {
			thread_cnt++;
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

				flop += y.size() * xcount * (107 + do_phi * 21);
				for (int i = 0; i < X.size(); i++) {
					for (int j = 0; j < y.size(); j++) {
						vect<simd_int> Y;
						vect<simd_float> dX;
						for (int dim = 0; dim < NDIM; dim++) {
							Y[dim] = simd_int(y[j][dim]);
						}
						if (opts.ewald) {
							for (int dim = 0; dim < NDIM; dim++) {
								dX[dim] = simd_float(simd_double(X[i][dim] - Y[dim]) * simd_double(POS_INV)); // 18
							}
						} else {
							for (int dim = 0; dim < NDIM; dim++) {
								dX[dim] = simd_float(simd_double(X[i][dim]) * simd_double(POS_INV) - simd_double(Y[dim]) * simd_double(POS_INV));
							}
						}
						const simd_float r2 = dX.dot(dX);								   // 5
						const simd_float r = sqrt(r2);								   // 7
						const simd_float rinv = simd_float(1) / max(r, halfH);								   //36
						const simd_float rinv3 = rinv * rinv * rinv;								   // 2
						simd_float sw1 = r > H;								   // 2
						simd_float sw3 = r < halfH;								   // 2
						simd_float sw2 = simd_float(1.0) - sw1 - sw3;								   // 3
						const simd_float roh = min(r * Hinv, 1);								   // 2
						const simd_float roh2 = roh * roh;								   // 1
						const simd_float roh3 = roh2 * roh;								   // 1

						const simd_float f1 = rinv3;

						simd_float f2 = simd_float(-32.0 / 3.0);
						f2 = fma(f2, roh, simd_float(+192.0 / 5.0));								   // 1
						f2 = fma(f2, roh, simd_float(-48.0));								   // 1
						f2 = fma(f2, roh, simd_float(+64.0 / 3.0));								   // 1
						f2 = fma(f2, roh3, simd_float(-1.0 / 15.0));								   // 1
						f2 *= rinv3;								   // 1

						simd_float f3 = simd_float(+32.0);
						f3 = fma(f3, roh, simd_float(-192.0 / 5.0));								   // 1
						f3 = fma(f3, roh2, simd_float(+32.0 / 3.0));								   // 1
						f3 *= H3inv;								   // 1

						simd_float f = sw1 * f1 + sw2 * f2 + sw3 * f3;								   // 5

						const auto dXM = dX * m;
						for (int dim = 0; dim < NDIM; dim++) {
							G[i][dim] -= simd_double(dXM[dim] * f);    						// 15
						}
//					printf( "----\n");
//					for( int k = 0; k < simd_float::size(); k++) {
//						if( sw2[k] == 1.0 || sw3[k]==1.0 && roh[k] != 0.0) {
//							printf( "%e\n", roh[k]);
//						}
//					}
						if (do_phi) {
							// 13S + 2D = 15
							const simd_float p1 = rinv;

							simd_float p2 = simd_float(+32.0 / 15.0);    						// 1
							p2 = fma(p2, roh, simd_float(-48.0 / 5.0));    						// 1
							p2 = fma(p2, roh, simd_float(+16.0));    						// 1
							p2 = fma(p2, roh, simd_float(-32.0 / 3.0));    						// 1
							p2 = fma(p2, roh2, simd_float(+16.0 / 5.0));    						// 1
							p2 = fma(p2, roh, simd_float(-1.0 / 15.0));    						// 1
							p2 *= rinv;    						// 1

							simd_float p3 = simd_float(-32.0 / 5.0);
							p3 = fma(p3, roh, simd_float(+48.0 / 5.0));    						// 1
							p3 = fma(p3, roh2, simd_float(-16.0 / 3.0));    						// 1
							p3 = fma(p3, roh2, simd_float(+14.0 / 5.0));    						// 1
							p3 *= Hinv;    						// 1

							Phi[i] -= simd_double((sw1 * p1 + sw2 * p2 + sw3 * p3) * m);    						// 10
						}

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
				thread_cnt--;
			}

		}
		std::vector<hpx::future<void>> futs;
		futs.reserve(entry.complete.size());
		for (auto &cfunc : entry.complete) {
			futs.push_back(cfunc());
		}
		hpx::wait_all(futs.begin(), futs.end());
		const int gi = id % GROUP_TABLE_SIZE;
		entry.free();

	}
	return flop;
}

void gwork_checkin(int id) {
	const int gi = id % GROUP_TABLE_SIZE;
	std::lock_guard<mutex_type> lock(groups_mtx[gi]);
	groups[gi][id].mcount++;
}

void gwork_reset() {
	if (myid == -1) {
		localities = hpx::find_all_localities();
		myid = hpx::get_locality_id();
	}
	std::vector<hpx::future<void>> futs;
	if (myid == 0) {
		for (int i = 1; i < localities.size(); i++) {
			futs.push_back(hpx::async < gwork_reset_action > (localities[i]));
		}
	}
	for (int i = 0; i < GROUP_TABLE_SIZE; i++) {
		groups[i].clear();
	}
	hpx::wait_all(futs.begin(), futs.end());
}

int gwork_assign_id() {
	int id;
	do {
		id = next_id_base++ * localities.size() + myid;
	} while (id == null_gwork_id);
	return id;
}
