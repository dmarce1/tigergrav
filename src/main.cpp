#include <tigergrav/defs.hpp>

#ifdef HPX_LITE
#include <hpx/hpx_lite.hpp>
#else
#include <hpx/hpx_init.hpp>
#endif

#include <tigergrav/initialize.hpp>
#include <tigergrav/options.hpp>
#include <tigergrav/part_vect.hpp>
#include <tigergrav/gravity.hpp>
#include <tigergrav/tree.hpp>
#include <tigergrav/groups.hpp>
#include <tigergrav/cosmo.hpp>
#include <tigergrav/map.hpp>

#include <algorithm>

#include <fenv.h>

double timer(void) {
	return std::chrono::duration_cast < std::chrono::milliseconds > (std::chrono::system_clock::now().time_since_epoch()).count() / 1000.0;
}

double fmm_time;
double parts_per_sec;
double fmm_time_total = 1.0e-10;
double fmm_parts_total = 0.0;

std::pair<kick_return, interaction_stats> solve_gravity(tree_client root_ptr, rung_type mrung, bool do_out, bool first_call = false) {
	auto start = timer();
	static const auto opts = options::get();
	auto mrc = root_ptr.compute_multipoles(mrung, do_out, null_gwork_id, 0);
//	gwork_show();
	auto root_list = std::vector<check_item>(1, root_ptr.get_check_item());
	if (do_out && !opts.solver_test && opts.groups) {
		groups_reset();
		auto tstart = timer();
		printf("Finding groups\n");
		part_vect_init_groups();
		tree::set_theta(0.99);
		root_ptr.find_groups(root_list, 0);
		do {
			printf(".\n");
		} while (groups_execute_finders());
		tree::set_theta(opts.theta);
		printf("Done finding groups in %e seconds\n", timer() - tstart);

	}
//	printf("Multipoles took %e seconds\n", timer() - start);
	start = timer();
	expansion<double> L;
	L = 0.0;
	kick_return rc;
	interaction_stats istats;
	if (opts.gravity) {
		fmm_time = timer();
		istats = root_ptr.kick_fmm(root_list, root_list, { { 0.5, 0.5, 0.5 } }, L, mrung, do_out, 0);
		fmm_time = timer() - fmm_time;
		rc = part_vect_kick_return();
		if (do_out && !opts.solver_test && opts.groups) {
			groups_finish1();
			part_vect_find_groups2();
			groups_finish2();
		}
		parts_per_sec = (double) mrc.m.num_active / fmm_time;
		fmm_time_total += fmm_time;
		fmm_parts_total += mrc.m.num_active;
	}
//	printf("fmm took %e seconds\n", timer() - start);
	return std::make_pair(rc, istats);
}

std::string to_kform(std::uint64_t num) {
	int cut = 10;
	if (num > cut * 1024 * 1024 * 1024) {
		return std::to_string(num / 1024 / 1024 / 1024) + "T";
	} else if (num > cut * 1024 * 1024) {
		return std::to_string(num / 1024 / 1024) + "M";
	} else if (num > cut * 1024) {
		return std::to_string(num / 1024) + "k";
	} else {
		return std::to_string(num) + " ";
	}
}

std::string to_kform(double num) {
	int cut = 10;
	if (num > cut * 1024 * 1024 * 1024) {
		return std::to_string((int) (num / 1024 / 1024 / 1024)) + "T";
	} else if (num > cut * 1024 * 1024) {
		return std::to_string((int) (num / 1024 / 1024)) + "M";
	} else if (num > cut * 1024) {
		return std::to_string(int(num / 1024)) + "k";
	} else {
		return std::to_string(int(num)) + " ";
	}
}

int hpx_main(int argc, char *argv[]) {
	printf("sizeof(particle) = %li\n", sizeof(particle));
	printf("sizeof(tree)     = %li\n", sizeof(tree));
//	printf("Hardware concurrency = %li\n", hpx::threads::hardware_concurrency());
	feenableexcept(FE_DIVBYZERO);
	feenableexcept(FE_INVALID);
	feenableexcept(FE_OVERFLOW);

	options opts;
	opts.process_options(argc, argv);

	double tau_max;
	double dtau_out;
	if (opts.cosmic) {
		cosmos cinit;
		cinit.advance_to_scale(1.0 / (1.0 + opts.z0));
		tau_max = -cinit.get_tau();
		const auto a0 = cinit.get_scale();
		const auto adot0 = cinit.get_Hubble() * a0;
		printf("Inializing with a = %e, adot = %e, tau = %e\n", a0, adot0, tau_max);
		cosmo_init(a0, adot0);
		if (opts.map) {
			map_init();
		}
	} else {
		cosmo_init(1.0, 0.0);
		tau_max = opts.t_max;
	}
	dtau_out = opts.t_max / opts.nout;

	printf("Output every %e\n", dtau_out);

	tree::set_theta(opts.theta);

	part_vect_init();

	if (opts.solver_test) {
		printf("Computing direct solution first\n");
		tree_client root_ptr = hpx::new_ < tree > (hpx::find_here(), 1, 0, opts.problem_size, 0).get();
		refine_return refine_rc;
		do {
			refine_rc = root_ptr.refine(0);
		} while (refine_rc.rc);
		tree::set_theta(1e-10);
		auto kr = solve_gravity(root_ptr, min_rung(0), true);
		std::sort(kr.first.out.begin(), kr.first.out.end());
		const auto direct = kr.first.out;
		printf("%11s %11s %11s %11s %11s %11s %11s %11s\n", "theta", "time", "GFLOPS", "error", "error99", "gx", "gy", "gz");
		for (double theta = 1.0; theta >= 0.17; theta -= 0.1) {
			root_ptr = hpx::new_ < tree > (hpx::find_here(), 1, 0, opts.problem_size, 0).get();
			refine_return refine_rc;
			do {
				refine_rc = root_ptr.refine(0);
			} while (refine_rc.rc);
			tree::set_theta(theta);
			tree::reset_flop();
			auto start = timer();
			kr = solve_gravity(root_ptr, min_rung(0), true);
			auto stop = timer();
			auto flops = tree::get_flop() / (stop - start + 1.0e-10) / std::pow(1024, 3);
			std::sort(kr.first.out.begin(), kr.first.out.end());
			const auto err = compute_error(kr.first.out, direct);
			printf("%11.4e %11.4e %11.4e %11.4e %11.4e %11.4e %11.4e %11.4e \n", theta, stop - start, flops, err.err, err.err99, err.g[0], err.g[1], err.g[2]);
		}
	} else {

		printf("Forming tree\n");
		auto tstart = timer();
		tree_client root_ptr = hpx::new_ < tree > (hpx::find_here(), 1, 0, opts.problem_size, 0).get();
		refine_return refine_rc;
		do {
			printf("Refining\n");
			refine_rc = root_ptr.refine(0);
		} while (refine_rc.rc);
		printf("Done forming tree took %e seconds\n", timer() - tstart);

		double t = 0.0;
		int iter = 0;
		double dt;
		std::pair<kick_return, interaction_stats> kr;
		time_type itime = 0;

		tstart = timer();

		double pec_energy = 0.0;
		double etot0;
		double last_ekin;
		double ekin;
		double epot;
		double last_epot;
		bool do_out = true;
		bool first_show = true;
		double z = 1.0 / cosmo_scale().second - 1.0;
		const auto show = [&]() {
			if (first_show) {
				last_ekin = ekin;
			}
			auto tmp = cosmo_scale();
			auto da = tmp.second - tmp.first;
			auto a = tmp.second;
			pec_energy += 0.5 * (ekin + last_ekin) * da;
			//			interaction_statistics istats = root_ptr->get_istats();
			if (iter % 25 == 0) {
				printf("%4s %11s %11s %3s %3s %5s %5s %11s %11s %11s %11s %11s %9s %9s %9s %11s %11s ", "i", "pps", "cps", "mind", "maxd", "avgd", "nnode", "t",
						"tau", "z", "a", "dt", "itime", "max rung", "min act.", "pct act", "GFLOP");
				printf(" %11s %11s %11s %11s  %11s %11s\n", "g", "p", "epot", "ekin", "epec", "etot");
			}
			const auto avg_depth = std::log(refine_rc.leaves) / std::log(2);
			const double cp_skew = (double) kr.second.CP_direct / std::max((double) kr.second.CC_direct, (double) 1);
			printf("%4i %11.4e %11.4e %4i %4i %5.1f %5s %11.4e  %11.4e %11.4e %11.4e %11.4e  ", iter, fmm_parts_total / fmm_time_total, cp_skew,
					refine_rc.min_depth, refine_rc.max_depth, avg_depth, to_kform(refine_rc.nodes).c_str(), t, cosmo_time(), z, a, dt);
			printf("%9x ", (int) itime);
			printf("%9i ", (int) kr.first.rung);
			printf("%9i ", (int) min_rung(itime));
			printf("%10.1f%% ", 100.0 * tree::get_pct_active());
			printf("%11.4e ", tree::get_flop() / (timer() - tstart + 1.0e-20) / pow(1024, 3));
//			tree::reset_flop();
//			tstart = timer();
//			FILE *fp = fopen("istats.txt", "at");
//			fprintf(fp, "%11.4e %6s %6s %6s %6s %6s %6s \n", parts_per_sec, to_kform((double) kr.second.CC_direct / fmm_time).c_str(),
//					to_kform((double) kr.second.CP_direct / fmm_time).c_str(), to_kform((double) kr.second.PP_direct / fmm_time).c_str(),
//					to_kform((double) kr.second.CC_ewald / fmm_time).c_str(), to_kform((double) kr.second.CP_ewald / fmm_time).c_str(),
//					to_kform((double) kr.second.PP_ewald / fmm_time).c_str());
//			fclose(fp);
			if (do_out) {
				printf("%11.4e ", abs(kr.first.stats.g));
				printf("%11.4e ", abs(kr.first.stats.p));
				const auto etot = a * (kr.first.stats.pot + kr.first.stats.kin) + pec_energy;
				if (first_show) {
					etot0 = etot;
					first_show = false;
				}
				const auto eden = std::max(a * ekin, std::abs(etot0));
				printf("%11.4e %11.4e %11.4e %11.4e %11.4e ", a * kr.first.stats.pot, a * ekin, pec_energy, etot,
						opts.glass ? (epot - last_epot) / epot : (etot - etot0) / eden);
				FILE *fp = fopen("energy.dat", "at");
				fprintf(fp, "%11.4e %11.4e %11.4e %11.4e %11.4e\n", a * kr.first.stats.pot, a * ekin, pec_energy, etot, (etot - etot0) / eden);
				fclose(fp);
			} else {
				printf("%11s ", "");
				printf("%11.4e ", abs(kr.first.stats.p));
				printf("%11s %11.4e %11.4e %11s ", "", a * ekin, pec_energy, "");
			}
//			printf("%f/%f %f/%f %f/%f %f/%f", istats.CC_direct_pct, istats.CC_ewald_pct, istats.CP_direct_pct, istats.CP_ewald_pct, istats.PC_direct_pct,
//					istats.PC_ewald_pct, istats.PP_direct_pct, istats.PP_ewald_pct);
			printf("\n");
			last_ekin = ekin;
		};
		int oi = 1;
		int si = 1;
		if (opts.cosmic) {
			cosmo_advance(0.0);
		}
		kr = solve_gravity(root_ptr, min_rung(0), do_out);
		if (!opts.gravity) {
			return hpx::finalize();
		}
		last_epot = epot = kr.first.stats.pot;
		ekin = kr.first.stats.kin;
		if (do_out) {
			output_particles(kr.first.out, "parts.0.silo");
			if (opts.groups) {
				groups_output(0);
			}
		}
		dt = rung_to_dt(kr.first.rung);
		while (t < opts.t_max) {
			show();
			auto ts = timer();
//			printf( "Drifting\n");
			if (opts.cosmic) {
				cosmo_advance(dt);
			}
			z = 1.0 / cosmo_scale().second - 1.0;
			if ((t + dt) / dtau_out >= oi) {
				do_out = true;
				printf("Doing output #%i\n", oi);
				oi++;
			} else {
				do_out = false;
			}
			if (opts.map) {
				map_reset(t + dt);
			}
			ekin = root_ptr.drift(t, kr.first.rung);
			static int mo = 0;
			if (opts.map) {
				map_output(t + dt);
			}
//			printf("drift took %e seconds\n", timer() - ts);
			ts = timer();
			root_ptr = hpx::invalid_id;
//			printf( "Forming tree\n");
			root_ptr = hpx::new_ < tree > (hpx::find_here(), 1, 0, opts.problem_size, 0).get();
			do {
				refine_rc = root_ptr.refine(0);
			} while (refine_rc.rc);
//			printf("Tree took %e seconds\n", timer() - ts);
			itime = inc(itime, kr.first.rung);
			if (time_to_double(itime) >= opts.t_max) {
				oi = opts.nout + 1;
				do_out = true;
			}
			kr = solve_gravity(root_ptr, min_rung(itime), do_out);
			if (do_out) {
				last_epot = epot;
				epot = kr.first.stats.pot;
				output_particles(kr.first.out, std::string("parts.") + std::to_string(oi - 1) + ".silo");
				if (opts.groups) {
					groups_output(oi - 1);
				}
			}
			t = time_to_double(itime);
			dt = rung_to_dt(kr.first.rung);
			iter++;
			if (iter > 10 && opts.glass || t >= opts.t_max) {
				if (std::abs(epot / last_epot - 1.0) < 5.0e-5 && ekin < 5.0e-11) {
					printf("Writing glass file\n");
					part_vect_write_glass();
					printf("Glass file written\n");
					break;
				}
			}
		}
		show();
//	root_ptr->output(t, oi);
	}
	return hpx::finalize();
}

#ifndef HPX_LITE
int main(int argc, char *argv[]) {

	std::vector<std::string> cfg = { "hpx.commandline.allow_unknown=1" };

	hpx::init(argc, argv, cfg);
}

#endif
