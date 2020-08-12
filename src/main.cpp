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
#include <tigergrav/cosmo.hpp>

#include <algorithm>

#include <fenv.h>

double timer(void) {
	return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count() / 1000.0;
}

kick_return solve_gravity(tree_client root_ptr, rung_type mrung, bool do_out) {
	auto start = timer();
	root_ptr.compute_multipoles(mrung, do_out, 0);
//	printf("Multipoles took %e seconds\n", timer() - start);
	start = timer();
	expansion<float> L;
	L = 0.0;
	auto root_list = std::vector<check_item>(1, root_ptr.get_check_item());
	auto rc = root_ptr.kick_fmm(root_list, root_list, { { 0.5, 0.5, 0.5 } }, L, mrung, do_out, 0);
//	printf("fmm took %e seconds\n", timer() - start);
	return rc;
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

	cosmos cinit;
	cinit.advance_to_time(-opts.t_max);

	const auto a0 = cinit.get_scale();
	const auto adot0 = cinit.get_Hubble() * a0;
	printf("Inializing with a = %e and adot = %e\n", a0, adot0);
	cosmo_init(a0, adot0);

	tree::set_theta(opts.theta);

	range root_box;
	for (int dim = 0; dim < NDIM; dim++) {
		root_box.min[dim] = 0.0;
		root_box.max[dim] = 1.0;
	}

	part_vect_init();

	if (opts.solver_test) {
		printf("Computing direct solution first\n");
		tree_client root_ptr = hpx::new_<tree>(hpx::find_here(), root_box, 0, opts.problem_size, 0).get();
		while (root_ptr.refine(0)) {
		}
		tree::set_theta(1e-10);
		auto kr = solve_gravity(root_ptr, min_rung(0), true);
		std::sort(kr.out.begin(), kr.out.end());
		const auto direct = kr.out;
		printf("%12s %12s %12s %12s %12s %12s %12s %12s\n", "theta", "time", "GFLOPS", "error", "error99", "gx", "gy", "gz");
		for (double theta = 1.0; theta >= 0.17; theta -= 0.1) {
			root_ptr = hpx::new_<tree>(hpx::find_here(), root_box, 0, opts.problem_size, 0).get();
			while (root_ptr.refine(0)) {
			}
			tree::set_theta(theta);
			tree::reset_flop();
			auto start = timer();
			kr = solve_gravity(root_ptr, min_rung(0), true);
			auto stop = timer();
			auto flops = tree::get_flop() / (stop - start + 1.0e-10) / std::pow(1024, 3);
			std::sort(kr.out.begin(), kr.out.end());
			const auto err = compute_error(kr.out, direct);
			printf("%12.5e %12.5e %12.5e %12.5e %12.5e %12.5e %12.5e %12.5e \n", theta, stop - start, flops, err.err, err.err99, err.g[0], err.g[1], err.g[2]);
		}
	} else {

		printf("Forming tree\n");
		auto tstart = timer();
		tree_client root_ptr = hpx::new_<tree>(hpx::find_here(), root_box, 0, opts.problem_size, 0).get();
		while (root_ptr.refine(0)) {
			printf("Refining\n");
		}
		printf("Done forming tree took %e seconds\n", timer() - tstart);

		double t = 0.0;
		int iter = 0;
		double dt;
		kick_return kr;
		time_type itime = 0;

		tstart = timer();

		const auto system_cmd = [](std::string cmd) {
			if (system(cmd.c_str()) != 0) {
				printf("Unable to execute system command %s\n", cmd.c_str());
				abort();
			}
		};
		float pec_energy = 0.0;
		double etot0;
		float last_ekin;
		float ekin;
		bool do_out = true;
		bool first_show = true;
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
				printf("%4s %12s %12s %12s %12s %12s %9s %9s %9s %12s ", "i", "t", "tau", "a", "H", "dt", "itime", "max rung", "min act.", "GFLOP");
				printf(" %12s %12s %12s %12s %12s %12s %12s %12s %12s %12s\n", "gx", "gy", "gz", "px", "py", "pz", "epot", "ekin", "epec", "etot");
			}
			printf("%4i %12.5e %12.5e %12.5e %12.5e %12.5e  ", iter, t, cosmo_time(), a, cosmo_Hubble(), dt);
			printf("%9x ", (int) itime);
			printf("%9i ", (int) kr.rung);
			printf("%9i ", (int) min_rung(itime));
			printf("%12.5e ", tree::get_flop() / (timer() - tstart + 1.0e-20) / pow(1024, 3));
//			tree::reset_flop();
//			tstart = timer();
			if (do_out) {
				for (int dim = 0; dim < NDIM; dim++) {
					printf("%12.5e ", kr.stats.g[dim]);
				}
				for (int dim = 0; dim < NDIM; dim++) {
					printf("%12.5e ", kr.stats.p[dim]);
				}
				const auto etot = a * (kr.stats.pot + kr.stats.kin) + pec_energy;
				if (first_show) {
					etot0 = etot;
					first_show = false;
				}
				const auto eden = std::max(a * ekin, 1.0e-20);
				printf("%12.5e %12.5e %12.5e %12.5e %12.5e ", a * kr.stats.pot, a * ekin, pec_energy, etot, (etot - etot0) / eden);
			} else {
				for (int dim = 0; dim < NDIM; dim++) {
					printf("%12s ", "");
				}
				for (int dim = 0; dim < NDIM; dim++) {
					printf("%12.5e ", kr.stats.p[dim]);
				}
				printf("%12s %12.5e %12.5e %12s ", "", a * ekin, pec_energy, "");
			}
//			printf("%f/%f %f/%f %f/%f %f/%f", istats.CC_direct_pct, istats.CC_ewald_pct, istats.CP_direct_pct, istats.CP_ewald_pct, istats.PC_direct_pct,
//					istats.PC_ewald_pct, istats.PP_direct_pct, istats.PP_ewald_pct);
			printf("\n");
			last_ekin = ekin;
		};
		int oi = 1;
		int si = 1;
		if (opts.ewald) {
			cosmo_advance(0.0);
		}
		kr = solve_gravity(root_ptr, min_rung(0), do_out);
		if (do_out) {
			output_particles(kr.out, "parts.0.silo");
		}
		dt = rung_to_dt(kr.rung);
		while (t < opts.t_max) {
			show();
			if ((t + dt) / opts.dt_out >= oi) {
				do_out = true;
				printf("Doing output\n");
				oi++;
			} else {
				do_out = false;
			}
			auto ts = timer();
//			printf( "Drifting\n");
			if (opts.ewald) {
				cosmo_advance(dt);
			}
			ekin = root_ptr.drift(dt);
//			printf("drift took %e seconds\n", timer() - ts);
			ts = timer();
			root_ptr = hpx::invalid_id;
//			printf( "Forming tree\n");
			root_ptr = hpx::new_<tree>(hpx::find_here(), root_box, 0, opts.problem_size, 0).get();
			while (root_ptr.refine(0)) {
			}
//			printf("Tree took %e seconds\n", timer() - ts);
			itime = inc(itime, kr.rung);
			kr = solve_gravity(root_ptr, min_rung(itime), do_out);
			if (do_out) {
				output_particles(kr.out, std::string("parts.") + std::to_string(oi - 1) + ".silo");
			}
			t = time_to_double(itime);
			dt = rung_to_dt(kr.rung);
			iter++;
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
