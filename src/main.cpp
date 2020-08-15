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

#include <algorithm>

#include <fenv.h>

double timer(void) {
	return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count() / 1000.0;
}

kick_return solve_gravity(tree_client root_ptr, rung_type mrung, bool do_out) {
	auto start = timer();
	static const auto opts = options::get();
	root_ptr.compute_multipoles(mrung, do_out, 0);
	auto root_list = std::vector<check_item>(1, root_ptr.get_check_item());
	if (do_out && ! opts.solver_test) {
		auto tstart = timer();
		printf( "Finding groups\n");
		part_vect_init_groups();
		tree::set_theta(1.0);
		do {
		} while (root_ptr.find_groups(root_list,0));
		groups_reset();
		tree::set_theta(opts.theta);
		printf( "Done finding groups in %e seconds\n", timer() - tstart);

	}
//	printf("Multipoles took %e seconds\n", timer() - start);
	start = timer();
	expansion<double> L;
	L = 0.0;
	auto rc = root_ptr.kick_fmm(root_list, root_list, { { 0.5, 0.5, 0.5 } }, L, mrung, do_out, 0);
	if( do_out&& ! opts.solver_test) {
		groups_finish1();
		part_vect_find_groups2();
		groups_finish2();
	}
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

	float tau_max;
	float dtau_out;
	if (opts.cosmic) {
		cosmos cinit;
		cinit.advance_to_scale(1.0 / (1.0 + opts.z0));
		tau_max = -cinit.get_tau();
		const auto a0 = cinit.get_scale();
		const auto adot0 = cinit.get_Hubble() * a0;
		printf("Inializing with a = %e, adot = %e, tau = %e\n", a0, adot0, tau_max);
		cosmo_init(a0, adot0);
	} else {
		cosmo_init(1.0, 0.0);
		tau_max = opts.t_max;
	}
	dtau_out = opts.t_max / opts.nout;

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
		printf("%11s %11s %11s %11s %11s %11s %11s %11s\n", "theta", "time", "GFLOPS", "error", "error99", "gx", "gy", "gz");
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
			printf("%11.4e %11.4e %11.4e %11.4e %11.4e %11.4e %11.4e %11.4e \n", theta, stop - start, flops, err.err, err.err99, err.g[0], err.g[1], err.g[2]);
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

		float pec_energy = 0.0;
		double etot0;
		float last_ekin;
		float ekin;
		float epot;
		float last_epot;
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
				printf("%4s %11s %11s %11s %11s %11s %11s %11s %9s %9s %9s %11s ", "i", "t", "tau", "z", "a", "H", "adotdot", "dt", "itime", "max rung",
						"min act.", "GFLOP");
				printf(" %11s %11s %11s %11s  %11s %11s\n", "g", "p", "epot", "ekin", "epec", "etot");
			}
			printf("%4i %11.4e %11.4e %11.4e %11.4e %11.4e %11.4e %11.4e  ", iter, t, cosmo_time(), z, a, cosmo_Hubble(), cosmo_adoubledot(), dt);
			printf("%9x ", (int) itime);
			printf("%9i ", (int) kr.rung);
			printf("%9i ", (int) min_rung(itime));
			printf("%11.4e ", tree::get_flop() / (timer() - tstart + 1.0e-20) / pow(1024, 3));
//			tree::reset_flop();
//			tstart = timer();
			if (do_out) {
				printf("%11.4e ", abs(kr.stats.g));
				printf("%11.4e ", abs(kr.stats.p));
				const auto etot = a * (kr.stats.pot + kr.stats.kin) + pec_energy;
				if (first_show) {
					etot0 = etot;
					first_show = false;
				}
				const auto eden = std::max(a * ekin, std::abs(etot0));
				printf("%11.4e %11.4e %11.4e %11.4e %11.4e ", a * kr.stats.pot, a * ekin, pec_energy, etot,
						opts.glass ? (epot - last_epot) / epot : (etot - etot0) / eden);
			} else {
				printf("%11s ", "");
				printf("%11.4e ", abs(kr.stats.p));
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
		last_epot = epot = kr.stats.pot;
		ekin = kr.stats.kin;
		if (do_out) {
			output_particles(kr.out, "parts.0.silo");
			groups_output(0);
		}
		dt = rung_to_dt(kr.rung);
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
			if (time_to_double(itime) >= opts.t_max) {
				oi = opts.nout + 1;
				do_out = true;
			}
			kr = solve_gravity(root_ptr, min_rung(itime), do_out);
			if (do_out) {
				last_epot = epot;
				epot = kr.stats.pot;
				output_particles(kr.out, std::string("parts.") + std::to_string(oi - 1) + ".silo");
				groups_output(oi-1);
			}
			t = time_to_double(itime);
			dt = rung_to_dt(kr.rung);
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
