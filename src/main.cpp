#include <hpx/hpx_init.hpp>

#include <tigergrav/initialize.hpp>
#include <tigergrav/options.hpp>
#include <tigergrav/particle.hpp>
#include <tigergrav/gravity.hpp>
#include <tigergrav/tree.hpp>

#include <fenv.h>

int hpx_main(int argc, char *argv[]) {

	feenableexcept(FE_DIVBYZERO);
	feenableexcept(FE_INVALID);
	feenableexcept(FE_OVERFLOW);

	options opts;
	opts.process_options(argc, argv);

	if (opts.ewald) {
		init_ewald();
	}
	auto parts = initial_particle_set(opts.problem, opts.problem_size);

	range root_box;
	for (int dim = 0; dim < NDIM; dim++) {
		root_box.min[dim] = 0.0;
		root_box.max[dim] = 1.0;
	}
	printf("Forming tree\n");
	tree_ptr root_ptr = tree::new_(root_box, parts.begin(), parts.end());
	printf("Done forming tree\n");

	float t = 0.0;
	int iter = 0;
	float dt;
#ifndef GLOBAL_DT
	rung_type rung;
	time_type itime = 0;
#endif

	const auto timer = []() {
		return std::chrono::duration_cast < std::chrono::milliseconds > (std::chrono::system_clock::now().time_since_epoch()).count() / 1000.0;
	};

	const auto tstart = timer();

	const auto show_stats = [&](stats s) {
		if (iter % 25 == 0) {
			printf("%4s %13s %13s %13s %13s %13s %13s ", "i", "t", "dt", "ek", "px", "py", "pz");
#ifdef STORE_G
		printf("%13s %13s %13s %13s %13s %13s ", "ep", "ax", "ay", "az", "etot", "virial");
#endif
#ifndef GLOBAL_DT
		printf("%9s ", "itime");
		printf("%9s ", "max rung");
		printf("%9s ", "min active");
#endif
		printf("%13s\n", "GFLOPS");
	}
	printf("%4i %13.6e %13.6e %13.6e %13.6e %13.6e %13.6e ", iter, t, dt, s.kin_tot, s.mom_tot[0], s.mom_tot[1], s.mom_tot[2]);
#ifdef STORE_G
		printf("%13.6e %13.6e %13.6e %13.6e %13.6e %13.6e ", s.pot_tot, s.acc_tot[0], s.acc_tot[1], s.acc_tot[2], s.ene_tot, s.virial_err);
#endif
#ifndef GLOBAL_DT
		printf("%9x ", (int) itime);
		printf("%9i ", (int) rung);
		printf("%9i ", (int) min_rung(itime));
#endif
		printf("%13.6e \n", s.flop / (timer() - tstart + std::numeric_limits<float>::min()) / std::pow(1024, 3));
	};

	const auto solve_gravity = [&]() {

	};

	int oi = 0;
	root_ptr->compute_monopoles();
#ifdef GLOBAL_DT
	dt = root_ptr->compute_gravity(std::vector<tree_ptr>(1, root_ptr), std::vector<source>(),std::vector<tree_ptr>(1, root_ptr), std::vector<source>());
#else
	const auto mrung = min_rung(0);
	root_ptr->active_particles(mrung);
	rung = root_ptr->kick(std::vector<tree_ptr>(1, root_ptr), std::vector<source>(), std::vector<tree_ptr>(1, root_ptr), std::vector<source>(),
			mrung);
	dt = rung_to_dt(rung);
#endif
	while (t < opts.t_max) {
		show_stats(root_ptr->statistics());
		if (t / opts.dt_max >= oi) {
			printf("output %i\n", oi);
			root_ptr->output(t, oi);
			oi++;
		}
#ifdef GLOBAL_DT
		root_ptr->kick(0.5 * dt);
		root_ptr->drift(dt);
		root_ptr = tree::new_(root_box, parts.begin(), parts.end());
		root_ptr->compute_monopoles();
		const float next_dt = root_ptr->compute_gravity(std::vector<tree_ptr>(1, root_ptr), std::vector<source>(),std::vector<tree_ptr>(1, root_ptr), std::vector<source>());
		root_ptr->kick(0.5 * dt);
		t += dt;
		dt = next_dt;
#else
		root_ptr->drift(dt);
		root_ptr = tree::new_(root_box, parts.begin(), parts.end());
		root_ptr->compute_monopoles();
		itime = inc(itime, rung);
		const auto mrung = min_rung(itime);
		root_ptr->active_particles(mrung);
		rung = root_ptr->kick(std::vector<tree_ptr>(1, root_ptr), std::vector<source>(), std::vector<tree_ptr>(1, root_ptr), std::vector<source>(),
				mrung);
		t = time_to_float(itime);
		dt = rung_to_dt(rung);
#endif
		iter++;
	}
	show_stats(root_ptr->statistics());
	root_ptr->output(t, oi);
	return hpx::finalize();
}

int main(int argc, char *argv[]) {

	std::vector<std::string> cfg = { "hpx.commandline.allow_unknown=1" };

	hpx::init(argc, argv, cfg);
}

