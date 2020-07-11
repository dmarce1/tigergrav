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
	auto parts = initial_particle_set(opts.problem, opts.problem_size, opts.out_parts);

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
	kick_return kr;
	time_type itime = 0;

	const auto timer = []() {
		return std::chrono::duration_cast < std::chrono::milliseconds > (std::chrono::system_clock::now().time_since_epoch()).count() / 1000.0;
	};

	const auto tstart = timer();

	const auto system_cmd = [](std::string cmd) {
		if (system(cmd.c_str()) != 0) {
			printf("Unable to execute system command %s\n", cmd.c_str());
			abort();
		}
	};
	bool do_stats = true;
	bool do_out = true;
	const auto show = [&]() {
		if (iter % 25 == 0) {
			printf("%4s %13s %13s %9s %9s %9s %13s ", "i", "t", "dt", "itime", "max rung", "min act.", "GFLOP");
			printf(" %13s %13s %13s %13s %13s %13s %13s %13s %13s\n", "gx", "gy", "gz", "px", "py", "pz", "epot", "ekin", "etot");
		}
		printf("%4i %13.6e %13.6e  ", iter, t, dt);
		printf("%9x ", (int) itime);
		printf("%9i ", (int) kr.rung);
		printf("%9i ", (int) min_rung(itime));
		printf("%13.6e ", root_ptr->get_flop() / (timer() - tstart + 1.0e-20) / pow(1024, 3));
		if (do_stats) {
			for (int dim = 0; dim < NDIM; dim++) {
				printf("%13.6e ", kr.stats.g[dim]);
			}
			for (int dim = 0; dim < NDIM; dim++) {
				printf("%13.6e ", kr.stats.p[dim]);
			}
			printf("%13.6e %13.6e %13.6e ", kr.stats.pot, kr.stats.kin, kr.stats.pot + kr.stats.kin);
		}
		printf("\n");
	};
	int oi = 1;
	int si = 1;
	root_ptr->compute_monopoles();
	const auto mrung = min_rung(0);
	root_ptr->active_particles(mrung, do_out);
	kr = root_ptr->kick(std::vector<tree_ptr>(1, root_ptr), std::vector<source>(), std::vector<tree_ptr>(1, root_ptr), std::vector<source>(), mrung, do_stats,
			do_out);
	if (do_out) {
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
		if ((t + dt) / opts.dt_stat >= si) {
			do_stats = true;
			si++;
		} else {
			do_stats = false;
		}
		root_ptr->drift(dt);
		root_ptr = tree::new_(root_box, parts.begin(), parts.end());
		root_ptr->compute_monopoles();
		itime = inc(itime, kr.rung);
		const auto mrung = min_rung(itime);
		root_ptr->active_particles(mrung, do_out);
		kr = root_ptr->kick(std::vector<tree_ptr>(1, root_ptr), std::vector<source>(), std::vector<tree_ptr>(1, root_ptr), std::vector<source>(), mrung,
				do_stats, do_out);
		if (do_out) {
		}
		t = time_to_float(itime);
		dt = rung_to_dt(kr.rung);
		iter++;
	}
	show();
//	root_ptr->output(t, oi);
	return hpx::finalize();
}

int main(int argc, char *argv[]) {

	std::vector<std::string> cfg = { "hpx.commandline.allow_unknown=1" };

	hpx::init(argc, argv, cfg);
}

