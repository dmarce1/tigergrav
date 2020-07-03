
#ifndef __AVX__
#error 'Must have AVX or higher'
#endif

#include <hpx/hpx_init.hpp>

#include <tigergrav/initialize.hpp>
#include <tigergrav/options.hpp>
#include <tigergrav/particle.hpp>
#include <tigergrav/tree.hpp>

int hpx_main(int argc, char *argv[]) {

	options opts;
	opts.process_options(argc, argv);

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
			printf("%13s\n", "GFLOPS");
		}
		printf("%4i %13.6e %13.6e %13.6e %13.6e %13.6e %13.6e ", iter, t, dt, s.kin_tot, s.mom_tot[0], s.mom_tot[1], s.mom_tot[2]);
#ifdef STORE_G
		printf("%13.6e %13.6e %13.6e %13.6e %13.6e %13.6e ", s.pot_tot, s.acc_tot[0], s.acc_tot[1], s.acc_tot[2], s.ene_tot, s.virial_err);
#endif
		printf("%13.6e \n", s.flop / (timer() - tstart) / std::pow(1024, 3));
	};

	int oi = 0;
#ifdef GLOBAL_DT
	root_ptr->compute_monopoles();
	printf("Solving gravity\n");
	dt = root_ptr->compute_gravity(std::vector<tree_ptr>(1, root_ptr), std::vector<source>());
#endif
	while (t < opts.t_max) {
		show_stats(root_ptr->statistics());
#ifdef GLOBAL_DT
		if (t / opts.dt_max >= oi) {
			root_ptr->output(t, oi);
			oi++;
		}
		root_ptr->kick(0.5 * dt);
		root_ptr->drift(dt);
		root_ptr->compute_monopoles();
		const float next_dt = root_ptr->compute_gravity(std::vector<tree_ptr>(1, root_ptr), std::vector<source>());
		root_ptr->kick(0.5 * dt);
		t += dt;
		dt = next_dt;
#else
		printf( "Variable dt not yet implemented\n");
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

