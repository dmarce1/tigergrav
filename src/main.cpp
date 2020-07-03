#include <hpx/hpx_init.hpp>

#include <tigergrav/initialize.hpp>
#include <tigergrav/options.hpp>
#include <tigergrav/particle.hpp>
#include <tigergrav/tree.hpp>

int hpx_main(int argc, char *argv[]) {

	options opts;
	opts.process_options(argc, argv);

	auto parts = initial_particle_set(opts.problem_size);

	range root_box;
	for (int dim = 0; dim < NDIM; dim++) {
		root_box.min[dim] = 0.0;
		root_box.max[dim] = 1.0;
	}
	printf("Forming tree\n");
	tree_ptr root_ptr = tree::new_(root_box, parts.begin(), parts.end());
	printf("Done forming tree\n");

	float t = 0.0;
	float dt;
	int oi = 0;
#ifdef GLOBAL_DT
	dt = root_ptr->compute_gravity(std::vector<tree_ptr>(1, root_ptr), std::vector<source>());
	while (t < opts.t_max) {
		if (t / opts.dt_max >= oi) {
			root_ptr->output(t, oi);
			oi++;
		}
		root_ptr->kick(0.5 * dt);
		root_ptr->drift(dt);
		const float next_dt = root_ptr->compute_gravity(std::vector<tree_ptr>(1, root_ptr), std::vector<source>());
		root_ptr->kick(0.5 * dt);
		t += dt;
		dt = next_dt;
	}
#else
	printf( "Variable dt not yet implemented\n");
#endif

	root_ptr->output(t, oi);
	return hpx::finalize();
}

int main(int argc, char *argv[]) {

	std::vector<std::string> cfg = { "hpx.commandline.allow_unknown=1" };

	hpx::init(argc, argv, cfg);
}

