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
	for (int i = 0; i < 2; i++) {
		printf("Forming tree\n");
		tree_ptr root_ptr = tree::new_(root_box, parts.begin(), parts.end());
		printf("Done forming tree\n");
	}
	return hpx::finalize();
}

int main(int argc, char *argv[]) {

	std::vector<std::string> cfg = { "hpx.commandline.allow_unknown=1" };

	hpx::init(argc, argv, cfg);
}

