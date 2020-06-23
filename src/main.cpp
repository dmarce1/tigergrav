#include <hpx/hpx_init.hpp>

#include <tigergrav/initialize.hpp>
#include <tigergrav/options.hpp>
#include <tigergrav/particle.hpp>
#include <tigergrav/tree.hpp>

int hpx_main(int argc, char *argv[]) {

	options opts;
	opts.process_options(argc, argv);

	auto parts = initial_particle_set(opts.problem_size);

	auto mid = bisect(parts.begin(), parts.end(), [](const particle &p) {
		return (double) p.x[0] < 0.5;
	});

	for (auto i = parts.begin(); i != mid; i++) {
		printf("%e\n", (double) i->x[0]);
	}
	printf("---------\n");
	for (auto i = mid; i != parts.end(); i++) {
		printf("%e\n", (double) i->x[0]);
	}

	return hpx::finalize();
}

int main(int argc, char *argv[]) {

	std::vector<std::string> cfg = { "hpx.commandline.allow_unknown=1" };

	hpx::init(argc, argv, cfg);
}

