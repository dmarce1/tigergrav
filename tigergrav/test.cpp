#include <hpx/hpx_init.hpp>
#include <immintrin.h>
#include <tigergrav/range.hpp>
#include <tigergrav/vect.hpp>



int hpx_main(int argc, char *argv[]) {
	box_id_type id = 0x1E;
	auto r = box_id_to_range(id);
	for (int d = 0; d < NDIM; d++) {
		printf("%e %e\n", r.min[d], r.max[d]);
	}

	return hpx::finalize();
}
int main(int argc, char *argv[]) {
	std::vector<std::string> cfg = { "hpx.commandline.allow_unknown=1" };
	hpx::init(argc, argv, cfg);
}

