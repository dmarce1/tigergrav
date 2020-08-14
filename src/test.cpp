#include <hpx/hpx_init.hpp>
#include <immintrin.h>
#include <tigergrav/simd.hpp>

int hpx_main(int argc, char *argv[]) {
//	simd_double x;
//	simd_double z;
//	for (double r = -2.0; r < 4.0; r += 0.01) {
//		x = simd_double(r);
//		const auto myy = erfexp(x,&z)[0];
//		const auto y = std::erf(r);
//		printf("%e %e %e %e\n", r, myy, y, std::abs((myy - y) / y));
//	}
//
	return hpx::finalize();
}
int main(int argc, char *argv[]) {
	std::vector<std::string> cfg = { "hpx.commandline.allow_unknown=1" };
	hpx::init(argc, argv, cfg);
}

