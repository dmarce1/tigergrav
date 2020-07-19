#include <hpx/hpx_init.hpp>
#include <immintrin.h>
#include <tigergrav/simd.hpp>



int hpx_main(int argc, char *argv[]) {
//	math_sincos sc;
//	double s, c;
//	for (double r = -10 * M_PI; r < -8 * M_PI; r += .1) {
//		sc(r, &s, &c);
//		__m256 _r = { r, r, r, r, r, r, r, r };
//		__m256 _s, _c;
//		sc(_r, &_s, &_c);
//		printf("%e %e %e %e\n", r, _c[7], std::sin(r), _c[7] - std::sin(r));
//	}
//	for (int i = 0; i < 100; i += 4) {
//		double x = i / 100.0;
//		auto y = func(_mm256_set_pd(x, x + 0.01, x + 0.02, x + 0.03));
//		printf("%e %e %e\n", x, y[0], sin(x));
//		printf("%e %e %e\n", x + 0.01, y[1], sin(x + 0.01));
//		printf("%e %e %e\n", x + 0.02, y[2], sin(x + 0.02));
//		printf("%e %e %e\n", x + 0.03, y[3], sin(x + 0.03));
//
//	}
	return hpx::finalize();
}
int main(int argc, char *argv[]) {
	std::vector < std::string > cfg = { "hpx.commandline.allow_unknown=1" };
	hpx::init(argc, argv, cfg);
}

