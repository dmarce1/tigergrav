#include <hpx/hpx_init.hpp>
#include <immintrin.h>
#include <tigergrav/simd.hpp>

float mysin(float x) {
	float coeffs[] = { -0.10132118,          // x
			0.0066208798,        // x^3
			-0.00017350505,       // x^5
			0.0000025222919,     // x^7
			-0.000000023317787,   // x^9
			0.00000000013291342, // x^11
			};
	float pi_major = 3.1415927;
	float pi_minor = -0.00000008742278;
	x = x - std::round(x / (2.0 * M_PI)) * (2.0 * M_PI);
	float x2 = x * x;
	float p11 = coeffs[5];
	float p9 = p11 * x2 + coeffs[4];
	float p7 = p9 * x2 + coeffs[3];
	float p5 = p7 * x2 + coeffs[2];
	float p3 = p5 * x2 + coeffs[1];
	float p1 = p3 * x2 + coeffs[0];
	return (x - pi_major - pi_minor) * (x + pi_major + pi_minor) * p1 * x;
}

simd_float mysin(simd_float x) {
	// From : http://mooooo.ooo/chebyshev-sine-approximation/
	simd_float coeffs[] = { simd_float(-0.10132118),          // x
	simd_float(0.0066208798),        // x^3
	simd_float(-0.00017350505),       // x^5
	simd_float(0.0000025222919),     // x^7
	simd_float(-0.000000023317787),   // x^9
	simd_float(0.00000000013291342), // x^11
			};
	simd_float pi_major(3.1415927);
	simd_float pi_minor(-0.00000008742278);
	x = x - round(x * (1.0 / (2.0 * M_PI))) * (2.0 * M_PI);
	simd_float x2 = x * x;
	simd_float p11 = coeffs[5];
	simd_float p9 = fma(p11, x2, coeffs[4]);
	simd_float p7 = fma(p9, x2, coeffs[3]);
	simd_float p5 = fma(p7, x2, coeffs[2]);
	simd_float p3 = fma(p5, x2, coeffs[1]);
	simd_float p1 = fma(p3, x2, coeffs[0]);
	return (x - pi_major - pi_minor) * (x + pi_major + pi_minor) * p1 * x;
}

simd_float mycos(simd_float x) {
	return mysin(x + simd_float(M_PI/2.0));
}

int hpx_main(int argc, char *argv[]) {
//	math_sincos sc;
	for (double r = -M_PI; r <= 20.0*M_PI; r += 0.1) {
		printf("%e %e %e %e\n", r, mycos(simd_float(r))[0], sin(r), mysin(r) - mysin(simd_float(r))[0]);
	}
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

