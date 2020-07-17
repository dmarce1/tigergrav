/*
 * test.cpp
 *
 *  Created on: Jul 14, 2020
 *      Author: dmarce1
 */

#include <hpx/hpx_init.hpp>
#include <immintrin.h>

#include <tigergrav/avx_mathfun.h>


int hpx_main(int argc, char *argv[]) {
	v8sf x;
	x[0] = 0.125;
	x[1] = 0.250;
	x[2] = 0.321;
	x[3] = -.412;
	x[4] = 2.341;
	x[5] = -10.342;
	x[6] = +0.00;
	x[7] = -0.01;
	auto y = exp256_ps(x);
	for( int i = 0; i < 8; i++) {
		printf( "%e %e %e %e\n", x[i], std::exp(x[i]), y[i], std::exp(x[i])- y[i]);
	}


	return hpx::finalize();
}
int main(int argc, char *argv[]) {
	std::vector<std::string> cfg = { "hpx.commandline.allow_unknown=1" };
	hpx::init(argc, argv, cfg);
}

