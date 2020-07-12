#include <tigergrav/expansion.hpp>

expansion<ireal> expansion_factor;

__attribute((constructor))
static void init_factors() {
	expansion_factor = 0.0;
	expansion_factor() += 1.0;
	for (int a = 0; a < NDIM; ++a) {
		expansion_factor(a) += 1.0;
		for (int b = 0; b < NDIM; ++b) {
			expansion_factor(a, b) += 0.5;
			for (int c = 0; c < NDIM; ++c) {
				expansion_factor(a, b, c) += 1.0 / 6.0;
				for (int d = 0; d < NDIM; ++d) {
					expansion_factor(a, b, c, d) += 1.0 / 24.0;
				}
			}
		}
	}
}
