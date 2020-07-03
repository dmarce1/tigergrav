#include <tigergrav/gravity.hpp>
#include <tigergrav/options.hpp>
#include <tigergrav/simd.hpp>

std::uint64_t gravity(std::vector<force> &f, const std::vector<vect<float>> &x, const std::vector<source> &y) {
	std::uint64_t flop = 0;
	static const auto h = options::get().soft_len;
	static const auto h2 = h * h;
	static const simd_vector H(h);
	static const simd_vector H2(h2);
	for (int i = 0; i < x.size(); i += SIMD_LEN) {
		const auto vlen = std::min(SIMD_LEN, (int) (x.size() - i));
		vect<simd_vector> G(0.0);
		simd_vector Phi(0.0);
		auto &g = f[i].g;
		const auto cnt = y.size();
		vect<simd_vector> X, Y;
		for (int dim = 0; dim < NDIM; dim++) {
			for (int k = 0; k < vlen; k++) {
				X[dim][k] = x[i + k][dim];
			}
		}
		for (int j = 0; j < cnt; j++) {
			simd_vector M(y[j].m);
			for (int dim = 0; dim < NDIM; dim++) {
				Y[dim] = simd_vector(y[j].x[dim]);
			}
			const vect<simd_vector> dX = X - Y;                   // 3 OP
			const simd_vector r2 = dX.dot(dX);                           // 5 OP
			const simd_vector rinv = simd_vector(1.0) / sqrt(r2 + H2);           // 3 OP
			const simd_vector rinv3 = rinv * rinv * rinv;                // 2 OP
			G = G - dX * (M * rinv3);                                 // 7 OP
			Phi = Phi - M * rinv;
			flop += 22 * vlen;
		}
		for (int k = 0; k < vlen; k++) {
			f[i + k].phi = Phi[k];
			for (int dim = 0; dim < NDIM; dim++) {
				f[i + k].g[dim] = G[dim][k];
			}
		}
	}
	return flop;
}
