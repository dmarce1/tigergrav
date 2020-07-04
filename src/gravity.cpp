#include <tigergrav/gravity.hpp>
#include <tigergrav/options.hpp>
#include <tigergrav/simd.hpp>

float ewald_separation(const vect<float> x) {
	float d = 0.0;
	for (int dim = 0; dim < NDIM; dim++) {
		const float absx = std::abs(x[dim]);
		float this_d = std::min(absx, (float) 1.0 - absx);
		d += this_d * this_d;
	}
	return std::sqrt(d);
}

std::uint64_t gravity(std::vector<force> &f, const std::vector<vect<float>> &x, std::vector<source> &y) {
	if (x.size() == 0) {
		return 0;
	}
	std::uint64_t flop = 0;
	static const auto opts = options::get();
	static const bool ewald = opts.ewald;
	static const auto h = opts.soft_len;
	static const auto h2 = h * h;
	static const simd_vector H(h);
	static const simd_vector H2(h2);
	static const auto one = simd_vector(1.0);
	static const auto half = simd_vector(0.5);
	vect<simd_vector> X, Y;
	simd_vector M;
	std::vector<vect<simd_vector>> G(x.size(), vect<float>(0.0));
	std::vector<simd_vector> Phi(x.size(), 0.0);
	const auto cnt1 = y.size();
	const auto cnt2 = ((cnt1 - 1 + SIMD_LEN) / SIMD_LEN) * SIMD_LEN;
	y.resize(cnt2);
	for (int j = cnt1; j < cnt2; j++) {
		y[j].m = 0.0;
		y[j].x = vect<float>(1.0);
	}
	for (int j = 0; j < cnt1; j += SIMD_LEN) {
		for (int k = 0; k < SIMD_LEN; k++) {
			M[k] = y[j].m;
			for (int dim = 0; dim < NDIM; dim++) {
				Y[dim][k] = y[j + k].x[dim];
			}
		}
		for (int i = 0; i < x.size(); i++) {
			for (int dim = 0; dim < NDIM; dim++) {
				X[dim] = simd_vector(x[i][dim]);
			}

			vect<simd_vector> dX = X - Y;             		// 3 OP
			if (ewald) {
				for (int dim = 0; dim < NDIM; dim++) {
					const auto absdx = abs(dX[dim]);										// 3 OP
					dX[dim] = copysign(dX[dim] * (half - absdx), min(absdx, one - absdx));  // 15 OP
				}
			}
			const simd_vector r2 = dX.dot(dX);              // 5 OP
			const simd_vector rinv = rsqrt(r2 + H2);        // 2 OP
			const simd_vector rinv3 = rinv * rinv * rinv;   // 2 OP
			G[i] -= dX * (M * rinv3);                       // 7 OP
			Phi[i] -= M * rinv;								// 2 OP
		}
	}
	for (int i = 0; i < x.size(); i++) {
		for (int dim = 0; dim < NDIM; dim++) {
			f[i].g[dim] = G[i][dim].sum();
		}
		f[i].phi = Phi[i].sum();
	}
	return (21 + ewald ? 18 : 0) * cnt1 * x.size();
}

