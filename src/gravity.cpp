#include <tigergrav/expansion.hpp>
#include <tigergrav/gravity.hpp>
#include <tigergrav/options.hpp>
#include <tigergrav/simd.hpp>

#include <hpx/include/async.hpp>

static const auto one = simd_float(1.0);
static const auto half = simd_float(0.5);
static const simd_float eps = simd_float(std::numeric_limits<float>::min());

float ewald_near_separation(const vect<float> x) {
	float d = 0.0;
	for (int dim = 0; dim < NDIM; dim++) {
		const float absx = std::abs(x[dim]);
		float this_d = std::min(absx, (float) 1.0 - absx);
		d += this_d * this_d;
	}
	return std::sqrt(d);
}

float ewald_far_separation(const vect<float> x, float r) {
	float d = 0.0;
	float maxd = 0.0;
	static const auto opts = options::get();
	const float cutoff_radius = opts.theta * 0.125;
	vect<float> y;
	for (int dim = 0; dim < NDIM; dim++) {
		const float absx = std::abs(x[dim]);
		const float this_d = std::min(absx, (float) 1.0 - absx);
		d += this_d * this_d;
		maxd = std::max(this_d, maxd);
		y[dim] = this_d;
	}
	float q;
	if (maxd > 0.0) {
		y = y / maxd;
		q = abs(y);
	} else {
		q = 0.0;
	}
	return std::max(std::sqrt(d), std::max(float(q * 0.5) - float(2 * r), 2 * cutoff_radius));
}

std::uint64_t gravity_PP_direct(std::vector<force> &f, const std::vector<vect<float>> &x, std::vector<vect<float>> &y) {
	if (x.size() == 0) {
		return 0;
	}
	std::uint64_t flop = 0;
	static const auto opts = options::get();
	static const simd_float M(1.0 / opts.problem_size);
	static const bool ewald = opts.ewald;
	static const auto h = opts.soft_len;
	static const auto h2 = h * h;
	static const simd_float huge = std::numeric_limits<float>::max() / 10.0;
	static const simd_float tiny = std::numeric_limits<float>::min() * 10.0;
	static const simd_float H(h);
	static const simd_float H2(h * h);
	static const simd_float Hinv(1.0 / h);
	static const simd_float H3inv(1.0 / h / h / h);
	static const simd_float H5inv(1.0 / h / h / h / h / h);
	static const simd_float H4(h2 * h2);
	static const auto _15o8 = simd_float(15.0 / 8.0);
	static const auto _10o8 = simd_float(10.0 / 8.0);
	static const auto _3o8 = simd_float(3.0 / 8.0);
	static const auto _2p5 = simd_float(2.5);
	static const auto _1p5 = simd_float(1.5);
	static const auto zero = simd_float(0);
	vect<simd_float> X, Y;
	std::vector<vect<simd_float>> G(x.size(), vect<float>(0.0));
	std::vector<simd_float> Phi(x.size(), 0.0);
	const auto cnt1 = y.size();
	const auto cnt2 = ((cnt1 - 1 + simd_float::size()) / simd_float::size()) * simd_float::size();
	y.resize(cnt2);
	for (int j = cnt1; j < cnt2; j++) {
		y[j] = vect<float>(1.0e+10);
	}
	for (int j = 0; j < cnt1; j += simd_float::size()) {
		for (int k = 0; k < simd_float::size(); k++) {
			for (int dim = 0; dim < NDIM; dim++) {
				Y[dim][k] = y[j + k][dim];
			}
		}
		for (int i = 0; i < x.size(); i++) {
			for (int dim = 0; dim < NDIM; dim++) {
				X[dim] = simd_float(x[i][dim]);
			}

			vect<simd_float> dX = X - Y;             																		// 3 OP
			if (ewald) {
				for (int dim = 0; dim < NDIM; dim++) {
					const auto absdx = abs(dX[dim]);																		// 3 OP
					dX[dim] = copysign(min(absdx, one - absdx), dX[dim] * (half - absdx));  								// 15 OP
				}
			}
			const simd_float r2 = dX.dot(dX);																				// 1 OP
			const simd_float zero_mask = r2 > simd_float(0);
			const simd_float rinv = zero_mask * rsqrt(r2 + tiny);       													// 2 OP
			const simd_float r = sqrt(r2);																					// 1 OP
			const simd_float rinv3 = rinv * rinv * rinv;   																	// 2 OP
			const simd_float sw_far = H < r;   																				// 1 OP
			const simd_float sw_near = simd_float(1) - sw_far;																// 1 OP
			const simd_float roverh = min(r * Hinv, 1);																		// 2 OP
			const simd_float roverh2 = roverh * roverh;																		// 1 OP
			const simd_float roverh4 = roverh2 * roverh2;																	// 1 OP
			const simd_float fnear = (_2p5 - _1p5 * roverh2) * H3inv;														// 3 OP
			for (int dim = 0; dim < NDIM; dim++) {
				G[i][dim] -= dX[dim] * M * (sw_far * rinv3 + sw_near * fnear);  											// 18 OP
			}
			const auto tmp = M * zero_mask; 	            																// 1 OP
			const auto near = (_15o8 - _10o8 * roverh2 + _3o8 * roverh4) * Hinv;											// 5 OP
			Phi[i] -= (sw_far * rinv + sw_near * near) * tmp;		        												// 5 OP
		}
	}
	for (int i = 0; i < x.size(); i++) {
		for (int dim = 0; dim < NDIM; dim++) {
			f[i].g[dim] += G[i][dim].sum();
		}
		f[i].phi += Phi[i].sum();
	}
	return 66 * cnt1 * x.size();
}

std::uint64_t gravity_PC_direct(std::vector<force> &f, const std::vector<vect<float>> &x, std::vector<multi_src> &y) {
	if (x.size() == 0) {
		return 0;
	}
	std::uint64_t flop = 0;
	static const auto opts = options::get();
	static const auto h = opts.soft_len;
	static const bool ewald = opts.ewald;
	static const auto h2 = h * h;
	static const simd_float H(h);
	static const simd_float H2(h2);
	vect<simd_float> X, Y;
	multipole<simd_float> M;
	std::vector<vect<simd_float>> G(x.size(), vect<float>(0.0));
	std::vector<simd_float> Phi(x.size(), 0.0);
	const auto cnt1 = y.size();
	const auto cnt2 = ((cnt1 - 1 + simd_float::size()) / simd_float::size()) * simd_float::size();
	y.resize(cnt2);
	for (int j = cnt1; j < cnt2; j++) {
		y[j].m = 0.0;
		y[j].x = vect<float>(1.0);
	}
	for (int j = 0; j < cnt1; j += simd_float::size()) {
		for (int k = 0; k < simd_float::size(); k++) {
			for (int n = 0; n < MP; n++) {
				M[n][k] = y[j + k].m[n];
			}
			for (int dim = 0; dim < NDIM; dim++) {
				Y[dim][k] = y[j + k].x[dim];
			}
		}
		for (int i = 0; i < x.size(); i++) {
			for (int dim = 0; dim < NDIM; dim++) {
				X[dim] = simd_float(x[i][dim]);
			}

			vect<simd_float> dX = X - Y;             		// 3 OP
			if (ewald) {
				for (int dim = 0; dim < NDIM; dim++) {
					const auto absdx = abs(dX[dim]);										// 3 OP
					dX[dim] = copysign(min(absdx, one - absdx), dX[dim] * (half - absdx));  // 15 OP
				}
			}
			auto this_f = multipole_interaction(M, dX);

			for (int dim = 0; dim < NDIM; dim++) {
				G[i][dim] += this_f.second[dim];  // 6 OP
			}
			Phi[i] += this_f.first;		        // 2 OP
		}
	}
	for (int i = 0; i < x.size(); i++) {
		for (int dim = 0; dim < NDIM; dim++) {
			f[i].g[dim] += G[i][dim].sum();
		}
		f[i].phi += Phi[i].sum();
	}
	return 26 * cnt1 * x.size();
}

std::uint64_t gravity_CC_direct(expansion<float> &L, const vect<ireal> &x, std::vector<multi_src> &y) {
	if (y.size() == 0) {
		return 0;
	}
	static const auto one = simd_real(1.0);
	static const auto half = simd_real(0.5);
	std::uint64_t flop = 0;
	static const auto opts = options::get();
	static const bool ewald = opts.ewald;
	static const auto h = opts.soft_len;
	static const auto h2 = h * h;
	static const simd_real H(h);
	static const simd_real H2(h2);
	vect<simd_real> X, Y;
	multipole<simd_real> M;
	expansion<simd_real> Lacc;
	Lacc = simd_real(0);
	const auto cnt1 = y.size();
	const auto cnt2 = ((cnt1 - 1 + simd_real::size()) / simd_real::size()) * simd_real::size();
	y.resize(cnt2);
	for (int j = cnt1; j < cnt2; j++) {
		y[j].m = 0.0;
		y[j].x = vect<ireal>(1.0);
	}

	for (int dim = 0; dim < NDIM; dim++) {
		X[dim] = simd_real(x[dim]);
	}
	for (int j = 0; j < cnt1; j += simd_real::size()) {
		for (int k = 0; k < simd_real::size(); k++) {
			for (int n = 0; n < MP; n++) {
				M[n][k] = y[j + k].m[n];
			}
			for (int dim = 0; dim < NDIM; dim++) {
				Y[dim][k] = y[j + k].x[dim];
			}
		}
		vect<simd_real> dX = X - Y;             										// 3 OP
		if (ewald) {
			for (int dim = 0; dim < NDIM; dim++) {
				const auto absdx = abs(dX[dim]);										// 3 OP
				dX[dim] = copysign(min(absdx, one - absdx), dX[dim] * (half - absdx));  // 15 OP
			}
		}
		multipole_interaction(Lacc, M, dX);												// 703 OP
	}

	for (int i = 0; i < LP; i++) {
		L[i] += Lacc[i].sum();
	}
	return 706 * cnt1;
}

std::uint64_t gravity_CP_direct(expansion<float> &L, const vect<ireal> &x, std::vector<vect<float>> &y) {
	if (y.size() == 0) {
		return 0;
	}
	static const auto one = simd_real(1.0);
	static const auto half = simd_real(0.5);
	std::uint64_t flop = 0;
	static const auto opts = options::get();
	static const auto m = 1.0 / opts.problem_size;
	static const bool ewald = opts.ewald;
	static const auto h = opts.soft_len;
	static const auto h2 = h * h;
	static const simd_real H(h);
	static const simd_real H2(h2);
	vect<simd_real> X, Y;
	simd_real M;
	expansion<simd_real> Lacc;
	Lacc = simd_real(0);
	const auto cnt1 = y.size();
	const auto cnt2 = ((cnt1 - 1 + simd_real::size()) / simd_real::size()) * simd_real::size();
	y.resize(cnt2);
	for (int j = cnt1; j < cnt2; j++) {
		y[j] = vect<ireal>(1.0);
	}

	for (int dim = 0; dim < NDIM; dim++) {
		X[dim] = simd_real(x[dim]);
	}
	M = m;
	for (int j = 0; j < cnt1; j += simd_real::size()) {
		for (int k = 0; k < simd_real::size(); k++) {
			for (int dim = 0; dim < NDIM; dim++) {
				Y[dim][k] = y[j + k][dim];
			}
		}
		if (j + simd_real::size() > cnt1) {
			for (int k = cnt1; k < cnt2; k++) {
				M[k - j] = 0.0;
			}
		}
		vect<simd_real> dX = X - Y;             										// 3 OP
		if (ewald) {
			for (int dim = 0; dim < NDIM; dim++) {
				const auto absdx = abs(dX[dim]);										// 3 OP
				dX[dim] = copysign(min(absdx, one - absdx), dX[dim] * (half - absdx));  // 15 OP
			}
		}
		multipole_interaction(Lacc, M, dX);												// 401 OP
	}

	for (int i = 0; i < LP; i++) {
		L[i] += Lacc[i].sum();
	}
	return 422 * cnt1;
}

std::uint64_t gravity_PP_ewald(std::vector<force> &f, const std::vector<vect<float>> &x, std::vector<vect<float>> &y) {
	if (x.size() == 0) {
		return 0;
	}
	std::uint64_t flop = 0;

	static const auto one = simd_real(1.0);
	static const auto half = simd_real(0.5);

	static const auto opts = options::get();
	static const simd_real M(1.0 / opts.problem_size);
	simd_real mask;
	static const bool ewald = opts.ewald;
	static const auto h = opts.soft_len;
	static const auto h2 = h * h;
	static const ewald_indices indices_real(5);
	static const ewald_indices indices_four(9);
	static const periodic_parts periodic;
	vect<simd_real> X, Y;
	std::vector<vect<simd_real>> G(x.size(), vect<float>(0.0));
	std::vector<simd_real> Phi(x.size(), 0.0);
	const auto cnt1 = y.size();
	const auto cnt2 = ((cnt1 - 1 + simd_real::size()) / simd_real::size()) * simd_real::size();
	y.resize(cnt2);
	for (int j = cnt1; j < cnt2; j++) {
		y[j] = vect<float>(0.0);
	}
	for (int J = 0; J < cnt1; J += simd_real::size()) {
		for (int k = 0; k < simd_real::size(); k++) {
			for (int dim = 0; dim < NDIM; dim++) {
				Y[dim][k] = y[J + k][dim];
			}
		}
		mask = simd_real(1);
		for (int k = std::min((int) (cnt1 - J), (int) simd_real::size()); k < simd_real::size(); k++) {
			mask[k] = 0.0;
		}
		for (int I = 0; I < x.size(); I++) {
			for (int dim = 0; dim < NDIM; dim++) {
				X[dim] = simd_real(x[I][dim]);
			}

			vect<simd_real> dX0 = X - Y;             		// 3 OP
			for (int dim = 0; dim < NDIM; dim++) {
				const auto absdx = abs(dX0[dim]);										// 3 OP
				dX0[dim] = copysign(min(absdx, one - absdx), dX0[dim] * (half - absdx));  // 15 OP
			}
			constexpr int nmax = 2;
			constexpr int hmax = 2;
			const float huge = std::numeric_limits<float>::max() / 10.0 / (nmax * nmax * nmax);
			const float tiny = std::numeric_limits<float>::min() * 10.0;
			vect<float> h;
			vect<simd_real> n;
			simd_real phi = 0.0;
			vect<simd_real> g;
			g = simd_real(0);
			for (int i = 0; i < indices_real.size(); i++) {
				h = indices_real[i];
				n = h;
				const vect<simd_real> dx = dX0 - n;                          // 3 OP
				const simd_real r2 = dx.dot(dx);
				const simd_real r = sqrt(r2);                      // 5 OP
				const simd_real rinv = r / (r2 + tiny);
				const simd_real r2inv = rinv * rinv;
				const simd_real r3inv = r2inv * rinv;
				const simd_real erfc = one - erf(2.0 * r);
				const simd_real d0 = -erfc * rinv;
				const simd_real expfactor = 4.0 * r * exp(-4.0 * r2) / sqrt(M_PI);
				const simd_real d1 = (expfactor + erfc) * r3inv;
				phi += d0; // 6 OP
				for (int a = 0; a < NDIM; a++) {
					g[a] -= (dX0[a] - n[a]) * d1;
				}
			}
			for (int i = 0; i < indices_four.size(); i++) {
				const expansion<float> &H = periodic[i];
				h = indices_four[i];
				const float h2 = h.dot(h);                     // 5 OP
				simd_real hdotdx = simd_real(0);
				for (int dim = 0; dim < NDIM; dim++) {
					hdotdx += dX0[dim] * h[dim];
				}
				const simd_real omega = 2.0 * M_PI * hdotdx;
				simd_real s, c;
				sincos(omega, &s, &c);
				phi += H() * c;
				for (int dim = 0; dim < NDIM; dim++) {
					g[dim] -= H(dim) * s;
				}
			}
			const simd_real r = abs(dX0);
			const simd_real rinv = r / (r * r + tiny);
			phi = simd_real(M_PI / 4.0) + phi + rinv;
			const simd_real sw = min(huge * r, 1.0);
			phi = 2.8372975 * (simd_real(1.0) - sw) + phi * sw;
			for (int dim = 0; dim < NDIM; dim++) {
				g[dim] = g[dim] + dX0[dim] * rinv * rinv * rinv;
			}
			Phi[I] += M * phi * mask;
			G[I] += g * M * mask;
		}
	}
	for (int i = 0; i < x.size(); i++) {
		for (int dim = 0; dim < NDIM; dim++) {
			f[i].g[dim] += G[i][dim].sum();
		}
		f[i].phi += Phi[i].sum();
	}
	return 26 * cnt1 * x.size();
}

std::uint64_t gravity_PC_ewald(std::vector<force> &f, const std::vector<vect<float>> &x, std::vector<multi_src> &y) {
	if (x.size() == 0) {
		return 0;
	}
	std::uint64_t flop = 0;

	static const auto one = simd_real(1.0);
	static const auto half = simd_real(0.5);
	static const auto opts = options::get();
	static const auto h = opts.soft_len;
	static const bool ewald = opts.ewald;
	static const auto h2 = h * h;
	static const simd_real H(h);
	static const simd_real H2(h2);
	vect<simd_real> X, Y;
	multipole<simd_real> M;
	std::vector<vect<simd_real>> G(x.size(), vect<float>(0.0));
	std::vector<simd_real> Phi(x.size(), 0.0);
	const auto cnt1 = y.size();
	const auto cnt2 = ((cnt1 - 1 + simd_real::size()) / simd_real::size()) * simd_real::size();
	y.resize(cnt2);
	for (int j = cnt1; j < cnt2; j++) {
		y[j].m = 0.0;
		y[j].x = vect<float>(1.0);
	}
	for (int j = 0; j < cnt1; j += simd_real::size()) {
		for (int k = 0; k < simd_real::size(); k++) {
			for (int n = 0; n < MP; n++) {
				M[n][k] = y[j + k].m[n];
			}
			for (int dim = 0; dim < NDIM; dim++) {
				Y[dim][k] = y[j + k].x[dim];
			}
		}
		for (int i = 0; i < x.size(); i++) {
			for (int dim = 0; dim < NDIM; dim++) {
				X[dim] = simd_real(x[i][dim]);
			}

			vect<simd_real> dX = X - Y;             		// 3 OP
			if (ewald) {
				for (int dim = 0; dim < NDIM; dim++) {
					const auto absdx = abs(dX[dim]);										// 3 OP
					dX[dim] = copysign(min(absdx, one - absdx), dX[dim] * (half - absdx));  // 15 OP
				}
			}
			auto this_f = multipole_interaction(M, dX, true);	//42826

			for (int dim = 0; dim < NDIM; dim++) {
				G[i][dim] += this_f.second[dim];  // 6 OP
			}
			Phi[i] += this_f.first;		        // 2 OP
		}
	}
	for (int i = 0; i < x.size(); i++) {
		for (int dim = 0; dim < NDIM; dim++) {
			f[i].g[dim] += G[i][dim].sum();
		}
		f[i].phi += Phi[i].sum();
	}
	return 42875 * cnt1 * x.size();
}

std::uint64_t gravity_CC_ewald(expansion<float> &L, const vect<ireal> &x, std::vector<multi_src> &y) {
	if (y.size() == 0) {
		return 0;
	}
	static const auto one = simd_real(1.0);
	static const auto half = simd_real(0.5);
	std::uint64_t flop = 0;
	static const auto opts = options::get();
	static const bool ewald = opts.ewald;
	static const auto h = opts.soft_len;
	static const auto h2 = h * h;
	static const simd_real H(h);
	static const simd_real H2(h2);
	vect<simd_real> X, Y;
	multipole<simd_real> M;
	expansion<simd_real> Lacc;
	Lacc = simd_real(0);
	const auto cnt1 = y.size();
	const auto cnt2 = ((cnt1 - 1 + simd_real::size()) / simd_real::size()) * simd_real::size();
	y.resize(cnt2);
	for (int j = cnt1; j < cnt2; j++) {
		y[j].m = 0.0;
		y[j].x = vect<ireal>(1.0);
	}

	for (int dim = 0; dim < NDIM; dim++) {
		X[dim] = simd_real(x[dim]);
	}
	for (int j = 0; j < cnt1; j += simd_real::size()) {
		for (int k = 0; k < simd_real::size(); k++) {
			for (int n = 0; n < MP; n++) {
				M[n][k] = y[j + k].m[n];
			}
			for (int dim = 0; dim < NDIM; dim++) {
				Y[dim][k] = y[j + k].x[dim];
			}
		}
		vect<simd_real> dX = X - Y;             										// 3 OP
		if (ewald) {
			for (int dim = 0; dim < NDIM; dim++) {
				const auto absdx = abs(dX[dim]);										// 3 OP
				dX[dim] = copysign(min(absdx, one - absdx), dX[dim] * (half - absdx));  // 15 OP
			}
		}
		multipole_interaction(Lacc, M, dX, true);												// 43009 OP
	}

	for (int i = 0; i < LP; i++) {
		L[i] += Lacc[i].sum();
	}
	return 43030 * cnt1;
}

std::uint64_t gravity_CP_ewald(expansion<float> &L, const vect<ireal> &x, std::vector<vect<float>> &y) {
	if (y.size() == 0) {
		return 0;
	}
	static const auto one = simd_real(1.0);
	static const auto half = simd_real(0.5);
	std::uint64_t flop = 0;
	static const auto opts = options::get();
	static const auto m = 1.0 / opts.problem_size;
	static const bool ewald = opts.ewald;
	static const auto h = opts.soft_len;
	static const auto h2 = h * h;
	static const simd_real H(h);
	static const simd_real H2(h2);
	vect<simd_real> X, Y;
	simd_real M;
	expansion<simd_real> Lacc;
	Lacc = simd_real(0);
	const auto cnt1 = y.size();
	const auto cnt2 = ((cnt1 - 1 + simd_real::size()) / simd_real::size()) * simd_real::size();
	y.resize(cnt2);
	for (int j = cnt1; j < cnt2; j++) {
		y[j] = vect<ireal>(1.0);
	}

	for (int dim = 0; dim < NDIM; dim++) {
		X[dim] = simd_real(x[dim]);
	}
	M = m;
	for (int j = 0; j < cnt1; j += simd_real::size()) {
		for (int k = 0; k < simd_real::size(); k++) {
			for (int dim = 0; dim < NDIM; dim++) {
				Y[dim][k] = y[j + k][dim];
			}
		}
		if (j + simd_real::size() > cnt1) {
			for (int k = cnt1; k < cnt2; k++) {
				M[k - j] = 0.0;
			}
		}
		vect<simd_real> dX = X - Y;             										// 3 OP
		if (ewald) {
			for (int dim = 0; dim < NDIM; dim++) {
				const auto absdx = abs(dX[dim]);										// 3 OP
				dX[dim] = copysign(min(absdx, one - absdx), dX[dim] * (half - absdx));  // 15 OP
			}
		}
		multipole_interaction(Lacc, M, dX, true);										// 42707 OP
	}

	for (int i = 0; i < LP; i++) {
		L[i] += Lacc[i].sum();
	}
	return 42728 * cnt1;
}

