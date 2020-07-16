#include <tigergrav/expansion.hpp>
#include <tigergrav/gravity.hpp>
#include <tigergrav/options.hpp>
#include <tigergrav/simd.hpp>

#include <hpx/include/async.hpp>


static const auto one = simd_float(1.0);
static const auto half = simd_float(0.5);
static const simd_float eps = simd_float(std::numeric_limits<float>::min());

double ewald_near_separation(const vect<double> x) {
	double d = 0.0;
	for (int dim = 0; dim < NDIM; dim++) {
		const double absx = std::abs(x[dim]);
		double this_d = std::min(absx, (double) 1.0 - absx);
		d += this_d * this_d;
	}
	return std::sqrt(d);
}

double ewald_far_separation(const vect<double> x) {
	double d = 0.0;
	for (int dim = 0; dim < NDIM; dim++) {
		const double absx = std::abs(x[dim]);
		const double this_d = std::min(absx, (double) 1.0 - absx);
		d += this_d * this_d;
	}
	return std::max(std::sqrt(d), double(0.25));
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
	static const simd_float H(h);
	static const simd_float H2(h2);
	vect<simd_float> X, Y;
	std::vector<vect<simd_float>> nG(x.size(), vect<float>(0.0));
	std::vector<simd_float> nPhi(x.size(), 0.0);
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

			vect<simd_float> dX = X - Y;             		// 3 OP
			if (ewald) {
				for (int dim = 0; dim < NDIM; dim++) {
					const auto absdx = abs(dX[dim]);										// 3 OP
					dX[dim] = copysign(min(absdx, one - absdx), dX[dim] * (half - absdx));  // 15 OP
				}
			}
			simd_float r2 = dX[0] * dX[0];					// 1 OP
			r2 = fma(dX[1], dX[1], r2);						// 2 OP
			r2 = fma(dX[2], dX[2], r2);                     // 2 OP
			const simd_float rinv = rsqrt(r2 + H2);        // 2 OP
			const simd_float rinv3 = rinv * rinv * rinv;   // 2 OP
			for (int dim = 0; dim < NDIM; dim++) {
				const auto tmp = M * rinv3;					// 3 OP
				nG[i][dim] = fma(dX[dim], tmp, nG[i][dim]);  // 6 OP
			}
			const simd_float kill_zero = r2 / (r2 + eps);  // 2 OP
			const auto tmp = M * kill_zero; 	            // 1 OP
			nPhi[i] = fma(rinv, tmp, nPhi[i]);		        // 2 OP
		}
	}
	for (int i = 0; i < x.size(); i++) {
		for (int dim = 0; dim < NDIM; dim++) {
			f[i].g[dim] += -nG[i][dim].sum();
		}
		f[i].phi += -nPhi[i].sum();
	}
	return 26 * cnt1 * x.size();
}

template<class SIMD, bool EWALD>
std::uint64_t gravity_PC(std::vector<force> &f, const std::vector<vect<float>> &x, std::vector<multi_src> &y) {
	if (x.size() == 0) {
		return 0;
	}

	static const auto one = SIMD(1.0);
	static const auto half = SIMD(0.5);
	std::uint64_t flop = 0;
	static const auto opts = options::get();
	static const auto h = opts.soft_len;
	static const bool ewald = opts.ewald;
	static const auto h2 = h * h;
	static const SIMD H(h);
	static const SIMD H2(h2);
	vect<SIMD> X, Y;
	multipole<SIMD> M;
	std::vector<vect<SIMD>> G(x.size(), vect<float>(0.0));
	std::vector<SIMD> Phi(x.size(), 0.0);
	const auto cnt1 = y.size();
	const auto cnt2 = ((cnt1 - 1 + SIMD::size()) / SIMD::size()) * SIMD::size();
	y.resize(cnt2);
	for (int j = cnt1; j < cnt2; j++) {
		y[j].m = 0.0;
		y[j].x = vect<float>(1.0);
	}
	for (int j = 0; j < cnt1; j += SIMD::size()) {
		for (int k = 0; k < SIMD::size(); k++) {
			for( int n = 0; n < MP; n++) {
				M[n][k] = y[j+k].m[n];
			}
			for (int dim = 0; dim < NDIM; dim++) {
				Y[dim][k] = y[j + k].x[dim];
			}
		}
		for (int i = 0; i < x.size(); i++) {
			for (int dim = 0; dim < NDIM; dim++) {
				X[dim] = SIMD(x[i][dim]);
			}

			vect<SIMD> dX = X - Y;             		// 3 OP
			if (ewald) {
				for (int dim = 0; dim < NDIM; dim++) {
					const auto absdx = abs(dX[dim]);										// 3 OP
					dX[dim] = copysign(min(absdx, one - absdx), dX[dim] * (half - absdx));  // 15 OP
				}
			}
			auto this_f = multipole_interaction(M, dX);   // 42826 , 520

			for (int dim = 0; dim < NDIM; dim++) {
				G[i][dim] += this_f.second[dim];  // 3 OP
			}
			Phi[i] += this_f.first;		        // 1 OP
		}
	}
	for (int i = 0; i < x.size(); i++) {
		for (int dim = 0; dim < NDIM; dim++) {
			f[i].g[dim] += G[i][dim].sum();
		}
		f[i].phi += Phi[i].sum();
	}
	return (EWALD ? 42851 : 545) * cnt1 * x.size();
}

std::uint64_t gravity_PC_direct(std::vector<force> &f, const std::vector<vect<float>> &x, std::vector<multi_src> &y) {
	return gravity_PC<simd_real,false>(f,x,y);
}


std::uint64_t gravity_PC_ewald(std::vector<force> &f, const std::vector<vect<float>> &x, std::vector<multi_src> &y) {
	return gravity_PC<simd_double,true>(f,x,y);
}


template<class SIMD, bool EWALD>
std::uint64_t gravity_CC(expansion<double> &L, const vect<ireal> &x, std::vector<multi_src> &y) {
	if (y.size() == 0) {
		return 0;
	}
	static const auto one = SIMD(1.0);
	static const auto half = SIMD(0.5);
	static const auto ewald = options::get().ewald;
	std::uint64_t flop = 0;
	vect<SIMD> X, Y;
	multipole<SIMD> M;
	expansion<SIMD> Lacc;
	Lacc = SIMD(0);
	const auto cnt1 = y.size();
	const auto cnt2 = ((cnt1 - 1 + SIMD::size()) / SIMD::size()) * SIMD::size();
	y.resize(cnt2);
	for (int j = cnt1; j < cnt2; j++) {
		y[j].m = 0.0;
		y[j].x = vect<ireal>(1.0);
	}

	for (int dim = 0; dim < NDIM; dim++) {
		X[dim] = SIMD(x[dim]);
	}
	for (int j = 0; j < cnt1; j += SIMD::size()) {
		for (int k = 0; k < SIMD::size(); k++) {
			for (int n = 0; n < MP; n++) {
				M[n][k] = y[j + k].m[n];
			}
			for (int dim = 0; dim < NDIM; dim++) {
				Y[dim][k] = y[j + k].x[dim];
			}
		}
		vect<SIMD> dX = X - Y;             										// 3 OP
		if (ewald) {
			for (int dim = 0; dim < NDIM; dim++) {
				const auto absdx = abs(dX[dim]);										// 3 OP
				dX[dim] = copysign(min(absdx, one - absdx), dX[dim] * (half - absdx));  // 15 OP
			}
		}
		multipole_interaction(Lacc, M, dX, EWALD);												// 703 OP
	}

	for (int i = 0; i < LP; i++) {
		L[i] += Lacc[i].sum();
	}
	return (EWALD ? 43030 : 724) * cnt1;
}


std::uint64_t gravity_CC_direct(expansion<double> &L, const vect<ireal> &x, std::vector<multi_src> &y) {
	return gravity_CC<simd_real,false>(L,x,y);
}

std::uint64_t gravity_CC_ewald(expansion<double> &L, const vect<ireal> &x, std::vector<multi_src> &y) {
	return gravity_CC<simd_double,true>(L,x,y);
}

template<class SIMD, bool EWALD>
std::uint64_t gravity_CP(expansion<double> &L, const vect<ireal> &x, std::vector<vect<float>> &y) {
	if (y.size() == 0) {
		return 0;
	}
	static const auto one = SIMD(1.0);
	static const auto half = SIMD(0.5);
	std::uint64_t flop = 0;
	static const auto opts = options::get();
	static const auto m = 1.0 / opts.problem_size;
	static const bool ewald = opts.ewald;
	static const auto h = opts.soft_len;
	static const auto h2 = h * h;
	static const SIMD H(h);
	static const SIMD H2(h2);
	vect<SIMD> X, Y;
	SIMD M;
	expansion<SIMD> Lacc;
	Lacc = SIMD(0);
	const auto cnt1 = y.size();
	const auto cnt2 = ((cnt1 - 1 + SIMD::size()) / SIMD::size()) * SIMD::size();
	y.resize(cnt2);
	for (int j = cnt1; j < cnt2; j++) {
		y[j] = vect<ireal>(1.0);
	}

	for (int dim = 0; dim < NDIM; dim++) {
		X[dim] = SIMD(x[dim]);
	}
	M = m;
	for (int j = 0; j < cnt1; j += SIMD::size()) {
		for (int k = 0; k < SIMD::size(); k++) {
			for (int dim = 0; dim < NDIM; dim++) {
				Y[dim][k] = y[j + k][dim];
			}
		}
		if (j + SIMD::size() > cnt1) {
			for (int k = cnt1; k < cnt2; k++) {
				M[k - j] = 0.0;
			}
		}
		vect<SIMD> dX = X - Y;             										// 3 OP
		if (ewald) {
			for (int dim = 0; dim < NDIM; dim++) {
				const auto absdx = abs(dX[dim]);										// 3 OP
				dX[dim] = copysign(min(absdx, one - absdx), dX[dim] * (half - absdx));  // 15 OP
			}
		}
		multipole_interaction(Lacc, M, dX, EWALD);												// 401 OP
	}

	for (int i = 0; i < LP; i++) {
		L[i] += Lacc[i].sum();
	}
	return (EWALD ? 42729 : 422) * cnt1;
}

std::uint64_t gravity_CP_direct(expansion<double> &L, const vect<ireal> &x, std::vector<vect<float>> &y) {
	return gravity_CP<simd_real,false>(L,x,y);
}

std::uint64_t gravity_CP_ewald(expansion<double> &L, const vect<ireal> &x, std::vector<vect<float>> &y) {
	return gravity_CP<simd_double,true>(L,x,y);
}


std::uint64_t gravity_PP_ewald(std::vector<force> &f, const std::vector<vect<float>> &x, std::vector<vect<float>> &y) {
	if (x.size() == 0) {
		return 0;
	}
	std::uint64_t flop = 0;

	static const auto one = simd_double(1.0);
	static const auto half = simd_double(0.5);

	static const auto opts = options::get();
	static const simd_double M(1.0 / opts.problem_size);
	simd_double mask;
	static const bool ewald = opts.ewald;
	static const auto h = opts.soft_len;
	static const auto h2 = h * h;
	static const ewald_indices indices; // size is 93
	static const periodic_parts periodic;
	vect<simd_double> X, Y;
	std::vector<vect<simd_double>> G(x.size(), vect<double>(0.0));
	std::vector<simd_double> Phi(x.size(), 0.0);
	const auto cnt1 = y.size();
	const auto cnt2 = ((cnt1 - 1 + simd_double::size()) / simd_double::size()) * simd_double::size();
	y.resize(cnt2);
	for (int j = cnt1; j < cnt2; j++) {
		y[j] = vect<double>(0.0);
	}
	for (int J = 0; J < cnt1; J += simd_double::size()) {
		for (int k = 0; k < simd_double::size(); k++) {
			for (int dim = 0; dim < NDIM; dim++) {
				Y[dim][k] = y[J + k][dim];
			}
		}
		mask = simd_double(1);
		for (int k = std::min((int) (cnt1 - J), (int) simd_double::size()); k < simd_double::size(); k++) {
			mask[k] = 0.0;
		}
		for (int I = 0; I < x.size(); I++) {
			for (int dim = 0; dim < NDIM; dim++) {
				X[dim] = simd_double(x[I][dim]);
			}

			vect<simd_double> dX0 = X - Y;             		// 3 OP
			for (int dim = 0; dim < NDIM; dim++) {
				const auto absdx = abs(dX0[dim]);										// 3 OP
				dX0[dim] = copysign(min(absdx, one - absdx), dX0[dim] * (half - absdx));  // 15 OP
			}
			constexpr int nmax = 2;
			constexpr int hmax = 2;
			const double huge = std::numeric_limits<double>::max() / 10.0 / (nmax * nmax * nmax);
			const double tiny = std::numeric_limits<double>::min() * 10.0;
			vect<double> h;
			vect<simd_double> n;
			simd_double phi = 0.0;
			vect<simd_double> g;
			g = simd_double(0);
			for (int i = 0; i < indices.size(); i++) {
				h = indices[i];
				const expansion<double> &H = periodic[i];
				n = h;
				const vect<simd_double> dx = dX0 - n;                          // 3 OP
				const simd_double r2 = dx.dot(dx);
				const simd_double r = sqrt(r2);                      // 5 OP
				const simd_double rinv = r / (r2 + tiny);
				const simd_double r2inv = rinv * rinv;
				const simd_double r3inv = r2inv * rinv;
				const simd_double erfc = one - erf(2.0 * r);
				const simd_double d0 = -erfc * rinv;
				const simd_double expfactor = 4.0 * r * exp(-4.0 * r2) / sqrt(M_PI);
				simd_double tmp = sin(-r2);
				const simd_double d1 = (expfactor + erfc) * r3inv;
				phi += d0; // 6 OP
				for (int a = 0; a < NDIM; a++) {
					g[a] -= (dX0[a] - n[a]) * d1;
				}
				const double h2 = h.dot(h);                     // 5 OP
				simd_double hdotdx = simd_double(0);
				for (int dim = 0; dim < NDIM; dim++) {
					hdotdx += dX0[dim] * h[dim];
				}
				const simd_double omega = 2.0 * M_PI * hdotdx;
				const simd_double c = cos(omega);
				const simd_double s = sin(omega);
				phi += H() * c;
				for (int dim = 0; dim < NDIM; dim++) {
					g[dim] -= H(dim) * s;
				}
			}
			const simd_double r = abs(dX0);
			const simd_double rinv = r / (r * r + tiny);
			phi = simd_double(M_PI / 4.0) + phi + rinv;
			const simd_double sw = min(huge * r, 1.0);
			phi = 2.8372975 * (simd_double(1.0) - sw) + phi * sw;
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

