#include <tigergrav/expansion.hpp>
#include <tigergrav/gravity.hpp>
#include <tigergrav/options.hpp>
#include <tigergrav/simd.hpp>

#include <hpx/include/async.hpp>

constexpr int NE = 64;
constexpr int NEP1 = NE + 1;
constexpr int NEP12 = NEP1 * NEP1;

using ewald_table_t = std::array<float,NEP1*NEP1*NEP1>;
static ewald_table_t epot;
static std::array<ewald_table_t, NDIM> eforce;
static std::array<std::array<ewald_table_t, NDIM>, NDIM> eforce2;

static const auto one = simd_svector(1.0);
static const auto half = simd_svector(0.5);
static const simd_svector eps = simd_svector(std::numeric_limits<float>::min());

std::vector<vect<double>> ewald_h;
std::vector<vect<double>> ewald_n;

force EW(general_vect<double, NDIM> x) {
	force f;
	constexpr int nmax = 2;
	constexpr int hmax = 2;
	const double huge = std::numeric_limits<double>::max() / 10.0 / (nmax * nmax * nmax);
	const double tiny = std::numeric_limits<double>::min() * 10.0;
	general_vect<double, NDIM> n, h;
	double phi = 0.0;
	vect<double> g = 0.0;
	for (int i = -nmax; i <= nmax; i++) {
		for (int j = -nmax; j <= nmax; j++) {
			for (int k = -nmax; k <= nmax; k++) {
				n[0] = i;
				n[1] = j;
				n[2] = k;
				const auto dx = x - n;                          // 3 OP
				double r = abs(x - n);                      // 5 OP
				const double rinv = r / (r * r + tiny);
				const double r2inv = rinv * rinv;
				const double r3inv = r2inv * rinv;
				const double erfc = 1.0 - erf(2.0 * r);
				const double d0 = -erfc * rinv;
				const double expfactor = 4.0 * r * exp(-4.0 * r * r) / sqrt(M_PI);
				const double d1 = (expfactor + erfc) * r3inv;
				phi += d0; // 6 OP
				for (int a = 0; a < NDIM; a++) {
					g[a] -= (x[a] - n[a]) * d1;
				}

				h = n;
				const double h2 = h.dot(h);                     // 5 OP
				if (h2 > 0) {
					const double hinv = 1.0 / h2;                  // 1 OP
					const double c0 = 1.0 / h2 * exp(-M_PI * M_PI * h2 / 4.0);
					const double omega = 2.0 * M_PI * h.dot(x);
					const double c = cos(omega);
					const double s = sin(omega);
					phi += -(1.0 / M_PI) * c0 * c;
					for (int dim = 0; dim < NDIM; dim++) {
						g[dim] += 2.0 * h[dim] * c0 * s;
					}
				}
			}
		}
	}
	const double r = abs(x);
	const double rinv = r / (r * r + tiny);
	f.phi = M_PI / 4.0 + phi + rinv;
	const double sw = std::min(huge * r, 1.0);
	f.phi = 2.8372975 * (1.0 - sw) + f.phi * sw;
	for (int dim = 0; dim < NDIM; dim++) {
		f.g[dim] = g[dim] + x[dim] * rinv * rinv * rinv;
	}
	return f;
}

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

std::uint64_t gravity_PP_ewald(std::vector<force> &f, const std::vector<vect<float>> &x, std::vector<vect<float>> &y) {
	if (x.size() == 0) {
		return 0;
	}
	std::uint64_t flop = 0;

	static const auto one = simd_dvector(1.0);
	static const auto half = simd_dvector(0.5);

	static const auto opts = options::get();
	static const simd_dvector M(1.0 / opts.problem_size);
	simd_dvector mask;
	static const bool ewald = opts.ewald;
	static const auto h = opts.soft_len;
	static const auto h2 = h * h;
	static const simd_dvector H(h);
	static const simd_dvector H2(h2);
	vect<simd_dvector> X, Y;
	std::vector<vect<simd_dvector>> G(x.size(), vect<double>(0.0));
	std::vector<simd_dvector> Phi(x.size(), 0.0);
	const auto cnt1 = y.size();
	const auto cnt2 = ((cnt1 - 1 + SIMD_DLEN) / SIMD_DLEN) * SIMD_DLEN;
	y.resize(cnt2);
	for (int j = cnt1; j < cnt2; j++) {
		y[j] = vect<double>(0.0);
	}
	for (int J = 0; J < cnt1; J += SIMD_DLEN) {
		for (int k = 0; k < SIMD_DLEN; k++) {
			for (int dim = 0; dim < NDIM; dim++) {
				Y[dim][k] = y[J + k][dim];
			}
		}
		mask = simd_dvector(1);
		for (int k = std::min((int) (cnt1 - J), (int) SIMD_DLEN); k < SIMD_DLEN; k++) {
			mask[k] = 0.0;
		}
		for (int I = 0; I < x.size(); I++) {
			for (int dim = 0; dim < NDIM; dim++) {
				X[dim] = simd_dvector(x[I][dim]);
			}

			vect<simd_dvector> dX0 = X - Y;             		// 3 OP
			for (int dim = 0; dim < NDIM; dim++) {
				const auto absdx = abs(dX0[dim]);										// 3 OP
				dX0[dim] = copysign(min(absdx, one - absdx), dX0[dim] * (half - absdx));  // 15 OP
			}
			constexpr int nmax = 2;
			constexpr int hmax = 2;
			const double huge = std::numeric_limits<double>::max() / 10.0 / (nmax * nmax * nmax);
			const double tiny = std::numeric_limits<double>::min() * 10.0;
			vect<double> h;
			vect<simd_dvector> n;
			simd_dvector phi = 0.0;
			vect<simd_dvector> g;
			g = simd_dvector(0);
			for (int i = -nmax; i <= nmax; i++) {
				for (int j = -nmax; j <= nmax; j++) {
					for (int k = -nmax; k <= nmax; k++) {
						n[0] = i;
						n[1] = j;
						n[2] = k;
						const vect<simd_dvector> dx = dX0 - n;                          // 3 OP
						const simd_dvector r2 = dx.dot(dx);
						const simd_dvector r = sqrt(r2);                      // 5 OP
						const simd_dvector rinv = r / (r2 + tiny);
						const simd_dvector r2inv = rinv * rinv;
						const simd_dvector r3inv = r2inv * rinv;
						const simd_dvector erfc = one - erf(2.0 * r);
						const simd_dvector d0 = -erfc * rinv;
						const simd_dvector expfactor = 4.0 * r * exp(-4.0 * r2) / sqrt(M_PI);
						simd_dvector tmp = sin(-r2);
						const simd_dvector d1 = (expfactor + erfc) * r3inv;
						phi += d0; // 6 OP
						for (int a = 0; a < NDIM; a++) {
							g[a] -= (dX0[a] - n[a]) * d1;
						}
						h[0] = i;
						h[1] = j;
						h[2] = k;
						const double h2 = h.dot(h);                     // 5 OP
						if (h2 > 0) {
							const double hinv = 1.0 / h2;                  // 1 OP
							const double c0 = 1.0 / h2 * exp(-M_PI * M_PI * h2 / 4.0);
							simd_dvector hdotdx = simd_dvector(0);
							for (int dim = 0; dim < NDIM; dim++) {
								hdotdx += dX0[dim] * h[dim];
							}
							const simd_dvector omega = 2.0 * M_PI * hdotdx;
							const simd_dvector c = cos(omega);
							const simd_dvector s = sin(omega);
							phi += -(1.0 / M_PI) * c0 * c;
							for (int dim = 0; dim < NDIM; dim++) {
								g[dim] += 2.0 * h[dim] * c0 * s;
							}
						}
					}
				}
			}
			const simd_dvector r = abs(dX0);
			const simd_dvector rinv = r / (r * r + tiny);
			phi = simd_dvector(M_PI / 4.0) + phi + rinv;
			const simd_dvector sw = min(huge * r, 1.0);
			phi = 2.8372975 * (simd_dvector(1.0) - sw) + phi * sw;
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
	return (26 + ewald ? 18 : 0) * cnt1 * x.size();
}

std::uint64_t gravity_PP(std::vector<force> &f, const std::vector<vect<float>> &x, std::vector<vect<float>> &y) {
	if (x.size() == 0) {
		return 0;
	}
	std::uint64_t flop = 0;
	static const auto opts = options::get();
	static const simd_svector M(1.0 / opts.problem_size);
	static const bool ewald = opts.ewald;
	static const auto h = opts.soft_len;
	static const auto h2 = h * h;
	static const simd_svector H(h);
	static const simd_svector H2(h2);
	vect<simd_svector> X, Y;
	std::vector<vect<simd_svector>> nG(x.size(), vect<float>(0.0));
	std::vector<simd_svector> nPhi(x.size(), 0.0);
	const auto cnt1 = y.size();
	const auto cnt2 = ((cnt1 - 1 + SIMD_SLEN) / SIMD_SLEN) * SIMD_SLEN;
	y.resize(cnt2);
	for (int j = cnt1; j < cnt2; j++) {
		y[j] = vect<float>(1.0e+10);
	}
	for (int j = 0; j < cnt1; j += SIMD_SLEN) {
		for (int k = 0; k < SIMD_SLEN; k++) {
			for (int dim = 0; dim < NDIM; dim++) {
				Y[dim][k] = y[j + k][dim];
			}
		}
		for (int i = 0; i < x.size(); i++) {
			for (int dim = 0; dim < NDIM; dim++) {
				X[dim] = simd_svector(x[i][dim]);
			}

			vect<simd_svector> dX = X - Y;             		// 3 OP
			if (ewald) {
				for (int dim = 0; dim < NDIM; dim++) {
					const auto absdx = abs(dX[dim]);										// 3 OP
					dX[dim] = copysign(min(absdx, one - absdx), dX[dim] * (half - absdx));  // 15 OP
				}
			}
			simd_svector r2 = dX[0] * dX[0];					// 1 OP
			r2 = fma(dX[1], dX[1], r2);						// 2 OP
			r2 = fma(dX[2], dX[2], r2);                     // 2 OP
			const simd_svector rinv = rsqrt(r2 + H2);        // 2 OP
			const simd_svector rinv3 = rinv * rinv * rinv;   // 2 OP
			for (int dim = 0; dim < NDIM; dim++) {
				const auto tmp = M * rinv3;					// 3 OP
				nG[i][dim] = fma(dX[dim], tmp, nG[i][dim]);  // 6 OP
			}
			const simd_svector kill_zero = r2 / (r2 + eps);  // 2 OP
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
	return (26 + ewald ? 18 : 0) * cnt1 * x.size();
}

std::uint64_t gravity_PC(std::vector<force> &f, const std::vector<vect<float>> &x, std::vector<multi_src> &y) {
	if (x.size() == 0) {
		return 0;
	}
	std::uint64_t flop = 0;
	static const auto opts = options::get();
	static const auto h = opts.soft_len;
	static const bool ewald = opts.ewald;
	static const auto h2 = h * h;
	static const simd_svector H(h);
	static const simd_svector H2(h2);
	vect<simd_svector> X, Y;
	multipole<simd_svector> M;
	std::vector<vect<simd_svector>> G(x.size(), vect<float>(0.0));
	std::vector<simd_svector> Phi(x.size(), 0.0);
	const auto cnt1 = y.size();
	const auto cnt2 = ((cnt1 - 1 + SIMD_SLEN) / SIMD_SLEN) * SIMD_SLEN;
	y.resize(cnt2);
	for (int j = cnt1; j < cnt2; j++) {
		y[j].m = 0.0;
		y[j].x = vect<float>(1.0);
	}
	for (int j = 0; j < cnt1; j += SIMD_SLEN) {
		for (int k = 0; k < SIMD_SLEN; k++) {
			M()[k] = y[j + k].m();
			for (int n = 0; n < NDIM; n++) {
				for (int l = 0; l <= n; l++) {
					M(n, l)[k] = y[j + k].m(n, l);
					for (int p = 0; p <= l; p++) {
						M(n, l, p)[k] = y[j + k].m(n, l, p);
#ifdef HEXAPOLE
						for (int q = 0; q <= p; q++) {
							M(n, l, p, q) = y[j + k].m(n, l, p, q);
						}
#endif
					}
				}
			}
			for (int dim = 0; dim < NDIM; dim++) {
				Y[dim][k] = y[j + k].x[dim];
			}
		}
		for (int i = 0; i < x.size(); i++) {
			for (int dim = 0; dim < NDIM; dim++) {
				X[dim] = simd_svector(x[i][dim]);
			}

			vect<simd_svector> dX = X - Y;             		// 3 OP
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
	return (26 + ewald ? 18 : 0) * cnt1 * x.size();
}

std::uint64_t gravity_CC(expansion<ireal> &L, const vect<ireal> &x, std::vector<multi_src> &y) {
	if (y.size() == 0) {
		return 0;
	}
	static const auto one = isimd_vector(1.0);
	static const auto half = isimd_vector(0.5);
	std::uint64_t flop = 0;
	static const auto opts = options::get();
	static const bool ewald = opts.ewald;
	static const auto h = opts.soft_len;
	static const auto h2 = h * h;
	static const isimd_vector H(h);
	static const isimd_vector H2(h2);
	vect<isimd_vector> X, Y;
	multipole<isimd_vector> M;
	expansion<isimd_vector> Lacc;
	Lacc = isimd_vector(0);
	const auto cnt1 = y.size();
	const auto cnt2 = ((cnt1 - 1 + ISIMD_LEN) / ISIMD_LEN) * ISIMD_LEN;
	y.resize(cnt2);
	for (int j = cnt1; j < cnt2; j++) {
		y[j].m = 0.0;
		y[j].x = vect<ireal>(1.0);
	}

	for (int dim = 0; dim < NDIM; dim++) {
		X[dim] = isimd_vector(x[dim]);
	}
	for (int j = 0; j < cnt1; j += ISIMD_LEN) {
		for (int k = 0; k < ISIMD_LEN; k++) {
			M()[k] = y[j + k].m();
			for (int n = 0; n < NDIM; n++) {
				for (int l = 0; l <= n; l++) {
					M(n, l)[k] = y[j + k].m(n, l);
					for (int p = 0; p <= l; p++) {
						M(n, l, p)[k] = y[j + k].m(n, l, p);
#ifdef HEXAPOLE
						for (int q = 0; q <= p; q++) {
							M(n, l, p, q) = y[j + k].m(n, l, p, q);
						}
#endif
					}
				}
			}
			for (int dim = 0; dim < NDIM; dim++) {
				Y[dim][k] = y[j + k].x[dim];
			}
		}
		vect<isimd_vector> dX = X - Y;             										// 3 OP
		if (ewald) {
			for (int dim = 0; dim < NDIM; dim++) {
				const auto absdx = abs(dX[dim]);										// 3 OP
				dX[dim] = copysign(min(absdx, one - absdx), dX[dim] * (half - absdx));  // 15 OP
			}
		}
		multipole_interaction(Lacc, M, dX);												// 701 OP
	}

	L() += Lacc().sum();
	for (int j = 0; j < NDIM; j++) {
		L(j) += Lacc(j).sum();
		for (int k = 0; k <= j; k++) {
			L(j, k) += Lacc(j, k).sum();
			for (int l = 0; l <= k; l++) {
				L(j, k, l) += Lacc(j, k, l).sum();
				for (int m = 0; m <= l; m++) {
					L(j, k, l, m) += Lacc(j, k, l, m).sum();
#ifdef HEXAPOLE
					for (int n = 0; n <= m; n++) {
						L(j, k, l, m, n) += Lacc(j, k, l, m, n).sum();
					}
#endif
				}
			}
		}
	}
	return (704 + ewald ? 18 : 0) * cnt1;
}

std::uint64_t gravity_CP(expansion<ireal> &L, const vect<ireal> &x, std::vector<vect<float>> &y) {
	if (y.size() == 0) {
		return 0;
	}
	static const auto one = isimd_vector(1.0);
	static const auto half = isimd_vector(0.5);
	std::uint64_t flop = 0;
	static const auto opts = options::get();
	static const auto m = 1.0 / opts.problem_size;
	static const bool ewald = opts.ewald;
	static const auto h = opts.soft_len;
	static const auto h2 = h * h;
	static const isimd_vector H(h);
	static const isimd_vector H2(h2);
	vect<isimd_vector> X, Y;
	multipole<isimd_vector> M;
	expansion<isimd_vector> Lacc;
	Lacc = isimd_vector(0);
	const auto cnt1 = y.size();
	const auto cnt2 = ((cnt1 - 1 + ISIMD_LEN) / ISIMD_LEN) * ISIMD_LEN;
	y.resize(cnt2);
	for (int j = cnt1; j < cnt2; j++) {
		y[j] = vect<ireal>(1.0);
	}

	for (int dim = 0; dim < NDIM; dim++) {
		X[dim] = isimd_vector(x[dim]);
	}
	for (int k = 0; k < ISIMD_LEN; k++) {
		M()[k] = m;
		for (int n = 0; n < NDIM; n++) {
			for (int l = 0; l <= n; l++) {
				M(n, l)[k] = 0.0;
				for (int p = 0; p <= l; p++) {
					M(n, l, p)[k] = 0.0;
				}
			}
		}
	}
	for (int j = 0; j < cnt1; j += ISIMD_LEN) {
		for (int k = 0; k < ISIMD_LEN; k++) {
			for (int dim = 0; dim < NDIM; dim++) {
				Y[dim][k] = y[j + k][dim];
			}
		}
		if (j + ISIMD_LEN > cnt1) {
			for (int k = cnt1; k < cnt2; k++) {
				M()[k - j] = 0.0;
			}
		}
		vect<isimd_vector> dX = X - Y;             										// 3 OP
		if (ewald) {
			for (int dim = 0; dim < NDIM; dim++) {
				const auto absdx = abs(dX[dim]);										// 3 OP
				dX[dim] = copysign(min(absdx, one - absdx), dX[dim] * (half - absdx));  // 15 OP
			}
		}
		multipole_interaction(Lacc, M, dX);												// 701 OP
	}

	L() += Lacc().sum();
	for (int j = 0; j < NDIM; j++) {
		L(j) += Lacc(j).sum();
		for (int k = 0; k <= j; k++) {
			L(j, k) += Lacc(j, k).sum();
			for (int l = 0; l <= k; l++) {
				L(j, k, l) += Lacc(j, k, l).sum();
				for (int m = 0; m <= l; m++) {
					L(j, k, l, m) += Lacc(j, k, l, m).sum();
#ifdef HEXAPOLE
					for (int n = 0; n <= m; n++) {
						L(j, k, l, m, n) += Lacc(j, k, l, m, n).sum();
					}
#endif
				}
			}
		}
	}
	return (704 + ewald ? 18 : 0) * cnt1;
}

std::uint64_t gravity_PC_ewald(std::vector<force> &f, const std::vector<vect<float>> &x, std::vector<multi_src> &y) {
	if (x.size() == 0) {
		return 0;
	}
	std::uint64_t flop = 0;

	static const auto one = simd_dvector(1.0);
	static const auto half = simd_dvector(0.5);
	static const auto opts = options::get();
	static const auto h = opts.soft_len;
	static const bool ewald = opts.ewald;
	static const auto h2 = h * h;
	static const simd_dvector H(h);
	static const simd_dvector H2(h2);
	vect<simd_dvector> X, Y;
	multipole<simd_dvector> M;
	std::vector<vect<simd_dvector>> G(x.size(), vect<float>(0.0));
	std::vector<simd_dvector> Phi(x.size(), 0.0);
	const auto cnt1 = y.size();
	const auto cnt2 = ((cnt1 - 1 + SIMD_DLEN) / SIMD_DLEN) * SIMD_DLEN;
	y.resize(cnt2);
	for (int j = cnt1; j < cnt2; j++) {
		y[j].m = 0.0;
		y[j].x = vect<float>(1.0);
	}
	for (int j = 0; j < cnt1; j += SIMD_DLEN) {
		for (int k = 0; k < SIMD_DLEN; k++) {
			M()[k] = y[j + k].m();
			for (int n = 0; n < NDIM; n++) {
				for (int l = 0; l <= n; l++) {
					M(n, l)[k] = y[j + k].m(n, l);
					for (int p = 0; p <= l; p++) {
						M(n, l, p)[k] = y[j + k].m(n, l, p);
					}
				}
			}
			for (int dim = 0; dim < NDIM; dim++) {
				Y[dim][k] = y[j + k].x[dim];
			}
		}
		for (int i = 0; i < x.size(); i++) {
			for (int dim = 0; dim < NDIM; dim++) {
				X[dim] = simd_dvector(x[i][dim]);
			}

			vect<simd_dvector> dX = X - Y;             		// 3 OP
			if (ewald) {
				for (int dim = 0; dim < NDIM; dim++) {
					const auto absdx = abs(dX[dim]);										// 3 OP
					dX[dim] = copysign(min(absdx, one - absdx), dX[dim] * (half - absdx));  // 15 OP
				}
			}
			auto this_f = multipole_interaction(M, dX, true);

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
	return (26 + ewald ? 18 : 0) * cnt1 * x.size();
}

std::uint64_t gravity_CC_ewald(expansion<ireal> &L, const vect<ireal> &x, std::vector<multi_src> &y) {
	if (y.size() == 0) {
		return 0;
	}
	static const auto one = isimd_vector(1.0);
	static const auto half = isimd_vector(0.5);
	std::uint64_t flop = 0;
	static const auto opts = options::get();
	static const bool ewald = opts.ewald;
	static const auto h = opts.soft_len;
	static const auto h2 = h * h;
	static const isimd_vector H(h);
	static const isimd_vector H2(h2);
	vect<isimd_vector> X, Y;
	multipole<isimd_vector> M;
	expansion<isimd_vector> Lacc;
	Lacc = isimd_vector(0);
	const auto cnt1 = y.size();
	const auto cnt2 = ((cnt1 - 1 + ISIMD_LEN) / ISIMD_LEN) * ISIMD_LEN;
	y.resize(cnt2);
	for (int j = cnt1; j < cnt2; j++) {
		y[j].m = 0.0;
		y[j].x = vect<ireal>(1.0);
	}

	for (int dim = 0; dim < NDIM; dim++) {
		X[dim] = isimd_vector(x[dim]);
	}
	for (int j = 0; j < cnt1; j += ISIMD_LEN) {
		for (int k = 0; k < ISIMD_LEN; k++) {
			M()[k] = y[j + k].m();
			for (int n = 0; n < NDIM; n++) {
				for (int l = 0; l <= n; l++) {
					M(n, l)[k] = y[j + k].m(n, l);
					for (int p = 0; p <= l; p++) {
						M(n, l, p)[k] = y[j + k].m(n, l, p);
					}
				}
			}
			for (int dim = 0; dim < NDIM; dim++) {
				Y[dim][k] = y[j + k].x[dim];
			}
		}
		vect<isimd_vector> dX = X - Y;             										// 3 OP
		if (ewald) {
			for (int dim = 0; dim < NDIM; dim++) {
				const auto absdx = abs(dX[dim]);										// 3 OP
				dX[dim] = copysign(min(absdx, one - absdx), dX[dim] * (half - absdx));  // 15 OP
			}
		}
		multipole_interaction(Lacc, M, dX, true);												// 701 OP
	}

	L() += Lacc().sum();
	for (int j = 0; j < NDIM; j++) {
		L(j) += Lacc(j).sum();
		for (int k = 0; k <= j; k++) {
			L(j, k) += Lacc(j, k).sum();
			for (int l = 0; l <= k; l++) {
				L(j, k, l) += Lacc(j, k, l).sum();
				for (int m = 0; m <= l; m++) {
					L(j, k, l, m) += Lacc(j, k, l, m).sum();
				}
			}
		}
	}
	return (704 + ewald ? 18 : 0) * cnt1;
}

std::uint64_t gravity_CP_ewald(expansion<ireal> &L, const vect<ireal> &x, std::vector<vect<float>> &y) {
	if (y.size() == 0) {
		return 0;
	}
	static const auto one = isimd_vector(1.0);
	static const auto half = isimd_vector(0.5);
	std::uint64_t flop = 0;
	static const auto opts = options::get();
	static const auto m = 1.0 / opts.problem_size;
	static const bool ewald = opts.ewald;
	static const auto h = opts.soft_len;
	static const auto h2 = h * h;
	static const isimd_vector H(h);
	static const isimd_vector H2(h2);
	vect<isimd_vector> X, Y;
	multipole<isimd_vector> M;
	expansion<isimd_vector> Lacc;
	Lacc = isimd_vector(0);
	const auto cnt1 = y.size();
	const auto cnt2 = ((cnt1 - 1 + ISIMD_LEN) / ISIMD_LEN) * ISIMD_LEN;
	y.resize(cnt2);
	for (int j = cnt1; j < cnt2; j++) {
		y[j] = vect<ireal>(1.0);
	}

	for (int dim = 0; dim < NDIM; dim++) {
		X[dim] = isimd_vector(x[dim]);
	}
	for (int k = 0; k < ISIMD_LEN; k++) {
		M()[k] = m;
		for (int n = 0; n < NDIM; n++) {
			for (int l = 0; l <= n; l++) {
				M(n, l)[k] = 0.0;
				for (int p = 0; p <= l; p++) {
					M(n, l, p)[k] = 0.0;
				}
			}
		}
	}
	for (int j = 0; j < cnt1; j += ISIMD_LEN) {
		for (int k = 0; k < ISIMD_LEN; k++) {
			for (int dim = 0; dim < NDIM; dim++) {
				Y[dim][k] = y[j + k][dim];
			}
		}
		if (j + ISIMD_LEN > cnt1) {
			for (int k = cnt1; k < cnt2; k++) {
				M()[k - j] = 0.0;
			}
		}
		vect<isimd_vector> dX = X - Y;             										// 3 OP
		if (ewald) {
			for (int dim = 0; dim < NDIM; dim++) {
				const auto absdx = abs(dX[dim]);										// 3 OP
				dX[dim] = copysign(min(absdx, one - absdx), dX[dim] * (half - absdx));  // 15 OP
			}
		}
		multipole_interaction(Lacc, M, dX, true);												// 701 OP
	}

	L() += Lacc().sum();
	for (int j = 0; j < NDIM; j++) {
		L(j) += Lacc(j).sum();
		for (int k = 0; k <= j; k++) {
			L(j, k) += Lacc(j, k).sum();
			for (int l = 0; l <= k; l++) {
				L(j, k, l) += Lacc(j, k, l).sum();
				for (int m = 0; m <= l; m++) {
					L(j, k, l, m) += Lacc(j, k, l, m).sum();
				}
			}
		}
	}
	return (704 + ewald ? 18 : 0) * cnt1;
}

void init_ewald() {

	force f;
	general_vect<double, NDIM> n, h;
	constexpr int nmax = 5;
	constexpr int hmax = 10;

	double sum1 = 0.0;
	vect<double> fsum1 = 0.0;
	for (int i = -nmax; i <= nmax; i++) {
		for (int j = -nmax; j <= nmax; j++) {
			for (int k = -nmax; k <= nmax; k++) {
				n[0] = i;
				n[1] = j;
				n[2] = k;
				ewald_n.push_back(n);
			}
		}
	}
	for (int i = -hmax; i <= hmax; i++) {
		for (int j = -hmax; j <= hmax; j++) {
			for (int k = -hmax; k <= hmax; k++) {
				h[0] = i;
				h[1] = j;
				h[2] = k;
				const double absh = abs(h);                     // 5 OP
				if (absh <= 10 && absh > 0) {
					ewald_h.push_back(h);
				}
			}
		}
	}

	FILE *fp = fopen("ewald.dat", "rb");
	if (fp) {
		int cnt = 0;
		printf("Found ewald.dat\n");
		const int sz = (NEP1) * (NEP1) * (NEP1);
		cnt += fread(&epot, sizeof(float), sz, fp);
		cnt += fread(&eforce, sizeof(float), NDIM * sz, fp);
		int expected = sz * (1 + NDIM);
		if (cnt != expected) {
			printf("ewald.dat is corrupt, read %i bytes, expected %i. Remove and re-run\n", cnt, expected);
			abort();
		}
		fclose(fp);
	} else {

		printf("ewald.dat not found\n");
		printf("Initializing Ewald (this may take some time)\n");

		const double dx0 = 0.5 / NE;
		for (int dim = 0; dim < NDIM; dim++) {
			eforce[dim][0] = 0.0;
		}
		epot[0] = 2.8372975;
		float n = 0;
		for (int i = 0; i <= NE; i++) {
			for (int j = 0; j <= NE; j++) {
				printf("%% %.2f complete\r", n / double(NEP1) / double(NEP1) * 100.0);
				n += 1.0;
				fflush(stdout);
				std::vector<hpx::future<void>> futs;
				for (int k = 0; k <= NE; k++) {
					const auto func = [i, j, k, dx0]() {
						general_vect<double, NDIM> x;
						x[0] = i * dx0;
						x[1] = j * dx0;
						x[2] = k * dx0;
						if (x.dot(x) != 0.0) {
							const double dx = 0.5 * dx0;
							const auto f = EW(x);
							epot[i * NEP12 + j * NEP1 + k] = f.phi;
							for (int dim = 0; dim < NDIM; dim++) {
								eforce[dim][i * NEP12 + j * NEP1 + k] = f.g[dim];
							}
						}
					};
					futs.push_back(hpx::async(func));
				}
				hpx::wait_all(futs);
			}
		}
		printf("\nDone initializing Ewald\n");
		fp = fopen("ewald.dat", "wb");
		const int sz = (NEP1) * (NEP1) * (NEP1);
		fwrite(&epot, sizeof(float), sz, fp);
		fwrite(&eforce, sizeof(float), NDIM * sz, fp);
		fclose(fp);
	}
	for (int dim = 0; dim < NDIM; dim++) {
		for (int i = 0; i < NE; i++) {
			for (int j = 0; j < NE; j++) {
				for (int k = 0; k < NE; k++) {
					const double dx0 = 0.5 / NE;
					const auto iii = i * NEP12 + j * NEP1 + k;
					eforce2[0][dim][iii] = (eforce[dim][iii + NEP12] - eforce[dim][iii]) / dx0;
					eforce2[1][dim][iii] = (eforce[dim][iii + NEP1] - eforce[dim][iii]) / dx0;
					eforce2[2][dim][iii] = (eforce[dim][iii + 1] - eforce[dim][iii]) / dx0;
				}
			}
		}
	}

}

