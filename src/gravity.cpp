#include <tigergrav/gravity.hpp>
#include <tigergrav/gravity_cuda.hpp>
#include <tigergrav/green.hpp>
#include <tigergrav/options.hpp>
#include <tigergrav/simd.hpp>

static const auto one = simd_float(1.0);
static const auto half = simd_float(0.5);
static const simd_float eps = simd_float(std::numeric_limits<float>::min());

double ewald_near_separation2(const vect<double> x) {
	double d = 0.0;
	for (int dim = 0; dim < NDIM; dim++) {
		const double absx = std::abs(x[dim]);
		double this_d = std::min(absx, (double) 1.0 - absx);
		d += this_d * this_d;
	}
	return d;
}

double ewald_far_separation2(const vect<double> x) {
	static const auto r2 = 0.25 * 0.25;
	return std::max(r2, ewald_near_separation2(x));
}

#include <tigergrav/interactions.hpp>

std::uint64_t gravity_PC_direct(std::vector<force> &f, const std::vector<vect<pos_type>> &x, std::vector<const multi_src*> &y) {
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
	vect<simd_int> X, Y;
	multipole<simd_float> M;
	std::vector<vect<simd_double>> G(x.size(), vect<simd_double>(simd_float(0)));
	std::vector<simd_double> Phi(x.size(), simd_double(0.0));
	const auto cnt1 = y.size();
	const auto cnt2 = ((cnt1 - 1 + simd_float::size()) / simd_float::size()) * simd_float::size();
	y.resize(cnt2);
	std::array<multi_src, simd_float::size()> ystage;
	for (int j = 0; j < cnt1; j += simd_float::size()) {
		for (int k = 0; k < simd_float::size(); k++) {
			if (j + k < cnt1) {
				ystage[k] = *y[j + k];
			} else {
				ystage[k].m = 0.0;
				ystage[k].x = y[cnt1 - 1]->x;
			}
		}
		for (int k = 0; k < simd_float::size(); k++) {
			for (int n = 0; n < MP; n++) {
				M[n][k] = ystage[k].m[n];
			}
		}
		for (int k = 0; k < simd_float::size(); k++) {
			for (int dim = 0; dim < NDIM; dim++) {
				Y[dim][k] = ystage[k].x[dim];
			}
		}
		for (int i = 0; i < x.size(); i++) {
			for (int dim = 0; dim < NDIM; dim++) {
				X[dim] = x[i][dim];
			}
			vect<simd_float> dX;
			if (opts.ewald) {
				for (int dim = 0; dim < NDIM; dim++) {
					dX[dim] = simd_float(X[dim] - Y[dim]) * simd_float(POS_INV); // 18
				}
			} else {
				for (int dim = 0; dim < NDIM; dim++) {
					dX[dim] = simd_float(X[dim]) * simd_float(POS_INV) - simd_float(Y[dim]) * simd_float(POS_INV);
				}
			}
			vect<simd_double> g;
			simd_double phi;
			multipole_interaction(g, phi, M, dX, false); // 516

			for (int dim = 0; dim < NDIM; dim++) {
				G[i][dim] += g[dim];  // 0 / 3
			}
			Phi[i] += phi;		          // 0 / 1
		}
	}
	for (int i = 0; i < x.size(); i++) {
		for (int dim = 0; dim < NDIM; dim++) {
			f[i].g[dim] += G[i][dim].sum();
		}
		f[i].phi += Phi[i].sum();
	}
	y.resize(cnt1);
	return 581 * cnt1 * x.size();
}

std::uint64_t gravity_CC_direct(expansion<double> &L, const vect<pos_type> &x, std::vector<const multi_src*> &y) {
	if (y.size() == 0) {
		return 0;
	}
	static const auto one = simd_float(1.0);
	static const auto half = simd_float(0.5);
	std::uint64_t flop = 0;
	static const auto opts = options::get();
	static const bool ewald = opts.ewald;
	static const auto h = opts.soft_len;
	static const auto h2 = h * h;
	static const simd_float H(h);
	static const simd_float H2(h2);
	vect<simd_int> X, Y;
	multipole<simd_float> M;
	expansion<simd_double> Lacc;
	Lacc = simd_double(0);
	const auto cnt1 = y.size();
	const auto cnt2 = ((cnt1 - 1 + simd_float::size()) / simd_float::size()) * simd_float::size();
	y.resize(cnt2);
	for (int dim = 0; dim < NDIM; dim++) {
		X[dim] = x[dim];
	}
	std::array<multi_src, simd_float::size()> ystage;
	for (int j = 0; j < cnt1; j += simd_float::size()) {
		for (int k = 0; k < simd_float::size(); k++) {
			if (j + k < cnt1) {
				ystage[k] = *y[j + k];
			} else {
				ystage[k].m = 0.0;
				ystage[k].x = y[cnt1 - 1]->x;
			}
		}
		for (int k = 0; k < simd_float::size(); k++) {
			for (int n = 0; n < MP; n++) {
				M[n][k] = ystage[k].m[n];
			}
		}
		for (int k = 0; k < simd_float::size(); k++) {
			for (int dim = 0; dim < NDIM; dim++) {
				Y[dim][k] = ystage[k].x[dim];
			}
		}

		vect<simd_float> dX;
		if (opts.ewald) {
			for (int dim = 0; dim < NDIM; dim++) {
				dX[dim] = simd_float(X[dim] - Y[dim]) * simd_float(POS_INV); // 18
			}
		} else {
			for (int dim = 0; dim < NDIM; dim++) {
				dX[dim] = simd_float(X[dim]) * simd_float(POS_INV) - simd_float(Y[dim]) * simd_float(POS_INV);
			}
		}
		multipole_interaction(Lacc, M, dX, false);												// 986
	}

	for (int i = 0; i < LP; i++) {
		L[i] += Lacc[i].sum();
	}
	y.resize(cnt1);
	return 1025 * cnt1;
}

std::uint64_t gravity_CP_direct(expansion<double> &L, const vect<pos_type> &x, std::vector<vect<pos_type>> y) {
	if (y.size() == 0) {
		return 0;
	}
	static const auto one = simd_float(1.0);
	static const auto half = simd_float(0.5);
	std::uint64_t flop = 0;
	static const auto opts = options::get();
	static const auto m = opts.m_tot / opts.problem_size;
	static const bool ewald = opts.ewald;
	static const auto h = opts.soft_len;
	static const auto h2 = h * h;
	static const simd_float H(h);
	static const simd_float H2(h2);
	vect<simd_int> Y, X;
	simd_float M;
	expansion<simd_double> Lacc;
	Lacc = simd_double(0);
	const auto cnt1 = y.size();
	const auto cnt2 = ((cnt1 - 1 + simd_float::size()) / simd_float::size()) * simd_float::size();
	y.resize(cnt2);
	for (int j = cnt1; j < cnt2; j++) {
		y[j] = y[cnt1 - 1];
	}

	for (int dim = 0; dim < NDIM; dim++) {
		X[dim] = x[dim];
	}
	M = m;
	for (int j = 0; j < cnt1; j += simd_float::size()) {
		for (int dim = 0; dim < NDIM; dim++) {
			for (int k = 0; k < simd_float::size(); k++) {
				Y[dim][k] = y[j + k][dim];
			}
		}
		if (j + simd_float::size() > cnt1) {
			for (int k = cnt1; k < cnt2; k++) {
				M[k - j] = 0.0;
			}
		}
		vect<simd_float> dX;
		if (opts.ewald) {
			for (int dim = 0; dim < NDIM; dim++) {
				dX[dim] = simd_float(X[dim] - Y[dim]) * simd_float(POS_INV); // 18
			}
		} else {
			for (int dim = 0; dim < NDIM; dim++) {
				dX[dim] = simd_float(X[dim]) * simd_float(POS_INV) - simd_float(Y[dim]) * simd_float(POS_INV);
			}
		}
		multipole_interaction(Lacc, M, dX);												// 	401
	}

	for (int i = 0; i < LP; i++) {
		L[i] += Lacc[i].sum();
	}
	y.resize(cnt1);
	return 455 * cnt1;
}

static const ewald_indices indices_real(EWALD_REAL_N2);
static const ewald_indices indices_four(EWALD_FOUR_N2);

std::uint64_t gravity_PP_ewald(std::vector<force> &f, const std::vector<vect<pos_type>> &x, std::vector<vect<pos_type>> y) {
	if (y.size() == 0) {
		return 0;
	}
	std::uint64_t flop = 0;
	static std::atomic<int> warning(0);
	if (warning++ == 0) {
		printf("gravity_PP_ewald detected - code is not optimized for this, consider a deeper tree\n");
	}

	static const auto one = simd_float(1.0);
	static const auto half = simd_float(0.5);

	static const auto opts = options::get();
	static const simd_double M(opts.m_tot / opts.problem_size);
	simd_double mask;
	static const bool ewald = opts.ewald;
	static const auto h = opts.soft_len;
	static const auto h2 = h * h;
//	printf( "%i %i \n", indices_real.size(), indices_four.size() );

	static const periodic_parts periodic;
	vect<simd_int> X, Y;
	std::vector<vect<simd_double>> G(x.size(), vect<simd_double>(simd_double(0)));
	std::vector<simd_double> Phi(x.size(), simd_double(0.0));
	const auto cnt1 = y.size();
	const auto cnt2 = ((cnt1 - 1 + simd_float::size()) / simd_float::size()) * simd_float::size();
	y.resize(cnt2);
	for (int j = cnt1; j < cnt2; j++) {
		y[j] = 0;
	}
	for (int J = 0; J < cnt1; J += simd_float::size()) {
		for (int dim = 0; dim < NDIM; dim++) {
			for (int k = 0; k < simd_float::size(); k++) {
				Y[dim][k] = y[J + k][dim];
			}
		}
		mask = simd_float(1);
		for (int k = std::min((int) (cnt1 - J), (int) simd_float::size()); k < simd_float::size(); k++) {
			mask[k] = 0.0;
		}
		for (int I = 0; I < x.size(); I++) {
			for (int dim = 0; dim < NDIM; dim++) {
				X[dim] = simd_int(x[I][dim]);
			}

			// 6S + 9D = 24
			vect<simd_float> dX0;
			for (int dim = 0; dim < NDIM; dim++) {
				dX0[dim] = simd_float(X[dim] - Y[dim]) * simd_float(POS_INV);
			}																	// 18
			constexpr int nmax = 2;
			constexpr int hmax = 2;
			static const simd_float two(2);
			static const simd_float rcut(1.0e-6);
			static const simd_float twopi(2 * M_PI);
			static const simd_float pioverfour(M_PI / 4.0);
			static const simd_float fouroversqrtpi(4.0 / sqrt(M_PI));
			static const simd_double phi0(2.8372975);
			vect<float> h;
			vect<simd_float> n;
			simd_double phi = 0.0;
			vect<simd_double> g;
			g = simd_double(0);
			const simd_float r = abs(dX0);										// 5
			simd_float cut_mask = r > rcut;										// 2

			// 162 x 305 = 49410
			for (int i = 0; i < indices_real.size(); i++) {
				h = indices_real[i];
				n = h;
				const vect<simd_float> dx = dX0 - n;                         	// 3
				const simd_float r2 = dx.dot(dx);								// 5
				const simd_float r = sqrt(r2);                      			// 7
				const simd_float mask = cut_mask * (r < 3.6);                   // 3
				simd_float rinv = mask * 1.0 / max(r, rcut);				    //37
				const simd_float r2inv = rinv * rinv;						    // 1
				const simd_float r3inv = r2inv * rinv;						    // 1
				simd_float expfac;
				const simd_float erfc = erfcexp(two * r, &expfac);				//77
				const simd_float d0 = -erfc * rinv;								// 2
				const simd_float expfactor = fouroversqrtpi * r * expfac;		// 2
				const simd_float d1 = (expfactor + erfc) * r3inv;				// 2
				phi += d0; 														// 4
				for (int a = 0; a < NDIM; a++) {
					g[a] -= (dX0[a] - n[a]) * d1;								// 18
				}
			}

			// 55 x 123 = 6765
			for (int i = 0; i < indices_four.size(); i++) {
				const expansion<float> &H = periodic[i];
				h = indices_four[i];
				simd_float hdotdx = dX0[0] * h[0];								// 1
				for (int dim = 1; dim < NDIM; dim++) {
					hdotdx += dX0[dim] * h[dim];								// 4
				}
				const simd_float omega = twopi * hdotdx;						// 1
				simd_float s, c;
				sincos(omega, &s, &c);											// 25
				phi += H() * c * cut_mask;										// 6
				for (int dim = 0; dim < NDIM; dim++) {
					g[dim] -= H(dim) * s * cut_mask;							// 18
				}
			}

			// 63
			simd_float rinv = 1.0 / max(r, rcut);								// 2
			phi += pioverfour + rinv;											// 6
			const simd_double sw = cut_mask;									// 2
			phi = phi0 * (simd_float(1.0) - sw) + phi * sw;						// 8
			const auto rinv3 = cut_mask * rinv * rinv * rinv;					// 3
			for (int dim = 0; dim < NDIM; dim++) {
				g[dim] += dX0[dim] * rinv3;										// 18
			}
			Phi[I] += M * phi * mask;											// 6
			G[I] += g * M * mask;												// 18
		}
	}
	for (int i = 0; i < x.size(); i++) {
		for (int dim = 0; dim < NDIM; dim++) {
			f[i].g[dim] += G[i][dim].sum();
		}
		f[i].phi += Phi[i].sum();
	}
	y.resize(cnt1);
	return 56245 * cnt1 * x.size();
}

std::uint64_t gravity_PC_ewald(std::vector<force> &f, const std::vector<vect<pos_type>> &x, std::vector<const multi_src*> &y) {
	if (x.size() == 0 || y.size() == 0) {
		return 0;
	}
	std::uint64_t flop = 0;
	static std::atomic<int> warning(0);
	if (warning++ == 0) {
		printf("gravity_PC_ewald detected - code is not optimized for this, consider a deeper tree\n");
	}

	static const auto one = simd_float(1.0);
	static const auto half = simd_float(0.5);
	static const auto opts = options::get();
	static const auto h = opts.soft_len;
	static const bool ewald = opts.ewald;
	static const auto h2 = h * h;
	static const simd_float H(h);
	static const simd_float H2(h2);
	vect<simd_int> X, Y;
	multipole<simd_float> M;
	std::vector<vect<simd_double>> G(x.size(), vect<float>(0.0));
	std::vector<simd_double> Phi(x.size(), simd_double(0.0));
	const auto cnt1 = y.size();
	const auto cnt2 = ((cnt1 - 1 + simd_float::size()) / simd_float::size()) * simd_float::size();
	y.resize(cnt2);
	std::array<multi_src, simd_float::size()> ystage;
	for (int j = 0; j < cnt1; j += simd_float::size()) {
		for (int k = 0; k < simd_float::size(); k++) {
			if (j + k < cnt1) {
				ystage[k] = *y[j + k];
			} else {
				ystage[k].m = 0.0;
				ystage[k].x = y[cnt1 - 1]->x;
			}
		}
		for (int k = 0; k < simd_float::size(); k++) {
			for (int n = 0; n < MP; n++) {
				M[n][k] = ystage[k].m[n];
			}
		}
		for (int k = 0; k < simd_float::size(); k++) {
			for (int dim = 0; dim < NDIM; dim++) {
				Y[dim][k] = ystage[k].x[dim];
			}
		}
		for (int i = 0; i < x.size(); i++) {
			for (int dim = 0; dim < NDIM; dim++) {
				X[dim] = x[i][dim];
			}

			vect<simd_float> dX;
			for (int dim = 0; dim < NDIM; dim++) {
				dX[dim] = simd_float(X[dim] - Y[dim]) * simd_float(POS_INV); // 18
			}
			vect<simd_double> g;
			simd_double phi;
			multipole_interaction(g, phi, M, dX, true); // 516

			for (int dim = 0; dim < NDIM; dim++) {
				G[i][dim] += g[dim];  // 0 / 3
			}
			Phi[i] += phi;		          // 0 / 1
		}
	}
	for (int i = 0; i < x.size(); i++) {
		for (int dim = 0; dim < NDIM; dim++) {
			f[i].g[dim] += G[i][dim].sum();
		}
		f[i].phi += Phi[i].sum();
	}
	y.resize(cnt1);
	return 251531 * cnt1 * x.size();
}

std::uint64_t gravity_CC_ewald(expansion<double> &L, const vect<pos_type> &x, std::vector<const multi_src*> &y) {
	if (y.size() == 0) {
		return 0;
	}
	static const auto opts = options::get();
	if (opts.cuda) {
//		return gravity_CC_ewald_cuda(L, x, y);
	}
	static const auto one = simd_float(1.0);
	static const auto half = simd_float(0.5);
	std::uint64_t flop = 0;
	static const bool ewald = opts.ewald;
	static const auto h = opts.soft_len;
	static const auto h2 = h * h;
	static const simd_float H(h);
	static const simd_float H2(h2);
	vect<simd_int> X, Y;
	multipole<simd_float> M;
	expansion<simd_double> Lacc;
	Lacc = simd_double(0);
	const auto cnt1 = y.size();
	const auto cnt2 = ((cnt1 - 1 + simd_float::size()) / simd_float::size()) * simd_float::size();
	y.resize(cnt2);

	for (int dim = 0; dim < NDIM; dim++) {
		X[dim] = x[dim];
	}
	std::array<multi_src, simd_float::size()> ystage;
	for (int j = 0; j < cnt1; j += simd_float::size()) {
		for (int k = 0; k < simd_float::size(); k++) {
			if (j + k < cnt1) {
				ystage[k] = *y[j + k];
			} else {
				ystage[k].m = 0.0;
				ystage[k].x = y[cnt1 - 1]->x;
			}
		}
		for (int k = 0; k < simd_float::size(); k++) {
			for (int n = 0; n < MP; n++) {
				M[n][k] = ystage[k].m[n];
			}
		}
		for (int k = 0; k < simd_float::size(); k++) {
			for (int dim = 0; dim < NDIM; dim++) {
				Y[dim][k] = ystage[k].x[dim];
			}
		}
		vect<simd_float> dX;
		for (int dim = 0; dim < NDIM; dim++) {
			dX[dim] = simd_float(X[dim] - Y[dim]) * simd_float(POS_INV); // 18
		}
		multipole_interaction(Lacc, M, dX, true);											// 251936
	}

	for (int i = 0; i < LP; i++) {
		L[i] += Lacc[i].sum();
	}
	y.resize(cnt1);
	return 251975 * cnt1;
}

std::uint64_t gravity_CP_ewald(expansion<double> &L, const vect<pos_type> &x, std::vector<vect<pos_type>> y) {
	if (y.size() == 0) {
		return 0;
	}
	static std::atomic<int> warning(0);
	if (warning++ == 0) {
		printf("gravity_CP_ewald detected - code is not optimized for this, consider a deeper tree\n");
	}

	static const auto one = simd_float(1.0);
	static const auto half = simd_float(0.5);
	std::uint64_t flop = 0;
	static const auto opts = options::get();
	static const auto m = opts.m_tot / opts.problem_size;
	static const bool ewald = opts.ewald;
	static const auto h = opts.soft_len;
	static const auto h2 = h * h;
	static const simd_float H(h);
	static const simd_float H2(h2);
	vect<simd_int> Y, X;
	simd_float M;
	expansion<simd_double> Lacc;
	Lacc = simd_double(0);
	const auto cnt1 = y.size();
	const auto cnt2 = ((cnt1 - 1 + simd_float::size()) / simd_float::size()) * simd_float::size();
	y.resize(cnt2);
	for (int j = cnt1; j < cnt2; j++) {
		y[j] = y[cnt1 - 1];
	}

	for (int dim = 0; dim < NDIM; dim++) {
		X[dim] = x[dim];
	}
	M = m;
	for (int j = 0; j < cnt1; j += simd_float::size()) {
		for (int dim = 0; dim < NDIM; dim++) {
			for (int k = 0; k < simd_float::size(); k++) {
				Y[dim][k] = y[j + k][dim];
			}
		}
		if (j + simd_float::size() > cnt1) {
			for (int k = cnt1; k < cnt2; k++) {
				M[k - j] = 0.0;
			}
		}
		vect<simd_float> dX;
		for (int dim = 0; dim < NDIM; dim++) {
			dX[dim] = simd_float(X[dim] - Y[dim]) * simd_float(POS_INV); // 18
		}
		multipole_interaction(Lacc, M, dX, true);										// 251176
	}

	for (int i = 0; i < LP; i++) {
		L[i] += Lacc[i].sum();
	}
	y.resize(cnt1);
	return 251233 * cnt1;
}

