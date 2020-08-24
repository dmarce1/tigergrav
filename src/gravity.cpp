#include <tigergrav/gravity.hpp>
#include <tigergrav/green.hpp>
#include <tigergrav/options.hpp>
#include <tigergrav/simd.hpp>

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

double ewald_far_separation(const vect<double> x, double r) {
	return std::max(0.25, ewald_near_separation(x));

}

// 43009,703
template<class DOUBLE, class SINGLE> // 986 // 251936
inline void multipole_interaction(expansion<DOUBLE> &L, const multipole<SINGLE> &M2, vect<SINGLE> dX, bool ewald = false) { // 670/700 + 418 * NREAL + 50 * NFOUR
	static const expansion_factors<SINGLE> expansion_factor;
	expansion<SINGLE> D;
	if (ewald) {
		D = green_ewald(dX);		// 251176
	} else {
		D = green_direct(dX);        // 226
	}

	// 760
	auto &L0 = L();
	L0 += M2() * D();																// 5
	for (int a = 0; a < 3; a++) {
		for (int b = a; b < 3; b++) {
			L0 += M2(a, b) * D(a, b) * expansion_factor(a, b);						// 36
			for (int c = b; c < 3; c++) {
				L0 -= M2(a, b, c) * D(a, b, c) * expansion_factor(a, b, c);			// 60
			}
		}
	}
	for (int a = 0; a < 3; a++) {
		auto &La = L(a);
		La += M2() * D(a);
		for (int b = 0; b < 3; b++) {
			for (int c = b; c < 3; c++) {
				La += M2(c, b) * D(a, b, c) * expansion_factor(c, b);				// 108
				for (int d = c; d < 3; d++) {
					La -= M2(b, c, d) * D(a, b, c, d) * expansion_factor(b, c, d);	//180
				}
			}
		}
	}

	for (int a = 0; a < 3; a++) {
		for (int b = a; b < 3; b++) {
			auto &Lab = L(a, b);
			Lab += M2() * D(a, b);												// 30
			for (int c = 0; c < 3; c++) {
				for (int d = c; d < 3; d++) {
					Lab += M2(c, d) * D(a, b, c, d) * expansion_factor(c, d);	 // 216
				}
			}
		}
	}

	for (int a = 0; a < 3; a++) {
		for (int b = a; b < 3; b++) {
			for (int c = b; c < 3; c++) {
				auto &Labc = L(a, b, c);
				Labc += M2() * D(a, b, c);										// 50
			}
		}
	}
	for (int a = 0; a < 3; a++) {
		for (int b = a; b < 3; b++) {
			for (int c = b; c < 3; c++) {
				for (int d = c; d < 3; d++) {
					auto &Labcd = L(a, b, c, d);
					Labcd += M2() * D(a, b, c, d);								// 75
				}
			}
		}
	}
}

template<class DOUBLE, class SINGLE> // 401 / 251351
inline void multipole_interaction(expansion<DOUBLE> &L, const SINGLE &M, vect<SINGLE> dX, bool ewald = false) { // 390 / 47301
	static const expansion_factors<SINGLE> expansion_factor;
	expansion<SINGLE> D;
	if (ewald) {
		D = green_ewald(dX);		// 251175
	} else {
		D = green_direct(dX);          // 226
	}

	// 175
	auto &L0 = L();
	L0 += M * D();													// 5
	for (int a = 0; a < 3; a++) {
		auto &La = L(a);
		La += M * D(a);												// 15
		for (int b = a; b < 3; b++) {
			auto &Lab = L(a, b);
			Lab += M * D(a, b);										// 30
			for (int c = b; c < 3; c++) {
				auto &Labc = L(a, b, c);
				Labc += M * D(a, b, c);								// 50
				for (int d = c; d < 3; d++) {
					auto &Labcd = L(a, b, c, d);
					Labcd += M * D(a, b, c, d);						// 75
				}
			}
		}
	}
}

template<class SINGLE, class DOUBLE> // 516 / 251466
inline void multipole_interaction(std::pair<DOUBLE, vect<DOUBLE>> &f, const multipole<SINGLE> &M, vect<SINGLE> dX, bool ewald = false) { // 517 / 47428
	static const expansion_factors<SINGLE> expansion_factor;
	expansion<SINGLE> D;
	if (ewald) {
		D = green_ewald(dX);				// 251176
	} else {
		D = green_direct(dX);				// 226
	}

	//290
	auto &ffirst = f.first;
	ffirst = M() * D();																// 5
	for (int a = 0; a < 3; a++) {
		for (int b = a; b < 3; b++) {
			ffirst += M(a, b) * D(a, b) * expansion_factor(a, b);					// 36
		}
	}
	for (int a = 0; a < 3; a++) {
		for (int b = 0; b < 3; b++) {
			for (int c = b; c < 3; c++) {
				ffirst -= M(a, b, c) * D(a, b, c) * expansion_factor(a, b, c);		// 60
			}
		}
	}
	f.second = DOUBLE(0);
	for (int a = 0; a < 3; a++) {
		f.second[a] -= M() * D(a);													// 15
	}
	for (int a = 0; a < 3; a++) {
		for (int b = 0; b < 3; b++) {
			for (int c = b; c < 3; c++) {
				f.second[a] -= M(c, b) * D(a, b, c) * expansion_factor(c, b);		// 108
			}
		}
	}
	for (int a = 0; a < 3; a++) {
		for (int b = 0; b < 3; b++) {
			for (int c = b; c < 3; c++) {
				for (int d = c; d < 3; d++) {
					auto &fseconda = f.second[a];
					fseconda += M(c, b, d) * D(a, b, c, d) * expansion_factor(b, c, d); // 180
				}
			}
		}
	}
}

std::uint64_t gravity_PC_direct(std::vector<force> &f, const std::vector<vect<pos_type>> &x, std::vector<multi_src> &y) {
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
	for (int j = cnt1; j < cnt2; j++) {
		y[j].m = 0.0;
		y[j].x = y[cnt1 - 1].x;
	}
	for (int j = 0; j < cnt1; j += simd_float::size()) {
		for (int n = 0; n < MP; n++) {
			for (int k = 0; k < simd_float::size(); k++) {
				M[n][k] = y[j + k].m[n];
			}
		}
		for (int dim = 0; dim < NDIM; dim++) {
			for (int k = 0; k < simd_float::size(); k++) {
				Y[dim][k] = y[j + k].x[dim];
			}
		}
		for (int i = 0; i < x.size(); i++) {
			for (int dim = 0; dim < NDIM; dim++) {
				X[dim] = x[i][dim];
			}
			vect<simd_float> dX;
			if (opts.ewald) {
				for (int dim = 0; dim < NDIM; dim++) {
					dX[dim] = simd_float(simd_double(X[dim] - Y[dim]) * simd_double(POS_INV)); // 18
				}
			} else {
				for (int dim = 0; dim < NDIM; dim++) {
					dX[dim] = simd_float(simd_double(X[dim]) * simd_double(POS_INV) - simd_double(Y[dim]) * simd_double(POS_INV));
				}
			}
			std::pair<simd_double, vect<simd_double>> this_f;
			multipole_interaction(this_f, M, dX); // 516

			for (int dim = 0; dim < NDIM; dim++) {
				G[i][dim] += this_f.second[dim];  // 0 / 3
			}
			Phi[i] += this_f.first;		          // 0 / 1
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

std::uint64_t gravity_CC_direct(expansion<double> &L, const vect<pos_type> &x, std::vector<multi_src> &y) {
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
	for (int j = cnt1; j < cnt2; j++) {
		y[j].m = 0.0;
		y[j].x = y[cnt1 - 1].x;
	}

	for (int dim = 0; dim < NDIM; dim++) {
		X[dim] = x[dim];
	}
	for (int j = 0; j < cnt1; j += simd_float::size()) {
		for (int n = 0; n < MP; n++) {
			for (int k = 0; k < simd_float::size(); k++) {
				M[n][k] = y[j + k].m[n];
			}
		}
		for (int dim = 0; dim < NDIM; dim++) {
			for (int k = 0; k < simd_float::size(); k++) {
				Y[dim][k] = y[j + k].x[dim];
			}
		}

		vect<simd_float> dX;
		if (opts.ewald) {
			for (int dim = 0; dim < NDIM; dim++) {
				dX[dim] = simd_float(simd_double(X[dim] - Y[dim]) * simd_double(POS_INV)); // 18
			}
		} else {
			for (int dim = 0; dim < NDIM; dim++) {
				dX[dim] = simd_float(simd_double(X[dim]) * simd_double(POS_INV) - simd_double(Y[dim]) * simd_double(POS_INV));
			}
		}
		multipole_interaction(Lacc, M, dX);												// 986
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
				dX[dim] = simd_float(simd_double(X[dim] - Y[dim]) * simd_double(POS_INV)); // 18
			}
		} else {
			for (int dim = 0; dim < NDIM; dim++) {
				dX[dim] = simd_float(simd_double(X[dim]) * simd_double(POS_INV) - simd_double(Y[dim]) * simd_double(POS_INV));
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
	if (x.size() == 0) {
		return 0;
	}
	std::uint64_t flop = 0;

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
				dX0[dim] = simd_float(simd_double(X[dim] - Y[dim]) * simd_double(POS_INV));
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

std::uint64_t gravity_PC_ewald(std::vector<force> &f, const std::vector<vect<pos_type>> &x, std::vector<multi_src> &y) {
	if (x.size() == 0) {
		return 0;
	}
	std::uint64_t flop = 0;

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
	for (int j = cnt1; j < cnt2; j++) {
		y[j].m = 0.0;
		y[j].x = y[cnt1 - 1].x;
	}
	for (int j = 0; j < cnt1; j += simd_float::size()) {
		for (int n = 0; n < MP; n++) {
			for (int k = 0; k < simd_float::size(); k++) {
				M[n][k] = y[j + k].m[n];
			}
		}
		for (int dim = 0; dim < NDIM; dim++) {
			for (int k = 0; k < simd_float::size(); k++) {
				Y[dim][k] = y[j + k].x[dim];
			}
		}
		for (int i = 0; i < x.size(); i++) {
			for (int dim = 0; dim < NDIM; dim++) {
				X[dim] = x[i][dim];
			}

			vect<simd_float> dX;
			for (int dim = 0; dim < NDIM; dim++) {
				dX[dim] = simd_float(simd_double(X[dim] - Y[dim]) * simd_double(POS_INV)); // 18
			}
			std::pair<simd_double, vect<simd_double>> this_f;
			multipole_interaction(this_f, M, dX, true);	// 251466

			for (int dim = 0; dim < NDIM; dim++) {
				G[i][dim] += this_f.second[dim];  // 0 / 3
			}
			Phi[i] += this_f.first;		        // 0 / 1
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

std::uint64_t gravity_CC_ewald(expansion<double> &L, const vect<pos_type> &x, std::vector<multi_src> &y) {
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
	for (int j = cnt1; j < cnt2; j++) {
		y[j].m = 0.0;
		y[j].x = y[cnt1 - 1].x;
	}

	for (int dim = 0; dim < NDIM; dim++) {
		X[dim] = x[dim];
	}
	for (int j = 0; j < cnt1; j += simd_float::size()) {
		for (int n = 0; n < MP; n++) {
			for (int k = 0; k < simd_float::size(); k++) {
				M[n][k] = y[j + k].m[n];
			}
		}
		for (int dim = 0; dim < NDIM; dim++) {
			for (int k = 0; k < simd_float::size(); k++) {
				Y[dim][k] = y[j + k].x[dim];
			}
		}
		vect<simd_float> dX;
		for (int dim = 0; dim < NDIM; dim++) {
			dX[dim] = simd_float(simd_double(X[dim] - Y[dim]) * simd_double(POS_INV)); // 18
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
			dX[dim] = simd_float(simd_double(X[dim] - Y[dim]) * simd_double(POS_INV)); // 18
		}
		multipole_interaction(Lacc, M, dX, true);										// 251176
	}

	for (int i = 0; i < LP; i++) {
		L[i] += Lacc[i].sum();
	}
	y.resize(cnt1);
	return 251233 * cnt1;
}

