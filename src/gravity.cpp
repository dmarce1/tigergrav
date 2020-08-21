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

double ewald_far_separation(const vect<double> x, double r, double l) {
	constexpr double r_e = 0.025;
	static const auto opts = options::get();
	if (x.dot(x) == 0.0) {
		if (r == 0.0 || r < l / 2.0) {
			return 4.0 * r_e;
		} else {
			return 8.0 * r_e * r / l;
		}
	} else {
		return std::max(0.25, ewald_near_separation(x));
	}

}

// 43009,703
template<class DOUBLE, class SINGLE> // 1006 // 1077 + 689 * NREAL + 64 * NFOUR
inline void multipole_interaction(expansion<DOUBLE> &L, const multipole<SINGLE> &M2, vect<SINGLE> dX, bool ewald = false) { // 670/700 + 418 * NREAL + 50 * NFOUR
	static const expansion_factors<SINGLE> expansion_factor;
	expansion<SINGLE> D;
	if (ewald) {
		D = green_ewald(dX);		// 317 + 689 * NREAL + 64 * NFOUR
	} else {
		D = green_direct(dX);        // 246
	}

	//232 S / 264D / 760
	auto &L0 = L();
	L0 += M2() * D();																// 1  / 2
	for (int a = 0; a < 3; a++) {
		for (int b = a; b < 3; b++) {
			L0 += M2(a, b) * D(a, b) * expansion_factor(a, b);						// 12 / 12
			for (int c = b; c < 3; c++) {
				L0 -= M2(a, b, c) * D(a, b, c) * expansion_factor(a, b, c);			// 20 / 20
			}
		}
	}
	for (int a = 0; a < 3; a++) {
		auto &La = L(a);
		La += M2() * D(a);
		for (int b = 0; b < 3; b++) {
			for (int c = b; c < 3; c++) {
				La += M2(c, b) * D(a, b, c) * expansion_factor(c, b);				// 36 / 36
				for (int d = c; d < 3; d++) {
					La -= M2(b, c, d) * D(a, b, c, d) * expansion_factor(b, c, d);	// 60 / 60
				}
			}
		}
	}

	for (int a = 0; a < 3; a++) {
		for (int b = a; b < 3; b++) {
			auto &Lab = L(a, b);
			Lab += M2() * D(a, b);													// 6  / 12
			for (int c = 0; c < 3; c++) {
				for (int d = c; d < 3; d++) {
					Lab += M2(c, d) * D(a, b, c, d) * expansion_factor(c, d);		// 72 / 72
				}
			}
		}
	}

	for (int a = 0; a < 3; a++) {
		for (int b = a; b < 3; b++) {
			for (int c = b; c < 3; c++) {
				auto &Labc = L(a, b, c);
				Labc += M2() * D(a, b, c);										// 10 / 20
			}
		}
	}
	for (int a = 0; a < 3; a++) {
		for (int b = a; b < 3; b++) {
			for (int c = b; c < 3; c++) {
				for (int d = c; d < 3; d++) {
					auto &Labcd = L(a, b, c, d);
					Labcd += M2() * D(a, b, c, d);								// 15 / 30
				}
			}
		}
	}
}

template<class DOUBLE, class SINGLE> // 423 / 494 + 689 * NREAL + 64 * NFOUR
inline void multipole_interaction(expansion<DOUBLE> &L, const SINGLE &M, vect<SINGLE> dX, bool ewald = false) { // 390 / 47301
	static const expansion_factors<SINGLE> expansion_factor;
	expansion<SINGLE> D;
	if (ewald) {
		D = green_ewald(dX);		// 317 + 689 * NREAL + 64 * NFOUR
	} else {
		D = green_direct(dX);          // 246
	}

	//37S + 70D = 177
	auto &L0 = L();
	L0 += M * D();													// 1 / 2
	for (int a = 0; a < 3; a++) {
		auto &La = L(a);
		La += M * D(a);												// 3 / 6
		for (int b = a; b < 3; b++) {
			auto &Lab = L(a, b);
			Lab += M * D(a, b);										// 6 / 12
			for (int c = b; c < 3; c++) {
				auto &Labc = L(a, b, c);
				Labc += M * D(a, b, c);								// 10 / 20
				for (int d = c; d < 3; d++) {
					auto &Labcd = L(a, b, c, d);
					Labcd += M * D(a, b, c, d);						// 15 / 30
				}
			}
		}
	}
}

template<class SINGLE, class DOUBLE> // 648 / 719 + 689 * NREAL + 64 * NFOUR
inline void multipole_interaction(std::pair<DOUBLE, vect<DOUBLE>> &f, const multipole<SINGLE> &M, vect<SINGLE> dX, bool ewald = false) { // 517 / 47428
	static const expansion_factors<SINGLE> expansion_factor;
	expansion<SINGLE> D;
	if (ewald) {
		D = green_ewald(dX);				// 317 + 689 * NREAL + 64 * NFOUR
	} else {
		D = green_direct(dX);				// 246
	}

	//132S + 135D = 402
	auto &ffirst = f.first;
	ffirst = M() * D();																// 1 /  1
	for (int a = 0; a < 3; a++) {
		for (int b = a; b < 3; b++) {
			ffirst += M(a, b) * D(a, b) * expansion_factor(a, b);					// 12 / 12
		}
	}
	for (int a = 0; a < 3; a++) {
		for (int b = 0; b < 3; b++) {
			for (int c = b; c < 3; c++) {
				ffirst -= M(a, b, c) * D(a, b, c) * expansion_factor(a, b, c);		// 20 / 20
			}
		}
	}
	f.second = DOUBLE(0);
	for (int a = 0; a < 3; a++) {
		f.second[a] -= M() * D(a);													// 3 / 6
	}
	for (int a = 0; a < 3; a++) {
		for (int b = 0; b < 3; b++) {
			for (int c = b; c < 3; c++) {
				f.second[a] -= M(c, b) * D(a, b, c) * expansion_factor(c, b);		// 36 / 36
			}
		}
	}
	for (int a = 0; a < 3; a++) {
		for (int b = 0; b < 3; b++) {
			for (int c = b; c < 3; c++) {
				for (int d = c; d < 3; d++) {
					auto &fseconda = f.second[a];
					fseconda += M(c, b, d) * D(a, b, c, d) * expansion_factor(b, c, d); // 60 / 60
				}
			}
		}
	}
}

std::uint64_t gravity_PP_direct(std::vector<force> &f, const std::vector<vect<pos_type>> &x, std::vector<vect<pos_type>> &y, bool do_phi) {
	if (x.size() == 0) {
		return 0;
	}
	std::uint64_t flop = 0;
	static const auto opts = options::get();
	static const bool ewald = opts.ewald;
	static const auto h = opts.soft_len;
	static const auto hinv = 1.0 / h;
	static const simd_float tiny = std::numeric_limits<float>::min();
	static const simd_float H(h);
	static const simd_float Hinv(1.0 / h);
	static const simd_float H3inv(1.0 / (h * h * h));
	static const auto zero = simd_float(0);
	vect<simd_int> X, Y;
	std::vector<vect<simd_double>> G(x.size());
	std::vector<simd_double> Phi(x.size());
	for (int i = 0; i < x.size(); i++) {
		G[i] = vect<simd_double>(0.0);
		Phi[i] = 0.0;
	}
	const auto cnt1 = y.size();
	const auto cnt2 = ((cnt1 - 1 + simd_float::size()) / simd_float::size()) * simd_float::size();
	y.resize(cnt2);
	for (int j = cnt1; j < cnt2; j++) {
		y[j] = 0;
	}
	simd_float M(opts.m_tot / opts.problem_size);
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

		for (int i = 0; i < x.size(); i++) {
			for (int dim = 0; dim < NDIM; dim++) {
				X[dim] = simd_int(x[i][dim]);
			}

			vect<simd_float> dX;
			if (opts.ewald) {
				for (int dim = 0; dim < NDIM; dim++) {
					dX[dim] = simd_float(simd_double(X[dim] - Y[dim]) * simd_double(POS_INV));
				}
			} else {
				for (int dim = 0; dim < NDIM; dim++) {
					dX[dim] = simd_float(simd_double(X[dim]) * simd_double(POS_INV) - simd_double(Y[dim]) * simd_double(POS_INV));
				}
			}
			const simd_float r2 = dX.dot(dX);																				// 5 OP
			const simd_float r = sqrt(r2);																					// 1
			const simd_float rinv = simd_float(1) / max(r, H);      														// 3 OP
			const simd_float rinv3 = rinv * rinv * rinv;   																	// 2 OP
			simd_float sw1 = r > H;   																						// 1 OP
			simd_float sw2 = simd_float(1.0) - sw1;									// 4 OP
			const simd_float roh = min(r * Hinv, 1);																		// 2 OP
			const simd_float roh2 = roh * roh;																				// 1 OP

			const simd_float f1 = rinv3;

			simd_float f2 = simd_float(-35.0 / 16.0);																// 1
			f2 = fmadd(f2, roh2, simd_float(+135.0 / 16.0));
			f2 = fmadd(f2, roh2, simd_float(-189.0 / 16.0));
			f2 = fmadd(f2, roh2, simd_float(+105.0 / 16.0));
			f2 *= H3inv;
			const auto dXM = dX * M;
			for (int dim = 0; dim < NDIM; dim++) {
				G[i][dim] -= simd_double(dXM[dim] * (sw1 * f1 + sw2 * f2));  													// 21 OP
			}

			if (do_phi) {
				const simd_float p1 = rinv;

				simd_float p2 = simd_float(+35.0 / 128.0);
				p2 = fmadd(p2, roh2, simd_float(-45.0 / 32.0));
				p2 = fmadd(p2, roh2, simd_float(189.0 / 64.0));
				p2 = fmadd(p2, roh2, simd_float(-105.0 / 32.0));
				p2 = fmadd(p2, roh2, simd_float(315.0 / 128.0));
				p2 * Hinv;

				Phi[i] -= simd_double((sw1 * p1 + sw2 * p2) * M);		        												// 7 OP
			}
		}
	}
	for (int i = 0; i < x.size(); i++) {
		for (int dim = 0; dim < NDIM; dim++) {
			f[i].g[dim] += G[i][dim].sum();
		}
		if (do_phi) {
			f[i].phi += Phi[i].sum();
			f[i].phi += 315.0 / 128.0 * hinv;
		}
	}

	y.resize(cnt1);
	return (110 * cnt1 + simd_float::size() * 4) * x.size();
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
	vect<simd_int> X;
	vect<simd_double> Y;
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
			vect<simd_float> dXimage;
			for (int dim = 0; dim < NDIM; dim++) {
				const simd_double dist = simd_double(X[dim]) * simd_double(POS_INV) + simd_double(0.5) - Y[dim];
				dX[dim] = simd_float(dist);             		// 3 OP
				dXimage[dim] = simd_float(simd_double(1) - abs(dist));
			}
			if (ewald) {
				for (int dim = 0; dim < NDIM; dim++) {
					const auto absdx = abs(dX[dim]);																		// 3 OP
					dX[dim] = copysign(min(absdx, dXimage[dim]), dX[dim] * (half - absdx));  								// 15 OP
				}
			}
			std::pair<simd_double, vect<simd_double>> this_f;
			multipole_interaction(this_f, M, dX); // 517 OP

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
	y.resize(cnt1);
	return (546 * cnt1 + simd_float::size() * 4) * x.size();
}

std::uint64_t gravity_CC_direct(expansion<double> &L, const vect<double> &x, std::vector<multi_src> &y) {
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
	vect<simd_double> X, Y;
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
		vect<simd_float> dXimage;
		for (int dim = 0; dim < NDIM; dim++) {
			const simd_double dist = X[dim] - Y[dim];
			dX[dim] = simd_float(dist);             		// 3 OP
			dXimage[dim] = simd_float(simd_double(1) - abs(dist));
		}
		if (ewald) {
			for (int dim = 0; dim < NDIM; dim++) {
				const auto absdx = abs(dX[dim]);																		// 3 OP
				dX[dim] = copysign(min(absdx, dXimage[dim]), dX[dim] * (half - absdx));  								// 15 OP
			}
		}
		multipole_interaction(Lacc, M, dX);												// 670 OP
	}

	for (int i = 0; i < LP; i++) {
		L[i] += Lacc[i].sum();
	}
	y.resize(cnt1);
	return 691 * cnt1 + LP * simd_float::size();
}

std::uint64_t gravity_CP_direct(expansion<double> &L, const vect<double> &x, std::vector<vect<pos_type>> &y) {
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
	vect<simd_int> Y;
	vect<simd_double> X;
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
		vect<simd_float> dXimage;
		for (int dim = 0; dim < NDIM; dim++) {
			const simd_double dist = X[dim] - (simd_double(Y[dim]) * simd_double(POS_INV) + simd_double(0.5));
			dX[dim] = simd_float(dist);             		// 3 OP
			dXimage[dim] = simd_float(simd_double(1) - abs(dist));
		}
		if (ewald) {
			for (int dim = 0; dim < NDIM; dim++) {
				const auto absdx = abs(dX[dim]);																		// 3 OP
				dX[dim] = copysign(min(absdx, dXimage[dim]), dX[dim] * (half - absdx));  								// 15 OP
			}
		}
		if (ewald) {
			for (int dim = 0; dim < NDIM; dim++) {
				const auto absdx = abs(dX[dim]);										// 3 OP
				dX[dim] = copysign(min(absdx, one - absdx), dX[dim] * (half - absdx));  // 15 OP
			}
		}
		multipole_interaction(Lacc, M, dX);												// 390 OP
	}

	for (int i = 0; i < LP; i++) {
		L[i] += Lacc[i].sum();
	}
	y.resize(cnt1);
	return 408 * cnt1 + simd_float::size() * LP;
}

static const ewald_indices indices_real(EWALD_REAL_N2);
static const ewald_indices indices_four(EWALD_FOUR_N2);

std::uint64_t gravity_PP_ewald(std::vector<force> &f, const std::vector<vect<pos_type>> &x, std::vector<vect<pos_type>> &y) {
	if (x.size() == 0) {
		return 0;
	}
	std::uint64_t flop = 0;

	static const auto one = simd_float(1.0);
	static const auto half = simd_float(0.5);

	static const auto opts = options::get();
	static const simd_float M(opts.m_tot / opts.problem_size);
	simd_float mask;
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

			vect<simd_float> dX0;
			for (int dim = 0; dim < NDIM; dim++) {
				dX0[dim] = simd_float(simd_double(X[dim] - Y[dim]) * simd_double(POS_INV));
			}
			constexpr int nmax = 2;
			constexpr int hmax = 2;
			static const simd_float tiny = std::numeric_limits<float>::min();
			static const simd_float maxrinv = 0.99 / std::pow(std::numeric_limits<float>::min(), 1.0 / 3.0) / M[0];
			static const simd_float two(2);
			static const simd_float twopi(2 * M_PI);
			static const simd_float pioverfour(M_PI / 4.0);
			static const simd_float fouroversqrtpi(4.0 / sqrt(M_PI));
			static const simd_float phi0(2.8372975);
			vect<float> h;
			vect<simd_float> n;
			simd_double phi = 0.0;
			vect<simd_double> g;
			g = simd_double(0);
			for (int i = 0; i < indices_real.size(); i++) {					// 78
				h = indices_real[i];
				n = h;
				const vect<simd_float> dx = dX0 - n;                         // 3 OP
				const simd_float r2 = dx.dot(dx);							// 3
				const simd_float r = sqrt(r2);                      			// 1
				const simd_float mask = r < 3.6;
				simd_float rinv = mask * r / (r2 + tiny);						// 2
				rinv = min(rinv, maxrinv);
				const simd_float r2inv = rinv * rinv;						// 1
				const simd_float r3inv = r2inv * rinv;						// 1
				simd_float expfac;
				const simd_float erfc = erfcexp(two * r, &expfac);		// 51
				const simd_float d0 = -erfc * rinv;							// 2
				const simd_float expfactor = fouroversqrtpi * r * expfac;	// 2
				const simd_float d1 = (expfactor + erfc) * r3inv;			// 2
				phi += d0; 													// 1
				for (int a = 0; a < NDIM; a++) {
					g[a] -= (dX0[a] - n[a]) * d1;							// 9
				}
			}
			for (int i = 0; i < indices_four.size(); i++) {					// 48
				const expansion<float> &H = periodic[i];
				h = indices_four[i];
				simd_float hdotdx = dX0[0] * h[0];							// 1
				for (int dim = 1; dim < NDIM; dim++) {
					hdotdx += dX0[dim] * h[dim];							// 4
				}
				const simd_float omega = twopi * hdotdx;						// 1
				simd_float s, c;
				sincos(omega, &s, &c);										// 34
				phi += H() * c;												// 2
				for (int dim = 0; dim < NDIM; dim++) {
					g[dim] -= H(dim) * s;									// 6
				}
			}
			const simd_float r = abs(dX0);									// 5
			simd_float rinv = r / (r * r + tiny);						// 3
			rinv = min(rinv, maxrinv);
			phi = pioverfour + phi + rinv;									// 2
			const simd_float sw = r > simd_float(0);							// 1
			phi = phi0 * (simd_float(1.0) - sw) + phi * sw;					// 4
			const auto rinv3 = rinv * rinv * rinv;							// 2
			for (int dim = 0; dim < NDIM; dim++) {
				g[dim] = g[dim] + dX0[dim] * rinv3;							// 6
			}
			Phi[I] += M * phi * mask;										// 3
			G[I] += g * M * mask;											// 9
//			printf( "%e %e %e %e %e\n", g[0][0], X[0][0], Y[0][0], X[0][0] - Y[0][0], dX0[0][0]);
		}
	}
	for (int i = 0; i < x.size(); i++) {
		for (int dim = 0; dim < NDIM; dim++) {
			f[i].g[dim] += G[i][dim].sum();
		}
		f[i].phi += Phi[i].sum();
	}
	y.resize(cnt1);
	return ((35 + indices_real.size() * 78 + indices_four.size() * 48) * cnt1 + simd_float::size() * 4) * x.size();
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
	vect<simd_int> X;
	vect<simd_double> Y;
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
			vect<simd_float> dXimage;
			for (int dim = 0; dim < NDIM; dim++) {
				const simd_double dist = simd_double(X[dim]) * simd_double(POS_INV) + simd_double(0.5) - Y[dim];
				dX[dim] = simd_float(dist);             		// 3 OP
				dXimage[dim] = simd_float(simd_double(1) - abs(dist));
			}
			for (int dim = 0; dim < NDIM; dim++) {
				const auto absdx = abs(dX[dim]);																		// 3 OP
				dX[dim] = copysign(min(absdx, dXimage[dim]), dX[dim] * (half - absdx));  								// 15 OP
			}
			std::pair<simd_double, vect<simd_double>> this_f;
			multipole_interaction(this_f, M, dX, true);	//700 + 418 * NREAL + 50 * NFOUR

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
	y.resize(cnt1);
	return ((729 + 418 * indices_real.size() + 50 * indices_four.size()) * cnt1 + simd_float::size() * 4) * x.size();
}

std::uint64_t gravity_CC_ewald(expansion<double> &L, const vect<double> &x, std::vector<multi_src> &y) {
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
	vect<simd_double> X, Y;
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
		vect<simd_float> dXimage;
		for (int dim = 0; dim < NDIM; dim++) {
			const simd_double dist = X[dim] - Y[dim];
			dX[dim] = simd_float(dist);             		// 3 OP
			dXimage[dim] = simd_float(simd_double(1) - abs(dist));
		}
		for (int dim = 0; dim < NDIM; dim++) {
			const auto absdx = abs(dX[dim]);																		// 3 OP
			dX[dim] = copysign(min(absdx, dXimage[dim]), dX[dim] * (half - absdx));  								// 15 OP
		}
		multipole_interaction(Lacc, M, dX, true);												// 700 + 418 * NREAL + 50 * NFOUR
	}

	for (int i = 0; i < LP; i++) {
		L[i] += Lacc[i].sum();
	}
	y.resize(cnt1);
	return (721 + 418 * indices_real.size() + 50 * indices_four.size()) * cnt1 + LP * simd_float::size();
}

std::uint64_t gravity_CP_ewald(expansion<double> &L, const vect<double> &x, std::vector<vect<pos_type>> &y) {
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
	vect<simd_int> Y;
	vect<simd_double> X;
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
		vect<simd_float> dXimage;
		for (int dim = 0; dim < NDIM; dim++) {
			const simd_double dist = X[dim] - (simd_double(Y[dim]) * simd_double(POS_INV) + simd_double(0.5));
			dX[dim] = simd_float(dist);             		// 3 OP
			dXimage[dim] = simd_float(simd_double(1) - abs(dist));
		}
		for (int dim = 0; dim < NDIM; dim++) {
			const auto absdx = abs(dX[dim]);																		// 3 OP
			dX[dim] = copysign(min(absdx, dXimage[dim]), dX[dim] * (half - absdx));  								// 15 OP
		}
		multipole_interaction(Lacc, M, dX, true);										// 700 + 418 * NREAL + 50 * NFOUR
	}

	for (int i = 0; i < LP; i++) {
		L[i] += Lacc[i].sum();
	}
	y.resize(cnt1);
	return (721 + 418 * indices_real.size() + 50 * indices_four.size()) * cnt1 + LP * simd_float::size();
}

