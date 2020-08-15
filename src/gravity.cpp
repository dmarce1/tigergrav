#include <tigergrav/expansion.hpp>
#include <tigergrav/gravity.hpp>
#include <tigergrav/options.hpp>
#include <tigergrav/simd.hpp>

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
	static const auto zero = simd_float(0);
	vect<simd_float> X, Y;
	std::vector<vect<simd_float>> G(x.size());
	std::vector<simd_float> Phi(x.size());
	for (int i = 0; i < x.size(); i++) {
		G[i] = vect<simd_float>(0.0);
		Phi[i] = 0.0;
	}
	const auto cnt1 = y.size();
	const auto cnt2 = ((cnt1 - 1 + simd_float::size()) / simd_float::size()) * simd_float::size();
	y.resize(cnt2);
	for (int j = cnt1; j < cnt2; j++) {
		y[j] = vect<float>(2.0);
	}
	simd_float M(opts.m_tot / opts.problem_size);
	for (int j = 0; j < cnt1; j += simd_float::size()) {
		for (int dim = 0; dim < NDIM; dim++) {
			for (int k = 0; k < simd_float::size(); k++) {
				Y[dim][k] = y[j + k][dim];
			}
		}
		if (j + simd_real::size() > cnt1) {
			for (int k = cnt1; k < cnt2; k++) {
				M[k - j] = 0.0;
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
			const simd_float r2 = dX.dot(dX);																				// 5 OP
			const simd_float zero_mask = r2 > simd_float(0);																// 1
			const simd_float r = sqrt(r2);																					// 1
			const simd_float rinv = zero_mask * r / (r2 + tiny);       														// 3 OP
			const simd_float rinv3 = rinv * rinv * rinv;   																	// 2 OP
			simd_float sw1 = r > H;   																						// 1 OP
			simd_float sw2 = (simd_float(1.0) - sw1) * (r > simd_float(0.5) * H);											// 4 OP
			simd_float sw3 = simd_float(1.0) - (sw1 + sw2);																	// 2
			const simd_float roh = min(r * Hinv, 1);																		// 2 OP
			const simd_float hor = h * min(rinv, simd_float(2.0) * Hinv);													// 2 OP
			const simd_float hor2 = hor * hor;																				// 1
			const simd_float hor3 = hor2 * hor;																				// 1
			const simd_float roh2 = roh * roh;																				// 1 OP
			const simd_float roh3 = roh2 * roh;																				// 1 OP
			const simd_float roh4 = roh2 * roh2;																			// 1 OP
			const simd_float roh5 = roh3 * roh2;																			// 1 OP

			const simd_float f1 = rinv3;

			simd_float f2 = simd_float(-1.0 / 15.0) * hor3;																	// 1
			f2 += simd_float(64.0 / 3.0);																					// 1
			f2 -= simd_float(48.0) * roh;																					// 2
			f2 += simd_float(192.0 / 5.0) * roh2;																			// 2
			f2 -= simd_float(32.0 / 3.0) * roh3;																			// 2
			f2 *= H3inv;																									// 1

			simd_float f3 = simd_float(32.0 / 3.0);
			f3 -= simd_float(192.0 / 5.0) * roh2;																			// 2
			f3 += simd_float(32.0) * roh3;																					// 2
			f3 *= H3inv;																									// 1

			const auto dXM = dX * M;
			for (int dim = 0; dim < NDIM; dim++) {
				G[i][dim] -= dXM[dim] * (sw1 * f1 + sw2 * f2 + sw3 * f3);  													// 21 OP
			}

			const simd_float p1 = rinv;

			simd_float p2 = simd_float(16.0 / 5.0);
			p2 -= simd_float(1.0 / 15.0) * hor;																				// 2
			p2 -= simd_float(32.0 / 3.0) * roh2;																			// 2
			p2 += simd_float(16.0) * roh3;																					// 1
			p2 -= simd_float(48.0 / 5.0) * roh4;																			// 2
			p2 += simd_float(32.0 / 5.0) * roh5;																			// 2
			p2 *= Hinv;																										// 1

			simd_float p3 = simd_float(14.0 / 15.0);
			p3 -= simd_float(16.0 / 3.0) * roh2;																			// 2
			p3 += simd_float(48.0 / 5.0) * roh4;																			// 2
			p3 -= simd_float(32.0 / 5.0) * roh5;																			// 2
			p3 *= Hinv;																										// 1

			const auto tmp = M * zero_mask; 	            																// 1 OP
			Phi[i] -= (sw1 * p1 + sw2 * p2 + sw3 * p3) * tmp;		        												// 7 OP
		}
	}
	for (int i = 0; i < x.size(); i++) {
		for (int dim = 0; dim < NDIM; dim++) {
			f[i].g[dim] += G[i][dim].sum();
		}
		f[i].phi += Phi[i].sum();
	}

	y.resize(cnt1);
	return (110 * cnt1 + simd_float::size() * 4) * x.size();
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
	std::vector<vect<simd_float>> G(x.size(), vect<simd_float>(simd_float(0)));
	std::vector<simd_float> Phi(x.size(), simd_float(0.0));
	const auto cnt1 = y.size();
	const auto cnt2 = ((cnt1 - 1 + simd_float::size()) / simd_float::size()) * simd_float::size();
	y.resize(cnt2);
	for (int j = cnt1; j < cnt2; j++) {
		y[j].m = 0.0;
		y[j].x = vect<float>(2.0);
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
				X[dim] = simd_float(x[i][dim]);
			}

			vect<simd_float> dX = X - Y;             		// 3 OP
			if (ewald) {
				for (int dim = 0; dim < NDIM; dim++) {
					const auto absdx = abs(dX[dim]);										// 3 OP
					dX[dim] = copysign(min(absdx, one - absdx), dX[dim] * (half - absdx));  // 15 OP
				}
			}
			auto this_f = multipole_interaction(M, dX); // 517 OP

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

std::uint64_t gravity_CC_direct(expansion<double> &L, const vect<ireal> &x, std::vector<multi_src> &y) {
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
		y[j].x = vect<ireal>(2.0);
	}

	for (int dim = 0; dim < NDIM; dim++) {
		X[dim] = simd_real(x[dim]);
	}
	for (int j = 0; j < cnt1; j += simd_real::size()) {
		for (int n = 0; n < MP; n++) {
			for (int k = 0; k < simd_real::size(); k++) {
				M[n][k] = y[j + k].m[n];
			}
		}
		for (int dim = 0; dim < NDIM; dim++) {
			for (int k = 0; k < simd_real::size(); k++) {
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
		multipole_interaction(Lacc, M, dX);												// 670 OP
	}

	for (int i = 0; i < LP; i++) {
		L[i] += Lacc[i].sum();
	}
	y.resize(cnt1);
	return 691 * cnt1 + LP * simd_float::size();
}

std::uint64_t gravity_CP_direct(expansion<double> &L, const vect<ireal> &x, std::vector<vect<float>> &y) {
	if (y.size() == 0) {
		return 0;
	}
	static const auto one = simd_real(1.0);
	static const auto half = simd_real(0.5);
	std::uint64_t flop = 0;
	static const auto opts = options::get();
	static const auto m = opts.m_tot / opts.problem_size;
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
		y[j] = vect<ireal>(2.0);
	}

	for (int dim = 0; dim < NDIM; dim++) {
		X[dim] = simd_real(x[dim]);
	}
	M = m;
	for (int j = 0; j < cnt1; j += simd_real::size()) {
		for (int dim = 0; dim < NDIM; dim++) {
			for (int k = 0; k < simd_real::size(); k++) {
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

std::uint64_t gravity_PP_ewald(std::vector<force> &f, const std::vector<vect<float>> &x, std::vector<vect<float>> &y) {
	if (x.size() == 0) {
		return 0;
	}
	std::uint64_t flop = 0;

	static const auto one = simd_real(1.0);
	static const auto half = simd_real(0.5);

	static const auto opts = options::get();
	static const simd_real M(opts.m_tot / opts.problem_size);
	simd_real mask;
	static const bool ewald = opts.ewald;
	static const auto h = opts.soft_len;
	static const auto h2 = h * h;
//	printf( "%i %i \n", indices_real.size(), indices_four.size() );

	static const periodic_parts periodic;
	vect<simd_real> X, Y;
	std::vector<vect<simd_real>> G(x.size(), vect<simd_float>(simd_float(0)));
	std::vector<simd_real> Phi(x.size(), simd_float(0.0));
	const auto cnt1 = y.size();
	const auto cnt2 = ((cnt1 - 1 + simd_real::size()) / simd_real::size()) * simd_real::size();
	y.resize(cnt2);
	for (int j = cnt1; j < cnt2; j++) {
		y[j] = vect<float>(0.0);
	}
	for (int J = 0; J < cnt1; J += simd_real::size()) {
		for (int dim = 0; dim < NDIM; dim++) {
			for (int k = 0; k < simd_real::size(); k++) {
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
			static const simd_float two(2);
			static const simd_float twopi(2 * M_PI);
			static const simd_float pioverfour(M_PI / 4.0);
			static const simd_float fouroversqrtpi(4.0 / sqrt(M_PI));
			static const simd_float phi0(2.8372975);
			vect<float> h;
			vect<simd_real> n;
			simd_real phi = 0.0;
			vect<simd_real> g;
			g = simd_real(0);
			for (int i = 0; i < indices_real.size(); i++) {					// 78
				h = indices_real[i];
				n = h;
				const vect<simd_real> dx = dX0 - n;                         // 3 OP
				const simd_real r2 = dx.dot(dx);							// 3
				const simd_real r = sqrt(r2);                      			// 1
				const simd_real rinv = r / (r2 + tiny);						// 2
				const simd_real r2inv = rinv * rinv;						// 1
				const simd_real r3inv = r2inv * rinv;						// 1
				simd_float expfac;
				const simd_real erfc = one - erfexp(two * r, &expfac);		// 51
				const simd_real d0 = -erfc * rinv;							// 2
				const simd_real expfactor = fouroversqrtpi * r * expfac;	// 2
				const simd_real d1 = (expfactor + erfc) * r3inv;			// 2
				phi += d0; 													// 1
				for (int a = 0; a < NDIM; a++) {
					g[a] -= (dX0[a] - n[a]) * d1;							// 9
				}
			}
			for (int i = 0; i < indices_four.size(); i++) {					// 48
				const expansion<float> &H = periodic[i];
				h = indices_four[i];
				simd_real hdotdx = dX0[0] * h[0];							// 1
				for (int dim = 1; dim < NDIM; dim++) {
					hdotdx += dX0[dim] * h[dim];							// 4
				}
				const simd_real omega = twopi * hdotdx;						// 1
				simd_real s, c;
				sincos(omega, &s, &c);										// 34
				phi += H() * c;												// 2
				for (int dim = 0; dim < NDIM; dim++) {
					g[dim] -= H(dim) * s;									// 6
				}
			}
			const simd_real r = abs(dX0);									// 5
			const simd_real rinv = r / (r * r + tiny);						// 3
			phi = pioverfour + phi + rinv;									// 2
			const simd_real sw = r > simd_float(0);							// 1
			phi = phi0 * (simd_real(1.0) - sw) + phi * sw;					// 4
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
	std::vector<simd_real> Phi(x.size(), simd_float(0.0));
	const auto cnt1 = y.size();
	const auto cnt2 = ((cnt1 - 1 + simd_real::size()) / simd_real::size()) * simd_real::size();
	y.resize(cnt2);
	for (int j = cnt1; j < cnt2; j++) {
		y[j].m = 0.0;
		y[j].x = vect<float>(1.0);
	}
	for (int j = 0; j < cnt1; j += simd_real::size()) {
		for (int n = 0; n < MP; n++) {
			for (int k = 0; k < simd_real::size(); k++) {
				M[n][k] = y[j + k].m[n];
			}
		}
		for (int dim = 0; dim < NDIM; dim++) {
			for (int k = 0; k < simd_real::size(); k++) {
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
			auto this_f = multipole_interaction(M, dX, true);	//700 + 418 * NREAL + 50 * NFOUR

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

std::uint64_t gravity_CC_ewald(expansion<double> &L, const vect<ireal> &x, std::vector<multi_src> &y) {
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
		for (int n = 0; n < MP; n++) {
			for (int k = 0; k < simd_real::size(); k++) {
				M[n][k] = y[j + k].m[n];
			}
		}
		for (int dim = 0; dim < NDIM; dim++) {
			for (int k = 0; k < simd_real::size(); k++) {
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
		multipole_interaction(Lacc, M, dX, true);												// 700 + 418 * NREAL + 50 * NFOUR
	}

	for (int i = 0; i < LP; i++) {
		L[i] += Lacc[i].sum();
	}
	y.resize(cnt1);
	return (721 + 418 * indices_real.size() + 50 * indices_four.size()) * cnt1 + LP * simd_float::size();
}

std::uint64_t gravity_CP_ewald(expansion<double> &L, const vect<ireal> &x, std::vector<vect<float>> &y) {
	if (y.size() == 0) {
		return 0;
	}
	static const auto one = simd_real(1.0);
	static const auto half = simd_real(0.5);
	std::uint64_t flop = 0;
	static const auto opts = options::get();
	static const auto m = opts.m_tot / opts.problem_size;
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
		for (int dim = 0; dim < NDIM; dim++) {
			for (int k = 0; k < simd_real::size(); k++) {
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
		multipole_interaction(Lacc, M, dX, true);										// 700 + 418 * NREAL + 50 * NFOUR
	}

	for (int i = 0; i < LP; i++) {
		L[i] += Lacc[i].sum();
	}
	y.resize(cnt1);
	return (721 + 418 * indices_real.size() + 50 * indices_four.size()) * cnt1 + LP * simd_float::size();
}

