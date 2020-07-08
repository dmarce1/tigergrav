#include <tigergrav/gravity.hpp>
#include <tigergrav/options.hpp>
#include <tigergrav/simd.hpp>

#include <hpx/include/async.hpp>

constexpr int NE = 64;
constexpr int NEP1 = 65;
constexpr int NEP12 = NEP1 * NEP1;

using ewald_table_t = std::array<float,NEP1*NEP1*NEP1>;
static ewald_table_t epot;
static std::array<ewald_table_t, NDIM> eforce;

static const auto one = simd_vector(1.0);
static const auto half = simd_vector(0.5);
static const simd_vector eps = simd_vector(std::numeric_limits<float>::min());

expansion<simd_vector> Green_direct(const vect<simd_vector> &r) {			 // 136 FLOP
	static const auto h = options::get().soft_len;
	static const simd_vector H2(h * h);
	expansion<simd_vector> D;
	const simd_vector r2 = r.dot(r);											// 5
	const simd_vector rinv = rsqrt(r2 + H2);								    // 2
	const simd_vector r2inv = rinv * rinv;										// 1
	const simd_vector r3inv = rinv * r2inv;										// 1
	const simd_vector r5inv = r3inv * r2inv;									// 1
	const simd_vector r7inv = r5inv * r2inv;									// 1

	D.zero();
	D() = -rinv;																// 1
	for (int i = 0; i < NDIM; i++) {
		D(i) = r[i] * r3inv;													// 3
		for (int j = 0; j <= i; j++) {
			D(i, j) = -simd_vector(3) * r[i] * r[j] * r5inv;					// 24
			if (i == j) {
				D(i, j) += D(i);												// 3
			}
			for (int k = 0; k <= j; k++) {
				D(i, j, k) = simd_vector(15) * r[i] * r[j] * r[k] * r7inv;       // 40
				if (i == j) {
					D(i, j, k) -= simd_vector(3) * r[k] * r5inv;				// 18
				}
				if (i == k) {
					D(i, j, k) -= simd_vector(3) * r[j] * r5inv;				// 18
				}
				if (j == k) {
					D(i, j, k) -= simd_vector(3) * r[i] * r5inv;				// 18
				}
			}
		}
	}
	return D;
}

double EW(general_vect<double, NDIM> x) {
	general_vect<double, NDIM> n, h;
	constexpr int nmax = 5;
	constexpr int hmax = 10;

	double sum1 = 0.0;
	for (int i = -nmax; i <= nmax; i++) {
		for (int j = -nmax; j <= nmax; j++) {
			for (int k = -nmax; k <= nmax; k++) {
				n[0] = i;
				n[1] = j;
				n[2] = k;
				const auto xmn = x - n;                          // 3 OP
				double absxmn = abs(x - n);                      // 5 OP
				if (absxmn < 3.6) {
					const double xmn2 = absxmn * absxmn;         // 1 OP
					const double xmn3 = xmn2 * absxmn;           // 1 OP
					sum1 += -(1.0 - erf(2.0 * absxmn)) / absxmn; // 6 OP
				}
			}
		}
	}
	double sum2 = 0.0;
	for (int i = -hmax; i <= hmax; i++) {
		for (int j = -hmax; j <= hmax; j++) {
			for (int k = -hmax; k <= hmax; k++) {
				h[0] = i;
				h[1] = j;
				h[2] = k;
				const double absh = abs(h);                     // 5 OP
				const double h2 = absh * absh;                  // 1 OP
				if (absh <= 10 && absh > 0) {
					sum2 += -(1.0 / M_PI) * (1.0 / h2 * exp(-M_PI * M_PI * h2 / 4.0) * cos(2.0 * M_PI * h.dot(x))); // 14 OP
				}
			}
		}
	}
	return M_PI / 4.0 + sum1 + sum2 + 1 / abs(x);
}

float ewald_near_separation(const vect<float> x) {
	float d = 0.0;
	for (int dim = 0; dim < NDIM; dim++) {
		const float absx = std::abs(x[dim]);
		float this_d = std::min(absx, (float) 1.0 - absx);
		d += this_d * this_d;
	}
	return std::sqrt(d);
}

float ewald_far_separation(const vect<float> x) {
	float d = 0.0;
	for (int dim = 0; dim < NDIM; dim++) {
		const float absx = std::abs(x[dim]);
		const float this_d = std::min(absx, (float) 1.0 - absx);
		d += this_d * this_d;
	}
	return std::max(std::sqrt(d), float(0.25));
}

std::uint64_t gravity_mono_mono(std::vector<force> &f, const std::vector<vect<float>> &x, std::vector<vect<float>> &y, const bool do_phi) {
	if (x.size() == 0) {
		return 0;
	}
	std::uint64_t flop = 0;
	static const auto opts = options::get();
	static const bool ewald = opts.ewald;
	static const auto m = 1.0 / opts.problem_size;
	static const auto h = opts.soft_len;
	static const auto h2 = h * h;
	static const simd_vector H(h);
	static const simd_vector H2(h2);
	static const simd_vector M(m);
	vect<simd_vector> X, Y;
	std::vector<vect<simd_vector>> nG(x.size(), vect<float>(0.0));
	std::vector<simd_vector> nPhi(x.size(), 0.0);
	const auto cnt1 = y.size();
	const auto cnt2 = ((cnt1 - 1 + SIMD_LEN) / SIMD_LEN) * SIMD_LEN;
	y.resize(cnt2);
	for (int j = cnt1; j < cnt2; j++) {
		y[j] = vect<float>(1.0e+10);
	}
	for (int j = 0; j < cnt1; j += SIMD_LEN) {
		for (int k = 0; k < SIMD_LEN; k++) {
			for (int dim = 0; dim < NDIM; dim++) {
				Y[dim][k] = y[j + k][dim];
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
			simd_vector r2 = dX[0] * dX[0];					// 1 OP
			r2 = fma(dX[1], dX[1], r2);						// 2 OP
			r2 = fma(dX[2], dX[2], r2);                     // 2 OP
			const simd_vector rinv = rsqrt(r2 + H2);        // 2 OP
			const simd_vector rinv3 = rinv * rinv * rinv;   // 2 OP
			for (int dim = 0; dim < NDIM; dim++) {
				const auto tmp = M * rinv3;					// 3 OP
				nG[i][dim] = fma(dX[dim], tmp, nG[i][dim]);  // 6 OP
			}
			if (do_phi) {
				const simd_vector kill_zero = r2 / (r2 + eps);  // 2 OP
				const auto tmp = M * kill_zero; 	            // 1 OP
				nPhi[i] = fma(rinv, tmp, nPhi[i]);		        // 2 OP
			}
		}
	}
	for (int i = 0; i < x.size(); i++) {
		for (int dim = 0; dim < NDIM; dim++) {
			f[i].g[dim] -= nG[i][dim].sum();
		}
		if (do_phi) {
			f[i].phi -= nPhi[i].sum();
		}
	}
	return (do_phi ? 26 : 21 + ewald ? 18 : 0) * cnt1 * x.size();
}

std::uint64_t gravity_mono_multi(std::vector<force> &f, const std::vector<vect<float>> &x, std::vector<multi_source> &y, const bool do_phi) {
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
	vect<simd_vector> X, Y;
	multipole<simd_vector> M;
	std::vector<vect<simd_vector>> G(x.size(), vect<float>(0.0));
	std::vector<simd_vector> Phi(x.size(), 0.0);
	const auto cnt1 = y.size();
	const auto cnt2 = ((cnt1 - 1 + SIMD_LEN) / SIMD_LEN) * SIMD_LEN;
	y.resize(cnt2);
	for (int j = cnt1; j < cnt2; j++) {
		y[j].m() = 0.0;
		for (int n = 0; n < NDIM; n++) {
			for (int m = 0; m <= n; m++) {
				y[j].m(n, m) = 0.0;
			}
		}
		y[j].x = vect<float>(1.0e+10);
	}
	for (int j = 0; j < cnt1; j += SIMD_LEN) {
		for (int k = 0; k < SIMD_LEN; k++) {
			M()[k] = y[j + k].m();
			for (int n = 0; n < NDIM; n++) {
				for (int m = 0; m <= n; m++) {
					M(n, m)[k] = y[j + k].m(n, m);
				}
			}
			for (int dim = 0; dim < NDIM; dim++) {
				Y[dim][k] = y[j + k].x[dim];
			}
		}
		for (int i = 0; i < x.size(); i++) {
			for (int dim = 0; dim < NDIM; dim++) {
				X[dim] = simd_vector(x[i][dim]);
			}

			vect<simd_vector> dX = X - Y;             										// 3 OP
			if (ewald) {
				for (int dim = 0; dim < NDIM; dim++) {
					const auto absdx = abs(dX[dim]);										// 3 OP
					dX[dim] = copysign(dX[dim] * (half - absdx), min(absdx, one - absdx));  // 15 OP
				}
			}
			expansion<simd_vector> D = Green_direct(dX);									// 135

			for (int dim = 0; dim < NDIM; dim++) {
				G[i][dim] -= D(dim) * M();													// 6
				for (int n = 0; n < NDIM; n++) {
					for (int m = 0; m <= n; m++) {
						const simd_vector c0 = n == m ? simd_vector(0.5) : simd_vector(1);
						G[i][dim] -= c0 * D(n, m, dim) * M(n, m);							// 54
					}
				}
			}
			if (do_phi) {
				Phi[i] += D() * M();
				for (int n = 0; n < NDIM; n++) {
					for (int m = 0; m <= n; m++) {
						const simd_vector c0 = n == m ? simd_vector(0.5) : simd_vector(1);
						Phi[i] += c0 * D(n, m) * M(n, m);									// 18
					}
				}
			}
		}
	}
	for (int i = 0; i < x.size(); i++) {
		for (int dim = 0; dim < NDIM; dim++) {
			f[i].g[dim] += G[i][dim].sum();
		}
		if (do_phi) {
			f[i].phi += Phi[i].sum();
		}
	}
	return (do_phi ? 216 : 198 + ewald ? 18 : 0) * cnt1 * x.size();
}

std::uint64_t gravity_multi_multi(expansion<double> &L, const vect<float> &x, std::vector<multi_source> &y) {
	std::uint64_t flop = 0;
	static const auto opts = options::get();
	static const bool ewald = opts.ewald;
	static const auto h = opts.soft_len;
	static const auto h2 = h * h;
	static const simd_vector H(h);
	static const simd_vector H2(h2);
	vect<simd_vector> X, Y;
	multipole<simd_vector> M;
	expansion<simd_vector> Lacc;
	Lacc.zero();

	const auto cnt1 = y.size();
	const auto cnt2 = ((cnt1 - 1 + SIMD_LEN) / SIMD_LEN) * SIMD_LEN;
	y.resize(cnt2);
	for (int j = cnt1; j < cnt2; j++) {
		y[j].m() = 0.0;
		for (int n = 0; n < NDIM; n++) {
			for (int m = 0; m <= n; m++) {
				y[j].m(n, m) = 0.0;
			}
		}
		y[j].x = vect<float>(1.0);
	}
	for (int dim = 0; dim < NDIM; dim++) {
		X[dim] = simd_vector(x[dim]);
	}
	for (int j = 0; j < cnt1; j += SIMD_LEN) {
		for (int k = 0; k < SIMD_LEN; k++) {
			M()[k] = y[j + k].m();
			for (int n = 0; n < NDIM; n++) {
				for (int m = 0; m <= n; m++) {
					M(n, m)[k] = y[j + k].m(n, m);
				}
			}
			for (int dim = 0; dim < NDIM; dim++) {
				Y[dim][k] = y[j + k].x[dim];
			}
		}

		vect<simd_vector> dX = X - Y;             										// 3 OP
		if (ewald) {
			for (int dim = 0; dim < NDIM; dim++) {
				const auto absdx = abs(dX[dim]);										// 3 OP
				dX[dim] = copysign(dX[dim] * (half - absdx), min(absdx, one - absdx));  // 15 OP
			}
		}
		expansion<simd_vector> D = Green_direct(dX);									// 135

		Lacc() += D() * M();															//  2
		for (int n = 0; n < NDIM; n++) {
			for (int m = 0; m <= n; m++) {
				const simd_vector c0 = n == m ? simd_vector(0.5) : simd_vector(1);
				Lacc() += c0 * D(n, m) * M(n, m);										// 18
			}
		}

		for (int n = 0; n < NDIM; n++) {
			Lacc(n) += D(n) * M();														// 6
			for (int m = 0; m < NDIM; m++) {
				for (int l = 0; l <= m; l++) {
					const simd_vector c0 = l == m ? simd_vector(0.5) : simd_vector(1);
					Lacc(n) += c0 * D(n, m, l) * M(m, l);								// 54
				}
			}
		}

		for (int n = 0; n < NDIM; n++) {
			for (int m = 0; m <= n; m++) {
				const simd_vector c0 = n == m ? simd_vector(0.5) : simd_vector(1);
				Lacc(n, m) += c0 * D(n, m) * M();										// 18
			}
		}

		for (int n = 0; n < NDIM; n++) {
			for (int m = 0; m < NDIM; m++) {
				for (int l = 0; l <= m; l++) {
					const simd_vector c0 = l == m ? simd_vector(1.0 / 6.0) : simd_vector(1.0 / 3.0);
					Lacc(n, m, l) += c0 * D(n, m, l) * M();								// 54
				}
			}
		}

	}

	L() += Lacc().sum();
	for (int n = 0; n < NDIM; n++) {
		L(n) += Lacc(n).sum();
		for (int m = 0; m <= n; m++) {
			L(n, m) += Lacc(n, m).sum();
			for (int l = 0; l <= m; l++) {
				L(n, m, l) += Lacc(n, m, l).sum();
			}
		}
	}

	return (290 + opts.ewald ? 18 : 0) * cnt1;
}

std::uint64_t gravity_multi_mono(expansion<double> &L, const vect<float> &x, std::vector<vect<float>> &y) {
	std::uint64_t flop = 0;
	static const auto opts = options::get();
	static const auto m = 1.0 / opts.problem_size;
	static const bool ewald = opts.ewald;
	static const auto h = opts.soft_len;
	static const auto h2 = h * h;
	static const simd_vector H(h);
	static const simd_vector H2(h2);
	vect<simd_vector> X, Y;
	static const simd_vector M(m);
	expansion<simd_vector> Lacc;
	Lacc.zero();

	const auto cnt1 = y.size();
	const auto cnt2 = ((cnt1 - 1 + SIMD_LEN) / SIMD_LEN) * SIMD_LEN;
	y.resize(cnt2);
	for (int j = cnt1; j < cnt2; j++) {
		y[j] = vect<float>(1.0e+10);
	}
	for (int dim = 0; dim < NDIM; dim++) {
		X[dim] = simd_vector(x[dim]);
	}
	for (int j = 0; j < cnt1; j += SIMD_LEN) {
		for (int k = 0; k < SIMD_LEN; k++) {
			for (int dim = 0; dim < NDIM; dim++) {
				Y[dim][k] = y[j + k][dim];
			}
		}

		vect<simd_vector> dX = X - Y;             										// 3 OP
		if (ewald) {
			for (int dim = 0; dim < NDIM; dim++) {
				const auto absdx = abs(dX[dim]);										// 3 OP
				dX[dim] = copysign(dX[dim] * (half - absdx), min(absdx, one - absdx));  // 15 OP
			}
		}
		expansion<simd_vector> D = Green_direct(dX);									// 135

		Lacc() += D() * M;																// 2

		for (int n = 0; n < NDIM; n++) {
			Lacc(n) += D(n) * M;														// 6
		}

		for (int n = 0; n < NDIM; n++) {
			for (int m = 0; m <= n; m++) {
				const simd_vector c0 = n == m ? simd_vector(0.5) : simd_vector(1);
				Lacc(n, m) += c0 * D(n, m) * M;											// 18
			}
		}

		for (int n = 0; n < NDIM; n++) {
			for (int m = 0; m < NDIM; m++) {
				for (int l = 0; l <= m; l++) {
					const simd_vector c0 = m == l ? simd_vector(1.0 / 6.0) : simd_vector(1.0 / 3.0);
					Lacc(n, m, l) += c0 * D(n, m, l) * M;								// 54
				}
			}
		}

	}

	L() += Lacc().sum();
	for (int n = 0; n < NDIM; n++) {
		L(n) += Lacc(n).sum();
		for (int m = 0; m <= n; m++) {
			L(n, m) += Lacc(n, m).sum();
			for (int l = 0; l <= m; l++) {
				L(n, m, l) += Lacc(n, m, l).sum();
			}
		}
	}

	return (200 + opts.ewald ? 18 : 0) * cnt1;
}

std::uint64_t gravity_ewald(std::vector<force> &f, const std::vector<vect<float>> &x, std::vector<mono_source> &y, const bool do_phi) {
	if (x.size() == 0) {
		return 0;
	}
	std::uint64_t flop = 0;
	static const auto opts = options::get();
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
			M[k] = y[j + k].m;
			for (int dim = 0; dim < NDIM; dim++) {
				Y[dim][k] = y[j + k].x[dim];
			}
		}
		for (int i = 0; i < x.size(); i++) {
			for (int dim = 0; dim < NDIM; dim++) {
				X[dim] = simd_vector(x[i][dim]);
			}

			vect<simd_vector> dX = X - Y;             										// 3 OP
			vect<simd_vector> sgn;
			for (int dim = 0; dim < NDIM; dim++) {
				const auto absdx = abs(dX[dim]);										// 3 OP
				sgn[dim] = copysign(dX[dim] * (half - absdx), 1.0);						// 9 OP
				dX[dim] = min(absdx, one - absdx);                                      // 6 OP
			}

			vect<simd_int_vector> I;
			vect<simd_vector> wm;
			vect<simd_vector> w;
			static const simd_vector dx0(0.5 / NE);
			static const simd_vector max_i(NE - 1);
			for (int dim = 0; dim < NDIM; dim++) {
				I[dim] = min(dX[dim] / dx0, max_i).to_int();								// 9 OP
				wm[dim] = (dX[dim] / dx0 - simd_vector(I[dim]));							// 9 OP
				w[dim] = simd_vector(1.0) - wm[dim];										// 3 OP
			}
			const simd_vector w00 = w[0] * w[1];											// 1 OP
			const simd_vector w01 = w[0] * wm[1];											// 1 OP
			const simd_vector w10 = wm[0] * w[1];											// 1 OP
			const simd_vector w11 = wm[0] * wm[1];											// 1 OP
			const simd_vector w000 = w00 * w[2];											// 1 OP
			const simd_vector w001 = w00 * wm[2];											// 1 OP
			const simd_vector w010 = w01 * w[2];											// 1 OP
			const simd_vector w011 = w01 * wm[2];											// 1 OP
			const simd_vector w100 = w10 * w[2];											// 1 OP
			const simd_vector w101 = w10 * wm[2];											// 1 OP
			const simd_vector w110 = w11 * w[2];											// 1 OP
			const simd_vector w111 = w11 * wm[2];											// 1 OP
			vect<simd_vector> F;
			simd_vector Pot;
			const simd_int_vector J = I[0] * simd_int_vector(NEP12) + I[1] * simd_int_vector(NEP1) + I[2];
			const simd_int_vector J000 = J;
			const simd_int_vector J001 = J + simd_int_vector(1);
			const simd_int_vector J010 = J + simd_int_vector(NEP1);
			const simd_int_vector J011 = J + simd_int_vector(1 + NEP1);
			const simd_int_vector J100 = J + simd_int_vector(NEP12);
			const simd_int_vector J101 = J + simd_int_vector(1 + NEP12);
			const simd_int_vector J110 = J + simd_int_vector(NEP1 + NEP12);
			const simd_int_vector J111 = J + simd_int_vector(1 + NEP1 + NEP12);
			simd_vector y000, y001, y010, y011, y100, y101, y110, y111;
			for (int dim = 0; dim < NDIM; dim++) {
				y000.gather(eforce[dim].data(), J000);
				y001.gather(eforce[dim].data(), J001);
				y010.gather(eforce[dim].data(), J010);
				y011.gather(eforce[dim].data(), J011);
				y100.gather(eforce[dim].data(), J100);
				y101.gather(eforce[dim].data(), J101);
				y110.gather(eforce[dim].data(), J110);
				y111.gather(eforce[dim].data(), J111);
				F[dim] = w000 * y000;															// 3 OP
				F[dim] = fma(w001, y001, F[dim]);												// 6 OP
				F[dim] = fma(w010, y010, F[dim]);												// 6 OP
				F[dim] = fma(w011, y011, F[dim]);												// 6 OP
				F[dim] = fma(w100, y100, F[dim]);												// 6 OP
				F[dim] = fma(w101, y101, F[dim]);												// 6 OP
				F[dim] = fma(w110, y110, F[dim]);												// 6 OP
				F[dim] = fma(w111, y111, F[dim]);												// 6 OP
			}
			if (do_phi) {
				y000.gather(epot.data(), J000);
				y001.gather(epot.data(), J001);
				y010.gather(epot.data(), J010);
				y011.gather(epot.data(), J011);
				y100.gather(epot.data(), J100);
				y101.gather(epot.data(), J101);
				y110.gather(epot.data(), J110);
				y111.gather(epot.data(), J111);
				Pot = w000 * y000;																// 1 OP
				Pot = fma(w001, y001, Pot);												// 2 OP
				Pot = fma(w010, y010, Pot);												// 2 OP
				Pot = fma(w011, y011, Pot);												// 2 OP
				Pot = fma(w100, y100, Pot);												// 2 OP
				Pot = fma(w101, y101, Pot);												// 2 OP
				Pot = fma(w110, y110, Pot);												// 2 OP
				Pot = fma(w111, y111, Pot);												// 2 OP
			}
			for (int dim = 0; dim < NDIM; dim++) {
				const auto tmp = M * sgn[dim];                                      // 3 OP
				G[i][dim] = fma(F[dim], tmp, G[i][dim]);                            // 6 OP
			}
			if (do_phi) {
				simd_vector r2 = dX[0] * dX[0];										     // 1 OP
				r2 = fma(dX[1], dX[1], r2);												 // 2 OP
				r2 = fma(dX[2], dX[2], r2);                     						 // 2 OP
				const simd_vector kill_zero = r2 / (r2 + eps); 							 // 2 OP
				const auto tmp = M * kill_zero;											 // 1 OP
				Phi[i] = fma(Pot, tmp, Phi[i]);											 // 2 OP
			}
		}
	}
	for (int i = 0; i < x.size(); i++) {
		for (int dim = 0; dim < NDIM; dim++) {
			f[i].g[dim] += G[i][dim].sum();
		}
		if (do_phi) {
			f[i].phi += Phi[i].sum();
		}
	}
	return (do_phi ? 133 : 108) * cnt1 * x.size();
}

void init_ewald() {
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
							const double dx = 0.05 * dx0;
							for (int dim = 0; dim < NDIM; dim++) {
								auto ym = x;
								auto yp = x;
								ym[dim] -= 0.5 * dx;
								yp[dim] += 0.5 * dx;
								const auto f = -(EW(yp) - EW(ym)) / dx;
								eforce[dim][i * NEP12 + j * NEP1 + k] = f;
							}
							const auto p = EW(x);
							epot[i * NEP12 + j * NEP1 + k] = p;
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

}

