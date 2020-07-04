#include <tigergrav/gravity.hpp>
#include <tigergrav/options.hpp>
#include <tigergrav/simd.hpp>

#include <hpx/include/async.hpp>

constexpr int EWALD_NBIN = 64;

using ewald_table_t = std::array<std::array<std::array<float, EWALD_NBIN + 1>, EWALD_NBIN + 1>, EWALD_NBIN + 1>;
static ewald_table_t epot;
static std::array<ewald_table_t, NDIM> eforce;

float EW(general_vect<double, NDIM> x) {
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

float ewald_separation(const vect<float> x) {
	float d = 0.0;
	for (int dim = 0; dim < NDIM; dim++) {
		const float absx = std::abs(x[dim]);
		float this_d = std::min(absx, (float) 1.0 - absx);
		d += this_d * this_d;
	}
	return std::sqrt(d);
}

std::uint64_t gravity_direct(std::vector<force> &f, const std::vector<vect<float>> &x, std::vector<source> &y) {
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
			M[k] = y[j + k].m;
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

std::uint64_t gravity_ewald(std::vector<force> &f, const std::vector<vect<float>> &x, std::vector<source> &y) {
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

			vect<std::array<std::uint32_t, SIMD_LEN>> I;
			vect<simd_vector> wm;
			vect<simd_vector> w;
			static const simd_vector dx0(0.5 / EWALD_NBIN);
			static const simd_vector max_i(EWALD_NBIN - 1);
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
			simd_vector y000, y001, y010, y011, y100, y101, y110, y111;
			vect<simd_vector> F;
			simd_vector Pot;
			for (int dim = 0; dim < NDIM; dim++) {
				for (int k = 0; k < SIMD_LEN; k++) {
					y000[k] = eforce[dim][I[0][k]][I[1][k]][I[2][k]];
					y001[k] = eforce[dim][I[0][k]][I[1][k]][I[2][k] + 1];
					y010[k] = eforce[dim][I[0][k]][I[1][k] + 1][I[2][k]];
					y011[k] = eforce[dim][I[0][k]][I[1][k] + 1][I[2][k] + 1];
					y100[k] = eforce[dim][I[0][k] + 1][I[1][k]][I[2][k]];
					y101[k] = eforce[dim][I[0][k] + 1][I[1][k]][I[2][k] + 1];
					y110[k] = eforce[dim][I[0][k] + 1][I[1][k] + 1][I[2][k]];
					y111[k] = eforce[dim][I[0][k] + 1][I[1][k] + 1][I[2][k] + 1];
				}
				F[dim] = w000 * y000;															// 3 OP
				F[dim] += w001 * y001;															// 6 OP
				F[dim] += w010 * y010;															// 6 OP
				F[dim] += w011 * y011;															// 6 OP
				F[dim] += w100 * y100;															// 6 OP
				F[dim] += w101 * y101;															// 6 OP
				F[dim] += w110 * y110;															// 6 OP
				F[dim] += w111 * y111;															// 6 OP
			}
			for (int k = 0; k < SIMD_LEN; k++) {
				y000[k] = epot[I[0][k]][I[1][k]][I[2][k]];
				y001[k] = epot[I[0][k]][I[1][k]][I[2][k] + 1];
				y010[k] = epot[I[0][k]][I[1][k] + 1][I[2][k]];
				y011[k] = epot[I[0][k]][I[1][k] + 1][I[2][k] + 1];
				y100[k] = epot[I[0][k] + 1][I[1][k]][I[2][k]];
				y101[k] = epot[I[0][k] + 1][I[1][k]][I[2][k] + 1];
				y110[k] = epot[I[0][k] + 1][I[1][k] + 1][I[2][k]];
				y111[k] = epot[I[0][k] + 1][I[1][k] + 1][I[2][k] + 1];
//				printf( "%i %i %i\n", I[0][k], I[1][k], I[2][k]);
			}
			Pot = w000 * y000;																// 1 OP
			Pot += w001 * y001;																// 2 OP
			Pot += w010 * y010;																// 2 OP
			Pot += w011 * y011;																// 2 OP
			Pot += w100 * y100;																// 2 OP
			Pot += w101 * y101;																// 2 OP
			Pot += w110 * y110;																// 2 OP
			Pot += w111 * y111;																// 2 OP
			for (int dim = 0; dim < NDIM; dim++) {
				G[i][dim] += F[dim] * M * sgn[dim];                       					// 9 OP
			}
			Phi[i] += M * Pot;																// 2 OP
		}
	}
	for (int i = 0; i < x.size(); i++) {
		for (int dim = 0; dim < NDIM; dim++) {
			f[i].g[dim] += G[i][dim].sum();
		}
		f[i].phi += Phi[i].sum();
	}
	return 125 * cnt1 * x.size();
}

void init_ewald() {
	FILE *fp = fopen("ewald.dat", "rb");
	if (fp) {
		int cnt = 0;
		printf("Found ewald.dat\n");
		const int sz = (EWALD_NBIN + 1) * (EWALD_NBIN + 1) * (EWALD_NBIN + 1);
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

		const double dx0 = 0.5 / EWALD_NBIN;
		for (int dim = 0; dim < NDIM; dim++) {
			eforce[dim][0][0][0] = 0.0;
		}
		epot[0][0][0] = 2.8372975;
		float n = 0;
		for (int i = 0; i <= EWALD_NBIN; i++) {
			for (int j = 0; j <= EWALD_NBIN; j++) {
				printf("%% %.2f complete\r", n / double(EWALD_NBIN + 1) / double(EWALD_NBIN + 1) * 100.0);
				n += 1.0;
				fflush(stdout);
				std::vector<hpx::future<void>> futs;
				for (int k = 0; k <= EWALD_NBIN; k++) {
					const auto func = [i, j, k, dx0]() {
						general_vect<double, NDIM> x;
						x[0] = i * dx0;
						x[1] = j * dx0;
						x[2] = k * dx0;
						if (x.dot(x) != 0.0) {
							const double dx = 0.01 * dx0;
							for (int dim = 0; dim < NDIM; dim++) {
								auto ym = x;
								auto yp = x;
								ym[dim] -= 0.5 * dx;
								yp[dim] += 0.5 * dx;
								const auto f = -(EW(yp) - EW(ym)) / dx;
								eforce[dim][i][j][k] = f;
							}
							const auto p = EW(x);
							epot[i][j][k] = p;
						}
					};
					futs.push_back(hpx::async(func));
				}
				hpx::wait_all(futs);
			}
		}
		printf("\nDone initializing Ewald\n");
		fp = fopen("ewald.dat", "wb");
		const int sz = (EWALD_NBIN + 1) * (EWALD_NBIN + 1) * (EWALD_NBIN + 1);
		fwrite(&epot, sizeof(float), sz, fp);
		fwrite(&eforce, sizeof(float), NDIM * sz, fp);
		fclose(fp);

	}

}

