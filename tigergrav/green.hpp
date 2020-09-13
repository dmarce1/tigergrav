#pragma once

constexpr float EWALD_REAL_N2 = 11;
constexpr float EWALD_FOUR_N2 = 7;
constexpr float EWALD_RADIUS_CUTOFF = 2.6;

#include <tigergrav/expansion.hpp>

struct ewald_indices: public std::vector<vect<float>> {
	ewald_indices(int n2max, bool nozero) {
		const int nmax = sqrt(n2max) + 1;
		vect<float> h;
		for (int i = -nmax; i <= nmax; i++) {
			for (int j = -nmax; j <= nmax; j++) {
				for (int k = -nmax; k <= nmax; k++) {
					if (i * i + j * j + k * k <= n2max) {
						h[0] = i;
						h[1] = j;
						h[2] = k;
						if (!nozero || h.dot(h) > 0) {
							this->push_back(h);
						}
					}
				}
			}
		}
		std::sort(this->begin(), this->end(), [](vect<float> &a, vect<float> &b) {
			return a.dot(a) > b.dot(b);
		});
	}
};

struct periodic_parts: public std::vector<expansion<float>> {
	periodic_parts() {
		static const ewald_indices indices(EWALD_FOUR_N2, true);
		for (auto i : indices) {
			vect<float> h = i;
			const float h2 = h.dot(h);                     // 5 OP
			expansion<float> D;
			D = 0.0;
			if (h2 > 0) {
				const float c0 = 1.0 / h2 * exp(-M_PI * M_PI * h2 / 4.0);
				D() = -(1.0 / M_PI) * c0;
				for (int a = 0; a < NDIM; a++) {
					D(a) = 2.0 * h[a] * c0;
					for (int b = 0; b <= a; b++) {
						D(a, b) = 4.0 * M_PI * h[a] * h[b] * c0;
						for (int c = 0; c <= b; c++) {
							D(a, b, c) = -8.0 * M_PI * M_PI * h[a] * h[b] * h[c] * c0;
							for (int d = 0; d <= c; d++) {
								D(a, b, c, d) = -16.0 * M_PI * M_PI * M_PI * h[a] * h[b] * h[c] * h[d] * c0;
							}
						}
					}

				}
			}
			this->push_back(D);
		}
	}
};

template<class T>  // 167
CUDA_EXPORT void green_deriv_direct(expansion<T> &D, const T &d0, const T &d1, const T &d2, const T &d3, const T &d4, const vect<T> &dx) {
	T dxadxb;
	T dxadxbdxc;
	D[0] = d0;
	D[1] = dx[0] * d1;
	dxadxb = dx[0] * dx[0];
	D[4] = dxadxb * d2;
	dxadxbdxc = dx[0] * dx[0] * dx[0];
	D[10] = dxadxbdxc * d3;
	D[20] = dxadxbdxc * dx[0] * d4;
	D[2] = dx[1] * d1;
	dxadxb = dx[1] * dx[0];
	D[5] = dxadxb * d2;
	dxadxbdxc = dx[1] * dx[0] * dx[0];
	D[11] = dxadxbdxc * d3;
	D[21] = dxadxbdxc * dx[0] * d4;
	dxadxb = dx[1] * dx[1];
	D[7] = dxadxb * d2;
	dxadxbdxc = dx[1] * dx[1] * dx[0];
	D[13] = dxadxbdxc * d3;
	D[23] = dxadxbdxc * dx[0] * d4;
	dxadxbdxc = dx[1] * dx[1] * dx[1];
	D[16] = dxadxbdxc * d3;
	D[26] = dxadxbdxc * dx[0] * d4;
	D[30] = dxadxbdxc * dx[1] * d4;
	D[3] = dx[2] * d1;
	dxadxb = dx[2] * dx[0];
	D[6] = dxadxb * d2;
	dxadxbdxc = dx[2] * dx[0] * dx[0];
	D[12] = dxadxbdxc * d3;
	D[22] = dxadxbdxc * dx[0] * d4;
	dxadxb = dx[2] * dx[1];
	D[8] = dxadxb * d2;
	dxadxbdxc = dx[2] * dx[1] * dx[0];
	D[14] = dxadxbdxc * d3;
	D[24] = dxadxbdxc * dx[0] * d4;
	dxadxbdxc = dx[2] * dx[1] * dx[1];
	D[17] = dxadxbdxc * d3;
	D[27] = dxadxbdxc * dx[0] * d4;
	D[31] = dxadxbdxc * dx[1] * d4;
	dxadxb = dx[2] * dx[2];
	D[9] = dxadxb * d2;
	dxadxbdxc = dx[2] * dx[2] * dx[0];
	D[15] = dxadxbdxc * d3;
	D[25] = dxadxbdxc * dx[0] * d4;
	dxadxbdxc = dx[2] * dx[2] * dx[1];
	D[18] = dxadxbdxc * d3;
	D[28] = dxadxbdxc * dx[0] * d4;
	D[32] = dxadxbdxc * dx[1] * d4;
	dxadxbdxc = dx[2] * dx[2] * dx[2];
	D[19] = dxadxbdxc * d3;
	D[29] = dxadxbdxc * dx[0] * d4;
	D[33] = dxadxbdxc * dx[1] * d4;
	D[34] = dxadxbdxc * dx[2] * d4;
	D[4] += d1;
	D[10] = fma(dx[0], d2, D[10]);
	D[20] = fma(dx[0] * dx[0], d3, D[20]);
	D[20] = fma(float(2.0), d2, D[20]);
	dxadxb = dx[0] * dx[0];
	D[10] = fma(dx[0], d2, D[10]);
	D[10] = fma(dx[0], d2, D[10]);
	D[20] = fma(dxadxb, d3, D[20]);
	D[20] = fma(dxadxb, d3, D[20]);
	D[20] += d2;
	D[20] = fma(dx[0], dx[0] * d3, D[20]);
	D[20] = fma(dxadxb, d3, D[20]);
	D[20] = fma(dx[0], dx[0] * d3, D[20]);
	D[7] += d1;
	D[16] = fma(dx[1], d2, D[16]);
	D[30] = fma(dx[1] * dx[1], d3, D[30]);
	D[30] = fma(float(2.0), d2, D[30]);
	dxadxb = dx[1] * dx[0];
	D[13] = fma(dx[0], d2, D[13]);
	D[11] = fma(dx[1], d2, D[11]);
	D[26] = fma(dxadxb, d3, D[26]);
	D[21] = fma(dxadxb, d3, D[21]);
	D[23] += d2;
	D[23] = fma(dx[0], dx[0] * d3, D[23]);
	D[21] = fma(dxadxb, d3, D[21]);
	D[21] = fma(dx[1], dx[0] * d3, D[21]);
	dxadxb = dx[1] * dx[1];
	D[16] = fma(dx[1], d2, D[16]);
	D[16] = fma(dx[1], d2, D[16]);
	D[30] = fma(dxadxb, d3, D[30]);
	D[30] = fma(dxadxb, d3, D[30]);
	D[30] += d2;
	D[26] = fma(dx[1], dx[0] * d3, D[26]);
	D[23] = fma(dxadxb, d3, D[23]);
	D[26] = fma(dx[1], dx[0] * d3, D[26]);
	D[30] = fma(dx[1], dx[1] * d3, D[30]);
	D[30] = fma(dxadxb, d3, D[30]);
	D[30] = fma(dx[1], dx[1] * d3, D[30]);
	D[9] += d1;
	D[19] = fma(dx[2], d2, D[19]);
	D[34] = fma(dx[2] * dx[2], d3, D[34]);
	D[34] = fma(float(2.0), d2, D[34]);
	dxadxb = dx[2] * dx[0];
	D[15] = fma(dx[0], d2, D[15]);
	D[12] = fma(dx[2], d2, D[12]);
	D[29] = fma(dxadxb, d3, D[29]);
	D[22] = fma(dxadxb, d3, D[22]);
	D[25] += d2;
	D[25] = fma(dx[0], dx[0] * d3, D[25]);
	D[22] = fma(dxadxb, d3, D[22]);
	D[22] = fma(dx[2], dx[0] * d3, D[22]);
	dxadxb = dx[2] * dx[1];
	D[18] = fma(dx[1], d2, D[18]);
	D[17] = fma(dx[2], d2, D[17]);
	D[33] = fma(dxadxb, d3, D[33]);
	D[31] = fma(dxadxb, d3, D[31]);
	D[32] += d2;
	D[28] = fma(dx[1], dx[0] * d3, D[28]);
	D[24] = fma(dxadxb, d3, D[24]);
	D[27] = fma(dx[2], dx[0] * d3, D[27]);
	D[32] = fma(dx[1], dx[1] * d3, D[32]);
	D[31] = fma(dxadxb, d3, D[31]);
	D[31] = fma(dx[2], dx[1] * d3, D[31]);
	dxadxb = dx[2] * dx[2];
	D[19] = fma(dx[2], d2, D[19]);
	D[19] = fma(dx[2], d2, D[19]);
	D[34] = fma(dxadxb, d3, D[34]);
	D[34] = fma(dxadxb, d3, D[34]);
	D[34] += d2;
	D[29] = fma(dx[2], dx[0] * d3, D[29]);
	D[25] = fma(dxadxb, d3, D[25]);
	D[29] = fma(dx[2], dx[0] * d3, D[29]);
	D[33] = fma(dx[2], dx[1] * d3, D[33]);
	D[32] = fma(dxadxb, d3, D[32]);
	D[33] = fma(dx[2], dx[1] * d3, D[33]);
	D[34] = fma(dx[2], dx[2] * d3, D[34]);
	D[34] = fma(dxadxb, d3, D[34]);
	D[34] = fma(dx[2], dx[2] * d3, D[34]);
}

template<class T>  // 576
CUDA_EXPORT void green_deriv_ewald(expansion<T> &D, const T &d0, const T &d1, const T &d2, const T &d3, const T &d4, const vect<T> &dx) {
	static const T two(2.0);
	T dxadxb;
	T dxadxbdxc;
	D[0] += d0;
	D[1] = fma(dx[0], d1, D[1]);
	dxadxb = dx[0] * dx[0];
	D[4] = fma(dxadxb, d2, D[4]);
	dxadxbdxc = dx[0] * dx[0] * dx[0];
	D[10] = fma(dxadxbdxc, d3, D[10]);
	D[20] = fma(dxadxbdxc * dx[0], d4, D[20]);
	D[2] = fma(dx[1], d1, D[2]);
	;
	dxadxb = dx[1] * dx[0];
	D[5] = fma(dxadxb, d2, D[5]);
	dxadxbdxc = dx[1] * dx[0] * dx[0];
	D[11] = fma(dxadxbdxc, d3, D[11]);
	D[21] = fma(dxadxbdxc * dx[0], d4, D[21]);
	dxadxb = dx[1] * dx[1];
	D[7] = fma(dxadxb, d2, D[7]);
	dxadxbdxc = dx[1] * dx[1] * dx[0];
	D[13] = fma(dxadxbdxc, d3, D[13]);
	D[23] = fma(dxadxbdxc * dx[0], d4, D[23]);
	dxadxbdxc = dx[1] * dx[1] * dx[1];
	D[16] = fma(dxadxbdxc, d3, D[16]);
	D[26] = fma(dxadxbdxc * dx[0], d4, D[26]);
	D[30] = fma(dxadxbdxc * dx[1], d4, D[30]);
	D[3] = fma(dx[2], d1, D[3]);
	dxadxb = dx[2] * dx[0];
	D[6] = fma(dxadxb, d2, D[6]);
	dxadxbdxc = dx[2] * dx[0] * dx[0];
	D[12] = fma(dxadxbdxc, d3, D[12]);
	D[22] = fma(dxadxbdxc * dx[0], d4, D[22]);
	dxadxb = dx[2] * dx[1];
	D[8] = fma(dxadxb, d2, D[8]);
	dxadxbdxc = dx[2] * dx[1] * dx[0];
	D[14] = fma(dxadxbdxc, d3, D[14]);
	D[24] = fma(dxadxbdxc * dx[0], d4, D[24]);
	dxadxbdxc = dx[2] * dx[1] * dx[1];
	D[17] = fma(dxadxbdxc, d3, D[17]);
	D[27] = fma(dxadxbdxc * dx[0], d4, D[27]);
	D[31] = fma(dxadxbdxc * dx[1], d4, D[31]);
	dxadxb = dx[2] * dx[2];
	D[9] = fma(dxadxb, d2, D[9]);
	dxadxbdxc = dx[2] * dx[2] * dx[0];
	D[15] = fma(dxadxbdxc, d3, D[15]);
	D[25] = fma(dxadxbdxc * dx[0], d4, D[25]);
	dxadxbdxc = dx[2] * dx[2] * dx[1];
	D[18] = fma(dxadxbdxc, d3, D[18]);
	D[28] = fma(dxadxbdxc * dx[0], d4, D[28]);
	D[32] = fma(dxadxbdxc * dx[1], d4, D[32]);
	dxadxbdxc = dx[2] * dx[2] * dx[2];
	D[19] = fma(dxadxbdxc, d3, D[19]);
	D[29] = fma(dxadxbdxc * dx[0], d4, D[29]);
	D[33] = fma(dxadxbdxc * dx[1], d4, D[33]);
	D[34] = fma(dxadxbdxc * dx[2], d4, D[34]);
	D[4] += d1;
	D[10] = fma(dx[0], d2, D[10]);
	D[20] = fma(dx[0] * dx[0], d3, D[20]);
	D[20] = fma(float(2.0), d2, D[20]);
	dxadxb = dx[0] * dx[0];
	D[10] = fma(dx[0], d2, D[10]);
	D[10] = fma(dx[0], d2, D[10]);
	D[20] = fma(dxadxb, d3, D[20]);
	D[20] = fma(dxadxb, d3, D[20]);
	D[20] += d2;
	D[20] = fma(dx[0], dx[0] * d3, D[20]);
	D[20] = fma(dxadxb, d3, D[20]);
	D[20] = fma(dx[0], dx[0] * d3, D[20]);
	D[7] += d1;
	D[16] = fma(dx[1], d2, D[16]);
	D[30] = fma(dx[1] * dx[1], d3, D[30]);
	D[30] = fma(float(2.0), d2, D[30]);
	dxadxb = dx[1] * dx[0];
	D[13] = fma(dx[0], d2, D[13]);
	D[11] = fma(dx[1], d2, D[11]);
	D[26] = fma(dxadxb, d3, D[26]);
	D[21] = fma(dxadxb, d3, D[21]);
	D[23] += d2;
	D[23] = fma(dx[0], dx[0] * d3, D[23]);
	D[21] = fma(dxadxb, d3, D[21]);
	D[21] = fma(dx[1], dx[0] * d3, D[21]);
	dxadxb = dx[1] * dx[1];
	D[16] = fma(dx[1], d2, D[16]);
	D[16] = fma(dx[1], d2, D[16]);
	D[30] = fma(dxadxb, d3, D[30]);
	D[30] = fma(dxadxb, d3, D[30]);
	D[30] += d2;
	D[26] = fma(dx[1], dx[0] * d3, D[26]);
	D[23] = fma(dxadxb, d3, D[23]);
	D[26] = fma(dx[1], dx[0] * d3, D[26]);
	D[30] = fma(dx[1], dx[1] * d3, D[30]);
	D[30] = fma(dxadxb, d3, D[30]);
	D[30] = fma(dx[1], dx[1] * d3, D[30]);
	D[9] += d1;
	D[19] = fma(dx[2], d2, D[19]);
	D[34] = fma(dx[2] * dx[2], d3, D[34]);
	D[34] = fma(float(2.0), d2, D[34]);
	dxadxb = dx[2] * dx[0];
	D[15] = fma(dx[0], d2, D[15]);
	D[12] = fma(dx[2], d2, D[12]);
	D[29] = fma(dxadxb, d3, D[29]);
	D[22] = fma(dxadxb, d3, D[22]);
	D[25] += d2;
	D[25] = fma(dx[0], dx[0] * d3, D[25]);
	D[22] = fma(dxadxb, d3, D[22]);
	D[22] = fma(dx[2], dx[0] * d3, D[22]);
	dxadxb = dx[2] * dx[1];
	D[18] = fma(dx[1], d2, D[18]);
	D[17] = fma(dx[2], d2, D[17]);
	D[33] = fma(dxadxb, d3, D[33]);
	D[31] = fma(dxadxb, d3, D[31]);
	D[32] += d2;
	D[28] = fma(dx[1], dx[0] * d3, D[28]);
	D[24] = fma(dxadxb, d3, D[24]);
	D[27] = fma(dx[2], dx[0] * d3, D[27]);
	D[32] = fma(dx[1], dx[1] * d3, D[32]);
	D[31] = fma(dxadxb, d3, D[31]);
	D[31] = fma(dx[2], dx[1] * d3, D[31]);
	dxadxb = dx[2] * dx[2];
	D[19] = fma(dx[2], d2, D[19]);
	D[19] = fma(dx[2], d2, D[19]);
	D[34] = fma(dxadxb, d3, D[34]);
	D[34] = fma(dxadxb, d3, D[34]);
	D[34] += d2;
	D[29] = fma(dx[2], dx[0] * d3, D[29]);
	D[25] = fma(dxadxb, d3, D[25]);
	D[29] = fma(dx[2], dx[0] * d3, D[29]);
	D[33] = fma(dx[2], dx[1] * d3, D[33]);
	D[32] = fma(dxadxb, d3, D[32]);
	D[33] = fma(dx[2], dx[1] * d3, D[33]);
	D[34] = fma(dx[2], dx[2] * d3, D[34]);
	D[34] = fma(dxadxb, d3, D[34]);
	D[34] = fma(dx[2], dx[2] * d3, D[34]);
}

template<class T>
CUDA_EXPORT inline expansion<T> green_direct(const vect<T> &dX) {		// 59  + 167 = 226
	static const T r0 = 1.0e-9;
//	static const T H = options::get().soft_len;
	static const T nthree(-3.0);
	static const T nfive(-5.0);
	static const T nseven(-7.0);
	const T r2 = dX.dot(dX);		// 5
	const T r = sqrt(r2);		// 7
	const T rinv = (r > r0) / max(r, r0);	// 38
	const T r2inv = rinv * rinv;			// 1
	const T d0 = -rinv;						// 1
	const T d1 = -d0 * r2inv;				// 2
	const T d2 = nthree * d1 * r2inv;		// 2
	const T d3 = nfive * d2 * r2inv;		// 2
	const T d4 = nseven * d3 * r2inv;		// 2
	expansion<T> D;
	green_deriv_direct(D, d0, d1, d2, d3, d4, dX);		// 167
	return D;
}
#ifdef __CUDA_ARCH__

__device__ const cuda_ewald_const& cuda_get_const();

CUDA_EXPORT expansion<float> green_ewald(const vect<float> &X) {
	const auto &cuda_const = cuda_get_const();
	const auto &four_indices = cuda_const.four_indices;
	const auto &real_indices = cuda_const.real_indices;
	const auto &hparts = cuda_const.periodic_parts;
	static const float three(3.0);
	const float fouroversqrtpi(4.0 / sqrt(M_PI));
	static const float two(2.0);
	static const float eight(8.0);
	static const float fifteen(15.0);
	static const float thirtyfive(35.0);
	static const float fourty(40.0);
	static const float fiftysix(56.0);
	static const float sixtyfour(64.0);
	static const float onehundredfive(105.0);
	static const float rcut(1.0e-6);
	const float r = abs(X);
	const float zmask = r > rcut;											// 2
	expansion<float> Dreal;
	expansion<float> Dfour;
	Dreal = 0.0;
	Dfour = 0.0;
	for (auto n : real_indices) {
		const vect<float> dx = X - vect<float>(n);				// 3
		const float r2 = dx.dot(dx);				// 5
		if (r2 < (EWALD_RADIUS_CUTOFF * EWALD_RADIUS_CUTOFF)) {
			const float r = sqrt(r2);					// 7
			const float r4 = r2 * r2;					// 1
			const float cmask = 1.0 - (n.dot(n) > 0.0);
			const float mask = (1.0 - (1.0 - zmask) * cmask);
			const float rinv = mask / max(r, rcut);		// 36
			const float r2inv = rinv * rinv;			// 1
			const float r3inv = r2inv * rinv;			// 1
			const float r5inv = r2inv * r3inv;			// 1
			const float r7inv = r2inv * r5inv;			// 1
			const float r9inv = r2inv * r7inv;			// 1
			const float erfc0 = erfcf(two * r);			// 76
			const float exp0 = expf(-two * two * r * r);
			const float expfactor = fouroversqrtpi * r * exp0; 	// 2
			const float d0 = -erfc0 * rinv;							// 2
			const float d1 = (expfactor + erfc0) * r3inv;			// 2
			const float d2 = -fma(expfactor, fma(eight, r2, three), three * erfc0) * r5inv;		// 5
			const float d3 = fma(expfactor, (fifteen + fma(fourty, r2, sixtyfour * r4)), fifteen * erfc0) * r7inv;		// 6
			const float d4 = -fma(expfactor, fma(eight * r2, (thirtyfive + fma(fiftysix, r2, sixtyfour * r4)), onehundredfive), onehundredfive * erfc0) * r9inv;// 9
			green_deriv_ewald(Dreal, d0, d1, d2, d3, d4, dx);			// 576
		}
	}
	static const float twopi = 2.0 * M_PI;

	for (int i = 0; i < EWALD_NFOUR; i++) {
		const auto &h = four_indices[i];
		const auto &hpart = hparts[i];
//		printf( "H = %e %e %e\n", h[0], h[1], h[2]);
		const float h2 = h.dot(h);
		const float hdotx = h.dot(X);
		float co;
		float so;
		sincosf(twopi * hdotx, &so, &co);
		Dfour() = fma(hpart(), co, Dfour());
		for (int a = 0; a < NDIM; a++) {
			Dfour(a) = fma(hpart(a), so, Dfour(a));
			for (int b = 0; b <= a; b++) {
				Dfour(a, b) = fma(hpart(a, b), co, Dfour(a, b));
				for (int c = 0; c <= b; c++) {
					Dfour(a, b, c) = fma(hpart(a, b, c), so, Dfour(a, b, c));
					for (int d = 0; d <= c; d++) {
						Dfour(a, b, c, d) = fma(hpart(a, b, c, d), co, Dfour(a, b, c, d));
					}
				}
			}
		}
	}
	expansion<float> &D = Dreal;
	for (int i = 0; i < LP; i++) {
		Dreal[i] += Dfour[i];
	}
	const auto D1 = green_direct(X);													// 167
	const float rinv = -D1();														// 2
	D() = (M_PI / 4.0) + D() + zmask * rinv;												// 2
	for (int a = 0; a < NDIM; a++) {
		D(a) = (D(a) - zmask * D1(a));												// 6
		for (int b = 0; b <= a; b++) {
			D(a, b) = (D(a, b) - zmask * D1(a, b));									// 12
			for (int c = 0; c <= b; c++) {
				D(a, b, c) = (D(a, b, c) - zmask * D1(a, b, c));					// 20
				for (int d = 0; d <= c; d++) {
					D(a, b, c, d) = (D(a, b, c, d) - zmask * D1(a, b, c, d));		// 30
				}
			}
		}
	}

	return D;

}

#else

template<class T>
inline expansion<T> green_ewald(const vect<T> &X) {		// 251176
	static const periodic_parts periodic;
	expansion<T> D;
	D = 0.0;
	vect<T> n;
	vect<float> h;
	static const ewald_indices indices_real(EWALD_REAL_N2, false);
	static const ewald_indices indices_four(EWALD_FOUR_N2, true);
//	printf( "%i %i\n", indices_real.size(), indices_four.size());
	static const T three(3.0);
	static const T fouroversqrtpi(4.0 / sqrt(M_PI));
	static const T two(2.0);
	static const T eight(8.0);
	static const T fifteen(15.0);
	static const T thirtyfive(35.0);
	static const T fourty(40.0);
	static const T fiftysix(56.0);
	static const T sixtyfour(64.0);
	static const T onehundredfive(105.0);
	static const simd_float rcut(1.0e-6);
//	printf("%i %i\n", indices_real.size(), indices_four.size());
	const T r = abs(X);															// 5
	const simd_float zmask = r > rcut;											// 2
	for (int i = 0; i < indices_real.size(); i++) {			// 739 * 305 		// 225395
		h = indices_real[i];
		n = h;
		const vect<T> dx = X - n;				// 3
		const T r2 = dx.dot(dx);				// 5
		const T r4 = r2 * r2;					// 1
		const T r6 = r2 * r4;					// 1
		const T r = sqrt(r2);					// 7
		const T cmask = T(1) - (n.dot(n) > 0.0);
		const T mask = (T(1) - (T(1) - zmask) * cmask) * (r < EWALD_RADIUS_CUTOFF);
		const T rinv = mask / max(r, rcut);		// 36
		const T r2inv = rinv * rinv;			// 1
		const T r3inv = r2inv * rinv;			// 1
		const T r5inv = r2inv * r3inv;			// 1
		const T r7inv = r2inv * r5inv;			// 1
		const T r9inv = r2inv * r7inv;			// 1
		T expfac;
		const T erfc = erfcexp(two * r, &expfac);			// 76
		const T expfactor = fouroversqrtpi * r * expfac; 	// 2
		const T d0 = -erfc * rinv;							// 2
		const T d1 = (expfactor + erfc) * r3inv;			// 2
		const T d2 = -fma(expfactor, fma(eight, T(r2), three), three * erfc) * r5inv;		// 5
		const T d3 = fma(expfactor, (fifteen + fma(fourty, T(r2), sixtyfour * T(r4))), fifteen * erfc) * r7inv;		// 6
		const T d4 = -fma(expfactor, fma(eight * T(r2), (thirtyfive + fma(fiftysix, r2, sixtyfour * r4)), onehundredfive), onehundredfive * erfc) * r9inv;	// 9
		green_deriv_ewald(D, d0, d1, d2, d3, d4, dx);			// 576
	}
	for (int i = 0; i < indices_four.size(); i++) {		// 207 * 123 = 			// 25461
		h = indices_four[i];
		const auto H = periodic[i];
		T hdotdx = X[0] * h[0];		// 1
		for (int a = 1; a < NDIM; a++) {
			hdotdx += X[a] * h[a];								// 4
		}
		static const T twopi = 2.0 * M_PI;
		const T omega = twopi * hdotdx;							// 1
		T co, si;
		sincos(omega, &si, &co);								// 25
		D() += H() * co;										// 5
		for (int a = 0; a < NDIM; a++) {
			D(a) += H(a) * si;									// 15
			for (int b = 0; b <= a; b++) {
				D(a, b) += H(a, b) * co;						// 30
				for (int c = 0; c <= b; c++) {
					D(a, b, c) += H(a, b, c) * si;				// 50
					for (int d = 0; d <= c; d++) {
						D(a, b, c, d) += H(a, b, c, d) * co; 	// 75
					}
				}

			}
		}
	}
	expansion<T> rcD;
	for (int i = 0; i < LP; i++) {
		rcD[i] = D[i];																	// 70
	}
	const auto D1 = green_direct(X);													// 167
	const T rinv = -D1();														// 2
	rcD() = T(M_PI / 4.0) + rcD() + zmask * rinv;												// 2
	for (int a = 0; a < NDIM; a++) {
		rcD(a) = (rcD(a) - zmask * D1(a));												// 6
		for (int b = 0; b <= a; b++) {
			rcD(a, b) = (rcD(a, b) - zmask * D1(a, b));									// 12
			for (int c = 0; c <= b; c++) {
				rcD(a, b, c) = (rcD(a, b, c) - zmask * D1(a, b, c));					// 20
				for (int d = 0; d <= c; d++) {
					rcD(a, b, c, d) = (rcD(a, b, c, d) - zmask * D1(a, b, c, d));		// 30
				}
			}
		}
	}
	return rcD;
}
#endif
