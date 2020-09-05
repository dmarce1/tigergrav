#include <tigergrav/options.hpp>
#include <tigergrav/cuda_export.hpp>
#include <tigergrav/cuda_check.hpp>
#include <tigergrav/gravity_cuda.hpp>
#include <tigergrav/green.hpp>

CUDA_EXPORT expansion<float> green_ewald(const vect<float> &X) {
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
	vect<int> n;
	expansion<double> D;
	D = 0.0;
	for (n[0] = -4; n[0] <= +4; n[0]++) {
		for (n[1] = -4; n[1] <= +4; n[1]++) {
			for (n[2] = -4; n[2] <= +4; n[2]++) {
				const vect<float> dx = X - vect<float>(n);				// 3
				const float r2 = dx.dot(dx);				// 5
				const float r4 = r2 * r2;					// 1
				const float r = sqrt(r2);					// 7
				if (r < 3.6) {
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
					const float d4 = -fma(expfactor, fma(eight * r2, (thirtyfive + fma(fiftysix, r2, sixtyfour * r4)), onehundredfive), onehundredfive * erfc0)
							* r9inv;		// 9
					green_deriv_ewald(D, d0, d1, d2, d3, d4, dx);			// 576
				}
			}
		}
	}
	static const float twopi = 2.0 * M_PI;
	for (n[0] = -3; n[0] <= 3; n[0]++) {
		for (n[1] = -3; n[1] <= 3; n[1]++) {
			for (n[2] = -3; n[2] <= 3; n[2]++) {
				if (n.dot(n) < 10) {
					vect<float> h = n;
					const float h2 = h.dot(h);
					const float hdotx = h.dot(X);
					if (h2 > 0.0) {
						const float co = cosf(twopi * hdotx);
						const float so = sinf(twopi * hdotx);
						float c0 = (-1.0 / M_PI) * expf(-M_PI * M_PI / 4.0 * h2) / h2;
						D() += c0 * co;
						for (int a = 0; a < NDIM; a++) {
							const float c1 = -twopi * c0 * h[a];
							D(a) += c1 * so;
							for (int b = 0; b <= a; b++) {
								const float c2 = +twopi * c1 * h[b];
								D(a, b) += c2 * co;
								for (int c = 0; c <= b; c++) {
									const float c3 = -twopi * c2 * h[c];
									D(a, b, c) += c3 * so;
									for (int d = 0; d <= c; d++) {
										const float c4 = +twopi * c3 * h[d];
										D(a, b, c, d) += c4 * co;
									}
								}
							}
						}
					}
				}
			}
		}
	}

	expansion<float> rcD;
//	if (r > rcut) {
	for (int i = 0; i < LP; i++) {
		rcD[i] = D[i];																	// 70
	}
	const auto D1 = green_direct(X);													// 167
	const float rinv = -D1();														// 2
	rcD() = (M_PI / 4.0) + rcD() + zmask * rinv;												// 2
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
//	} else {
//		for (int i = 0; i < LP; i++) {
//			rcD[i] = 0.0;																	// 70
//		}
//
//	}

	return rcD;

}

#include <tigergrav/interactions.hpp>

#include <stack>
#include <atomic>
void yield_to_hpx();

static vect<pos_type> *y_vect;
static part_iter y_begin;
static part_iter y_end;
static bool first_call = true;
static std::atomic<int> thread_cnt(0);

bool cuda_thread_count() {
	return thread_cnt;
}

void cuda_copy_particle_image(part_iter part_begin, part_iter part_end, const std::vector<vect<pos_type>> &parts) {
	y_begin = part_begin;
	y_end = part_end;
	const auto size = part_end - part_begin;
	if (first_call) {
		CUDA_CHECK(cudaMalloc((void** ) &y_vect, sizeof(vect<pos_type> ) * size));
		first_call = false;
	}
	CUDA_CHECK(cudaMemcpy(y_vect, parts.data(), size * sizeof(vect<pos_type> ), cudaMemcpyHostToDevice));
}

#define EWALD_MAX_TBSIZE 64

#define WORKSIZE 128
#define NODESIZE 64
#define NWARP (WORKSIZE/WARPSIZE)
#define WARPSIZE 32

__global__ void CC_ewald_kernel(expansion<double> *lptr, const vect<pos_type> X, const multi_src *y, int ysize) {
	int l = threadIdx.x;
	int tb_size = blockDim.x;
	auto &L = *lptr;

	__shared__ double value[EWALD_MAX_TBSIZE];
//	double *value = reinterpret_cast<double*>(shmem);
	expansion<double> Lacc;
	const int ymax = ((ysize - 1) / tb_size + 1) * tb_size;
	for (int yi = l; yi < ymax; yi += tb_size) {
		for (int i = 0; i < LP; i++) {
			Lacc[i] = 0.0;
		}
		if (yi < ysize) {
			vect<pos_type> Y = y[yi].x;
			multipole<float> M = y[yi].m;
			vect<float> dX;
			for (int dim = 0; dim < NDIM; dim++) {
				dX[dim] = float(X[dim] - Y[dim]) * float(POS_INV); // 18
			}
			multipole_interaction(Lacc, M, dX, true);											// 251936
		}
		for (int i = 0; i < LP; i++) {
			value[l] = Lacc[i];
			__syncthreads();
			for (int N = tb_size / 2; N > 0; N >>= 1) {
				if (l < N) {
					value[l] += value[l + N];
				}
				__syncthreads();
			}
			if (l == 0) {
				L[i] += value[0];
			}
		}
	}
}

struct cuda_context_ewald {
	int ysize;
	cudaStream_t stream;
	expansion<double> *L;
	multi_src *y;
	expansion<double> *Lp;
	multi_src *yp;
	cuda_context_ewald(int ys) {
		ysize = 1;
		while (ysize < ys) {
			ysize *= 2;
		}
		CUDA_CHECK(cudaMalloc(&L, sizeof(expansion<double> )));
		CUDA_CHECK(cudaMalloc(&y, sizeof(multi_src) * ysize));
		CUDA_CHECK(cudaMallocHost(&Lp, sizeof(expansion<double> )));
		CUDA_CHECK(cudaMallocHost(&yp, sizeof(multi_src) * ysize));
		CUDA_CHECK(cudaStreamCreate(&stream));
	}
	void resize(int ys) {
		if (ys > ysize) {
			while (ysize < ys) {
				ysize *= 2;
			}
			CUDA_CHECK(cudaFree(y));
			CUDA_CHECK(cudaMalloc(&y, sizeof(multi_src) * ysize));
			CUDA_CHECK(cudaFreeHost(yp));
			CUDA_CHECK(cudaMallocHost(&yp, sizeof(multi_src) * ysize));
		}
	}
};

static std::atomic<int> lock_ewald(0);
static std::stack<cuda_context_ewald> stack_ewald;

cuda_context_ewald pop_context_ewald(int ys) {
	while (lock_ewald++ != 0) {
		lock_ewald--;
	}
	if (stack_ewald.empty()) {
		lock_ewald--;
		return cuda_context_ewald(ys);
	} else {
		auto ctx = stack_ewald.top();
		stack_ewald.pop();
		lock_ewald--;
		ctx.resize(ys);
		return ctx;
	}
}

void push_context_ewald(cuda_context_ewald ctx) {
	while (lock_ewald++ != 0) {
		lock_ewald--;
	}
	stack_ewald.push(ctx);
	lock_ewald--;
}

std::uint64_t gravity_CC_ewald_cuda(expansion<double> &L, const vect<pos_type> &x, std::vector<const multi_src*> &y) {
	auto ctx = pop_context_ewald(y.size());
	int k = 0;
	for (int i = 0; i < y.size(); i++) {
		ctx.yp[k++] = *y[i];
	}
	*ctx.Lp = L;
	CUDA_CHECK(cudaMemcpyAsync(ctx.y, ctx.yp, sizeof(multi_src) * y.size(), cudaMemcpyHostToDevice, ctx.stream));
	CUDA_CHECK(cudaMemcpyAsync(ctx.L, ctx.Lp, sizeof(expansion<double> ), cudaMemcpyHostToDevice, ctx.stream));

	const int tb_max = EWALD_MAX_TBSIZE;
	int tb_size;
	if (y.size() <= tb_max) {
		tb_size = (((y.size() - 1) / WARPSIZE) + 1) * WARPSIZE;
	} else {
		int nperthread = (y.size() - 1) / tb_max + 1;
		tb_size = (y.size() - 1) / nperthread + 1;
		tb_size = (((tb_size - 1) / WARPSIZE) + 1) * WARPSIZE;
	}
	if( tb_size > EWALD_MAX_TBSIZE) {
		printf( "Error ewald\n");
		abort();
	}

CC_ewald_kernel<<<dim3(1,1,1),dim3(tb_size,1,1),0,ctx.stream>>>(ctx.L, x, ctx.y, y.size());

																				CUDA_CHECK(cudaMemcpyAsync(ctx.Lp, ctx.L, sizeof(expansion<double> ), cudaMemcpyDeviceToHost, ctx.stream));

	while (cudaStreamQuery(ctx.stream) != cudaSuccess) {
		yield_to_hpx();
	}
	L = *ctx.Lp;
	push_context_ewald(std::move(ctx));
}

__global__ void PPPC_direct_kernel(force *F, const vect<pos_type> *x, const vect<pos_type> *y, const std::pair<part_iter, part_iter> *yiters,
		const multi_src *z, int *xindex, int *yindex, int *zindex, float m, float h, bool ewald) {
//	printf("sizeof(force) = %li\n", sizeof(force));

	const int iwarp = threadIdx.y;
	const int ui = blockIdx.x;
	const int l = iwarp * blockDim.x + threadIdx.x;
	const int n = threadIdx.x;

	__shared__ vect<pos_type>
	X[NODESIZE];
	__shared__ force
	G[NWARP][WARPSIZE];

	const auto yb = yindex[ui];
	const auto ye = yindex[ui + 1];
	const auto xb = xindex[ui];
	const auto xe = xindex[ui + 1];
	const auto xsize = xe - xb;
	{
		const float Hinv = 1.0 / h;
		const float H3inv = Hinv * Hinv * Hinv;
		const auto ymax = ((ye - yb - 1) / WORKSIZE + 1) * WORKSIZE + yb;
		if (l < xsize) {
			X[l] = x[xb + l];
		}
		__syncthreads();
		for (int yi = yb + l; yi < ymax; yi += WORKSIZE) {
			int jb, je;
			if (yi < ye) {
				jb = yiters[yi].first;
				je = yiters[yi].second;
//			memcpy(Ymem[l], y + jb, (je - jb) * sizeof(vect<pos_type> ));
			}
			for (int i = xb; i < xe; i++) {
				G[iwarp][n].phi = 0.0;
				G[iwarp][n].g = vect<float>(0.0);
				if (yi < ye) {
					for (int j = jb; j < je; j++) {
						const vect<pos_type> Y = y[j];
						vect<float> dX;
						if (ewald) {
							for (int dim = 0; dim < NDIM; dim++) {
								dX[dim] = float(X[i - xb][dim] - Y[dim]) * float(POS_INV);
							}
						} else {
							for (int dim = 0; dim < NDIM; dim++) {
								dX[dim] = (float(X[i - xb][dim]) - float(Y[dim])) * float(POS_INV);  // 15
							}
						}
						const float r2 = dX.dot(dX);								   // 5
						const float r = sqrt(r2);									   // 1
						const float rinv = float(1) / max(r, 0.5 * h);             	   // 2
						const float rinv3 = rinv * rinv * rinv;                        // 2
						float f, p;
						if (r > h) {
							f = rinv3;
							p = rinv;
						} else if (r > 0.5 * h) {
							const float roh = min(r * Hinv, 1.0);                           // 2
							const float roh2 = roh * roh;                                 // 1
							const float roh3 = roh2 * roh;                                // 1
							f = float(-32.0 / 3.0);
							f = f * roh + float(+192.0 / 5.0);						// 1
							f = f * roh + float(-48.0);								// 1
							f = f * roh + float(+64.0 / 3.0);						// 1
							f = f * roh3 + float(-1.0 / 15.0);						// 1
							f *= rinv3;														// 1
							p = float(+32.0 / 15.0);						// 1
							p = p * roh, float(-48.0 / 5.0);					// 1
							p = p * roh, float(+16.0);							// 1
							p = p * roh, float(-32.0 / 3.0);					// 1
							p = p * roh2, float(+16.0 / 5.0);					// 1
							p = p * roh, float(-1.0 / 15.0);					// 1
							p *= rinv;                                                    	// 1
						} else {
							const float roh = min(r * Hinv, 1.0);                           // 2
							const float roh2 = roh * roh;                                 // 1
							f = float(+32.0);
							f = f * roh + float(-192.0 / 5.0);						// 1
							f = f * roh2 + float(+32.0 / 3.0);						// 1
							f *= H3inv;                                                       	// 1
							p = float(-32.0 / 5.0);
							p = p * roh, float(+48.0 / 5.0);					// 1
							p = p * roh2, float(-16.0 / 3.0);					// 1
							p = p * roh2, float(+14.0 / 5.0);					// 1
							p *= Hinv;														// 1
						}
						const auto dXM = dX * m;								// 3
						for (int dim = 0; dim < NDIM; dim++) {
							G[iwarp][n].g[dim] -= dXM[dim] * f;    				// 6
						}
						// 13S + 2D = 15
						G[iwarp][n].phi -= p * m;    						// 2
					}
				}
				for (int N = WARPSIZE / 2; N > 0; N >>= 1) {
					if (n < N) {
						G[iwarp][n].g += G[iwarp][n + N].g;
						G[iwarp][n].phi += G[iwarp][n + N].phi;
					}
				}
				__syncthreads();
				for (int N = NWARP / 2; N > 0; N >>= 1) {
					if (l < N) {
						G[l][0].g += G[l + N][0].g;
						G[l][0].phi += G[l + N][0].phi;
					}
					__syncthreads();
				}
				if (l == 0) {
					for (int dim = 0; dim < NDIM; dim++) {
						F[i].g[dim] += G[0][0].g[dim];
					}
					F[i].phi += G[0][0].phi;
				}
				__syncthreads();
			}
		}
	}
	{
		const int zmax = ((zindex[ui + 1] - 1) / WORKSIZE + 1) * WORKSIZE;
		for (int zi = zindex[ui] + l; zi < zmax; zi += WORKSIZE) {
			for (int i = xb; i < xe; i++) {
				G[iwarp][n].phi = 0.0;
				G[iwarp][n].g = vect<float>(0.0);
				if (zi < zindex[ui + 1]) {
					const multipole<float> &M = z[zi].m;
					const vect<pos_type> &Y = z[zi].x;
					vect<float> dX;
					if (ewald) {
						for (int dim = 0; dim < NDIM; dim++) {
							dX[dim] = float(X[i - xb][dim] - Y[dim]) * float(POS_INV); // 18
						}
					} else {
						for (int dim = 0; dim < NDIM; dim++) {
							dX[dim] = float(X[i - xb][dim]) * float(POS_INV) - float(Y[dim]) * float(POS_INV);
						}
					}

					vect<double> g;
					double phi;
					multipole_interaction(g, phi, M, dX); // 516
					G[iwarp][n].g += g;  // 0 / 3
					G[iwarp][n].phi += phi;		          // 0 / 1
				}
				__syncthreads();
				for (int N = NWARP / 2; N > 0; N >>= 1) {
					if (l < N) {
						G[l][0].g += G[l + N][0].g;
						G[l][0].phi += G[l + N][0].phi;
					}
					__syncthreads();
				}
				if (l == 0) {
					for (int dim = 0; dim < NDIM; dim++) {
						F[i].g[dim] += G[0][0].g[dim];
					}
					F[i].phi += G[0][0].phi;
				}
				__syncthreads();
			}
		}
	}
}

struct cuda_context {
	int xsize, ysize, zsize, isize;
	cudaStream_t stream;
	force *f;
	vect<pos_type> *x;
	std::pair<part_iter, part_iter> *y;
	multi_src *z;
	int *xi;
	int *yi;
	int *zi;
	force *fp;
	vect<pos_type> *xp;
	multi_src *zp;
	std::pair<part_iter, part_iter> *yp;
	int *xip;
	int *yip;
	int *zip;
	cuda_context(int xs, int ys, int zs, int is) {
		xsize = 1;
		ysize = 1;
		zsize = 1;
		isize = 1;
		while (xsize < xs) {
			xsize *= 2;
		}
		while (zsize < zs) {
			zsize *= 2;
		}
		while (ysize < ys) {
			ysize *= 2;
		}
		while (isize < is) {
			isize *= 2;
		}
		CUDA_CHECK(cudaMalloc(&f, sizeof(force) * xsize));
		CUDA_CHECK(cudaMalloc(&x, sizeof(vect<pos_type> ) * xsize));
		CUDA_CHECK(cudaMalloc(&y, sizeof(std::pair<part_iter, part_iter>) * ysize));
		CUDA_CHECK(cudaMalloc(&z, sizeof(multi_src) * zsize));
		CUDA_CHECK(cudaMalloc(&xi, sizeof(int) * isize));
		CUDA_CHECK(cudaMalloc(&yi, sizeof(int) * isize));
		CUDA_CHECK(cudaMalloc(&zi, sizeof(int) * isize));
		CUDA_CHECK(cudaMallocHost(&fp, sizeof(force) * xsize));
		CUDA_CHECK(cudaMallocHost(&xp, sizeof(vect<pos_type> ) * xsize));
		CUDA_CHECK(cudaMallocHost(&yp, sizeof(std::pair<part_iter, part_iter>) * ysize));
		CUDA_CHECK(cudaMallocHost(&zp, sizeof(multi_src) * zsize));
		CUDA_CHECK(cudaMallocHost(&xip, sizeof(int) * isize));
		CUDA_CHECK(cudaMallocHost(&yip, sizeof(int) * isize));
		CUDA_CHECK(cudaMallocHost(&zip, sizeof(int) * isize));
		CUDA_CHECK(cudaStreamCreate(&stream));
	}
	void resize(int xs, int ys, int zs, int is) {
		if (xs > xsize) {
			while (xsize < xs) {
				xsize *= 2;
			}
			CUDA_CHECK(cudaFree(x));
			CUDA_CHECK(cudaFree(f));
			CUDA_CHECK(cudaMalloc(&f, sizeof(force) * xsize));
			CUDA_CHECK(cudaMalloc(&x, sizeof(vect<pos_type> ) * xsize));
			CUDA_CHECK(cudaFreeHost(xp));
			CUDA_CHECK(cudaFreeHost(fp));
			CUDA_CHECK(cudaMallocHost(&fp, sizeof(force) * xsize));
			CUDA_CHECK(cudaMallocHost(&xp, sizeof(vect<pos_type> ) * xsize));
		}
		if (ys > ysize) {
			while (ysize < ys) {
				ysize *= 2;
			}
			CUDA_CHECK(cudaFree(y));
			CUDA_CHECK(cudaMalloc(&y, sizeof(std::pair<part_iter, part_iter>) * ysize));
			CUDA_CHECK(cudaFreeHost(yp));
			CUDA_CHECK(cudaMallocHost(&yp, sizeof(std::pair<part_iter, part_iter>) * ysize));
		}
		if (zs > zsize) {
			while (zsize < zs) {
				zsize *= 2;
			}
			CUDA_CHECK(cudaFree(z));
			CUDA_CHECK(cudaMalloc(&z, sizeof(multi_src) * zsize));
			CUDA_CHECK(cudaFreeHost(zp));
			CUDA_CHECK(cudaMallocHost(&zp, sizeof(multi_src) * zsize));
		}
		if (is > isize) {
			while (isize < is) {
				isize *= 2;
			}
			CUDA_CHECK(cudaFree(xi));
			CUDA_CHECK(cudaFree(yi));
			CUDA_CHECK(cudaFree(zi));
			CUDA_CHECK(cudaMalloc(&xi, sizeof(int) * isize));
			CUDA_CHECK(cudaMalloc(&yi, sizeof(int) * isize));
			CUDA_CHECK(cudaMalloc(&zi, sizeof(int) * isize));
			CUDA_CHECK(cudaFreeHost(xip));
			CUDA_CHECK(cudaFreeHost(yip));
			CUDA_CHECK(cudaFreeHost(zip));
			CUDA_CHECK(cudaMallocHost(&xip, sizeof(int) * isize));
			CUDA_CHECK(cudaMallocHost(&yip, sizeof(int) * isize));
			CUDA_CHECK(cudaMallocHost(&zip, sizeof(int) * isize));
		}
	}
};

static std::atomic<int> lock(0);
static std::stack<cuda_context> stack;

cuda_context pop_context(int xs, int ys, int zs, int is) {
	while (lock++ != 0) {
		lock--;
	}
	if (stack.empty()) {
		lock--;
		return cuda_context(xs, ys, zs, is);
	} else {
		auto ctx = stack.top();
		stack.pop();
		lock--;
		ctx.resize(xs, ys, zs, is);
		return ctx;
	}
}

void push_context(cuda_context ctx) {
	while (lock++ != 0) {
		lock--;
	}
	stack.push(ctx);
	lock--;
}

std::uint64_t gravity_PP_direct_cuda(std::vector<cuda_work_unit> &&units) {
	thread_cnt++;

	static const auto opts = options::get();
	static const float m = opts.m_tot / opts.problem_size;
	static thread_local std::vector<int> xindex;
	static thread_local std::vector<int> yindex;
	static thread_local std::vector<int> zindex;
	static thread_local std::vector<force> f;
	static thread_local std::vector<vect<pos_type>> x;
	static thread_local std::vector<std::pair<part_iter, part_iter>> y;
	static thread_local std::vector<multi_src> z;
	xindex.resize(0);
	yindex.resize(0);
	zindex.resize(0);
	f.resize(0);
	x.resize(0);
	y.resize(0);
	z.resize(0);

	int xi = 0;
	int yi = 0;
	int zi = 0;
	std::uint64_t interactions = 0;
	for (const auto &unit : units) {
		xindex.push_back(xi);
		yindex.push_back(yi);
		zindex.push_back(zi);
		xi += unit.xptr->size();
		yi += unit.yiters.size();
		zi += unit.z.size();
		f.insert(f.end(), unit.fptr->begin(), unit.fptr->end());
		x.insert(x.end(), unit.xptr->begin(), unit.xptr->end());
		for (int j = 0; j < unit.yiters.size(); j++) {
			std::pair<part_iter, part_iter> iter = unit.yiters[j];
			iter.first -= y_begin;
			iter.second -= y_begin;
			interactions += unit.xptr->size() * (iter.second - iter.first);
			y.push_back(iter);
		}
		for (int j = 0; j < unit.z.size(); j++) {
			z.push_back(*unit.z[j]);
		}
	}
	xindex.push_back(xi);
	yindex.push_back(yi);
	zindex.push_back(zi);
	const auto fbytes = sizeof(force) * f.size();
	const auto xbytes = sizeof(vect<pos_type> ) * x.size();
	const auto ybytes = sizeof(std::pair<part_iter, part_iter>) * y.size();
	const auto zbytes = sizeof(multi_src) * z.size();
	const auto xibytes = sizeof(int) * xindex.size();
	const auto yibytes = sizeof(int) * yindex.size();
	const auto zibytes = sizeof(int) * zindex.size();

	auto ctx = pop_context(x.size(), y.size(), z.size(), zindex.size());
	memcpy(ctx.fp, f.data(), fbytes);
	memcpy(ctx.xp, x.data(), xbytes);
	memcpy(ctx.yp, y.data(), ybytes);
	memcpy(ctx.zp, z.data(), zbytes);
	memcpy(ctx.xip, xindex.data(), xibytes);
	memcpy(ctx.yip, yindex.data(), yibytes);
	memcpy(ctx.zip, zindex.data(), zibytes);
	CUDA_CHECK(cudaMemcpyAsync(ctx.f, ctx.fp, fbytes, cudaMemcpyHostToDevice, ctx.stream));
	CUDA_CHECK(cudaMemcpyAsync(ctx.y, ctx.yp, ybytes, cudaMemcpyHostToDevice, ctx.stream));
	if (zbytes != 0) {
//		printf( "%li %lli %lli\n", zbytes, ctx.z, ctx.zp);
		CUDA_CHECK(cudaMemcpyAsync(ctx.z, ctx.zp, zbytes, cudaMemcpyHostToDevice, ctx.stream));
	}
	CUDA_CHECK(cudaMemcpyAsync(ctx.x, ctx.xp, xbytes, cudaMemcpyHostToDevice, ctx.stream));
	CUDA_CHECK(cudaMemcpyAsync(ctx.yi, ctx.yip, yibytes, cudaMemcpyHostToDevice, ctx.stream));
	CUDA_CHECK(cudaMemcpyAsync(ctx.xi, ctx.xip, xibytes, cudaMemcpyHostToDevice, ctx.stream));
	CUDA_CHECK(cudaMemcpyAsync(ctx.zi, ctx.zip, zibytes, cudaMemcpyHostToDevice, ctx.stream));

PPPC_direct_kernel<<<dim3(units.size(),1,1),dim3(WARPSIZE,NWARP,1),0,ctx.stream>>>(ctx.f,ctx.x,y_vect, ctx.y,ctx.z,ctx.xi,ctx.yi,ctx.zi, m, opts.soft_len, opts.ewald);

										CUDA_CHECK(cudaMemcpyAsync(ctx.fp, ctx.f, fbytes, cudaMemcpyDeviceToHost, ctx.stream));
	while (cudaStreamQuery(ctx.stream) != cudaSuccess) {
		yield_to_hpx();
	}
	int k = 0;
	for (const auto &unit : units) {
		for (auto &this_f : *unit.fptr) {
			this_f = ctx.fp[k];
			k++;
		}
	}
	push_context(ctx);
	thread_cnt--;
	return interactions * 36;
}

