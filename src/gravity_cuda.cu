#include <tigergrav/options.hpp>
#include <tigergrav/cuda_export.hpp>
#include <tigergrav/cuda_check.hpp>
#include <tigergrav/gravity_cuda.hpp>
#include <tigergrav/green.hpp>
#include <tigergrav/interactions.hpp>

#include <stack>
#include <atomic>
void yield_to_hpx();

static vect<pos_type> *y_vect;
static part_iter y_begin;
static part_iter y_end;
static bool first_call = true;
static std::atomic<int> thread_cnt(0);

__device__ __constant__ cuda_ewald_const cuda_ewald;

__device__ const cuda_ewald_const& cuda_get_const() {
	return cuda_ewald;
}

void cuda_init() {
	static std::atomic<int> lock(0);
	static bool init = false;
	while (lock++ != 0) {
		lock--;
	}
	if (!init) {
		static const float efs[LP + 1] = { 1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 5.00000000e-01, 1.00000000e+00, 1.00000000e+00,
				5.00000000e-01, 1.00000000e+00, 5.00000000e-01, 1.66666667e-01, 5.00000000e-01, 5.00000000e-01, 5.00000000e-01, 1.00000000e+00, 5.00000000e-01,
				1.66666667e-01, 5.00000000e-01, 5.00000000e-01, 1.66666667e-01, 4.16666667e-02, 1.66666667e-01, 1.66666667e-01, 2.50000000e-01, 5.00000000e-01,
				2.50000000e-01, 1.66666667e-01, 5.00000000e-01, 5.00000000e-01, 1.66666667e-01, 4.16666667e-02, 1.66666667e-01, 2.50000000e-01, 1.66666667e-01,
				4.16666667e-02, 0.0 };

		cuda_ewald_const c;
		const ewald_indices indices_real(EWALD_REAL_N2, false);
		const ewald_indices indices_four(EWALD_FOUR_N2, true);
		const periodic_parts periodic;
		for (int i = 0; i < indices_real.size(); i++) {
			c.real_indices[i] = indices_real[i];
		}
		for (int i = 0; i < indices_four.size(); i++) {
			c.four_indices[i] = indices_four[i];
			c.periodic_parts[i] = periodic[i];
		}
		for (int i = 0; i < LP; i++) {
			c.exp_factors[i] = efs[i];
		}

		CUDA_CHECK(cudaMemcpyToSymbol(cuda_ewald, &c, sizeof(cuda_ewald_const)));
		init = true;
	}
	lock--;
}

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

#define WARPSIZE 32
#define CCSIZE 32

#define WORKSIZE 256
#define PCWORKSIZE 96
#define NODESIZE 64
#define NWARP (WORKSIZE/WARPSIZE)
#define PCNWARP (PCWORKSIZE/WARPSIZE)
#define WARPSIZE 32

__global__ void CC_ewald_kernel(expansion<double> *lptr, const vect<pos_type> X, const multi_src *y, int ysize) {

	int l = threadIdx.x + blockDim.x * blockIdx.x;
	int n = threadIdx.x;
	int tb_size = blockDim.x;
	auto &L = *lptr;

	__shared__ expansion<double>
	Lacc[CCSIZE];
	for (int i = 0; i < LP; i++) {
		Lacc[n][i] = 0.0;
	}
	for (int yi = l; yi < ysize; yi += tb_size * gridDim.x) {
		if (yi < ysize) {
			vect<float> dX;
			for (int dim = 0; dim < NDIM; dim++) {
				dX[dim] = float(X[dim] - y[yi].x[dim]) * float(POS_INV); // 18
			}
			multipole_interaction(Lacc[n], y[yi].m, dX, true);											// 251936
		}
	}
//	__syncthreads();
	for (int N = tb_size / 2; N > 0; N >>= 1) {
		if (n < N) {
			for (int i = 0; i < LP; i++) {
				Lacc[n][i] += Lacc[n + N][i];
			}
		}
//		__syncthreads();
	}
	if (n == 0) {
		for (int i = 0; i < LP; i++) {
			atomicAdd(&L[i], Lacc[0][i]);
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

	cuda_init();

	auto ctx = pop_context_ewald(y.size());
	int k = 0;
	for (int i = 0; i < y.size(); i++) {
		ctx.yp[k++] = *y[i];
	}
	*ctx.Lp = L;
	CUDA_CHECK(cudaMemcpyAsync(ctx.y, ctx.yp, sizeof(multi_src) * y.size(), cudaMemcpyHostToDevice, ctx.stream));
	CUDA_CHECK(cudaMemcpyAsync(ctx.L, ctx.Lp, sizeof(expansion<double> ), cudaMemcpyHostToDevice, ctx.stream));

	int tb_size = (((y.size() - 1) / CCSIZE) + 1) * CCSIZE;

CC_ewald_kernel<<<dim3(tb_size/CCSIZE,1,1),dim3(CCSIZE,1,1),0,ctx.stream>>>(ctx.L, x, ctx.y, y.size());

																						CUDA_CHECK(cudaMemcpyAsync(ctx.Lp, ctx.L, sizeof(expansion<double> ), cudaMemcpyDeviceToHost, ctx.stream));
	while (cudaStreamQuery(ctx.stream) != cudaSuccess) {
		yield_to_hpx();
	}
	L = *ctx.Lp;
	push_context_ewald(std::move(ctx));
}

__global__ void PP_direct_kernel(force *F, const vect<pos_type> *x, const vect<pos_type> *y, const std::pair<part_iter, part_iter> *yiters, int *xindex,
		int *yindex, float m, float h, bool ewald) {
//	printf("sizeof(force) = %li\n", sizeof(force));

	const int iwarp = threadIdx.y;
	const int ui = blockIdx.x;
	const int l = iwarp * blockDim.x + threadIdx.x;
	const int n = threadIdx.x;
	const float Hinv = 1.0 / h;
	const float H3inv = Hinv * Hinv * Hinv;

	__shared__ vect<pos_type>
	X[NODESIZE];
	__shared__ force
	G[NWARP][WARPSIZE];
	__shared__ vect<pos_type>
	Ymem[NWARP][WARPSIZE][SYNCRATE];

	const auto yb = yindex[ui];
	const auto ye = yindex[ui + 1];
	const auto xb = xindex[ui];
	const auto xe = xindex[ui + 1];
	const auto xsize = xe - xb;
	if (l < xsize) {
		X[l] = x[xb + l];
	}
	__syncthreads();
	{
		const auto ymax = ((ye - yb - 1) / WORKSIZE + 1) * WORKSIZE + yb;
		for (int yi = yb + l; yi < ymax; yi += WORKSIZE) {
			int jb, je;
			if (yi < ye) {
				jb = yiters[yi].first;
				je = yiters[yi].second;
			}
			for (int k = 0; k < WARPSIZE; k++) {
				auto *Yptr = reinterpret_cast<float*>(Ymem[iwarp][k]);
				const int this_yi = ((yi - yb) / WARPSIZE) * WARPSIZE + k + yb;
				if (this_yi < ye) {
					const int jb = yiters[this_yi].first;
					const int je = yiters[this_yi].second;
					const int size = (je - jb) * NDIM;
					if (n < size) {
						Yptr[n] = reinterpret_cast<const float*>(y + jb)[n];
					}
				}
			}
			for (int i = xb; i < xe; i++) {
				const auto this_x = X[i - xb];
				G[iwarp][n].phi = 0.0;
				G[iwarp][n].g = vect<float>(0.0);
				if (yi < ye) {
#pragma loop unroll SYNCRATE
					for (int j0 = 0; j0 < SYNCRATE; j0++) {
						const int j = j0 + jb;
						if (j < je) {
							const vect<pos_type> Y = Ymem[iwarp][n][j0];
							vect<float> dX;
							if (ewald) {
								for (int dim = 0; dim < NDIM; dim++) {
									dX[dim] = float(this_x[dim] - Y[dim]) * float(POS_INV);
								}
							} else {
								for (int dim = 0; dim < NDIM; dim++) {
									dX[dim] = (float(this_x[dim]) - float(Y[dim])) * float(POS_INV);  // 15
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
								p = p * roh + float(-48.0 / 5.0);					// 1
								p = p * roh + float(+16.0);							// 1
								p = p * roh + float(-32.0 / 3.0);					// 1
								p = p * roh2 + float(+16.0 / 5.0);					// 1
								p = p * roh + float(-1.0 / 15.0);					// 1
								p *= rinv;                                                    	// 1
							} else {
								const float roh = min(r * Hinv, 1.0);                           // 2
								const float roh2 = roh * roh;                                 // 1
								f = float(+32.0);
								f = f * roh + float(-192.0 / 5.0);						// 1
								f = f * roh2 + float(+32.0 / 3.0);						// 1
								f *= H3inv;                                                       	// 1
								p = float(-32.0 / 5.0);
								p = p * roh + float(+48.0 / 5.0);					// 1
								p = p * roh2 + float(-16.0 / 3.0);					// 1
								p = p * roh2 + float(+14.0 / 5.0);					// 1
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
				}
				for (int N = WARPSIZE / 2; N > 0; N >>= 1) {
					if (n < N) {
						G[iwarp][n].g += G[iwarp][n + N].g;
						G[iwarp][n].phi += G[iwarp][n + N].phi;
					}
				}
				if (n == 0) {
					for (int dim = 0; dim < NDIM; dim++) {
						atomicAdd(&F[i].g[dim], G[iwarp][0].g[dim]);
					}
					atomicAdd(&F[i].phi, G[iwarp][0].phi);
				}
			}
		}
	}
}

__global__
void PC_direct_kernel(force *F, const vect<pos_type> *x, const multi_src *z, int *xindex, int *zindex, bool ewald) {
//	printf("sizeof(force) = %li\n", sizeof(force));

	const int iwarp = threadIdx.y;
	const int ui = blockIdx.x;
	const int l = iwarp * blockDim.x + threadIdx.x;
	const int n = threadIdx.x;

	__shared__ vect<pos_type>
	X[NODESIZE];
	__shared__ force
	G[PCNWARP][WARPSIZE];

	const auto xb = xindex[ui];
	const auto xe = xindex[ui + 1];
	const auto xsize = xe - xb;
	if (l < xsize) {
		X[l] = x[xb + l];
	}
	__syncthreads();
	int zmax = ((zindex[ui + 1] - zindex[ui] - 1) / PCWORKSIZE + 1) * PCWORKSIZE + zindex[ui];
	for (int zi = zindex[ui] + l; zi < zmax; zi += PCWORKSIZE) {
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
				G[iwarp][n].g += g; // 0 / 3
				G[iwarp][n].phi += phi; // 0 / 1
			}
			for (int N = WARPSIZE / 2; N > 0; N >>= 1) {
				if (n < N) {
					G[iwarp][n].g += G[iwarp][n + N].g;
					G[iwarp][n].phi += G[iwarp][n + N].phi;
				}
			}
			if (n == 0) {
				for (int dim = 0; dim < NDIM; dim++) {
					atomicAdd(&F[i].g[dim], G[iwarp][0].g[dim]);
				}
				atomicAdd(&F[i].phi, G[iwarp][0].phi);
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
	int *xi;
	int *yi;
	force *fp;
	vect<pos_type> *xp;
	std::pair<part_iter, part_iter> *yp;
	int *xip;
	int *yip;
	bool empty;
	cuda_context() {
		empty = true;
	}
	cuda_context(int xs, int ys, int is) {
		empty = false;
		xsize = 1;
		ysize = 1;
		zsize = 1;
		isize = 1;
		while (xsize < xs) {
			xsize *= 2;
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
		CUDA_CHECK(cudaMalloc(&xi, sizeof(int) * isize));
		CUDA_CHECK(cudaMalloc(&yi, sizeof(int) * isize));
		CUDA_CHECK(cudaMallocHost(&fp, sizeof(force) * xsize));
		CUDA_CHECK(cudaMallocHost(&xp, sizeof(vect<pos_type> ) * xsize));
		CUDA_CHECK(cudaMallocHost(&yp, sizeof(std::pair<part_iter, part_iter>) * ysize));
		CUDA_CHECK(cudaMallocHost(&xip, sizeof(int) * isize));
		CUDA_CHECK(cudaMallocHost(&yip, sizeof(int) * isize));
		CUDA_CHECK(cudaStreamCreate(&stream));
	}
	void resize(int xs, int ys, int is) {
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
		if (is > isize) {
			while (isize < is) {
				isize *= 2;
			}
			CUDA_CHECK(cudaFree(xi));
			CUDA_CHECK(cudaFree(yi));
			CUDA_CHECK(cudaMalloc(&xi, sizeof(int) * isize));
			CUDA_CHECK(cudaMalloc(&yi, sizeof(int) * isize));
			CUDA_CHECK(cudaFreeHost(xip));
			CUDA_CHECK(cudaFreeHost(yip));
			CUDA_CHECK(cudaMallocHost(&xip, sizeof(int) * isize));
			CUDA_CHECK(cudaMallocHost(&yip, sizeof(int) * isize));
		}
	}
};

static std::atomic<int> lock(0);
static std::stack<cuda_context> stack;

cuda_context pop_context(int xs, int ys, int is) {
	while (lock++ != 0) {
		lock--;
	}
	if (stack.empty()) {
		lock--;
		return cuda_context(xs, ys, is);
	} else {
		auto ctx = stack.top();
		stack.pop();
		lock--;
		ctx.resize(xs, ys, is);
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

struct cuda_context_pc {
	int xsize, ysize, zsize, isize;
	cudaStream_t stream;
	force *f;
	vect<pos_type> *x;
	multi_src *z;
	int *xi;
	int *zi;
	force *fp;
	vect<pos_type> *xp;
	multi_src *zp;
	int *xip;
	int *zip;
	bool empty;
	cuda_context_pc() {
		empty = true;
	}
	cuda_context_pc(int xs, int zs, int is) {
		empty = false;
		xsize = 1;
		zsize = 1;
		isize = 1;
		while (xsize < xs) {
			xsize *= 2;
		}
		while (zsize < zs) {
			zsize *= 2;
		}
		while (isize < is) {
			isize *= 2;
		}
		CUDA_CHECK(cudaMalloc(&f, sizeof(force) * xsize));
		CUDA_CHECK(cudaMalloc(&x, sizeof(vect<pos_type> ) * xsize));
		CUDA_CHECK(cudaMalloc(&z, sizeof(multi_src) * zsize));
		CUDA_CHECK(cudaMalloc(&xi, sizeof(int) * isize));
		CUDA_CHECK(cudaMalloc(&zi, sizeof(int) * isize));
		CUDA_CHECK(cudaMallocHost(&fp, sizeof(force) * xsize));
		CUDA_CHECK(cudaMallocHost(&xp, sizeof(vect<pos_type> ) * xsize));
		CUDA_CHECK(cudaMallocHost(&zp, sizeof(multi_src) * zsize));
		CUDA_CHECK(cudaMallocHost(&xip, sizeof(int) * isize));
		CUDA_CHECK(cudaMallocHost(&zip, sizeof(int) * isize));
		CUDA_CHECK(cudaStreamCreate(&stream));
	}
	void resize(int xs, int zs, int is) {
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
			CUDA_CHECK(cudaFree(zi));
			CUDA_CHECK(cudaMalloc(&xi, sizeof(int) * isize));
			CUDA_CHECK(cudaMalloc(&zi, sizeof(int) * isize));
			CUDA_CHECK(cudaFreeHost(xip));
			CUDA_CHECK(cudaFreeHost(zip));
			CUDA_CHECK(cudaMallocHost(&xip, sizeof(int) * isize));
			CUDA_CHECK(cudaMallocHost(&zip, sizeof(int) * isize));
		}
	}
};

static std::atomic<int> lock_pc(0);
static std::stack<cuda_context_pc> stack_pc;

cuda_context_pc pop_context_pc(int xs, int zs, int is) {
	while (lock_pc++ != 0) {
		lock_pc--;
	}
	if (stack_pc.empty()) {
		lock_pc--;
		return cuda_context_pc(xs, zs, is);
	} else {
		auto ctx = stack_pc.top();
		stack_pc.pop();
		lock_pc--;
		ctx.resize(xs, zs, is);
		return ctx;
	}
}

void push_context_pc(cuda_context_pc ctx) {
	while (lock_pc++ != 0) {
		lock_pc--;
	}
	stack_pc.push(ctx);
	lock_pc--;
}

std::uint64_t gravity_PP_direct_cuda(std::vector<cuda_work_unit> &&units) {
	static const auto opts = options::get();
	static const float m = opts.m_tot / opts.problem_size;
	cuda_init();
	std::uint64_t interactions = 0;
	cuda_context ctx;
	cuda_context_pc ctx_pc;
	{
		static thread_local std::vector<int> xindex;
		static thread_local std::vector<int> yindex;
		static thread_local std::vector<force> f;
		static thread_local std::vector<vect<pos_type>> x;
		static thread_local std::vector<std::pair<part_iter, part_iter>> y;
		xindex.resize(0);
		yindex.resize(0);
		f.resize(0);
		x.resize(0);
		y.resize(0);
		int xi = 0;
		int yi = 0;
		for (const auto &unit : units) {
			xindex.push_back(xi);
			yindex.push_back(yi);
			xi += unit.xptr->size();
			yi += unit.yiters.size();
			f.insert(f.end(), unit.fptr->begin(), unit.fptr->end());
			x.insert(x.end(), unit.xptr->begin(), unit.xptr->end());
			for (int j = 0; j < unit.yiters.size(); j++) {
				std::pair<part_iter, part_iter> iter = unit.yiters[j];
				iter.first -= y_begin;
				iter.second -= y_begin;
				interactions += unit.xptr->size() * (iter.second - iter.first);
				y.push_back(iter);
			}
		}
		xindex.push_back(xi);
		yindex.push_back(yi);
		const auto fbytes = sizeof(force) * f.size();
		const auto xbytes = sizeof(vect<pos_type> ) * x.size();
		const auto ybytes = sizeof(std::pair<part_iter, part_iter>) * y.size();
		const auto xibytes = sizeof(int) * xindex.size();
		const auto yibytes = sizeof(int) * yindex.size();

		ctx = pop_context(x.size(), y.size(), xindex.size());
		memcpy(ctx.fp, f.data(), fbytes);
		memcpy(ctx.xp, x.data(), xbytes);
		memcpy(ctx.yp, y.data(), ybytes);
		memcpy(ctx.xip, xindex.data(), xibytes);
		memcpy(ctx.yip, yindex.data(), yibytes);
		CUDA_CHECK(cudaMemcpyAsync(ctx.f, ctx.fp, fbytes, cudaMemcpyHostToDevice, ctx.stream));
		CUDA_CHECK(cudaMemcpyAsync(ctx.y, ctx.yp, ybytes, cudaMemcpyHostToDevice, ctx.stream));
		CUDA_CHECK(cudaMemcpyAsync(ctx.x, ctx.xp, xbytes, cudaMemcpyHostToDevice, ctx.stream));
		CUDA_CHECK(cudaMemcpyAsync(ctx.yi, ctx.yip, yibytes, cudaMemcpyHostToDevice, ctx.stream));
		CUDA_CHECK(cudaMemcpyAsync(ctx.xi, ctx.xip, xibytes, cudaMemcpyHostToDevice, ctx.stream));

PP_direct_kernel<<<dim3(units.size(),1,1),dim3(WARPSIZE,NWARP,1),0,ctx.stream>>>(ctx.f,ctx.x,y_vect, ctx.y,ctx.xi,ctx.yi, m, opts.soft_len, opts.ewald);

											CUDA_CHECK(cudaMemcpyAsync(ctx.fp, ctx.f, fbytes, cudaMemcpyDeviceToHost, ctx.stream));
	}
	{
		static thread_local std::vector<int> xindex;
		static thread_local std::vector<int> zindex;
		static thread_local std::vector<force> f;
		static thread_local std::vector<vect<pos_type>> x;
		static thread_local std::vector<multi_src> z;
		xindex.resize(0);
		zindex.resize(0);
		f.resize(0);
		x.resize(0);
		z.resize(0);

		int xi = 0;
		int zi = 0;
		int size = 0;
		std::uint64_t interactions = 0;
		for (const auto &unit : units) {
			if (unit.z.size()) {
				xindex.push_back(xi);
				zindex.push_back(zi);
				xi += unit.xptr->size();
				zi += unit.z.size();
				f.insert(f.end(), unit.fptr->begin(), unit.fptr->end());
				x.insert(x.end(), unit.xptr->begin(), unit.xptr->end());
				for (int j = 0; j < unit.z.size(); j++) {
					z.push_back(*unit.z[j]);
				}
				size++;
			}
		}
		xindex.push_back(xi);
		zindex.push_back(zi);
		if (z.size()) {
			const auto fbytes = sizeof(force) * f.size();
			const auto xbytes = sizeof(vect<pos_type> ) * x.size();
			const auto zbytes = sizeof(multi_src) * z.size();
			const auto xibytes = sizeof(int) * xindex.size();
			const auto zibytes = sizeof(int) * zindex.size();

			ctx_pc = pop_context_pc(x.size(), z.size(), zindex.size());
			memcpy(ctx_pc.fp, f.data(), fbytes);
			memcpy(ctx_pc.xp, x.data(), xbytes);
			memcpy(ctx_pc.zp, z.data(), zbytes);
			memcpy(ctx_pc.xip, xindex.data(), xibytes);
			memcpy(ctx_pc.zip, zindex.data(), zibytes);
			CUDA_CHECK(cudaMemcpyAsync(ctx_pc.f, ctx_pc.fp, fbytes, cudaMemcpyHostToDevice, ctx_pc.stream));
//		printf( "%li %lli %lli\n", zbytes, ctx_pc.z, ctx_pc.zp);
			CUDA_CHECK(cudaMemcpyAsync(ctx_pc.z, ctx_pc.zp, zbytes, cudaMemcpyHostToDevice, ctx_pc.stream));
			CUDA_CHECK(cudaMemcpyAsync(ctx_pc.x, ctx_pc.xp, xbytes, cudaMemcpyHostToDevice, ctx_pc.stream));
			CUDA_CHECK(cudaMemcpyAsync(ctx_pc.xi, ctx_pc.xip, xibytes, cudaMemcpyHostToDevice, ctx_pc.stream));
			CUDA_CHECK(cudaMemcpyAsync(ctx_pc.zi, ctx_pc.zip, zibytes, cudaMemcpyHostToDevice, ctx_pc.stream));

PC_direct_kernel<<<dim3(size,1,1),dim3(WARPSIZE,PCNWARP,1),0,ctx_pc.stream>>>(ctx_pc.f,ctx_pc.x,ctx_pc.z,ctx_pc.xi,ctx_pc.zi,opts.ewald);

																																		CUDA_CHECK(cudaMemcpyAsync(ctx_pc.fp, ctx_pc.f, fbytes, cudaMemcpyDeviceToHost, ctx.stream));
		}
	}


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
	if (!ctx_pc.empty) {
		while (cudaStreamQuery(ctx_pc.stream) != cudaSuccess) {
			yield_to_hpx();
		}
		int k = 0;
		for (const auto &unit : units) {
			if (unit.z.size()) {
				for (auto &this_f : *unit.fptr) {
					this_f = ctx_pc.fp[k];
					k++;
				}
			}
		}
		push_context_pc(ctx_pc);
	}
	return interactions * 36;
}

