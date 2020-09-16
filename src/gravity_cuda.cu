#include <tigergrav/options.hpp>
#include <tigergrav/cuda_export.hpp>
#include <tigergrav/cuda_check.hpp>
#include <tigergrav/gravity_cuda.hpp>
#include <tigergrav/green.hpp>
#include <tigergrav/interactions.hpp>

#include <stack>
#include <atomic>
void yield_to_hpx();


__device__ __constant__ cuda_ewald_const cuda_ewald;

__device__ const cuda_ewald_const& cuda_get_const() {
	return cuda_ewald;
}

double *flop_ptr;

double cuda_reset_flop() {
	double result;
	double zero = 0.0;
	CUDA_CHECK(cudaMemcpy(&result, flop_ptr, sizeof(double), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(flop_ptr, &zero, sizeof(double), cudaMemcpyHostToDevice));
	return result;
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

		static cuda_ewald_const c;
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
		CUDA_CHECK(cudaMallocHost((void** )&flop_ptr, sizeof(double)));
		cuda_reset_flop();
//		CUDA_CHECK(cudaThreadSetLimit(cudaLimitStackSize, 2048));
	}
	lock--;
}


#define WARPSIZE 32
#define CCSIZE 32

#define WORKSIZE 256
#define PCWORKSIZE 96
#define NODESIZE 64
#define NWARP (WORKSIZE/WARPSIZE)
#define PCNWARP (PCWORKSIZE/WARPSIZE)
#define WARPSIZE 32

#include <cstdint>

__global__ void CC_ewald_kernel(expansion<float> *lptr, const vect<pos_type> X, const multi_src *y, int ysize, bool do_phi, double *flop_ptr) {

	int l = threadIdx.x + blockDim.x * blockIdx.x;
	int n = threadIdx.x;
	int tb_size = blockDim.x;
	auto &L = *lptr;

	__shared__ expansion<float>
	Lacc[CCSIZE];
	__shared__ std::uint64_t
	flop[CCSIZE];
	flop[n] = 0;
	for (int i = 0; i < LP; i++) {
		Lacc[n][i] = 0.0;
	}
	for (int yi = l; yi < ysize; yi += tb_size * gridDim.x) {
		if (yi < ysize) {
			vect<float> dX;
			for (int dim = 0; dim < NDIM; dim++) {
				dX[dim] = float(X[dim] - y[yi].x[dim]) * float(POS_INV); 		// 3
			}
			flop[n] += 3 + multipole_interaction(Lacc[n], y[yi].m, dX, true, do_phi);
		}
	}
	for (int N = tb_size / 2; N > 0; N >>= 1) {
		if (n < N) {
			for (int i = 0; i < LP; i++) {
				Lacc[n][i] += Lacc[n + N][i];
			}
			flop[n] += LP;
		}
	}
	if (n == 0) {
		for (int i = 0; i < LP; i++) {
			atomicAdd(&L[i], Lacc[0][i]);
		}
		flop[n] += LP;
	}
	for (int N = tb_size / 2; N > 0; N >>= 1) {
		if (n < N) {
			flop[n] += flop[n + N];
		}
	}
	if (n == 0) {
		atomicAdd(flop_ptr, flop[0]);
	}
}

struct cuda_context_ewald {
	int ysize;
	cudaStream_t stream;
	expansion<float> *L;
	multi_src *y;
	expansion<float> *Lp;
	multi_src *yp;
	cuda_context_ewald(int ys) {
		ysize = 1;
		while (ysize < ys) {
			ysize *= 2;
		}
		CUDA_CHECK(cudaMalloc(&L, sizeof(expansion<float> )));
		CUDA_CHECK(cudaMalloc(&y, sizeof(multi_src) * ysize));
		CUDA_CHECK(cudaMallocHost(&Lp, sizeof(expansion<float> )));
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

void gravity_CC_ewald_cuda(expansion<float> &L, const vect<pos_type> &x, std::vector<const multi_src*> &y, bool do_phi) {

	cuda_init();

	auto ctx = pop_context_ewald(y.size());
	int k = 0;
	for (int i = 0; i < y.size(); i++) {
		ctx.yp[k++] = *y[i];
	}
	*ctx.Lp = L;
	CUDA_CHECK(cudaMemcpyAsync(ctx.y, ctx.yp, sizeof(multi_src) * y.size(), cudaMemcpyHostToDevice, ctx.stream));
	CUDA_CHECK(cudaMemcpyAsync(ctx.L, ctx.Lp, sizeof(expansion<float> ), cudaMemcpyHostToDevice, ctx.stream));

	int tb_size = (((y.size() - 1) / CCSIZE) + 1) * CCSIZE;

	/**/CC_ewald_kernel<<<dim3(tb_size/CCSIZE,1,1),dim3(CCSIZE,1,1),0,ctx.stream>>>(ctx.L, x, ctx.y, y.size(), do_phi, flop_ptr);

	CUDA_CHECK(cudaMemcpyAsync(ctx.Lp, ctx.L, sizeof(expansion<float> ), cudaMemcpyDeviceToHost, ctx.stream));
	while (cudaStreamQuery(ctx.stream) != cudaSuccess) {
		yield_to_hpx();
	}
	L = *ctx.Lp;
	push_context_ewald(std::move(ctx));
}

template<bool DO_PHI>
/**/__global__ /**/
void PP_direct_kernel(force *F, const vect<pos_type> *x, const vect<pos_type> *y, const std::pair<part_iter, part_iter> *yiters, int *xindex,
		int *yindex, float m, float h, bool ewald, double *flop_ptr) {
//	printf("sizeof(force) = %li\n", sizeof(force));

	const int iwarp = threadIdx.y;
	const int ui = blockIdx.x;
	const int l = iwarp * blockDim.x + threadIdx.x;
	const int n = threadIdx.x;
	const float Hinv = 1.0 / h;
	const float H3inv = Hinv * Hinv * Hinv;
	const float halfh = 0.5 * h;

	__shared__ vect<pos_type>
	X[NODESIZE];
	__shared__ force
	G[NWARP][WARPSIZE];
	__shared__ vect<pos_type>
	Ymem[NWARP][WARPSIZE][SYNCRATE];

	__shared__ std::uint64_t
	flop[NWARP][WARPSIZE];

	flop[iwarp][n] = 0;

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
									dX[dim] = float(this_x[dim] - Y[dim]) * float(POS_INV);			// 3
								}
								flop[iwarp][n] += 3;
							} else {
								for (int dim = 0; dim < NDIM; dim++) {
									dX[dim] = (float(this_x[dim]) - float(Y[dim])) * float(POS_INV);  // 12
								}
								flop[iwarp][n] += 12;
							}
							const float r2 = dX.dot(dX);								   // 5
							const float r = sqrt(r2);// 1
							const float rinv = float(1) / max(r, halfh);// 2
							const float rinv3 = rinv * rinv * rinv;// 2
							flop[iwarp][n] += DO_PHI ? 21 : 19;
							float f, p;
							if (r > h) {
								f = rinv3;
								p = rinv;
							} else if (r > 0.5 * h) {
								const float roh = min(r * Hinv, 1.0);                         // 2
								const float roh2 = roh * roh;// 1
								const float roh3 = roh2 * roh;// 1
								f = float(-32.0 / 3.0);
								f = fma(f, roh, float(+192.0 / 5.0));// 2
								f = fma(f, roh, float(-48.0));// 2
								f = fma(f, roh, float(+64.0 / 3.0));// 2
								f = fma(f, roh3, float(-1.0 / 15.0));// 2
								f *= rinv3;// 1
								flop[iwarp][n] += 13;
								if (DO_PHI) {
									p = float(+32.0 / 15.0);
									p = fma(p, roh, float(-48.0 / 5.0));                                 // 2
									p = fma(p, roh, float(+16.0));// 2
									p = fma(p, roh, float(-32.0 / 3.0));// 2
									p = fma(p, roh2, float(+16.0 / 5.0));// 2
									p = fma(p, roh, float(-1.0 / 15.0));// 2
									p *= rinv;// 1
									flop[iwarp][n] += 11;
								}
							} else {
								const float roh = min(r * Hinv, 1.0);                           // 2
								const float roh2 = roh * roh;// 1
								f = float(+32.0);
								f = fma(f, roh, float(-192.0 / 5.0));// 2
								f = fma(f, roh2, float(+32.0 / 3.0));// 2
								f *= H3inv;// 1
								flop[iwarp][n] += 7;
								if (DO_PHI) {
									p = float(-32.0 / 5.0);
									p = fma(p, roh, float(+48.0 / 5.0));							// 2
									p = fma(p, roh2, float(-16.0 / 3.0));// 2
									p = fma(p, roh2, float(+14.0 / 5.0));// 2
									p *= Hinv;// 1
									flop[iwarp][n] += 8;
								}
							}
							const auto dXM = dX * m;								// 3
							for (int dim = 0; dim < NDIM; dim++) {
								G[iwarp][n].g[dim] -= dXM[dim] * f;    				// 6
							}
							// 13S + 2D = 15
							if( DO_PHI ) {
								G[iwarp][n].phi -= p * m;    							// 2
							}
						}
					}
				}
				for (int N = WARPSIZE / 2; N > 0; N >>= 1) {
					if (n < N) {
						G[iwarp][n].g += G[iwarp][n + N].g;
						flop[iwarp][n] += 4;
						if( DO_PHI ) {
							G[iwarp][n].phi += G[iwarp][n + N].phi;
							flop[iwarp][n] += 1;
						}
					}
				}
				if (n == 0) {
					for (int dim = 0; dim < NDIM; dim++) {
						atomicAdd(&F[i].g[dim], G[iwarp][0].g[dim]);
					}
					flop[iwarp][n] += 4;
					if( DO_PHI ) {
						atomicAdd(&F[i].phi, G[iwarp][0].phi);
						flop[iwarp][n] += 1;
					}
				}
			}
		}
	}
	for (int N = WARPSIZE / 2; N > 0; N >>= 1) {
		if (n < N) {
			flop[iwarp][n] += flop[iwarp][n + N];
		}
	}
	if (n == 0) {
		atomicAdd(flop_ptr, flop[iwarp][0]);
	}
}

__global__
void PC_direct_kernel(force *F, const vect<pos_type> *x, const multi_src *z, int *xindex, int *zindex, bool ewald, bool do_phi, double *flop_ptr) {
//	printf("sizeof(force) = %li\n", sizeof(force));

	const int iwarp = threadIdx.y;
	const int ui = blockIdx.x;
	const int l = iwarp * blockDim.x + threadIdx.x;
	const int n = threadIdx.x;

	__shared__ vect<pos_type>
	X[NODESIZE];
	__shared__ force
	G[PCNWARP][WARPSIZE];
	__shared__ std::uint64_t
	flop[NWARP][WARPSIZE];

	flop[iwarp][n] = 0;

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
						dX[dim] = float(X[i - xb][dim] - Y[dim]) * float(POS_INV); // 3
					}
					flop[iwarp][n] += 3;
				} else {
					for (int dim = 0; dim < NDIM; dim++) {
						dX[dim] = (float(X[i - xb][dim]) - float(Y[dim])) * float(POS_INV); // 12
					}
					flop[iwarp][n] += 12;
				}

				vect<double> g;
				double phi;
				flop[iwarp][n] += 4 + multipole_interaction(g, phi, M, dX, false, do_phi); // 516
				G[iwarp][n].g += g; // 0 / 3
				G[iwarp][n].phi += phi; // 0 / 1
			}
			for (int N = WARPSIZE / 2; N > 0; N >>= 1) {
				if (n < N) {
					G[iwarp][n].g += G[iwarp][n + N].g;
					G[iwarp][n].phi += G[iwarp][n + N].phi;
					flop[iwarp][n] += 4;
				}
			}
			if (n == 0) {
				for (int dim = 0; dim < NDIM; dim++) {
					atomicAdd(&F[i].g[dim], G[iwarp][0].g[dim]);
				}
				atomicAdd(&F[i].phi, G[iwarp][0].phi);
				flop[iwarp][0] += 4;
			}
		}
	}
	for (int N = WARPSIZE / 2; N > 0; N >>= 1) {
		if (n < N) {
			flop[iwarp][n] += flop[iwarp][n + N];
		}
	}
	if (n == 0) {
		atomicAdd(flop_ptr, flop[iwarp][0]);
	}
}

struct cuda_context {
	int xsize, ysize, zsize, isize, ypsize;
	cudaStream_t stream;
	force *f;
	vect<pos_type> *x;
	std::pair<part_iter, part_iter> *y;
	vect<pos_type> *ypos;
	multi_src *z;
	int *xi;
	int *yi;
	int *zi;
	force *fp;
	cuda_context(int xs, int ys, int zs, int is, int yps) {
		xsize = 1;
		ysize = 1;
		zsize = 1;
		isize = 1;
		ypsize = 1;
		while (xsize < xs) {
			xsize *= 2;
		}
		while (zsize < zs) {
			zsize *= 2;
		}
		while (ysize < ys) {
			ysize *= 2;
		}
		while (ypsize < yps) {
			ypsize *= 2;
		}
		while (isize < is) {
			isize *= 2;
		}
		CUDA_CHECK(cudaMalloc(&f, sizeof(force) * xsize));
		CUDA_CHECK(cudaMalloc(&x, sizeof(vect<pos_type> ) * xsize));
		CUDA_CHECK(cudaMalloc(&y, sizeof(std::pair<part_iter, part_iter>) * ysize));
		CUDA_CHECK(cudaMalloc(&ypos, sizeof(vect<pos_type> ) * ypsize));
		CUDA_CHECK(cudaMalloc(&z, sizeof(multi_src) * zsize));
		CUDA_CHECK(cudaMalloc(&xi, sizeof(int) * isize));
		CUDA_CHECK(cudaMalloc(&yi, sizeof(int) * isize));
		CUDA_CHECK(cudaMalloc(&zi, sizeof(int) * isize));
		CUDA_CHECK(cudaMallocHost(&fp, sizeof(force) * xsize));
		CUDA_CHECK(cudaStreamCreate(&stream));
	}
	void resize(int xs, int ys, int zs, int is, int yps) {
		if (yps > ypsize) {
			while (ypsize < yps) {
				ypsize *= 2;
			}
			CUDA_CHECK(cudaFree(ypos));
			CUDA_CHECK(cudaMalloc(&ypos, sizeof(vect<pos_type> ) * ypsize));
		}
		if (xs > xsize) {
			while (xsize < xs) {
				xsize *= 2;
			}
			CUDA_CHECK(cudaFree(x));
			CUDA_CHECK(cudaFree(f));
			CUDA_CHECK(cudaMalloc(&f, sizeof(force) * xsize));
			CUDA_CHECK(cudaMalloc(&x, sizeof(vect<pos_type> ) * xsize));
			CUDA_CHECK(cudaFreeHost(fp));
			CUDA_CHECK(cudaMallocHost(&fp, sizeof(force) * xsize));
		}
		if (ys > ysize) {
			while (ysize < ys) {
				ysize *= 2;
			}
			CUDA_CHECK(cudaFree(y));
			CUDA_CHECK(cudaMalloc(&y, sizeof(std::pair<part_iter, part_iter>) * ysize));
		}
		if (zs > zsize) {
			while (zsize < zs) {
				zsize *= 2;
			}
			CUDA_CHECK(cudaFree(z));
			CUDA_CHECK(cudaMalloc(&z, sizeof(multi_src) * zsize));
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
		}
	}
};

static std::atomic<int> lock(0);
static std::stack<cuda_context> stack;

cuda_context pop_context(int xs, int ys, int zs, int is, int yps) {
	while (lock++ != 0) {
		lock--;
	}
	if (stack.empty()) {
		lock--;
		return cuda_context(xs, ys, zs, is, yps);
	} else {
		auto ctx = stack.top();
		stack.pop();
		lock--;
		ctx.resize(xs, ys, zs, is, yps);
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

void gravity_PP_direct_cuda(std::vector<cuda_work_unit> &&units, const pinned_vector<vect<pos_type>> &ydata, bool do_phi) {
	static const auto opts = options::get();
	static const float m = opts.m_tot / opts.problem_size;
	cuda_init();
	std::uint64_t interactions = 0;
	{
		static thread_local pinned_vector<int> xindex;
		static thread_local pinned_vector<int> yindex;
		static thread_local pinned_vector<force> f;
		static thread_local pinned_vector<vect<pos_type>> x;
		static thread_local pinned_vector<std::pair<part_iter, part_iter>> y;
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
			for (const auto &this_f : *(unit.fptr)) {
				f.push_back(this_f);
			}
			for (const auto &this_x : *(unit.xptr)) {
				x.push_back(this_x);
			}
			for (int j = 0; j < unit.yiters.size(); j++) {
				std::pair<part_iter, part_iter> iter = unit.yiters[j];
				interactions += unit.xptr->size() * (iter.second - iter.first);
				y.push_back(iter);
			}
		}
		xindex.push_back(xi);
		yindex.push_back(yi);
		const auto fbytes = sizeof(force) * f.size();
		const auto xbytes = sizeof(vect<pos_type> ) * x.size();
		const auto ybytes = sizeof(std::pair<part_iter, part_iter>) * y.size();
		const auto ypbytes = sizeof(vect<pos_type> ) * ydata.size();
		const auto xibytes = sizeof(int) * xindex.size();
		const auto yibytes = sizeof(int) * yindex.size();

		auto ctx = pop_context(x.size(), y.size(), 0, xindex.size(), ydata.size());
		CUDA_CHECK(cudaMemcpyAsync(ctx.f, f.data(), fbytes, cudaMemcpyHostToDevice, ctx.stream));
		CUDA_CHECK(cudaMemcpyAsync(ctx.y, y.data(), ybytes, cudaMemcpyHostToDevice, ctx.stream));
		CUDA_CHECK(cudaMemcpyAsync(ctx.ypos, ydata.data(), ypbytes, cudaMemcpyHostToDevice, ctx.stream));
		CUDA_CHECK(cudaMemcpyAsync(ctx.x, x.data(), xbytes, cudaMemcpyHostToDevice, ctx.stream));
		CUDA_CHECK(cudaMemcpyAsync(ctx.yi, yindex.data(), yibytes, cudaMemcpyHostToDevice, ctx.stream));
		CUDA_CHECK(cudaMemcpyAsync(ctx.xi, xindex.data(), xibytes, cudaMemcpyHostToDevice, ctx.stream));
		if (do_phi) {
			PP_direct_kernel<true><<<dim3(units.size(),1,1),dim3(WARPSIZE,NWARP,1),0,ctx.stream>>>(ctx.f,ctx.x,ctx.ypos, ctx.y,ctx.xi,ctx.yi, m, opts.soft_len, opts.ewald, flop_ptr);
		} else {
			PP_direct_kernel<false><<<dim3(units.size(),1,1),dim3(WARPSIZE,NWARP,1),0,ctx.stream>>>(ctx.f,ctx.x,ctx.ypos, ctx.y,ctx.xi,ctx.yi, m, opts.soft_len, opts.ewald, flop_ptr);
		}

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

	}
	{
		static thread_local pinned_vector<int> xindex;
		static thread_local pinned_vector<int> zindex;
		static thread_local pinned_vector<force> f;
		static thread_local pinned_vector<vect<pos_type>> x;
		static thread_local pinned_vector<multi_src> z;
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
				for (const auto &this_f : *(unit.fptr)) {
					f.push_back(this_f);
				}
				for (const auto &this_x : *(unit.xptr)) {
					x.push_back(this_x);
				}
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

			auto ctx = pop_context(x.size(), 0, z.size(), zindex.size(), 0);
			CUDA_CHECK(cudaMemcpyAsync(ctx.f, f.data(), fbytes, cudaMemcpyHostToDevice, ctx.stream));
//		printf( "%li %lli %lli\n", zbytes, ctx.z, ctx.zp);
			CUDA_CHECK(cudaMemcpyAsync(ctx.z, z.data(), zbytes, cudaMemcpyHostToDevice, ctx.stream));
			CUDA_CHECK(cudaMemcpyAsync(ctx.x, x.data(), xbytes, cudaMemcpyHostToDevice, ctx.stream));
			CUDA_CHECK(cudaMemcpyAsync(ctx.xi, xindex.data(), xibytes, cudaMemcpyHostToDevice, ctx.stream));
			CUDA_CHECK(cudaMemcpyAsync(ctx.zi, zindex.data(), zibytes, cudaMemcpyHostToDevice, ctx.stream));

			/**/PC_direct_kernel<<<dim3(size,1,1),dim3(WARPSIZE,PCNWARP,1),0,ctx.stream>>>(ctx.f,ctx.x,ctx.z,ctx.xi,ctx.zi,opts.ewald, do_phi, flop_ptr);

			CUDA_CHECK(cudaMemcpyAsync(ctx.fp, ctx.f, fbytes, cudaMemcpyDeviceToHost, ctx.stream));
			while (cudaStreamQuery(ctx.stream) != cudaSuccess) {
				yield_to_hpx();
			}
			int k = 0;
			for (const auto &unit : units) {
				if (unit.z.size()) {
					for (auto &this_f : *unit.fptr) {
						this_f = ctx.fp[k];
						k++;
					}
				}
			}
			push_context(ctx);
		}
	}
}

