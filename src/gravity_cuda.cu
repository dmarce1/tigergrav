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

#define WORKSIZE 128
#define NODESIZE 64
#define NWARP (WORKSIZE/WARPSIZE)
#define WARPSIZE 32

__global__ void CC_ewald_kernel(expansion<double> *lptr, const vect<pos_type> X, const multi_src *y, int ysize) {
	int l = threadIdx.x;
	auto &L = *lptr;
	__shared__ expansion<double>
	Lacc[WARPSIZE];
	for (int i = 0; i < LP; i++) {
		Lacc[l][i] = 0.0;
	}
	for (int yi = l; yi < ysize; yi += WARPSIZE) {
		vect<pos_type> Y = y[l].x;
		multipole<float> M = y[l].m;
		vect<float> dX;
		for (int dim = 0; dim < NDIM; dim++) {
			dX[dim] = float(X[dim] - Y[dim]) * float(POS_INV); // 18
		}
		multipole_interaction(Lacc[l], M, dX, true);											// 251936
	}
	for (int N = WARPSIZE / 2; N > 0; N >>= 1) {
		for (int i = 0; i < LP; i++) {
			Lacc[l][i] += Lacc[l + N][i];
		}
	}
	if (l == 0) {
		for (int i = 0; i < LP; i++) {
			L[i] += Lacc[0][i];
		}
	}
}

std::uint64_t gravity_CC_ewald_cuda(expansion<double>& L, const vect<pos_type> &x, std::vector<const multi_src*> &y) {
	expansion<double> *Ldev;
	multi_src *Ypinned;
	multi_src *Ydev;
	cudaStream_t stream;
	CUDA_CHECK(cudaMallocHost(&Ypinned, sizeof(multi_src) * y.size()));
	int k = 0;
	for (int i = 0; i < y.size(); i++) {
		Ypinned[k++] = *y[i];
	}
	CUDA_CHECK(cudaStreamCreate(&stream));
	CUDA_CHECK(cudaMalloc((void**) &Ldev, sizeof(expansion<double> )));
	CUDA_CHECK(cudaMalloc((void**) &Ydev, sizeof(multi_src) * y.size()));
	CUDA_CHECK(cudaMemcpy(Ydev, Ypinned, sizeof(multi_src), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(Ldev, &L, sizeof(expansion<double>), cudaMemcpyHostToDevice));

	CC_ewald_kernel<<<dim3(1,1,1),dim3(WARPSIZE,1,1),0,stream>>>(Ldev, x, Ydev, y.size());
	while (cudaStreamQuery(stream) != cudaSuccess) {
		yield_to_hpx();
	}
	CUDA_CHECK(cudaMemcpy(&L, Ldev, sizeof(expansion<double>), cudaMemcpyDeviceToHost));

	CUDA_CHECK(cudaStreamDestroy(stream));
	CUDA_CHECK(cudaFreeHost(Ypinned));
	CUDA_CHECK(cudaFree(Ldev));
	CUDA_CHECK(cudaFree (Ydev));
}

__global__ void PP_direct_kernel(force *F, vect<pos_type> *x, vect<pos_type> *y, std::pair<part_iter, part_iter> *yiters, int *xindex, int *yindex, float m,
		float h, bool ewald) {
	const int iwarp = threadIdx.y;
	const int ui = blockIdx.x;
	const int l = iwarp * blockDim.x + threadIdx.x;
	const int n = threadIdx.x;
	const float Hinv = 1.0 / h;
	const float H3inv = Hinv * Hinv * Hinv;

	__shared__ vect<pos_type>
	X[NODESIZE];
//	__shared__ vect<pos_type>
//	Ymem[WORKSIZE][SYNCRATE];
	__shared__ force
	G[NWARP][WARPSIZE];

	const auto yb = yindex[ui];
	const auto ye = yindex[ui + 1];
	const auto xb = xindex[ui];
	const auto xe = xindex[ui + 1];
	const auto xsize = xe - xb;
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
			if (l == 0) {
				for (int iw = 0; iw < NWARP; iw++) {
					for (int dim = 0; dim < NDIM; dim++) {
						F[i].g[dim] += G[iw][0].g[dim];
					}
					F[i].phi += G[iw][0].phi;
				}
			}
			__syncthreads();
		}
	}

}

struct cuda_context {
	int xsize, ysize, isize;
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
	cuda_context(int xs, int ys, int is) {
		xsize = 1;
		ysize = 1;
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

std::uint64_t gravity_PP_direct_cuda(std::vector<cuda_work_unit> &&units) {
	thread_cnt++;

	static const auto opts = options::get();
	static const float m = opts.m_tot / opts.problem_size;
	std::vector<int> xindex;
	std::vector<int> yindex;
	std::vector<force> f;
	std::vector < vect < pos_type >> x;
	std::vector<std::pair<part_iter, part_iter>> y;

	int xi = 0;
	int yi = 0;
	std::uint64_t interactions = 0;
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

	auto ctx = pop_context(x.size(), y.size(), xindex.size());
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

