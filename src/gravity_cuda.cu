#include <tigergrav/options.hpp>
#include <tigergrav/cuda_check.hpp>
#include <tigergrav/gravity_cuda.hpp>

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
#define SYNCRATE 8

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
	__shared__ force
	G[NWARP][WARPSIZE];
	__shared__ force
	Gstore[NWARP][NODESIZE];

	const auto yb = yindex[ui];
	const auto ye = yindex[ui + 1];
	const auto xb = xindex[ui];
	const auto xe = xindex[ui + 1];
	const auto xsize = xe - xb;
	const auto ymax = ((ye - yb - 1) / WORKSIZE + 1) * WORKSIZE + yb;
	if (l < xsize) {
		X[l] = x[xb + l];
	}
	if (l < NODESIZE) {
		for( int i = 0; i < NWARP; i++) {
			Gstore[i][l].g = vect<float>(0.0);
			Gstore[i][l].phi = 0.0;
		}
	}
	__syncthreads();
	for (int yi = yb + l; yi < ymax; yi += WORKSIZE) {
		for (int i = xb; i < xe; i++) {
			G[iwarp][n].phi = 0.0;
			G[iwarp][n].g = vect<float>(0.0);
			if (yi < ye) {
				const auto &iter = yiters[yi];
				const int yb = iter.first;
				const int ye = iter.second;
				for (int j = yb; j < ye; j++) {
					const vect<pos_type> Y = y[j];
					vect<float> dX;
					if (ewald) {
						for (int dim = 0; dim < NDIM; dim++) {
							dX[dim] = float(double(X[i - xindex[ui]][dim] - Y[dim]) * double(POS_INV));
						}
					} else {
						for (int dim = 0; dim < NDIM; dim++) {
							dX[dim] = float(double(X[i - xindex[ui]][dim]) * double(POS_INV) - double(Y[dim]) * double(POS_INV));
						}
					}
					const float r2 = dX.dot(dX);								   // 5
					const float r = sqrt(r2);									   // 7
					const float rinv = float(1) / max(r, 0.5 * h);             //36
					const float rinv3 = rinv * rinv * rinv;                       // 2
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
					const auto dXM = dX * m;
					for (int dim = 0; dim < NDIM; dim++) {
						G[iwarp][n].g[dim] += double(-dXM[dim] * f);    						// 15
					}
					// 13S + 2D = 15
					G[iwarp][n].phi += double(-p * m);    						// 10
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
					Gstore[iwarp][i - xb].g[dim] += G[iwarp][0].g[dim];
				}
				Gstore[iwarp][i - xb].phi += G[iwarp][0].phi;
			}
		}
	}
	__syncthreads();
	if (l < xsize) {
		for (int i = 0; i < NWARP; i++) {
			for (int dim = 0; dim < NDIM; dim++) {
				atomicAdd(&F[l + xb].g[dim], Gstore[i][l].g[dim]);
			}
			atomicAdd(&F[l + xb].phi, Gstore[i][l].phi);
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
	static const double m = opts.m_tot / opts.problem_size;
	std::vector<int> xindex;
	std::vector<int> yindex;
	std::vector<force> f;
	std::vector<vect<pos_type>> x;
	std::vector<std::pair<part_iter, part_iter>> y;

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
	return 0;
}

//#define MAXWORKSIZE 512
//
//__global__ void PP_direct_kernel(force *F, vect<pos_type> *x, vect<pos_type> *y, std::pair<int, int> *yiters, int yiter_start, float m, float h, bool ewald) {
//	const int i = blockIdx.x;
//	const int WORKSIZE = blockDim.x;
//	const int l = threadIdx.x;
//	const float Hinv = 1.0 / h;
//	const float H3inv = 1.0 / (h * h * h);
//	const vect<pos_type> X = x[i];
//
//	__shared__ force
//	this_g[MAXBLOCKSIZE];
//	for (int dim = 0; dim < NDIM; dim++) {
//		this_g[l].g[dim] = 0.0;
//	}
//	this_g[l].phi = 0.0;
//
//	const auto yb = yiters[yiter_start + l].first;
//	const auto ye = yiters[yiter_start + l].second;
//	for (part_iter yi = yb; yi < ye; yi++) {
//
//		const vect<pos_type> Y = y[yi];
//		vect<float> dX;
//		if (ewald) {
//			for (int dim = 0; dim < NDIM; dim++) {
//				dX[dim] = float(double(X[dim] - Y[dim]) * double(POS_INV));
//			}
//		} else {
//			for (int dim = 0; dim < NDIM; dim++) {
//				dX[dim] = float(double(X[dim]) * double(POS_INV) - double(Y[dim]) * double(POS_INV));
//			}
//		}
//		const float r2 = dX.dot(dX);								   // 5
//		const float r = sqrt(r2);									   // 7
//		const float rinv = float(1) / max(r, 0.5 * h);             //36
//		const float rinv3 = rinv * rinv * rinv;                       // 2
//		float f, p;
//		if (r > h) {
//			f = rinv3;
//			p = rinv;
//		} else if (r > 0.5 * h) {
//			const float roh = min(r * Hinv, 1.0);                           // 2
//			const float roh2 = roh * roh;                                 // 1
//			const float roh3 = roh2 * roh;                                // 1
//			f = float(-32.0 / 3.0);
//			f = f * roh + float(+192.0 / 5.0);						// 1
//			f = f * roh + float(-48.0);								// 1
//			f = f * roh + float(+64.0 / 3.0);						// 1
//			f = f * roh3 + float(-1.0 / 15.0);						// 1
//			f *= rinv3;														// 1
//			p = float(+32.0 / 15.0);						// 1
//			p = p * roh, float(-48.0 / 5.0);					// 1
//			p = p * roh, float(+16.0);							// 1
//			p = p * roh, float(-32.0 / 3.0);					// 1
//			p = p * roh2, float(+16.0 / 5.0);					// 1
//			p = p * roh, float(-1.0 / 15.0);					// 1
//			p *= rinv;                                                    	// 1
//		} else {
//			const float roh = min(r * Hinv, 1.0);                           // 2
//			const float roh2 = roh * roh;                                 // 1
//			f = float(+32.0);
//			f = f * roh + float(-192.0 / 5.0);						// 1
//			f = f * roh2 + float(+32.0 / 3.0);						// 1
//			f *= H3inv;                                                       	// 1
//			p = float(-32.0 / 5.0);
//			p = p * roh, float(+48.0 / 5.0);					// 1
//			p = p * roh2, float(-16.0 / 3.0);					// 1
//			p = p * roh2, float(+14.0 / 5.0);					// 1
//			p *= Hinv;														// 1
//		}
//		const auto dXM = dX * m;
//		for (int dim = 0; dim < NDIM; dim++) {
//			this_g[l].g[dim] -= double(dXM[dim] * f);    						// 15
//		}
//		// 13S + 2D = 15
//		this_g[l].phi -= p * m;    						// 10
//	}
//	int P = blocksize;
//	P--;
//	P |= P >> 1;
//	P |= P >> 2;
//	P |= P >> 4;
//	P |= P >> 8;
//	P++;
//	P >>= 1;
//	__syncthreads();
//	if (l + P < blocksize) {
//		this_g[l].g += this_g[l + P].g;
//		this_g[l].phi += this_g[l + P].phi;
//	}
//	__syncthreads();
//	for (int N = P / 2; N > 0; N >>= 1) {
//		if (l < N) {
//			this_g[l].g += this_g[l + N].g;
//			this_g[l].phi += this_g[l + N].phi;
//		}
//		__syncthreads();
//	}
//	for (int dim = 0; dim < NDIM; dim++) {
//		F[i].g[dim] += this_g[0].g[dim]; // 1 OP
//	}
//	F[i].phi += this_g[0].phi; // 1 OP
//}
//
//struct cuda_context {
//	force *f;
//	vect<pos_type> *x;
//	std::pair<int, int> *yiter;
//	force *fpin;
//	vect<pos_type> *xpin;
//	std::pair<int, int> *yiterpin;
//	cudaStream_t stream;
//	cudaEvent_t event;
//	int xsize;
//	int ysize;
//	void allocate() {
//		CUDA_CHECK(cudaMalloc(&f, sizeof(force) * xsize));
//		CUDA_CHECK(cudaMalloc(&x, sizeof(vect<pos_type> ) * xsize));
//		CUDA_CHECK(cudaMalloc(&yiter, sizeof(std::pair<int, int>) * ysize));
//		CUDA_CHECK(cudaMallocHost(&fpin, sizeof(force) * xsize));
//		CUDA_CHECK(cudaMallocHost(&xpin, sizeof(vect<pos_type> ) * xsize));
//		CUDA_CHECK(cudaMallocHost(&yiterpin, sizeof(std::pair<int, int>) * ysize));
//	}
//	cuda_context(int xs, int ys) {
//		xsize = xs;
//		ysize = ys;
//		allocate();
//		CUDA_CHECK(cudaEventCreate(&event));
//		CUDA_CHECK(cudaStreamCreate(&stream));
//	}
//	void resize(int xs, int ys) {
//		if (xs > xsize || ys > ysize) {
//			CUDA_CHECK(cudaFree(x));
//			CUDA_CHECK(cudaFree(yiter));
//			CUDA_CHECK(cudaFree(f));
//			CUDA_CHECK(cudaFreeHost(xpin));
//			CUDA_CHECK(cudaFreeHost(yiterpin));
//			CUDA_CHECK(cudaFreeHost(fpin));
//			xsize = std::max(xs, xsize);
//			ysize = std::max(ys, ysize);
//			allocate();
//		}
//	}
//	void copy_to_pinned(const std::vector<force> &this_f, const std::vector<vect<pos_type>> &this_x,
//			const std::vector<std::pair<part_iter, part_iter>> &these_yiters) {
//		memcpy(fpin, this_f.data(), this_f.size() * sizeof(force));
//		memcpy(xpin, this_x.data(), this_x.size() * sizeof(vect<pos_type> ));
//		for (int i = 0; i < these_yiters.size(); i++) {
//			yiterpin[i].first = these_yiters[i].first - y_begin;
//			yiterpin[i].second = these_yiters[i].second - y_begin;
//		}
//	}
//};
//
//static std::atomic<int> lock(0);
//static std::stack<cuda_context> stack;
//
//cuda_context pop_context(int xs, int ys) {
//	while (lock++ != 0) {
//		lock--;
//	}
//	if (stack.empty()) {
//		lock--;
//		return cuda_context(xs, ys);
//	} else {
//		auto ctx = stack.top();
//		ctx.resize(xs, ys);
//		stack.pop();
//		lock--;
//		return ctx;
//	}
//}
//
//void push_context(cuda_context ctx) {
//	while (lock++ != 0) {
//		lock--;
//	}
//	stack.push(ctx);
//	lock--;
//}
//
//std::uint64_t gravity_PP_direct_cuda(std::vector<force> &f, const std::vector<vect<pos_type>> &x, std::vector<std::pair<part_iter, part_iter>> &yiter,
//		bool do_phi) {
//
//	static const auto opts = options::get();
//	const double m = opts.m_tot / opts.problem_size;
//	cuda_context ctx = pop_context(x.size(), yiter.size());
//	ctx.copy_to_pinned(f, x, yiter);
//	CUDA_CHECK(cudaMemcpyAsync(ctx.f, ctx.fpin, x.size() * sizeof(force), cudaMemcpyHostToDevice, ctx.stream));
//	CUDA_CHECK(cudaMemcpyAsync(ctx.x, ctx.xpin, x.size() * sizeof(vect<pos_type> ), cudaMemcpyHostToDevice, ctx.stream));
//	CUDA_CHECK(cudaMemcpyAsync(ctx.yiter, ctx.yiterpin, yiter.size() * sizeof(std::pair<int, int>), cudaMemcpyHostToDevice, ctx.stream));
//	for (int j = 0; j < yiter.size(); j += MAXBLOCKSIZE) {
//		const int this_block_size = std::min((int) yiter.size(), j + MAXBLOCKSIZE) - j;
//	PP_direct_kernel<<<x.size(),this_block_size,0,ctx.stream>>>(ctx.f,ctx.x,y_vect, ctx.yiter,j, m, opts.soft_len, opts.ewald);
//	CUDA_CHECK(cudaMemcpyAsync(f.data(), ctx.f, x.size() * sizeof(force), cudaMemcpyDeviceToHost, ctx.stream));
//}
//CUDA_CHECK(cudaEventRecord(ctx.event, ctx.stream));
//while (cudaEventQuery(ctx.event) != cudaSuccess) {
//	yield_to_hpx();
//}
//push_context(ctx);
//return 0;
//}
////#define P 512
////
////__global__ void PP_direct_kernel(force *F, vect<pos_type> *x, vect<pos_type> *y, int ysize, float m, float h, bool ewald);
////
////struct cuda_context {
////	force *f;
////	vect<pos_type> *x;
////	vect<pos_type> *y;
////	cudaStream_t stream;
////	cudaEvent_t event;
////	int xsize;
////	int ysize;
////	void allocate() {
////		cudaMalloc(&f, sizeof(force) * xsize);
////		cudaMalloc(&x, sizeof(vect<pos_type> ) * xsize);
////		cudaMalloc(&y, sizeof(vect<pos_type> ) * ysize);
////	}
////	cuda_context(int xs, int ys) {
////		xsize = xs;
////		ysize = ys;
////		allocate();
////		cudaEventCreate(&event);
////		cudaStreamCreate(&stream);
////	}
////	void resize(int xs, int ys) {
////		if (xs > xsize || ys > ysize) {
////			cudaFree(x);
////			cudaFree(y);
////			cudaFree(f);
////			xsize = std::max(xs, xsize);
////			ysize = std::max(ys, ysize);
////			allocate();
////		}
////	}
////};
////
////static std::atomic<int> lock(0);
////static std::stack<cuda_context> stack;
////
////cuda_context pop_context(int xs, int ys) {
////	while (lock++ != 0) {
////		lock--;
////	}
////	if (stack.empty()) {
////		lock--;
////		return cuda_context(xs, ys);
////	} else {
////		auto ctx = stack.top();
////		ctx.resize(xs, ys);
////		stack.pop();
////		lock--;
////		return ctx;
////	}
////}
////
////void push_context(cuda_context ctx) {
////	while (lock++ != 0) {
////		lock--;
////	}
////	stack.push(ctx);
////	lock--;
////}
////
////std::uint64_t gravity_PP_direct_cuda(std::vector<force> &f, const std::vector<vect<pos_type>> &x, std::vector<vect<pos_type>> y, bool do_phi) {
////
////	static const auto opts = options::get();
////	const double m = opts.m_tot / opts.problem_size;
////	cuda_context ctx = pop_context(x.size(), y.size());
////	cudaMemcpy(ctx.f, f.data(), x.size() * sizeof(force), cudaMemcpyHostToDevice);
////	cudaMemcpy(ctx.x, x.data(), x.size() * sizeof(vect<pos_type> ), cudaMemcpyHostToDevice);
////	cudaMemcpy(ctx.y, y.data(), y.size() * sizeof(vect<pos_type> ), cudaMemcpyHostToDevice);
////	PP_direct_kernel<<<x.size(),P,0,ctx.stream>>>(ctx.f,ctx.x,ctx.y,y.size(), m, opts.soft_len, opts.ewald);
////	cudaEventRecord(ctx.event, ctx.stream);
////	while (cudaEventQuery(ctx.event) != cudaSuccess) {
////		yield_to_hpx();
////	}
////	cudaMemcpy(f.data(), ctx.f, x.size() * sizeof(force), cudaMemcpyDeviceToHost);
////	push_context(ctx);
////	return 0;
////}
////
////__global__ void PP_direct_kernel(force *F, vect<pos_type> *x, vect<pos_type> *y, int ysize, float m, float h, bool ewald) {
////	const int i = blockIdx.x;
////	const int l = threadIdx.x;
////	const float Hinv = 1.0 / h;
////	const float H3inv = 1.0 / (h * h * h);
////	const vect<pos_type> X = x[i];
////
////	__shared__ force
////	this_g[P];
////	for (int dim = 0; dim < NDIM; dim++) {
////		this_g[l].g[dim] = 0.0;
////	}
////	this_g[l].phi = 0.0;
////	for (int j = l; j < ysize; j += P) {
////		const vect<pos_type> Y = y[j];
////		vect<float> dX;
////		if (ewald) {
////			for (int dim = 0; dim < NDIM; dim++) {
////				dX[dim] = float(double(X[dim] - Y[dim]) * double(POS_INV));
////			}
////		} else {
////			for (int dim = 0; dim < NDIM; dim++) {
////				dX[dim] = float(double(X[dim]) * double(POS_INV) - double(Y[dim]) * double(POS_INV));
////			}
////		}
////		const float r2 = dX.dot(dX);								   // 5
////		const float r = sqrt(r2);									   // 7
////		const float rinv = float(1) / max(r, 0.5 * h);             //36
////		const float rinv3 = rinv * rinv * rinv;                       // 2
////		float f, p;
////		if (r > h) {
////			f = rinv3;
////			p = rinv;
////		} else if (r > 0.5 * h) {
////			const float roh = min(r * Hinv, 1.0);                           // 2
////			const float roh2 = roh * roh;                                 // 1
////			const float roh3 = roh2 * roh;                                // 1
////			f = float(-32.0 / 3.0);
////			f = f * roh + float(+192.0 / 5.0);						// 1
////			f = f * roh + float(-48.0);								// 1
////			f = f * roh + float(+64.0 / 3.0);						// 1
////			f = f * roh3 + float(-1.0 / 15.0);						// 1
////			f *= rinv3;														// 1
////			p = float(+32.0 / 15.0);						// 1
////			p = p * roh, float(-48.0 / 5.0);					// 1
////			p = p * roh, float(+16.0);							// 1
////			p = p * roh, float(-32.0 / 3.0);					// 1
////			p = p * roh2, float(+16.0 / 5.0);					// 1
////			p = p * roh, float(-1.0 / 15.0);					// 1
////			p *= rinv;                                                    	// 1
////		} else {
////			const float roh = min(r * Hinv, 1.0);                           // 2
////			const float roh2 = roh * roh;                                 // 1
////			f = float(+32.0);
////			f = f * roh + float(-192.0 / 5.0);						// 1
////			f = f * roh2 + float(+32.0 / 3.0);						// 1
////			f *= H3inv;                                                       	// 1
////			p = float(-32.0 / 5.0);
////			p = p * roh, float(+48.0 / 5.0);					// 1
////			p = p * roh2, float(-16.0 / 3.0);					// 1
////			p = p * roh2, float(+14.0 / 5.0);					// 1
////			p *= Hinv;														// 1
////		}
////		const auto dXM = dX * m;
////		for (int dim = 0; dim < NDIM; dim++) {
////			this_g[l].g[dim] -= double(dXM[dim] * f);    						// 15
////		}
////		// 13S + 2D = 15
////		this_g[l].phi -= p * m;    						// 10
////	}
////	__syncthreads();
////	for (int N = P / 2; N > 0; N >>= 1) {
////		if (l < N) {
////			this_g[l].g += this_g[l + N].g;
////			this_g[l].phi += this_g[l + N].phi;
////		}
////		__syncthreads();
////	}
////	for (int dim = 0; dim < NDIM; dim++) {
////		F[i].g[dim] += this_g[0].g[dim]; // 1 OP
////	}
////	F[i].phi += this_g[0].phi; // 1 OP
////}
