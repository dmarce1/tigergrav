/*
 * test.cpp
 *
 *  Created on: Jul 14, 2020
 *      Author: dmarce1
 */

#include <hpx/hpx_init.hpp>
#include <immintrin.h>

//
//struct interpolating_function {
//	std::vector<double> F;
//	std::vector<double> dFdx;
//	int N;
//	double dx;
//	double xmax;
//	double xmin;
//	__m256d _xmin;
//	__m256d _dx;
//	__m256d _dxinv;
//	__m128i _ione;
//	__m256d _dzero;
//	__m256d _done;
//	std::function<double(double)> y;
//	template<class T>
//	std::function<T(T)> get_scalar_function() const {
//		const auto func = [=](double x) {
//			const T x0 = (x - xmin) / dx;
//			const int i1 = x0;
//			const int i2 = i1 + 1;
//			const T t2 = x0 - i1;
//			const T t1 = T(1) - t2;
//			const T y1 = F[i1];
//			const T y2 = F[i2];
//			const T k1 = dFdx[i1];
//			const T k2 = dFdx[i2];
//			const T dy = y2 - y1;
//			const T a = k1 * dx - dy;
//			const T b = -k2 * dx + dy;
//			return t1 * y1 + t2 * y2 + t1 * t2 * (t1 * a + t2 * b);
//		};
//		return func;
//	}
//	double test() {
//		constexpr int N = 1000000;
//		auto f = get_scalar_function<double>();
//		double max_err = 0.0;
//		for (int i = 0; i < N; i++) {
//			double x = xmin + (double) i / N * (xmax - xmin);
//			double err = abs(0.5 * (f(x) - y(x)) / (abs(f(x)) + abs(y(x))));
//			max_err = std::max(err, max_err);
//		}
//		return max_err;
//	}
//	interpolating_function(std::function<double(double)> f, std::function<double(double)> dfdx, double __xmin, double __xmax, double toler) {
//		y = f;
//		xmin = __xmin;
//		xmax = __xmax;
//		N = 2;
//
//		do {
//			N *= 2;
//			F.resize(N + 1);
//			dFdx.resize(N + 1);
//			dx = (xmax - xmin) / N;
//			for (int n = 0; n <= N; n++) {
//				F[n] = f(n * dx + xmin);
//				dFdx[n] = dfdx(n * dx + xmin);
//			}
//			printf("%e\n", test());
//			if (N > 1024 * 1024 * 1240) {
//				printf("unable to generate table\n");
//			}
//		} while (test() > toler);
//		_xmin = _mm256_set_pd(xmin, xmin, xmin, xmin);
//		_dx = _mm256_set_pd(dx, dx, dx, dx);
//		_dxinv = _mm256_set_pd(1.0 / dx, 1.0 / dx, 1.0 / dx, 1.0 / dx);
//		_ione = _mm_set_epi32(1, 1, 1, 1);
//		_dzero = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);
//		_done = _mm256_set_pd(1.0, 1.0, 1.0, 1.0);
//		printf("Generated interpolating table with %i values\n", N + 1);
//	}
//	std::function<__m256d(__m256d)> get_avx2_double_function() {
//		const auto func = [=](__m256d _x) {
//			const __m256d _x0 = _mm256_mul_pd(_mm256_sub_pd(_x, _xmin), _dxinv);					// 2
//			auto _i0 = _mm256_cvtpd_epi32(_x0);											// 1
//			auto _i1 = _mm_add_epi32(_i0, _ione);
//			const auto _t2 = _mm256_sub_pd(_x0, _mm256_cvtepi32_pd(_i0));						// 2
//			const auto _t1 = _mm256_sub_pd(_done, _t2);											// 2
//			__m256d _y1, _y2, _k1, _k2;
//			for (int i = 0; i < 4; i++) {
//				_y1[i] = F[(reinterpret_cast<int*>(&_i0))[i]];
//
//			}
//			for (int i = 0; i < 4; i++) {
//				_y2[i] = F[(reinterpret_cast<int*>(&_i1))[i]];
//
//			}
//			for (int i = 0; i < 4; i++) {
//				_k1[i] = dFdx[(reinterpret_cast<int*>(&_i0))[i]];
//
//			}
//			for (int i = 0; i < 4; i++) {
//				_k2[i] = dFdx[(reinterpret_cast<int*>(&_i1))[i]];
//
//			}
////			_y1 = _mm256_i32gather_pd(F.data(), _i0, 1);
////			_y2 = _mm256_i32gather_pd(F.data(), _i1, 1);
////			_k1 = _mm256_i32gather_pd(dFdx.data(), _i0, 1);
////			_k2 = _mm256_i32gather_pd(dFdx.data(), _i1, 1);
//			const auto _dy = _mm256_sub_pd(_y2, _y1);											// 1
//			const auto _a = _mm256_fmadd_pd(_k1, _dx, _mm256_sub_pd(_dzero, _dy));				// 3
//			const auto _b = _mm256_fmadd_pd(_mm256_sub_pd(_dzero, _k1), _dx, _dy);				// 3
//			auto _tmp = _mm256_mul_pd(_mm256_mul_pd(_t1, _t2), _mm256_add_pd(_mm256_mul_pd(_t1, _a), _mm256_mul_pd(_t2, _b))); //5
//			_tmp = _mm256_fmadd_pd(_t2, _y2, _tmp);												// 2
//			_tmp = _mm256_fmadd_pd(_t1, _y1, _tmp);												// 2
//////			return _mm256_cvtepi32_pd(_i1);
//			return _tmp;
//		};
//		return func;
//	}
//};

int hpx_main(int argc, char *argv[]) {
//	interpolating_function exp_table([](double x) {
//		return std::exp(x);
//	},[](double x) {
//		return std::exp(x);
//	},-64, +64, 1.0e-10);
//	interpolating_function sin_table([](double x) {
//		return std::sin(x);
//	},[](double x) {
//		return std::cos(x);
//	},-64, +64, 1.0e-10);
//	interpolating_function cos_table([](double x) {
//		return std::cos(x);
//	},[](double x) {
//		return -std::sin(x);
//	},-64, +64, 1.0e-10);
//
//	auto func = sin_table.get_avx2_double_function();
//
//	for (int i = 0; i < 100; i += 4) {
//		double x = i / 100.0;
//		auto y = func(_mm256_set_pd(x, x + 0.01, x + 0.02, x + 0.03));
//		printf("%e %e %e\n", x, y[0], sin(x));
//		printf("%e %e %e\n", x + 0.01, y[1], sin(x + 0.01));
//		printf("%e %e %e\n", x + 0.02, y[2], sin(x + 0.02));
//		printf("%e %e %e\n", x + 0.03, y[3], sin(x + 0.03));
//
//	}
	return hpx::finalize();
}
int main(int argc, char *argv[]) {
	std::vector<std::string> cfg = { "hpx.commandline.allow_unknown=1" };
	hpx::init(argc, argv, cfg);
}

