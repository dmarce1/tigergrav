/*
 * simd.hpp
 *
 *  Created on: Jul 3, 2020
 *      Author: dmarce1
 */

#ifndef TIGERGRAV_SIMD_HPP_
#define TIGERGRAV_SIMD_HPP_

#include <tigergrav/defs.hpp>

#include <immintrin.h>

#include <cmath>

#ifdef USE_AVX512
#define SIMDALIGN                  __attribute__((aligned(64)))
#define SIMD_VLEN 16
#define _simd_float                 __m512
#define _simd_double 				__m512d
#define _simd_int                   __m512i
#define _mmx_add_ps(a,b)            _mm512_add_ps((a),(b))
#define _mmx_sub_ps(a,b)            _mm512_sub_ps((a),(b))
#define _mmx_mul_ps(a,b)            _mm512_mul_ps((a),(b))
#define _mmx_div_ps(a,b)            _mm512_div_ps((a),(b))
#define _mmx_add_pd(a,b)            _mm512_add_pd((a),(b))
#define _mmx_sub_pd(a,b)            _mm512_sub_pd((a),(b))
#define _mmx_mul_pd(a,b)            _mm512_mul_pd((a),(b))
#define _mmx_div_pd(a,b)            _mm512_div_pd((a),(b))
#define _mmx_sqrt_ps(a)             _mm512_sqrt_ps(a)
#define _mmx_min_ps(a, b)           _mm512_min_ps((a),(b))
#define _mmx_max_ps(a, b)           _mm512_max_ps((a),(b))
#define _mmx_max_pd(a, b)           _mm512_max_pd((a),(b))
#define _mmx_or_ps(a, b)            _mm512_or_ps((a),(b))
#define _mmx_and_ps(a, b)           _mm512_and_ps((a),(b))
#define _mmx_andnot_ps(a, b)        _mm512_andnot_ps((a),(b))
#define _mmx_rsqrt_ps(a)            _mm512_rsqrt14_ps(a)
#define _mmx_add_epi32(a,b)         _mm512_add_epi32((a),(b))
#define _mmx_sub_epi32(a,b)         _mm512_sub_epi32((a),(b))
#define _mmx_mul_epi32(a,b)         _mm512_mullo_epi32((a),(b))
#define _mmx_cvtps_epi32(a)         _mm512_cvtps_epi32((a))
#define _mmx_fmadd_ps(a,b,c)        _mm512_fmadd_ps ((a),(b),(c))
#define _mmx_cmp_ps(a,b,c)       _mm512_cmp_ps_mask(a,b,c)
#endif

#ifdef USE_AVX2
#define SIMDALIGN                  __attribute__((aligned(32)))
#define SIMD_VLEN 8
#define _simd_float                 __m256
#define _simd_int                   __m256i
#define _simd_double                __m256d
#define _mmx_add_ps(a,b)            _mm256_add_ps((a),(b))
#define _mmx_sub_ps(a,b)            _mm256_sub_ps((a),(b))
#define _mmx_mul_ps(a,b)            _mm256_mul_ps((a),(b))
#define _mmx_div_ps(a,b)            _mm256_div_ps((a),(b))
#define _mmx_add_pd(a,b)            _mm256_add_pd((a),(b))
#define _mmx_sub_pd(a,b)            _mm256_sub_pd((a),(b))
#define _mmx_mul_pd(a,b)            _mm256_mul_pd((a),(b))
#define _mmx_div_pd(a,b)            _mm256_div_pd((a),(b))
#define _mmx_sqrt_ps(a)             _mm256_sqrt_ps(a)
#define _mmx_min_ps(a, b)           _mm256_min_ps((a),(b))
#define _mmx_max_ps(a, b)           _mm256_max_ps((a),(b))
#define _mmx_max_pd(a, b)           _mm256_max_pd((a),(b))
#define _mmx_or_ps(a, b)            _mm256_or_ps((a),(b))
#define _mmx_and_ps(a, b)           _mm256_and_ps((a),(b))
#define _mmx_andnot_ps(a, b)        _mm256_andnot_ps((a),(b))
#define _mmx_rsqrt_ps(a)            _mm256_rsqrt_ps(a)
#define _mmx_add_epi32(a,b)         _mm256_add_epi32((a),(b))
#define _mmx_sub_epi32(a,b)         _mm256_sub_epi32((a),(b))
#define _mmx_mul_epi32(a,b)         _mm256_mullo_epi32((a),(b))
#define _mmx_cvtps_epi32(a)         _mm256_cvtps_epi32((a))
#define _mmx_fmadd_ps(a,b,c)        _mm256_fmadd_ps ((a),(b),(c))
#define _mmx_cmp_ps(a,b,c)        	_mm256_cmp_ps(a,b,c)
#endif

#ifdef USE_AVX
#define SIMDALIGN                  __attribute__((aligned(16)))
#define SIMD_VLEN 4
#define _simd_float                 __m128
#define _simd_int                   __m128i
#define _simd_double                 __m128d
#define _mmx_add_ps(a,b)            _mm_add_ps((a),(b))
#define _mmx_sub_ps(a,b)            _mm_sub_ps((a),(b))
#define _mmx_mul_ps(a,b)            _mm_mul_ps((a),(b))
#define _mmx_div_ps(a,b)            _mm_div_ps((a),(b))
#define _mmx_add_pd(a,b)            _mm_add_pd((a),(b))
#define _mmx_sub_pd(a,b)            _mm_sub_pd((a),(b))
#define _mmx_mul_pd(a,b)            _mm_mul_pd((a),(b))
#define _mmx_div_pd(a,b)            _mm_div_ps((a),(b))
#define _mmx_sqrt_ps(a)             _mm_sqrt_ps(a)
#define _mmx_min_ps(a, b)           _mm_min_ps((a),(b))
#define _mmx_max_ps(a, b)           _mm_max_ps((a),(b))
#define _mmx_max_pd(a, b)           _mm_max_pd((a),(b))
#define _mmx_or_ps(a, b)            _mm_or_ps((a),(b))
#define _mmx_and_ps(a, b)           _mm_and_ps((a),(b))
#define _mmx_andnot_ps(a, b)        _mm_andnot_ps((a),(b))
#define _mmx_rsqrt_ps(a)            _mm_rsqrt_ps(a)
#define _mmx_add_epi32(a,b)         _mm_add_epi32((a),(b))
#define _mmx_sub_epi32(a,b)         _mm_sub_epi32((a),(b))
#define _mmx_mul_epi32(a,b)         _mm_mullo_epi32((a),(b))
#define _mmx_cvtps_epi32(a)         _mm_cvtps_epi32((a))
#define _mmx_cmp_ps(a,b,c)        	_mm_cmp_ps(a,b,c)
#endif

class simd_float;
class simd_int;
class simd_double;

class simd_float {
private:
	_simd_float v[2];
public:
	static constexpr std::size_t size() {
		return 2 * SIMD_VLEN;
	}
	simd_float() = default;
	inline ~simd_float() = default;
	simd_float(const simd_float&) = default;
	inline simd_float(float d) {
#ifdef USE_AVX512
        v[0] = _mm512_set_ps(d, d, d, d, d, d, d, d, d, d, d, d, d, d, d, d);
        v[1] = _mm512_set_ps(d, d, d, d, d, d, d, d, d, d, d, d, d, d, d, d);
#endif
#ifdef USE_AVX2
		v[0] = _mm256_set_ps(d, d, d, d, d, d, d, d);
		v[1] = _mm256_set_ps(d, d, d, d, d, d, d, d);
#endif
#ifdef USE_AVX
		v[0] = _mm_set_ps(d, d, d, d);
		v[1] = _mm_set_ps(d, d, d, d);
#endif
	}
	inline float sum() const {
		float sum = 0.0;
		for (int i = 0; i < 2 * SIMD_VLEN; i++) {
			sum += (*this)[i];
		}
		return sum;
	}

	inline simd_float(simd_int i);

	inline simd_float& operator=(const simd_float &other) = default;
	simd_float& operator=(simd_float &&other) = default;
	inline simd_float operator+(const simd_float &other) const {
		simd_float r;
		r.v[0] = _mmx_add_ps(v[0], other.v[0]);
		r.v[1] = _mmx_add_ps(v[1], other.v[1]);
		return r;
	}
	inline simd_float operator-(const simd_float &other) const {
		simd_float r;
		r.v[0] = _mmx_sub_ps(v[0], other.v[0]);
		r.v[1] = _mmx_sub_ps(v[1], other.v[1]);
		return r;
	}
	inline simd_float operator*(const simd_float &other) const {
		simd_float r;
		r.v[0] = _mmx_mul_ps(v[0], other.v[0]);
		r.v[1] = _mmx_mul_ps(v[1], other.v[1]);
		return r;
	}
	inline simd_float operator/(const simd_float &other) const {
		simd_float r;
		r.v[0] = _mmx_div_ps(v[0], other.v[0]);
		r.v[1] = _mmx_div_ps(v[1], other.v[1]);
		return r;
	}
	inline simd_float operator+() const {
		return *this;
	}
	inline simd_float operator-() const {
		return simd_float(0.0) - *this;
	}
	inline simd_float& operator+=(const simd_float &other) {
		*this = *this + other;
		return *this;
	}
	inline simd_float& operator-=(const simd_float &other) {
		*this = *this - other;
		return *this;
	}
	inline simd_float& operator*=(const simd_float &other) {
		*this = *this * other;
		return *this;
	}
	inline simd_float& operator/=(const simd_float &other) {
		*this = *this / other;
		return *this;
	}

	inline simd_float operator*(float d) const {
		const simd_float other = d;
		return other * *this;
	}
	inline simd_float operator/(float d) const {
		const simd_float other = 1.0 / d;
		return *this * other;
	}

	inline simd_float operator*=(float d) {
		*this = *this * d;
		return *this;
	}
	inline simd_float operator/=(float d) {
		*this = *this * (1.0 / d);
		return *this;
	}
	inline float& operator[](std::size_t i) {
		return reinterpret_cast<float*>(&v)[i];
	}
	inline float operator[](std::size_t i) const {
		return reinterpret_cast<const float*>(&v)[i];
	}

	friend simd_float copysign(const simd_float&, const simd_float&);
	friend simd_float sqrt(const simd_float&);
	friend simd_float rsqrt(const simd_float&);
	friend simd_float operator*(float, const simd_float &other);
	friend simd_float operator/(float, const simd_float &other);
	friend simd_float max(const simd_float &a, const simd_float &b);
	friend simd_float min(const simd_float &a, const simd_float &b);
	friend simd_float fmadd(const simd_float &a, const simd_float &b, const simd_float &c);
	friend simd_float round(const simd_float);

	friend simd_float sin(const simd_float &a);
	friend simd_float cos(const simd_float &a);
	friend simd_float abs(const simd_float &a);
	friend simd_float erfexp(const simd_float &a, simd_float*);
	friend simd_float gather(void*, int, int);

	friend simd_float two_pow(const simd_float &r);
	friend void sincos(const simd_float &x, simd_float *s, simd_float *c);
	simd_float operator<(simd_float other) const {
		simd_float rc;
		static const simd_float one(1);
		static const simd_float zero(0);
		auto mask0 = _mmx_cmp_ps(v[0], other.v[0], _CMP_LT_OQ);
		auto mask1 = _mmx_cmp_ps(v[1], other.v[1], _CMP_LT_OQ);
#ifdef USE_AVX512
		rc.v[0] = _mm512_mask_mov_ps(zero.v[0],mask0,one.v[0]);
		rc.v[1] = _mm512_mask_mov_ps(zero.v[1],mask1,one.v[1]);
#else
		rc.v[0] = _mmx_and_ps(mask0, one.v[0]);
		rc.v[1] = _mmx_and_ps(mask1, one.v[1]);
#endif
		return rc;
	}
	simd_float operator>(simd_float other) const {
		simd_float rc;
		static const simd_float one(1);
		static const simd_float zero(0);
		auto mask0 = _mmx_cmp_ps(v[0], other.v[0], _CMP_GT_OQ);
		auto mask1 = _mmx_cmp_ps(v[1], other.v[1], _CMP_GT_OQ);
#ifdef USE_AVX512
		rc.v[0] = _mm512_mask_mov_ps(zero.v[0],mask0,one.v[0]);
		rc.v[1] = _mm512_mask_mov_ps(zero.v[1],mask1,one.v[1]);
#else
		rc.v[0] = _mmx_and_ps(mask0, one.v[0]);
		rc.v[1] = _mmx_and_ps(mask1, one.v[1]);
#endif
		return rc;
	}
	friend class simd_double;
}SIMDALIGN;
;

class simd_int {
private:
	_simd_int v[2];
public:
	static constexpr std::size_t size() {
		return 2 * SIMD_VLEN;
	}
	simd_int() = default;
	inline ~simd_int() = default;
	simd_int(const simd_int&) = default;
	inline simd_int(int d) {
#ifdef USE_AVX512
        v[0] = _mm512_set_epi32(d, d, d, d, d, d, d, d, d, d, d, d, d, d, d, d);
        v[1] = _mm512_set_epi32(d, d, d, d, d, d, d, d, d, d, d, d, d, d, d, d);
#endif
#ifdef USE_AVX2
		v[0] = _mm256_set_epi32(d, d, d, d, d, d, d, d);
		v[1] = _mm256_set_epi32(d, d, d, d, d, d, d, d);
#endif
#ifdef USE_AVX
		v[0] = _mm_set_epi32(d, d, d, d);
		v[1] = _mm_set_epi32(d, d, d, d);
#endif
	}
	inline simd_int& operator=(const simd_int &other) = default;
	simd_int& operator=(simd_int &&other) = default;
	inline simd_int operator+(const simd_int &other) const {
		simd_int r;
		r.v[0] = _mmx_add_epi32(v[0], other.v[0]);
		r.v[1] = _mmx_add_epi32(v[1], other.v[1]);
		return r;
	}
	inline simd_int operator-(const simd_int &other) const {
		simd_int r;
		r.v[0] = _mmx_sub_epi32(v[0], other.v[0]);
		r.v[1] = _mmx_sub_epi32(v[1], other.v[1]);
		return r;
	}
	inline simd_int operator*(const simd_int &other) const {
		simd_int r;
		r.v[0] = _mmx_mul_epi32(v[0], other.v[0]);
		r.v[1] = _mmx_mul_epi32(v[1], other.v[1]);
		return r;
	}
	inline simd_int operator+() const {
		return *this;
	}
	inline simd_int operator-() const {
		return simd_int(0.0) - *this;
	}
	inline simd_int operator/(int d) const {
		const simd_int other = 1.0 / d;
		return *this * other;
	}

	inline simd_int operator*=(int d) {
		*this = *this * d;
		return *this;
	}
	inline int& operator[](std::size_t i) {
		return reinterpret_cast<int*>(&v)[i];
	}
	inline int operator[](std::size_t i) const {
		return reinterpret_cast<const int*>(&v)[i];
	}
	friend class simd_double;
	friend class simd_float;

}SIMDALIGN;

inline simd_float::simd_float(simd_int i) {
#ifdef USE_AVX512
	v[0] = _mm512_cvtepi32_ps(i.v[0]);
	v[1] = _mm512_cvtepi32_ps(i.v[1]);
#endif
#ifdef USE_AVX2
	v[0] = _mm256_cvtepi32_ps(i.v[0]);
	v[1] = _mm256_cvtepi32_ps(i.v[1]);
#endif

}
class simd_double {
private:
	_simd_double a[4];
public:
	static constexpr std::size_t size() {
		return 2 * SIMD_VLEN;
	}
	simd_double() = default;
	inline ~simd_double() = default;
	simd_double(const simd_double&) = default;
	inline simd_double(double d) {
#ifdef USE_AVX512
        a[0] = _mm512_set_pd(d, d, d, d, d, d, d, d);
        a[1] = _mm512_set_pd(d, d, d, d, d, d, d, d);
        a[2] = _mm512_set_pd(d, d, d, d, d, d, d, d);
        a[3] = _mm512_set_pd(d, d, d, d, d, d, d, d);
#endif
#ifdef USE_AVX2
		a[0] = _mm256_set_pd(d, d, d, d);
		a[1] = _mm256_set_pd(d, d, d, d);
		a[2] = _mm256_set_pd(d, d, d, d);
		a[3] = _mm256_set_pd(d, d, d, d);
#endif
	}

	union int_union {
#ifdef USE_AVX512
		__m256i m1[4];
		__m512i m2[2];
#endif
#ifdef USE_AVX2
		__m128i m1[4];
		__m256i m2[2];
#endif
#ifdef USE_AVX
#error 'AVX not complete'
#endif
	};

	union float_union {
#ifdef USE_AVX512
		__m256 m1[4];
		__m512 m2[2];
#endif
#ifdef USE_AVX2
		__m128 m1[4];
		__m256 m2[2];
#endif
#ifdef USE_AVX
#error 'AVX not complete'
#endif
	};

	inline simd_double(simd_int i) {
		int_union u;
		u.m2[0] = i.v[0];
		u.m2[1] = i.v[1];
#ifdef USE_AVX512
		a[0] = _mm512_cvtepi32_pd(u.m1[0]);
		a[1] = _mm512_cvtepi32_pd(u.m1[1]);
		a[2] = _mm512_cvtepi32_pd(u.m1[2]);
		a[3] = _mm512_cvtepi32_pd(u.m1[3]);
#endif
#ifdef USE_AVX2
		a[0] = _mm256_cvtepi32_pd(u.m1[0]);
		a[1] = _mm256_cvtepi32_pd(u.m1[1]);
		a[2] = _mm256_cvtepi32_pd(u.m1[2]);
		a[3] = _mm256_cvtepi32_pd(u.m1[3]);
#endif

	}

	inline simd_double(simd_float i) {
		float_union u;
		u.m2[0] = i.v[0];
		u.m2[1] = i.v[1];
#ifdef USE_AVX512
		a[0] = _mm512_cvtps_pd(u.m1[0]);
		a[1] = _mm512_cvtps_pd(u.m1[1]);
		a[2] = _mm512_cvtps_pd(u.m1[2]);
		a[3] = _mm512_cvtps_pd(u.m1[3]);
#endif
#ifdef USE_AVX2
		a[0] = _mm256_cvtps_pd(u.m1[0]);
		a[1] = _mm256_cvtps_pd(u.m1[1]);
		a[2] = _mm256_cvtps_pd(u.m1[2]);
		a[3] = _mm256_cvtps_pd(u.m1[3]);
#endif

	}

	inline operator simd_float() const {
		float_union u;
#ifdef USE_AVX512
		u.m1[0] = _mm512_cvtpd_ps(a[0]);
		u.m1[1] = _mm512_cvtpd_ps(a[1]);
		u.m1[2] = _mm512_cvtpd_ps(a[2]);
		u.m1[3] = _mm512_cvtpd_ps(a[3]);
#endif
#ifdef USE_AVX2
		u.m1[0] = _mm256_cvtpd_ps(a[0]);
		u.m1[1] = _mm256_cvtpd_ps(a[1]);
		u.m1[2] = _mm256_cvtpd_ps(a[2]);
		u.m1[3] = _mm256_cvtpd_ps(a[3]);
#endif
		simd_float f;
		f.v[0] = u.m2[0];
		f.v[1] = u.m2[1];
		return f;
	}

	inline simd_double& operator=(const simd_double &other) = default;
	simd_double& operator=(simd_double &&other) = default;
	inline simd_double operator+(const simd_double &other) const {
		simd_double r;
		r.a[0] = _mmx_add_pd(a[0], other.a[0]);
		r.a[1] = _mmx_add_pd(a[1], other.a[1]);
		r.a[2] = _mmx_add_pd(a[2], other.a[2]);
		r.a[3] = _mmx_add_pd(a[3], other.a[3]);
		return r;
	}
	inline simd_double operator-(const simd_double &other) const {
		simd_double r;
		r.a[0] = _mmx_sub_pd(a[0], other.a[0]);
		r.a[1] = _mmx_sub_pd(a[1], other.a[1]);
		r.a[2] = _mmx_sub_pd(a[2], other.a[2]);
		r.a[3] = _mmx_sub_pd(a[3], other.a[3]);
		return r;
	}
	inline simd_double operator*(const simd_double &other) const {
		simd_double r;
		r.a[0] = _mmx_mul_pd(a[0], other.a[0]);
		r.a[1] = _mmx_mul_pd(a[1], other.a[1]);
		r.a[2] = _mmx_mul_pd(a[2], other.a[2]);
		r.a[3] = _mmx_mul_pd(a[3], other.a[3]);
		return r;
	}
	inline simd_double operator/(const simd_double &other) const {
		simd_double r;
		r.a[0] = _mmx_div_pd(a[0], other.a[0]);
		r.a[1] = _mmx_div_pd(a[1], other.a[1]);
		r.a[2] = _mmx_div_pd(a[2], other.a[2]);
		r.a[3] = _mmx_div_pd(a[3], other.a[3]);
		return r;
	}
	inline simd_double operator+() const {
		return *this;
	}
	inline simd_double operator-() const {
		return simd_double(0.0) - *this;
	}
	inline simd_double operator/(double d) const {
		const simd_double other = 1.0 / d;
		return *this * other;
	}

	inline simd_double operator*=(double d) {
		*this = *this * d;
		return *this;
	}
	inline simd_double& operator+=(const simd_double &other) {
		*this = *this + other;
		return *this;
	}
	inline simd_double& operator-=(const simd_double &other) {
		*this = *this - other;
		return *this;
	}
	inline simd_double& operator*=(const simd_double &other) {
		*this = *this * other;
		return *this;
	}
	inline simd_double& operator/=(const simd_double &other) {
		*this = *this / other;
		return *this;
	}

	double sum() const {
		double s = 0.0;
		for (int i = 0; i < 4; i++) {
			s += a[0][i];
			s += a[1][i];
			s += a[2][i];
			s += a[3][i];
		}
		return s;
	}

	inline double& operator[](std::size_t i) {
		return (reinterpret_cast<double*>(&a))[i];
	}
	inline int operator[](std::size_t i) const {
		return (reinterpret_cast<const double*>(&a))[i];
	}

	friend simd_double max(const simd_double &a, const simd_double &b);
	friend simd_double abs(const simd_double &a);

}SIMDALIGN;

inline simd_float two_pow(const simd_float &r) {											// 21
	static const simd_float zero = simd_float(0.0);
	static const simd_float one = simd_float(1.0);
	static const simd_float c1 = simd_float(std::log(2));
	static const simd_float c2 = simd_float((0.5) * std::pow(std::log(2), 2));
	static const simd_float c3 = simd_float((1.0 / 6.0) * std::pow(std::log(2), 3));
	static const simd_float c4 = simd_float((1.0 / 24.0) * std::pow(std::log(2), 4));
	static const simd_float c5 = simd_float((1.0 / 120.0) * std::pow(std::log(2), 5));
	static const simd_float c6 = simd_float((1.0 / 720.0) * std::pow(std::log(2), 6));
	static const simd_float c7 = simd_float((1.0 / 5040.0) * std::pow(std::log(2), 7));
	static const simd_float c8 = simd_float((1.0 / 40320.0) * std::pow(std::log(2), 8));
	simd_float r0;
#ifdef USE_AVX512
    __m512i n[2];
#endif
#ifdef USE_AVX2
	__m256i n[2];
#endif
#ifdef USE_AVX512
	r0.v[0] = _mm512_roundscale_ps(r.v[0], _MM_FROUND_TO_NEAREST_INT);
	r0.v[1] = _mm512_roundscale_ps(r.v[1], _MM_FROUND_TO_NEAREST_INT);
	n[0] = _mm512_cvtps_epi32(r0.v[0]);
	n[1] = _mm512_cvtps_epi32(r0.v[1]);
	r0.v[0] = _mm512_cvtepi32_ps(n[0]);
	r0.v[1] = _mm512_cvtepi32_ps(n[1]);
#endif
#ifdef USE_AVX2
	r0.v[0] = _mm256_round_ps(r.v[0], _MM_FROUND_TO_NEAREST_INT);							// 1
	r0.v[1] = _mm256_round_ps(r.v[1], _MM_FROUND_TO_NEAREST_INT);							// 1
	n[0] = _mm256_cvtps_epi32(r0.v[0]);														// 1
	n[1] = _mm256_cvtps_epi32(r0.v[1]);														// 1
	r0.v[0] = _mm256_cvtepi32_ps(n[0]);														// 1
	r0.v[1] = _mm256_cvtepi32_ps(n[1]);														// 1
#endif
	auto x = r - r0;
	auto y = c8;
	y = fmadd(y, x, c7);																		// 2
	y = fmadd(y, x, c6);																		// 2
	y = fmadd(y, x, c5);																		// 2
	y = fmadd(y, x, c4);																		// 2
	y = fmadd(y, x, c3);																		// 2
	y = fmadd(y, x, c2);																		// 2
	y = fmadd(y, x, c1);																		// 2
	y = fmadd(y, x, one);																		// 2
#ifdef USE_AVX512
	static const auto sevenf =  _mm512_set_epi32(0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f);
	auto imm00 = _mm512_add_epi32(n[0], sevenf); // 1
	auto imm01 = _mm512_add_epi32(n[1], sevenf); // 1
	imm00 = _mm512_slli_epi32(imm00, 23);
	imm01 = _mm512_slli_epi32(imm01, 23);
	r0.v[0] = _mm512_castsi512_ps(imm00);
	r0.v[1] = _mm512_castsi512_ps(imm01);
#endif
#ifdef USE_AVX2
	static const auto sevenf = _mm256_set_epi32(0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f);
	auto imm00 = _mm256_add_epi32(n[0], sevenf);
	auto imm01 = _mm256_add_epi32(n[1], sevenf);
	imm00 = _mm256_slli_epi32(imm00, 23);
	imm01 = _mm256_slli_epi32(imm01, 23);
	r0.v[0] = _mm256_castsi256_ps(imm00);
	r0.v[1] = _mm256_castsi256_ps(imm01);
#endif
	auto res = y * r0;																			// 1
	return res;
}

inline simd_float round(const simd_float a) {
	simd_float v;
#ifdef USE_AVX512
	v.v[0] = _mm512_roundscale_ps(a.v[0], 0);
	v.v[1] = _mm512_roundscale_ps(a.v[1], 0);
#endif
#ifdef USE_AVX2
	v.v[0] = _mm256_round_ps(a.v[0], _MM_FROUND_TO_NEAREST_INT);
	v.v[1] = _mm256_round_ps(a.v[1], _MM_FROUND_TO_NEAREST_INT);
#endif
	return v;
}

inline simd_float sin(const simd_float &x0) {						// 17
	auto x = x0;
	// From : http://mooooo.ooo/chebyshev-sine-approximation/
	static const simd_float pi_major(3.1415927);
	static const simd_float pi_minor(-0.00000008742278);
	x = x - round(x * (1.0 / (2.0 * M_PI))) * (2.0 * M_PI);			// 4
	const simd_float x2 = x * x;									// 1
	simd_float p = simd_float(0.00000000013291342);
	p = fmadd(p, x2, simd_float(-0.000000023317787));				// 2
	p = fmadd(p, x2, simd_float(0.0000025222919));					// 2
	p = fmadd(p, x2, simd_float(-0.00017350505));					// 2
	p = fmadd(p, x2, simd_float(0.0066208798));						// 2
	p = fmadd(p, x2, simd_float(-0.10132118));						// 2
	const auto x1 = (x - pi_major - pi_minor);						// 2
	const auto x3 = (x + pi_major + pi_minor);						// 2
	auto res = x1 * x3 * p * x;										// 3
	return res;
}

inline simd_float cos(const simd_float &x) {
	return sin(x + simd_float(M_PI / 2.0));
}

inline void sincos(const simd_float &x, simd_float *s, simd_float *c) {
//#ifdef __AVX512F__
//	s->v = _mm512_sincos_ps(&(c->v),x.v);
//#else
	*s = sin(x);
	*c = cos(x);
//#endif
}

inline simd_float exp(simd_float a) { 	// 24
	static const simd_float c0 = 1.0 / std::log(2);
	static const auto hi = simd_float(88);
	static const auto lo = simd_float(-88);
	a = min(a, hi);
	a = max(a, lo);
	return two_pow(a * c0);
}

inline simd_float erfcexp(const simd_float &x, simd_float *e) {				// 50
	simd_float v;
	const simd_float p(0.3275911);
	const simd_float a1(0.254829592);
	const simd_float a2(-0.284496736);
	const simd_float a3(1.421413741);
	const simd_float a4(-1.453152027);
	const simd_float a5(1.061405429);
	const simd_float t1 = simd_float(1) / (simd_float(1) + p * x);			// 3
	const simd_float t2 = t1 * t1;											// 1
	const simd_float t3 = t2 * t1;											// 2
	const simd_float t4 = t2 * t2;											// 3
	const simd_float t5 = t2 * t3;											// 4
	*e = exp(-x * x);														// 24
	return (a1 * t1 + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5) * *e; // 11
	return v;
}

inline simd_float fmadd(const simd_float &a, const simd_float &b, const simd_float &c) {

	simd_float v;
	v.v[0] = _mmx_fmadd_ps(a.v[0], b.v[0], c.v[0]);
	v.v[1] = _mmx_fmadd_ps(a.v[1], b.v[1], c.v[1]);
	return v;
}

inline simd_float copysign(const simd_float &y, const simd_float &x) {
// From https://stackoverflow.com/questions/57870896/writing-a-portable-sse-avx-version-of-stdcopysign

	simd_float v;
	constexpr float signbit = -0.f;
	static simd_float const avx_signbit = simd_float(signbit);
	const auto tmp0 = _mmx_andnot_ps(avx_signbit.v[0], y.v[0]);
	const auto tmp1 = _mmx_andnot_ps(avx_signbit.v[1], y.v[1]);
	const auto tmp2 = _mmx_and_ps(avx_signbit.v[0], x.v[0]);
	const auto tmp3 = _mmx_and_ps(avx_signbit.v[1], x.v[1]);
	v.v[0] = _mmx_or_ps(tmp2, tmp0); // (avx_signbit & from) | (~avx_signbit & to)
	v.v[1] = _mmx_or_ps(tmp3, tmp1); // (avx_signbit & from) | (~avx_signbit & to)
	return v;
}

inline simd_float sqrt(const simd_float &vec) {
	simd_float r;
	r.v[0] = _mmx_sqrt_ps(vec.v[0]);
	r.v[1] = _mmx_sqrt_ps(vec.v[1]);
	return r;

}

inline simd_float operator*(float d, const simd_float &other) {
	const simd_float a = d;
	return a * other;
}

inline simd_float operator/(float d, const simd_float &other) {
	const simd_float a = d;
	return a / other;
}
inline simd_float min(const simd_float &a, const simd_float &b) {
	simd_float r;
	r.v[0] = _mmx_min_ps(a.v[0], b.v[0]);
	r.v[1] = _mmx_min_ps(a.v[1], b.v[1]);
	return r;
}

inline simd_float max(const simd_float &a, const simd_float &b) {
	simd_float r;
	r.v[0] = _mmx_max_ps(a.v[0], b.v[0]);
	r.v[1] = _mmx_max_ps(a.v[1], b.v[1]);
	return r;
}

inline simd_float abs(const simd_float &a) {
	return max(a, -a);
}

inline simd_double max(const simd_double &c, const simd_double &b) {
	simd_double r;
	r.a[0] = _mmx_max_pd(c.a[0], b.a[0]);
	r.a[1] = _mmx_max_pd(c.a[1], b.a[1]);
	r.a[2] = _mmx_max_pd(c.a[2], b.a[2]);
	r.a[3] = _mmx_max_pd(c.a[3], b.a[3]);
	return r;
}

inline simd_double abs(const simd_double &a) {
	return max(a, -a);
}

#endif /* TIGERGRAV_SIMD_HPP_ */
