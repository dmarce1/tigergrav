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

#ifdef DOUBLE_SIMD
#define NCHUNK 2
#else
#define NCHUNK 1
#endif

#ifdef USE_AVX512
#define SIMDALIGN                  __attribute__((aligned(64)))
#define SIMD_VLEN 16
#define _simd_float                 __m512
#define _simd_int                   __m512i
#define _mmx_add_ps(a,b)            _mm512_add_ps((a),(b))
#define _mmx_sub_ps(a,b)            _mm512_sub_ps((a),(b))
#define _mmx_mul_ps(a,b)            _mm512_mul_ps((a),(b))
#define _mmx_div_ps(a,b)            _mm512_div_ps((a),(b))
#define _mmx_sqrt_ps(a)             _mm512_sqrt_ps(a)
#define _mmx_min_ps(a, b)           _mm512_min_ps((a),(b))
#define _mmx_max_ps(a, b)           _mm512_max_ps((a),(b))
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
#else
#ifdef USE_AVX2
#define SIMDALIGN                  __attribute__((aligned(32)))
#define SIMD_VLEN 8
#define _simd_float                 __m256
#define _simd_int                   __m256i
#define _mmx_add_ps(a,b)            _mm256_add_ps((a),(b))
#define _mmx_sub_ps(a,b)            _mm256_sub_ps((a),(b))
#define _mmx_mul_ps(a,b)            _mm256_mul_ps((a),(b))
#define _mmx_div_ps(a,b)            _mm256_div_ps((a),(b))
#define _mmx_sqrt_ps(a)             _mm256_sqrt_ps(a)
#define _mmx_min_ps(a, b)           _mm256_min_ps((a),(b))
#define _mmx_max_ps(a, b)           _mm256_max_ps((a),(b))
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
#else
#define NOSIMD
#error 'Error - Compiling without SIMD support!'
#endif

#endif
class simd_int;
class simd_float;

class simd_float {
private:
	_simd_float v[NCHUNK];
public:
	static constexpr std::size_t size() {
		return NCHUNK * SIMD_VLEN;
	}
	simd_float() = default;
	inline ~simd_float() = default;
	simd_float(const simd_float&) = default;
	inline simd_float(float d) {
		for (int i = 0; i < NCHUNK; i++) {
#ifdef USE_AVX512
        v[i] = _mm512_set_ps(d, d, d, d, d, d, d, d, d, d, d, d, d, d, d, d);
#else
			v[i] = _mm256_set_ps(d, d, d, d, d, d, d, d);
#endif
		}
	}
	union union_mm {
#ifdef USE_AVX512
		__m512 m16[1];
		__m256 m8[2];
		__m128 m4[4];
		float m1[16];
#else
		__m256 m8[1];
		__m128 m4[2];
		float m1[8];
#endif
	};
	inline float sum() const {
		union_mm s;
#ifdef USE_AVX512
#ifdef DOUBLE_SIMD
		s.m16[0] = _mm512_add_ps(v[0],v[1]);
#else
		s.m16[0] = v[0];
#endif
		s.m8[0] = _mm256_add_ps(s.m8[0],s.m8[1]);
#else
#ifdef DOUBLE_SIMD
		s.m8[0] = _mm256_add_ps(v[0],v[1]);
#else
		s.m8[0] = v[0];
#endif
#endif
		s.m4[0] = _mm_add_ps(s.m4[0], s.m4[1]);
		s.m1[4] = s.m1[2];
		s.m1[5] = s.m1[3];
		s.m4[0] = _mm_add_ps(s.m4[0], s.m4[1]);
		return s.m1[0] + s.m1[1];
	}
	inline simd_float(const simd_int &other);
	inline simd_int to_int() const;
	inline simd_float& operator=(const simd_float &other) = default;
	simd_float& operator=(simd_float &&other) = default;
	inline simd_float operator+(const simd_float &other) const {
		simd_float r;
		for (int i = 0; i < NCHUNK; i++) {
			r.v[i] = _mmx_add_ps(v[i], other.v[i]);
		}
		return r;
	}
	inline simd_float operator-(const simd_float &other) const {
		simd_float r;
		for (int i = 0; i < NCHUNK; i++) {
			r.v[i] = _mmx_sub_ps(v[i], other.v[i]);
		}
		return r;
	}
	inline simd_float operator*(const simd_float &other) const {
		simd_float r;
		for (int i = 0; i < NCHUNK; i++) {
			r.v[i] = _mmx_mul_ps(v[i], other.v[i]);
		}
		return r;
	}
	inline simd_float operator/(const simd_float &other) const {
		simd_float r;
		for (int i = 0; i < NCHUNK; i++) {
			r.v[i] = _mmx_div_ps(v[i], other.v[i]);
		}
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
		float *a = reinterpret_cast<float*>(&v[i / SIMD_VLEN]);
		return a[i % SIMD_VLEN];
	}
	inline float operator[](std::size_t i) const {
		const float *a = reinterpret_cast<const float*>(&v[i / SIMD_VLEN]);
		return a[i % SIMD_VLEN];
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

	friend simd_float exp(const simd_float &a);
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
		for (int i = 0; i < NCHUNK; i++) {
			auto mask = _mmx_cmp_ps(v[i], other.v[i], _CMP_LT_OQ);
#ifdef USE_AVX512
			rc.v[i] = _mm512_mask_mov_ps(zero.v[i],mask,one.v[i]);
#else
			rc.v[i] = _mmx_and_ps(mask, one.v[i]);
#endif
		}
		return rc;
	}
	simd_float operator>(simd_float other) const {
		simd_float rc;
		static const simd_float one(1);
		static const simd_float zero(0);
		for (int i = 0; i < NCHUNK; i++) {
			auto mask = _mmx_cmp_ps(v[i], other.v[i], _CMP_GT_OQ);
#ifdef USE_AVX512
			rc.v[i] = _mm512_mask_mov_ps(zero.v[i],mask,one.v[i]);
#else
			rc.v[i] = _mmx_and_ps(mask, one.v[i]);
#endif
		}
		return rc;
	}
}
SIMDALIGN;

inline simd_float gather(void *base, int s, int o) {
	simd_float v;
	const static __m256i i0 = _mm256_set_epi32(0, 1, 2, 3, 4, 5, 6, 7);
	__m256i j0 = _mm256_mul_epi32(i0, _mm256_set_epi32(s, s, s, s, s, s, s, s));
	j0 = _mm256_add_epi32(j0, _mm256_set_epi32(o, o, o, o, o, o, o, o));
	v.v[0] = _mm256_i32gather_ps((const float*)base, j0, 1);
#ifdef DOUBLE_SIMD
	const static __m256i i1 = _mm256_set_epi32(8, 9, 10, 11, 12, 13, 14, 15);
	__m256i j0 = _mm256_mul_epi32(i0, _mm256_set_epi32(s, s, s, s, s, s, s, s));
	j0 = _mm256_add_epi32(j0, _mm256_set_epi32(o, o, o, o, o, o, o, o));
	v.v[1] = _mm256_i32gather_ps(base, j0, 1);
#endif
	return v;
}

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
        __m512i n[NCHUNK];
#else
	__m256i n[NCHUNK];
#endif
	for (int i = 0; i < NCHUNK; i++) {
#ifdef USE_AVX512
		r0.v[i] = _mm512_roundscale_ps(r.v[i], _MM_FROUND_TO_NEAREST_INT);
		n[i] = _mm512_cvtps_epi32(r0.v[i]);
		r0.v[i] = _mm512_cvtepi32_ps(n[i]);
#else
		r0.v[i] = _mm256_round_ps(r.v[i], _MM_FROUND_TO_NEAREST_INT);							// 1
		n[i] = _mm256_cvtps_epi32(r0.v[i]);														// 1
		r0.v[i] = _mm256_cvtepi32_ps(n[i]);														// 1
#endif
	}
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
	for (int i = 0; i < NCHUNK; i++) {
#ifdef USE_AVX512
		auto imm0 = _mm512_add_epi32(n[i], _mm512_set_epi32(0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f)); // 1
		imm0 = _mm512_slli_epi32(imm0, 23);
		r0.v[i] = _mm512_castsi512_ps(imm0);
#else
		auto imm0 = _mm256_add_epi32(n[i], _mm256_set_epi32(0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f));
		imm0 = _mm256_slli_epi32(imm0, 23);
		r0.v[i] = _mm256_castsi256_ps(imm0);
#endif
	}
	auto res = y * r0;																			// 1
	return res;
}

inline simd_float round(const simd_float a) {
	simd_float v;
	for (int i = 0; i < NCHUNK; i++) {
#ifdef USE_AVX512
		v.v[i] = _mm512_roundscale_ps(a.v[i], 0);
#else
		v.v[i] = _mm256_round_ps(a.v[i], _MM_FROUND_TO_NEAREST_INT);
#endif
	}
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

inline simd_float exp(const simd_float &a) { 	// 22
	static const simd_float c0 = 1.0 / std::log(2);
	return two_pow(a * c0);
//	simd_float v;
//	for (int i = 0; i < NCHUNK; i++) {
//#ifdef USE_AVX512
//		v.v[i] = exp512_ps(a.v[i]);
//#else
//		v.v[i] = exp256_ps(a.v[i]);
//#endif
//	}
//	return v;
}

inline simd_float erfexp(const simd_float &x, simd_float *e) {				// 48
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
	return simd_float(1) - (a1 * t1 + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5) * *e; // 11
	return v;
}

inline simd_float fmadd(const simd_float &a, const simd_float &b, const simd_float &c) {

	simd_float v;
	for (int i = 0; i < NCHUNK; i++) {
		v.v[i] = _mmx_fmadd_ps(a.v[i], b.v[i], c.v[i]);
	}
	return v;
}

inline simd_float copysign(const simd_float &y, const simd_float &x) {
// From https://stackoverflow.com/questions/57870896/writing-a-portable-sse-avx-version-of-stdcopysign

	simd_float v;
	for (int i = 0; i < NCHUNK; i++) {
		constexpr float signbit = -0.f;
		static simd_float const avx_signbit = simd_float(signbit);
		v.v[i] = _mmx_or_ps(_mmx_and_ps(avx_signbit.v[i], x.v[i]), _mmx_andnot_ps(avx_signbit.v[i], y.v[i])); // (avx_signbit & from) | (~avx_signbit & to)
	}
	return v;
}

inline simd_float sqrt(const simd_float &vec) {
	simd_float r;
	for (int i = 0; i < NCHUNK; i++) {
		r.v[i] = _mmx_sqrt_ps(vec.v[i]);
	}
	return r;

}

inline simd_float rsqrt(const simd_float &vec) {
	simd_float r;
	for (int i = 0; i < NCHUNK; i++) {
		r.v[i] = _mmx_rsqrt_ps(vec.v[i]);
	}
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

inline simd_float max(const simd_float &a, const simd_float &b) {
	simd_float r;
	for (int i = 0; i < NCHUNK; i++) {
		r.v[i] = _mmx_max_ps(a.v[i], b.v[i]);
	}
	return r;
}

inline simd_float min(const simd_float &a, const simd_float &b) {
	simd_float r;
	for (int i = 0; i < NCHUNK; i++) {
		r.v[i] = _mmx_min_ps(a.v[i], b.v[i]);
	}
	return r;
}

inline simd_float abs(const simd_float &a) {
	return max(a, -a);
}

#endif /* TIGERGRAV_SIMD_HPP_ */
