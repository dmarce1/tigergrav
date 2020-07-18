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

#include <tigergrav/avx_mathfun.h>

#include <cmath>
//#define CHECK_ALIGN

#ifdef USE_AVX512
#define SIMDALIGN                  __attribute__((aligned(64)))
#define SIMD_FLOAT_LEN 16
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
#define SIMD_FLOAT_LEN 8
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
#warning 'Warning - Compiling without SIMD support!'
#endif

#endif
	class simd_int;
	class simd_float;

#ifdef CHECK_ALIGN
#ifdef __AVX512__
	inline void assert_align(const _simd_float& v) {
		if (reinterpret_cast<std::size_t>(&(v)) % 64 != 0) {
			printf("Alignment failure %s %i\n", __FILE__, __LINE__);
			abort();
		}
	}
#else
	inline void assert_align(const _simd_float& v) {
		if (reinterpret_cast<std::size_t>(&(v)) % 32 != 0) {
			printf("Alignment failure %s %i\n", __FILE__, __LINE__);
			abort();
		}
	}
#endif
#else
#define assert_align(a)
#endif

	class simd_float {
	private:
#ifdef NOSIMD
	float v;
#else
		_simd_float v;
#endif
	public:
		static constexpr std::size_t size() {
#ifdef NOSIMD
		return 1;
#else
			return SIMD_FLOAT_LEN;
#endif
		}
		simd_float() = default;
		inline ~simd_float() = default;
		simd_float(const simd_float&) = default;
		inline simd_float(float d) {
			assert_align(v);
#ifdef NOSIMD
		v = d;
#else
#ifdef USE_AVX512
        v = _mm512_set_ps(d, d, d, d, d, d, d, d, d, d, d, d, d, d, d, d);
#else
			v = _mm256_set_ps(d, d, d, d, d, d, d, d);
#endif
#endif
		}
		inline float sum() const {
			assert_align(v);
#ifdef NOSIMD
		return v;
#else
			float sum = 0.0;
			for (int i = 0; i < SIMD_FLOAT_LEN; i++) {
				sum += (*this)[i];
			}
			return sum;
#endif
		}
		inline simd_float(const simd_int &other);
		inline simd_int to_int() const;
		inline simd_float& operator=(const simd_float &other) = default;
		simd_float& operator=(simd_float &&other) {
			assert_align(v);
			v = std::move(other.v);
			return *this;
		}
		inline simd_float operator+(const simd_float &other) const {
#ifdef NOSIMD
		return v + other.v;
#else
			assert_align(v);
			assert_align(other.v);
			simd_float r;
			r.v = _mmx_add_ps(v, other.v);
			return r;
#endif
		}
		inline simd_float operator-(const simd_float &other) const {
#ifdef NOSIMD
		return v - other.v;
#else
			assert_align(v);
			assert_align(other.v);
			simd_float r;
			r.v = _mmx_sub_ps(v, other.v);
			return r;
#endif
		}
		inline simd_float operator*(const simd_float &other) const {
#ifdef NOSIMD
		return v * other.v;
#else
			assert_align(v);
			assert_align(other.v);
			simd_float r;
			r.v = _mmx_mul_ps(v, other.v);
			return r;
#endif
		}
		inline simd_float operator/(const simd_float &other) const {
#ifdef NOSIMD
		return v / other.v;
#else
			assert_align(v);
			assert_align(other.v);
			simd_float r;
			r.v = _mmx_div_ps(v, other.v);
			return r;
#endif
		}
		inline simd_float operator+() const {
			assert_align(v);
			return *this;
		}
		inline simd_float operator-() const {
			assert_align(v);
			return simd_float(0.0) - *this;
		}
		inline simd_float& operator+=(const simd_float &other) {
			assert_align(v);
			assert_align(other.v);
			*this = *this + other;
			return *this;
		}
		inline simd_float& operator-=(const simd_float &other) {
			assert_align(v);
			assert_align(other.v);
			*this = *this - other;
			return *this;
		}
		inline simd_float& operator*=(const simd_float &other) {
			assert_align(v);
			assert_align(other.v);
			*this = *this * other;
			return *this;
		}
		inline simd_float& operator/=(const simd_float &other) {
			assert_align(v);
			assert_align(other.v);
			*this = *this / other;
			return *this;
		}

		inline simd_float operator*(float d) const {
			const simd_float other = d;
			assert_align(v);
			assert_align(other.v);
			return other * *this;
		}
		inline simd_float operator/(float d) const {
			const simd_float other = 1.0 / d;
			assert_align(v);
			assert_align(other.v);
			return *this * other;
		}

		inline simd_float operator*=(float d) {
			assert_align(v);
			*this = *this * d;
			return *this;
		}
		inline simd_float operator/=(float d) {
			assert_align(v);
			*this = *this * (1.0 / d);
			return *this;
		}
		inline float& operator[](std::size_t i) {
			assert_align(v);
#ifdef NOSIMD
		return v;
#else
			float *a = reinterpret_cast<float*>(&v);
			return a[i];
#endif
		}
		inline float operator[](std::size_t i) const {
			assert_align(v);
#ifdef NOSIMD
		return v;
#else
			const float *a = reinterpret_cast<const float*>(&v);
			return a[i];
#endif
		}

		friend simd_float copysign(const simd_float&, const simd_float&);
		friend simd_float sqrt(const simd_float&);
		friend simd_float rsqrt(const simd_float&);
		friend simd_float operator*(float, const simd_float &other);
		friend simd_float operator/(float, const simd_float &other);
		friend simd_float max(const simd_float &a, const simd_float &b);
		friend simd_float min(const simd_float &a, const simd_float &b);
		friend simd_float fma(const simd_float &a, const simd_float &b, const simd_float &c);
		friend simd_float round(const simd_float);

		friend simd_float exp(const simd_float &a);
		friend simd_float sin(const simd_float& a);
		friend simd_float cos(const simd_float& a);
		friend simd_float abs(const simd_float&a);
		friend simd_float erfexp(const simd_float &a, simd_float*);

		friend void sincos(const simd_float& x, simd_float *s, simd_float *c);
#ifdef CHECK_ALIGN
		friend void assert_align(void*);
#endif

		simd_float operator<(simd_float other) const {
#ifdef NOSIMD
		return v < other.v ? 1.0 : 0.0;
#else
			assert_align(v);
			assert_align(other.v);
			static const simd_float one(1);
			static const simd_float zero(0);
			auto mask = _mmx_cmp_ps(v, other.v, _CMP_LT_OQ);
			simd_float v;
#ifdef USE_AVX512
		auto rc = _mm512_mask_mov_ps(zero.v,mask,one.v);
#else
			auto rc = _mmx_and_ps(mask, one.v);
#endif
			v.v = rc;
			return v;
#endif
		}

		simd_float operator>(simd_float other) const {
#ifdef NOSIMD
		return v > other.v ? 1.0 : 0.0;
#else
			assert_align(v);
			assert_align(other.v);
			static const simd_float one(1);
			static const simd_float zero(0);
			auto mask = _mmx_cmp_ps(v, other.v, _CMP_GT_OQ);
			simd_float v;
#ifdef USE_AVX512
		auto rc = _mm512_mask_mov_ps(zero.v,mask,one.v);
#else
			auto rc = _mmx_and_ps(mask, one.v);
#endif
			v.v = rc;
			return v;
#endif
		}

	} SIMDALIGN;

inline simd_float round(const simd_float a) {
	simd_float v;
#ifdef NOSIMD
	return std::round(a.v);
#else
	assert_align(a.v);
#ifdef USE_AVX512
	v.v = _mm512_roundscale_ps(a.v, 0);
#else
	v.v = _mm256_round_ps(a.v, _MM_FROUND_TO_NEAREST_INT);
#endif
	return v;
#endif
}

inline simd_float sin(const simd_float& x0) {
	assert_align(x0.v);
	auto x = x0;
	// From : http://mooooo.ooo/chebyshev-sine-approximation/
	static const simd_float pi_major(3.1415927);
	static const simd_float pi_minor(-0.00000008742278);
	x = x - round(x * (1.0 / (2.0 * M_PI))) * (2.0 * M_PI);
	const simd_float x2 = x * x;
	simd_float p = simd_float(0.00000000013291342);
	p = fma(p, x2, simd_float(-0.000000023317787));
	p = fma(p, x2, simd_float(0.0000025222919));
	p = fma(p, x2, simd_float(-0.00017350505));
	p = fma(p, x2, simd_float(0.0066208798));
	p = fma(p, x2, simd_float(-0.10132118));
	const auto x1 = (x - pi_major - pi_minor);
	const auto x3= (x + pi_major + pi_minor);
	auto res = x1 * x3 * p * x;
	return res;
}

inline simd_float cos(const simd_float& x) {
	assert_align(x.v);
	return sin(x + simd_float(M_PI / 2.0));
}

inline void sincos(const simd_float& x, simd_float *s, simd_float *c) {
	assert_align(x.v);
	assert_align(s->v);
	assert_align(c->v);
#ifdef USE_AVX512
	sincos512_ps(x.v, &s->v, &c->v);
#else
	*s = sin(x);
	*c = cos(x);
#endif
}

inline simd_float exp(const simd_float &a) {
	simd_float v;
	assert_align(a.v);
#ifdef NOSIMD
	return std::exp(a.v);
#else
#ifdef USE_AVX512
	v.v = exp512_ps(a.v);
#else
	v.v = exp256_ps(a.v);
#endif
	return v;
#endif
}

inline simd_float erfexp(const simd_float &x, simd_float *e) {
	assert_align(x.v);
	assert_align(e->v);
	simd_float v;
	const simd_float p(0.3275911);
	const simd_float a1(0.254829592);
	const simd_float a2(-0.284496736);
	const simd_float a3(1.421413741);
	const simd_float a4(-1.453152027);
	const simd_float a5(1.061405429);
	const simd_float t1 = simd_float(1) / (simd_float(1) + p * x);
	const simd_float t2 = t1 * t1;
	const simd_float t3 = t2 * t1;
	const simd_float t4 = t2 * t2;
	const simd_float t5 = t2 * t3;
	*e = exp(-x * x);
	return simd_float(1) - (a1 * t1 + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5) * *e;
	return v;
}

inline simd_float fma(const simd_float &a, const simd_float &b, const simd_float &c) {
	assert_align(a.v);
	assert_align(b.v);
	assert_align(c.v);
#ifdef NOSIMD
	return a.v * b.v + c.v;
#else
	simd_float v;
	v.v = _mmx_fmadd_ps(a.v, b.v, c.v);
	return v;
#endif
}

inline simd_float copysign(const simd_float &y, const simd_float &x) {
// From https://stackoverflow.com/questions/57870896/writing-a-portable-sse-avx-version-of-stdcopysign
	assert_align(x.v);
	assert_align(y.v);

#ifdef NOSIMD
	return std::copysign(y.v, x.v);
#else

	constexpr float signbit = -0.f;
	static auto const avx_signbit = simd_float(signbit).v;
	simd_float v;
	v.v = _mmx_or_ps(_mmx_and_ps(avx_signbit, x.v), _mmx_andnot_ps(avx_signbit, y.v)); // (avx_signbit & from) | (~avx_signbit & to)
	return v;
#endif
}

inline simd_float sqrt(const simd_float &vec) {
	assert_align(vec.v);

#ifdef NOSIMD
	return std::sqrt(vec.v);
#else
	simd_float r;
	r.v = _mmx_sqrt_ps(vec.v);
	return r;
#endif
}

inline simd_float rsqrt(const simd_float &vec) {
	assert_align(vec.v);
#ifdef NOSIMD
	return 1.0 / std::sqrt(vec.v);
#else
	simd_float r;
	r.v = _mmx_rsqrt_ps(vec.v);
	return r;
#endif
}

inline simd_float operator*(float d, const simd_float &other) {
	assert_align(other.v);
	const simd_float a = d;
	return a * other;
}

inline simd_float operator/(float d, const simd_float &other) {
	assert_align(other.v);
	const simd_float a = d;
	return a / other;
}

inline simd_float max(const simd_float &a, const simd_float &b) {
	assert_align(a.v);
	assert_align(b.v);
#ifdef NOSIMD
	return std::max(a.v, b.v);
#else
	simd_float r;
	r.v = _mmx_max_ps(a.v, b.v);
	return r;
#endif
}

inline simd_float min(const simd_float &a, const simd_float &b) {
	assert_align(a.v);
	assert_align(b.v);
#ifdef NOSIMD
	return std::min(a.v, b.v);
#else
	simd_float r;
	r.v = _mmx_min_ps(a.v, b.v);
	return r;
#endif
}

inline simd_float abs(const simd_float &a) {
	assert_align(a.v);
	return max(a, -a);
}

#endif /* TIGERGRAV_SIMD_HPP_ */
