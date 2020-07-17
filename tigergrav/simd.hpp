/*
 * simd.hpp
 *
 *  Created on: Jul 3, 2020
 *      Author: dmarce1
 */

#ifndef TIGERGRAV_SIMD_HPP_
#define TIGERGRAV_SIMD_HPP_

#include <immintrin.h>

#include <tigergrav/avx_mathfun.h>

#include <cmath>



#if defined(__AVX512F__)

#define SIMD_FLOAT_LEN 16
#define _simd_float                 __m512
#define _simd_int                   __m512i
#define _mmx_set_ps(d)              _mm512_set_ps(d, d, d, d, d, d, d, d,d, d, d, d, d, d, d, d)
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
#define _mmx_rsqrt_ps(a)            _mm512_rsqrt23_ps(a)
#define _mmx_add_epi32(a,b)         _mm512_add_epi32((a),(b))
#define _mmx_sub_epi32(a,b)         _mm512_sub_epi32((a),(b))
#define _mmx_mul_epi32(a,b)         _mm512_mullo_epi32((a),(b))
#define _mmx_cvtps_epi32(a)         _mm512_cvtps_epi32((a))
#define _mmx_fmadd_ps(a,b,c)        _mm512_fmadd_ps ((a),(b),(c))
#define _mmx_cmp_ps(a,b,c)        	_mm512_cmp_ps(a,b,c)

#else


#if defined(__AVX2__)
#define SIMD_FLOAT_LEN 8
#define _simd_float                 __m256
#define _simd_int                   __m256i
#define _mmx_set_ps(d)              _mm256_set_ps(d, d, d, d, d, d, d, d)
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
#error 'Do not have SIMD instructions for this processor'
#endif
#endif

class simd_int;

class simd_float {
private:
	_simd_float v;
public:
	static constexpr std::size_t size() {
		return SIMD_FLOAT_LEN;
	}
	simd_float() = default;
	inline ~simd_float() = default;
	simd_float(const simd_float&) = default;
	inline simd_float(float d) {
		v = _mmx_set_ps(d);
	}
	inline float sum() const {
		float sum = 0.0;
		for (int i = 0; i < SIMD_FLOAT_LEN; i++) {
			sum += (*this)[i];
		}
		return sum;
	}
	inline simd_float(const simd_int &other);
	inline simd_int to_int() const;
	inline simd_float& operator=(const simd_float &other) = default;
	simd_float& operator=(simd_float &&other) {
		v = std::move(other.v);
		return *this;
	}
	inline simd_float operator+(const simd_float &other) const {
		simd_float r;
		r.v = _mmx_add_ps(v, other.v);
		return r;
	}
	inline simd_float operator-(const simd_float &other) const {
		simd_float r;
		r.v = _mmx_sub_ps(v, other.v);
		return r;
	}
	inline simd_float operator*(const simd_float &other) const {
		simd_float r;
		r.v = _mmx_mul_ps(v, other.v);
		return r;
	}
	inline simd_float operator/(const simd_float &other) const {
		simd_float r;
		r.v = _mmx_div_ps(v, other.v);
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
		float *a = reinterpret_cast<float*>(&v);
		return a[i];
	}
	inline float operator[](std::size_t i) const {
		const float *a = reinterpret_cast<const float*>(&v);
		return a[i];
	}

	float max() const {
		const float a = std::max((*this)[0], (*this)[1]);
		const float b = std::max((*this)[2], (*this)[3]);
		const float c = std::max((*this)[4], (*this)[5]);
		const float d = std::max((*this)[6], (*this)[7]);
		const float e = std::max(a, b);
		const float f = std::max(c, d);
		return std::max(e, f);
	}
	float min() const {
		const float a = std::min((*this)[0], (*this)[1]);
		const float b = std::min((*this)[2], (*this)[3]);
		const float c = std::min((*this)[4], (*this)[5]);
		const float d = std::min((*this)[6], (*this)[7]);
		const float e = std::min(a, b);
		const float f = std::min(c, d);
		return std::min(e, f);
	}
	friend simd_float copysign(const simd_float&, const simd_float&);
	friend simd_float sqrt(const simd_float&);
	friend simd_float rsqrt(const simd_float&);
	friend simd_float operator*(float, const simd_float &other);
	friend simd_float operator/(float, const simd_float &other);
	friend simd_float max(const simd_float &a, const simd_float &b);
	friend simd_float min(const simd_float &a, const simd_float &b);
	friend simd_float fma(const simd_float &a, const simd_float &b, const simd_float &c);

	friend simd_float exp(const simd_float &a);
	friend simd_float erfexp(const simd_float &a, simd_float* e);
	friend void sincos(simd_float x, simd_float *s, simd_float *c);


	// 2 OPS
	simd_float operator<(simd_float other) const {
			static const simd_float one(1);
			static const simd_float zero(0);
			auto mask = _mmx_cmp_ps(v, other.v, _CMP_LT_OQ);
			auto rc = _mmx_and_ps(mask, one.v);
			simd_float v;
			v.v = rc;
			return v;
		}

	// 2 OPS
	simd_float operator<=(simd_float other) const {
		static const simd_float one(1);
		static const simd_float zero(0);
		auto mask = _mmx_cmp_ps(v, other.v, _CMP_LE_OQ);
		auto rc = _mmx_and_ps(mask, one.v);
		simd_float v;
		v.v = rc;
		return v;
	}

	// 2 OPS
	simd_float operator!=(simd_float other) const {
		static const simd_float one(1);
		static const simd_float zero(0);
		auto mask = _mmx_cmp_ps(v, other.v, _CMP_NEQ_OQ);
		auto rc = _mmx_and_ps(mask, one.v);
		simd_float v;
		v.v = rc;
		return v;
	}

	// 2 OPS
	simd_float operator==(simd_float other) const {
		static const simd_float one(1);
		static const simd_float zero(0);
		auto mask = _mmx_cmp_ps(v, other.v, _CMP_EQ_OQ);
		auto rc = _mmx_and_ps(mask, one.v);
		simd_float v;
		v.v = rc;
		return v;
	}

	// 2 OPS
	simd_float operator>(simd_float other) const {
		static const simd_float one(1);
		static const simd_float zero(0);
		auto mask = _mmx_cmp_ps(v, other.v, _CMP_GT_OQ);
		auto rc = _mmx_and_ps(mask, one.v);
		simd_float v;
		v.v = rc;
		return v;
	}

	// 2 OPS
	simd_float operator>=(simd_float other) const {
		static const simd_float one(1);
		static const simd_float zero(0);
		auto mask = _mmx_cmp_ps(v, other.v, _CMP_GE_OQ);
		auto rc = _mmx_and_ps(mask, one.v);
		simd_float v;
		v.v = rc;
		return v;
	}




};

// 41 OPS
inline void sincos(simd_float x, simd_float *s, simd_float *c) {
#if defined(__AVX512F__)
	sincos512_ps(x.v, &(s->v), &(c->v));
#else
	sincos256_ps(x.v, &(s->v), &(c->v));
#endif
}

// 30 OPS
inline simd_float exp(const simd_float &a) {
	simd_float v;

#if defined(__AVX512F__)
	v.v = exp512_ps(a.v);
#else
	v.v = exp256_ps(a.v);
#endif
	return v;
}

// 50 OPS
inline simd_float erfexp(const simd_float &x, simd_float* e) {
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
	simd_float v;
	v.v = _mmx_fmadd_ps(a.v, b.v, c.v);
	return v;
}

inline simd_float copysign(const simd_float &y, const simd_float &x) {
// From https://stackoverflow.com/questions/57870896/writing-a-portable-sse-avx-version-of-stdcopysign
	constexpr float signbit = -0.f;
	static auto const avx_signbit = simd_float(signbit).v;
	simd_float v;
	v.v = _mmx_or_ps(_mmx_and_ps(avx_signbit, x.v), _mmx_andnot_ps(avx_signbit, y.v)); // (avx_signbit & from) | (~avx_signbit & to)
	return v;
}

inline simd_float sqrt(const simd_float &vec) {
	simd_float r;
	r.v = _mmx_sqrt_ps(vec.v);
	return r;
}

inline simd_float rsqrt(const simd_float &vec) {
	simd_float r;
	r.v = _mmx_rsqrt_ps(vec.v);
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

inline void simd_pack(simd_float *dest, float *src, int src_len, int pos) {
	for (int i = 0; i != src_len; ++i) {
		dest[i][pos] = src[i];
	}
}

inline void simd_unpack(float *dest, simd_float *src, int src_len, int pos) {
	for (int i = 0; i != src_len; ++i) {
		dest[i] = src[i][pos];
	}
}

inline simd_float max(const simd_float &a, const simd_float &b) {
	simd_float r;
	r.v = _mmx_max_ps(a.v, b.v);
	return r;
}

inline simd_float min(const simd_float &a, const simd_float &b) {
	simd_float r;
	r.v = _mmx_min_ps(a.v, b.v);
	return r;
}

inline simd_float abs(const simd_float &a) {
	return max(a, -a);
}

#endif /* TIGERGRAV_SIMD_HPP_ */
