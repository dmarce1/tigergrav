/*
 * simd.hpp
 *
 *  Created on: Jul 3, 2020
 *      Author: dmarce1
 */

#ifndef TIGERGRAV_SIMD_HPP_
#define TIGERGRAV_SIMD_HPP_

#include <immintrin.h>

#if defined(__AVX512F__)
#define SIMD_LEN 16
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
#define _mmx_cvtepi32_ps(a)         _mm512_cvtepi32_ps((a))
#define _mmx_cvtps_epi32(a)         _mm512_cvtps_epi32((a))
#define _mmx_i32gather_ps(a,b,c)    _mm512_i32gather_ps((b),(a),(c))
#define _mmx_fmadd_ps(a,b,c)        _mm512_fmadd_ps ((a),(b),(c))

#elif defined(__AVX2__)
#define SIMD_LEN 8
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
#define _mmx_cvtepi32_ps(a)         _mm256_cvtepi32_ps((a))
#define _mmx_cvtps_epi32(a)         _mm256_cvtps_epi32((a))
#define _mmx_i32gather_ps(a,b,c)    _mm256_i32gather_ps((a),(b),(c))
#define _mmx_fmadd_ps(a,b,c)        _mm256_fmadd_ps ((a),(b),(c))
#else
#error 'Do not have SIMD instructions for this processor'
#endif

class simd_int_vector;

class simd_vector {
private:
	_simd_float v;
public:
	simd_vector() = default;
	inline simd_vector gather(float *base, simd_int_vector indices);
	inline ~simd_vector() = default;
	simd_vector(const simd_vector&) = default;
	inline simd_vector(float d) {
#ifdef __AVX512F__
		v = _mm512_set_ps(d, d, d, d, d, d, d, d, d, d, d, d, d, d, d, d);
#else
		v = _mm256_set_ps(d, d, d, d, d, d, d, d);
#endif
	}
	inline float sum() const {
		float sum = 0.0;
		for (int i = 0; i < SIMD_LEN; i++) {
			sum += (*this)[i];
		}
		return sum;
	}
	inline simd_vector(const simd_int_vector &other);
	inline simd_int_vector to_int() const;
	inline simd_vector& operator=(const simd_vector &other) = default;
	simd_vector& operator=(simd_vector &&other) {
		v = std::move(other.v);
		return *this;
	}
	inline simd_vector operator+(const simd_vector &other) const {
		simd_vector r;
		r.v = _mmx_add_ps(v, other.v);
		return r;
	}
	inline simd_vector operator-(const simd_vector &other) const {
		simd_vector r;
		r.v = _mmx_sub_ps(v, other.v);
		return r;
	}
	inline simd_vector operator*(const simd_vector &other) const {
		simd_vector r;
		r.v = _mmx_mul_ps(v, other.v);
		return r;
	}
	inline simd_vector operator/(const simd_vector &other) const {
		simd_vector r;
		r.v = _mmx_div_ps(v, other.v);
		return r;
	}
	inline simd_vector operator+() const {
		return *this;
	}
	inline simd_vector operator-() const {
		return simd_vector(0.0) - *this;
	}
	inline simd_vector& operator+=(const simd_vector &other) {
		*this = *this + other;
		return *this;
	}
	inline simd_vector& operator-=(const simd_vector &other) {
		*this = *this - other;
		return *this;
	}
	inline simd_vector& operator*=(const simd_vector &other) {
		*this = *this * other;
		return *this;
	}
	inline simd_vector& operator/=(const simd_vector &other) {
		*this = *this / other;
		return *this;
	}

	inline simd_vector operator*(float d) const {
		const simd_vector other = d;
		return other * *this;
	}
	inline simd_vector operator/(float d) const {
		const simd_vector other = 1.0 / d;
		return *this * other;
	}

	inline simd_vector operator*=(float d) {
		*this = *this * d;
		return *this;
	}
	inline simd_vector operator/=(float d) {
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
	friend simd_vector copysign(const simd_vector&, const simd_vector&);
	friend simd_vector sqrt(const simd_vector&);
	friend simd_vector rsqrt(const simd_vector&);
	friend simd_vector operator*(float, const simd_vector &other);
	friend simd_vector operator/(float, const simd_vector &other);
	friend simd_vector max(const simd_vector &a, const simd_vector &b);
	friend simd_vector min(const simd_vector &a, const simd_vector &b);
	friend simd_vector fma(const simd_vector &a, const simd_vector &b, const simd_vector &c);

};

inline simd_vector fma(const simd_vector &a, const simd_vector &b, const simd_vector &c) {
	simd_vector v;
	v.v =_mmx_fmadd_ps(a.v, b.v, c.v);
	return v;
}

inline simd_vector copysign(const simd_vector &x, const simd_vector &y) {
	// From https://stackoverflow.com/questions/57870896/writing-a-portable-sse-avx-version-of-stdcopysign
	constexpr float signbit = -0.f;
	static auto const avx_signbit = simd_vector(signbit).v;
	simd_vector v;
	v.v = _mmx_or_ps(_mmx_and_ps(avx_signbit, x.v), _mmx_andnot_ps(avx_signbit, y.v)); // (avx_signbit & from) | (~avx_signbit & to)
	return v;
}

inline simd_vector sqrt(const simd_vector &vec) {
	simd_vector r;
	r.v = _mmx_sqrt_ps(vec.v);
	return r;
}

inline simd_vector rsqrt(const simd_vector &vec) {
	simd_vector r;
	r.v = _mmx_rsqrt_ps(vec.v);
	return r;
}

inline simd_vector operator*(float d, const simd_vector &other) {
	const simd_vector a = d;
	return a * other;
}

inline simd_vector operator/(float d, const simd_vector &other) {
	const simd_vector a = d;
	return a / other;
}

inline void simd_pack(simd_vector *dest, float *src, int src_len, int pos) {
	for (int i = 0; i != src_len; ++i) {
		dest[i][pos] = src[i];
	}
}

inline void simd_unpack(float *dest, simd_vector *src, int src_len, int pos) {
	for (int i = 0; i != src_len; ++i) {
		dest[i] = src[i][pos];
	}
}

inline simd_vector max(const simd_vector &a, const simd_vector &b) {
	simd_vector r;
	r.v = _mmx_max_ps(a.v, b.v);
	return r;
}

inline simd_vector min(const simd_vector &a, const simd_vector &b) {
	simd_vector r;
	r.v = _mmx_min_ps(a.v, b.v);
	return r;
}

inline simd_vector abs(const simd_vector &a) {
	return max(a, -a);
}

class simd_int_vector {
private:
	_simd_int v;
public:
	simd_int_vector() {
		*this = 0;
	}
	inline ~simd_int_vector() = default;
	simd_int_vector(const simd_int_vector&) = default;
	inline simd_int_vector(std::int32_t d) {
#ifdef __AVX512F__
		v = _mm512_set_epi32(d, d, d, d, d, d, d, d, d, d, d, d, d, d, d, d);
#else
		v = _mm256_set_epi32(d, d, d, d, d, d, d, d);
#endif
	}
	inline simd_int_vector(simd_int_vector &&other) {
		*this = std::move(other);
	}
	inline simd_int_vector& operator=(const simd_int_vector &other) = default;
	simd_int_vector& operator=(simd_int_vector &&other) {
		v = std::move(other.v);
		return *this;
	}
	inline simd_int_vector operator+(const simd_int_vector &other) const {
		simd_int_vector r;
		r.v = _mmx_add_epi32(v, other.v);
		return r;
	}
	inline simd_int_vector operator-(const simd_int_vector &other) const {
		simd_int_vector r;
		r.v = _mmx_sub_epi32(v, other.v);
		return r;
	}
	inline simd_int_vector operator*(const simd_int_vector &other) const {
		simd_int_vector r;
		r.v = _mmx_mul_epi32(v, other.v);
		return r;
	}
	inline simd_int_vector operator+() const {
		return *this;
	}
	inline simd_int_vector operator-() const {
		return simd_int_vector(0.0) - *this;
	}
	inline simd_int_vector& operator+=(const simd_int_vector &other) {
		*this = *this + other;
		return *this;
	}
	inline simd_int_vector& operator-=(const simd_int_vector &other) {
		*this = *this - other;
		return *this;
	}
	inline simd_int_vector& operator*=(const simd_int_vector &other) {
		*this = *this * other;
		return *this;
	}
	inline simd_int_vector operator*(std::int32_t d) const {
		const simd_int_vector other = d;
		return other * *this;
	}
	inline simd_int_vector operator*=(std::int32_t d) {
		*this = *this * d;
		return *this;
	}
	inline simd_int_vector operator/=(std::int32_t d) {
		*this = *this * (1.0 / d);
		return *this;
	}
	inline std::int32_t& operator[](std::size_t i) {
		std::int32_t *a = reinterpret_cast<std::int32_t*>(&v);
		return a[i];
	}
	inline std::int32_t operator[](std::size_t i) const {
		const std::int32_t *a = reinterpret_cast<const std::int32_t*>(&v);
		return a[i];
	}
	friend class simd_vector;
};

inline simd_vector::simd_vector(const simd_int_vector &other) {
	v = _mmx_cvtepi32_ps(other.v);
}

inline simd_int_vector simd_vector::to_int() const {
	simd_int_vector a;
	a.v = _mmx_cvtps_epi32(v);
	return a;
}

inline simd_vector simd_vector::gather(float *base, simd_int_vector indices) {
	v = _mmx_i32gather_ps(base, indices.v, sizeof(float));
	return *this;
}

#endif /* TIGERGRAV_SIMD_HPP_ */
