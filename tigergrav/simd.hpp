/*
 * simd.hpp
 *
 *  Created on: Jul 3, 2020
 *      Author: dmarce1
 */

#ifndef TIGERGRAV_SIMD_HPP_
#define TIGERGRAV_SIMD_HPP_

#include <immintrin.h>

#define SIMD_LEN  8
#define __mxs __m256
#define _mmx_set_ps(d)    _mm256_set_ps((d),(d),(d),(d),(d),(d),(d),(d))
#define _mmx_add_ps(a,b)  _mm256_add_ps((a),(b))
#define _mmx_sub_ps(a,b)  _mm256_sub_ps((a),(b))
#define _mmx_mul_ps(a,b)  _mm256_mul_ps((a),(b))
#define _mmx_div_ps(a,b)  _mm256_div_ps((a),(b))
#define _mmx_sqrt_ps(a)   _mm256_sqrt_ps(a)
#define _mmx_rsqrt_ps(a)  _mm256_rsqrt_ps(a)
#define _mmx_max_ps(a, b) _mm256_max_ps((a),(b))
#define _mmx_min_ps(a, b) _mm256_min_ps((a),(b))

class simd_vector {
private:
	__mxs v;
public:
	simd_vector() {
		*this = 0;
	}
	inline ~simd_vector() = default;
	simd_vector(const simd_vector&) = default;
	inline simd_vector(float d) {
		v = _mmx_set_ps(d);
	}
	inline float sum() const {
		const float r0 = (*this)[0] + (*this)[4];
		const float r1 = (*this)[1] + (*this)[5];
		const float r2 = (*this)[2] + (*this)[6];
		const float r3 = (*this)[3] + (*this)[7];
		const float r4 = r0 + r1;
		const float r5 = r2 + r3;
		return r4 + r5;
	}
	inline simd_vector(simd_vector &&other) {
		*this = std::move(other);
	}
	inline simd_vector(const std::array<std::uint32_t,SIMD_LEN> &other) {
		const auto *ptr = reinterpret_cast<const __m256i*>(&other);
		v = _mm256_cvtepi32_ps(*ptr);
	}
	inline std::array<std::uint32_t,SIMD_LEN> to_int() const {
		std::array<std::uint32_t, SIMD_LEN> a;
		auto *ptr = reinterpret_cast<__m256i*>(&a);
		*ptr = _mm256_cvtps_epi32(v);
		return a;
	}
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

};

inline simd_vector copysign(const simd_vector &x, const simd_vector &y) {
	// From https://stackoverflow.com/questions/57870896/writing-a-portable-sse-avx-version-of-stdcopysign
	constexpr float signbit = -0.f;
	static auto const avx_signbit = _mm256_broadcast_ss(&signbit);
	simd_vector v;
	v.v = _mm256_or_ps(_mm256_and_ps(avx_signbit, x.v), _mm256_andnot_ps(avx_signbit, y.v)); // (avx_signbit & from) | (~avx_signbit & to)
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

#endif /* TIGERGRAV_SIMD_HPP_ */
