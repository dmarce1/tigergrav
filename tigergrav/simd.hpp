/*
 * simd.hpp
 *
 *  Created on: Jul 3, 2020
 *      Author: dmarce1
 */

#ifndef TIGERGRAV_SIMD_HPP_
#define TIGERGRAV_SIMD_HPP_

#include <immintrin.h>
#include <vector>
#include <algorithm>

#if defined(__AVX2__)
#define SIMD_SLEN 8
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

#define SIMD_DLEN 4
#define _simd_double                 __m256d
#define _mmx_add_pd(a,b)            _mm256_add_pd((a),(b))
#define _mmx_sub_pd(a,b)            _mm256_sub_pd((a),(b))
#define _mmx_mul_pd(a,b)            _mm256_mul_pd((a),(b))
#define _mmx_div_pd(a,b)            _mm256_div_pd((a),(b))
#define _mmx_sqrt_pd(a)             _mm256_sqrt_pd(a)
#define _mmx_min_pd(a, b)           _mm256_min_pd((a),(b))
#define _mmx_max_pd(a, b)           _mm256_max_pd((a),(b))
#define _mmx_or_pd(a, b)            _mm256_or_pd((a),(b))
#define _mmx_and_pd(a, b)           _mm256_and_pd((a),(b))
#define _mmx_andnot_pd(a, b)        _mm256_andnot_pd((a),(b))
#define _mmx_fmadd_pd(a,b,c)        _mm256_fmadd_pd ((a),(b),(c))
#else
#error 'Do not have SIMD instructions for this processor'
#endif

class simd_int_vector;

class simd_svector {
private:
	_simd_float v;
public:
	simd_svector() = default;
	inline simd_svector gather(float *base, simd_int_vector indices);
	inline ~simd_svector() = default;
	simd_svector(const simd_svector&) = default;
	inline simd_svector(float d) {
		v = _mm256_set_ps(d, d, d, d, d, d, d, d);
	}
	inline float sum() const {
		float sum = 0.0;
		for( int i = 0; i < SIMD_SLEN; i++) {
			sum += (*this)[i];
		}
		return sum;
	}
	inline simd_svector(const simd_int_vector &other);
	inline simd_int_vector to_int() const;
	inline simd_svector& operator=(const simd_svector &other) = default;
	simd_svector& operator=(simd_svector &&other) {
		v = std::move(other.v);
		return *this;
	}
	inline simd_svector operator+(const simd_svector &other) const {
		simd_svector r;
		r.v = _mmx_add_ps(v, other.v);
		return r;
	}
	inline simd_svector operator-(const simd_svector &other) const {
		simd_svector r;
		r.v = _mmx_sub_ps(v, other.v);
		return r;
	}
	inline simd_svector operator*(const simd_svector &other) const {
		simd_svector r;
		r.v = _mmx_mul_ps(v, other.v);
		return r;
	}
	inline simd_svector operator/(const simd_svector &other) const {
		simd_svector r;
		r.v = _mmx_div_ps(v, other.v);
		return r;
	}
	inline simd_svector operator+() const {
		return *this;
	}
	inline simd_svector operator-() const {
		return simd_svector(0.0) - *this;
	}
	inline simd_svector& operator+=(const simd_svector &other) {
		*this = *this + other;
		return *this;
	}
	inline simd_svector& operator-=(const simd_svector &other) {
		*this = *this - other;
		return *this;
	}
	inline simd_svector& operator*=(const simd_svector &other) {
		*this = *this * other;
		return *this;
	}
	inline simd_svector& operator/=(const simd_svector &other) {
		*this = *this / other;
		return *this;
	}

	inline simd_svector operator*(float d) const {
		const simd_svector other = d;
		return other * *this;
	}
	inline simd_svector operator/(float d) const {
		const simd_svector other = 1.0 / d;
		return *this * other;
	}

	inline simd_svector operator*=(float d) {
		*this = *this * d;
		return *this;
	}
	inline simd_svector operator/=(float d) {
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
	friend simd_svector copysign(const simd_svector&, const simd_svector&);
	friend simd_svector sqrt(const simd_svector&);
	friend simd_svector rsqrt(const simd_svector&);
	friend simd_svector operator*(float, const simd_svector &other);
	friend simd_svector operator/(float, const simd_svector &other);
	friend simd_svector max(const simd_svector &a, const simd_svector &b);
	friend simd_svector min(const simd_svector &a, const simd_svector &b);
	friend simd_svector fma(const simd_svector &a, const simd_svector &b, const simd_svector &c);

};

inline simd_svector fma(const simd_svector &a, const simd_svector &b, const simd_svector &c) {
	simd_svector v;
	v.v = _mmx_fmadd_ps(a.v, b.v, c.v);
	return v;
}

inline simd_svector copysign(const simd_svector &x, const simd_svector &y) {
	// From https://stackoverflow.com/questions/57870896/writing-a-portable-sse-avx-version-of-stdcopysign
	constexpr float signbit = -0.f;
	static auto const avx_signbit = simd_svector(signbit).v;
	simd_svector v;
	v.v = _mmx_or_ps(_mmx_and_ps(avx_signbit, x.v), _mmx_andnot_ps(avx_signbit, y.v)); // (avx_signbit & from) | (~avx_signbit & to)
	return v;
}

inline simd_svector sqrt(const simd_svector &vec) {
	simd_svector r;
	r.v = _mmx_sqrt_ps(vec.v);
	return r;
}

inline simd_svector rsqrt(const simd_svector &vec) {
	simd_svector r;
	r.v = _mmx_rsqrt_ps(vec.v);
	return r;
}

inline simd_svector operator*(float d, const simd_svector &other) {
	const simd_svector a = d;
	return a * other;
}

inline simd_svector operator/(float d, const simd_svector &other) {
	const simd_svector a = d;
	return a / other;
}

inline void simd_pack(simd_svector *dest, float *src, int src_len, int pos) {
	for (int i = 0; i != src_len; ++i) {
		dest[i][pos] = src[i];
	}
}

inline void simd_unpack(float *dest, simd_svector *src, int src_len, int pos) {
	for (int i = 0; i != src_len; ++i) {
		dest[i] = src[i][pos];
	}
}

inline simd_svector max(const simd_svector &a, const simd_svector &b) {
	simd_svector r;
	r.v = _mmx_max_ps(a.v, b.v);
	return r;
}

inline simd_svector min(const simd_svector &a, const simd_svector &b) {
	simd_svector r;
	r.v = _mmx_min_ps(a.v, b.v);
	return r;
}

inline simd_svector abs(const simd_svector &a) {
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
		v = _mm256_set_epi32(d, d, d, d, d, d, d, d);
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
	friend class simd_svector;
};

inline simd_svector::simd_svector(const simd_int_vector &other) {
	v = _mmx_cvtepi32_ps(other.v);
}

inline simd_int_vector simd_svector::to_int() const {
	simd_int_vector a;
	a.v = _mmx_cvtps_epi32(v);
	return a;
}

inline simd_svector simd_svector::gather(float *base, simd_int_vector indices) {
	v = _mmx_i32gather_ps(base, indices.v, sizeof(float));
	return *this;
}

class simd_dvector {
private:
	_simd_double v;
public:
	simd_dvector() = default;
	inline simd_dvector gather(double *base, simd_int_vector indices);
	inline ~simd_dvector() = default;
	simd_dvector(const simd_dvector&) = default;
	inline simd_dvector(double d) {
		v = _mm256_set_pd(d, d, d, d);
	}
	inline double sum() const {
		double sum = 0.0;
		for (int i = 0; i < SIMD_DLEN; i++) {
			sum += (*this)[i];
		}
		return sum;
	}
	inline simd_dvector(const simd_int_vector &other);
	inline simd_int_vector to_int() const;
	inline simd_dvector& operator=(const simd_dvector &other) = default;
	simd_dvector& operator=(simd_dvector &&other) {
		v = std::move(other.v);
		return *this;
	}
	inline simd_dvector operator+(const simd_dvector &other) const {
		simd_dvector r;
		r.v = _mmx_add_pd(v, other.v);
		return r;
	}
	inline simd_dvector operator-(const simd_dvector &other) const {
		simd_dvector r;
		r.v = _mmx_sub_pd(v, other.v);
		return r;
	}
	inline simd_dvector operator*(const simd_dvector &other) const {
		simd_dvector r;
		r.v = _mmx_mul_pd(v, other.v);
		return r;
	}
	inline simd_dvector operator/(const simd_dvector &other) const {
		simd_dvector r;
		r.v = _mmx_div_pd(v, other.v);
		return r;
	}
	inline simd_dvector operator+() const {
		return *this;
	}
	inline simd_dvector operator-() const {
		return simd_dvector(0.0) - *this;
	}
	inline simd_dvector& operator+=(const simd_dvector &other) {
		*this = *this + other;
		return *this;
	}
	inline simd_dvector& operator-=(const simd_dvector &other) {
		*this = *this - other;
		return *this;
	}
	inline simd_dvector& operator*=(const simd_dvector &other) {
		*this = *this * other;
		return *this;
	}
	inline simd_dvector& operator/=(const simd_dvector &other) {
		*this = *this / other;
		return *this;
	}

	inline simd_dvector operator*(double d) const {
		const simd_dvector other = d;
		return other * *this;
	}
	inline simd_dvector operator/(double d) const {
		const simd_dvector other = 1.0 / d;
		return *this * other;
	}

	inline simd_dvector operator*=(double d) {
		*this = *this * d;
		return *this;
	}
	inline simd_dvector operator/=(double d) {
		*this = *this * (1.0 / d);
		return *this;
	}
	inline double& operator[](std::size_t i) {
		double *a = reinterpret_cast<double*>(&v);
		return a[i];
	}
	inline double operator[](std::size_t i) const {
		const double *a = reinterpret_cast<const double*>(&v);
		return a[i];
	}

	double max() const {
		const double a = std::max((*this)[0], (*this)[1]);
		const double b = std::max((*this)[2], (*this)[3]);
		return std::max(a, b);
	}
	double min() const {
		const double a = std::min((*this)[0], (*this)[1]);
		const double b = std::min((*this)[2], (*this)[3]);
		return std::min(a, b);
	}
	friend simd_dvector copysign(const simd_dvector&, const simd_dvector&);
	friend simd_dvector sqrt(const simd_dvector&);
	friend simd_dvector rsqrt(const simd_dvector&);
	friend simd_dvector operator*(double, const simd_dvector &other);
	friend simd_dvector operator/(double, const simd_dvector &other);
	friend simd_dvector max(const simd_dvector &a, const simd_dvector &b);
	friend simd_dvector min(const simd_dvector &a, const simd_dvector &b);
	friend simd_dvector fma(const simd_dvector &a, const simd_dvector &b, const simd_dvector &c);

};

inline simd_dvector fma(const simd_dvector &a, const simd_dvector &b, const simd_dvector &c) {
	simd_dvector v;
	v.v = _mmx_fmadd_pd(a.v, b.v, c.v);
	return v;
}

inline simd_dvector copysign(const simd_dvector &x, const simd_dvector &y) {
	// From https://stackoverflow.com/questions/57870896/writing-a-portable-sse-avx-version-of-stdcopysign
	constexpr double signbit = -0.f;
	static auto const avx_signbit = simd_dvector(signbit).v;
	simd_dvector v;
	v.v = _mmx_or_pd(_mmx_and_pd(avx_signbit, x.v), _mmx_andnot_pd(avx_signbit, y.v)); // (avx_signbit & from) | (~avx_signbit & to)
	return v;
}

inline simd_dvector sqrt(const simd_dvector &vec) {
	simd_dvector r;
	r.v = _mmx_sqrt_pd(vec.v);
	return r;
}

inline simd_dvector rsqrt(const simd_dvector &vec) {
	return 1.0 / vec;
}

inline simd_dvector operator*(double d, const simd_dvector &other) {
	const simd_dvector a = d;
	return a * other;
}

inline simd_dvector operator/(double d, const simd_dvector &other) {
	const simd_dvector a = d;
	return a / other;
}

inline void simd_pack(simd_dvector *dest, double *src, int src_len, int pos) {
	for (int i = 0; i != src_len; ++i) {
		dest[i][pos] = src[i];
	}
}

inline void simd_unpack(double *dest, simd_dvector *src, int src_len, int pos) {
	for (int i = 0; i != src_len; ++i) {
		dest[i] = src[i][pos];
	}
}

inline simd_dvector max(const simd_dvector &a, const simd_dvector &b) {
	simd_dvector r;
	r.v = _mmx_max_pd(a.v, b.v);
	return r;
}

inline simd_dvector min(const simd_dvector &a, const simd_dvector &b) {
	simd_dvector r;
	r.v = _mmx_min_pd(a.v, b.v);
	return r;
}

inline simd_dvector abs(const simd_dvector &a) {
	return max(a, -a);
}

#endif /* TIGERGRAV_SIMD_HPP_ */
