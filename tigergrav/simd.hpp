/*
 * simd.hpp
 *
 *  Created on: Jul 3, 2020
 *      Author: dmarce1
 */

#ifndef TIGERGRAV_SIMD_HPP_
#define TIGERGRAV_SIMD_HPP_

#include <icc/immintrin.h>

#if defined(__AVX2__)
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

#define SIMD_DOUBLE_LEN 4
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

class simd_int;

class simd_float {
private:
	_simd_float v;
public:
	simd_float() = default;
	inline ~simd_float() = default;
	simd_float(const simd_float&) = default;
	inline simd_float(float d) {
		v = _mm256_set_ps(d, d, d, d, d, d, d, d);
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

	friend simd_float exp(const simd_float& a);
	friend simd_float sin(const simd_float& a);
	friend simd_float cos(const simd_float& a);
	friend simd_float erf(const simd_float& a);

};



inline simd_float exp(const simd_float& a) {
	simd_float v;
	v.v = _mm256_exp_ps(a.v);
	return v;
}

inline simd_float sin(const simd_float& a) {
	simd_float v;
	v.v = _mm256_sin_ps(a.v);
	return v;
}

inline simd_float cos(const simd_float& a) {
	simd_float v;
	v.v = _mm256_cos_ps(a.v);
	return v;
}

inline simd_float erf(const simd_float& a) {
	simd_float v;
	v.v = _mm256_erf_ps(a.v);
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

class simd_double {
private:
	_simd_double v;
public:
	simd_double() = default;
	inline ~simd_double() = default;
	simd_double(const simd_double&) = default;
	inline simd_double(double d) {
		v = _mm256_set_pd(d, d, d, d);
	}
	inline double sum() const {
		double sum = 0.0;
		for (int i = 0; i < SIMD_DOUBLE_LEN; i++) {
			sum += (*this)[i];
		}
		return sum;
	}
	inline simd_double(const simd_int &other);
	inline simd_int to_int() const;
	inline simd_double& operator=(const simd_double &other) = default;
	simd_double& operator=(simd_double &&other) {
		v = std::move(other.v);
		return *this;
	}
	inline simd_double operator+(const simd_double &other) const {
		simd_double r;
		r.v = _mmx_add_pd(v, other.v);
		return r;
	}
	inline simd_double operator-(const simd_double &other) const {
		simd_double r;
		r.v = _mmx_sub_pd(v, other.v);
		return r;
	}
	inline simd_double operator*(const simd_double &other) const {
		simd_double r;
		r.v = _mmx_mul_pd(v, other.v);
		return r;
	}
	inline simd_double operator/(const simd_double &other) const {
		simd_double r;
		r.v = _mmx_div_pd(v, other.v);
		return r;
	}
	inline simd_double operator+() const {
		return *this;
	}
	inline simd_double operator-() const {
		return simd_double(0.0) - *this;
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

	inline simd_double operator*(double d) const {
		const simd_double other = d;
		return other * *this;
	}
	inline simd_double operator/(double d) const {
		const simd_double other = 1.0 / d;
		return *this * other;
	}

	inline simd_double operator*=(double d) {
		*this = *this * d;
		return *this;
	}
	inline simd_double operator/=(double d) {
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
	friend simd_double copysign(const simd_double&, const simd_double&);
	friend simd_double sqrt(const simd_double&);
	friend simd_double rsqrt(const simd_double&);
	friend simd_double operator*(double, const simd_double &other);
	friend simd_double operator/(double, const simd_double &other);
	friend simd_double max(const simd_double &a, const simd_double &b);
	friend simd_double min(const simd_double &a, const simd_double &b);
	friend simd_double fma(const simd_double &a, const simd_double &b, const simd_double &c);
	friend simd_double exp(const simd_double& a);
	friend simd_double erf(const simd_double& a);
	friend simd_double sin(const simd_double& a);
	friend simd_double cos(const simd_double& a);

};


inline simd_double exp(const simd_double& a) {
	simd_double v;
	v.v = _mm256_exp_pd(a.v);
	return v;
}

inline simd_double sin(const simd_double& a) {
	simd_double v;
	v.v = _mm256_sin_pd(a.v);
	return v;
}

inline simd_double cos(const simd_double& a) {
	simd_double v;
	v.v = _mm256_cos_pd(a.v);
	return v;
}

inline simd_double erf(const simd_double& a) {
	simd_double v;
	v.v = _mm256_erf_pd(a.v);
	return v;
}


inline simd_double fma(const simd_double &a, const simd_double &b, const simd_double &c) {
	simd_double v;
	v.v = _mmx_fmadd_pd(a.v, b.v, c.v);
	return v;
}

inline simd_double copysign(const simd_double &y, const simd_double &x) {
	// From https://stackoverflow.com/questions/57870896/writing-a-portable-sse-avx-version-of-stdcopysign
	constexpr double signbit = -0.f;
	static auto const avx_signbit = simd_double(signbit).v;
	simd_double v;
	v.v = _mmx_or_pd(_mmx_and_pd(avx_signbit, x.v), _mmx_andnot_pd(avx_signbit, y.v)); // (avx_signbit & from) | (~avx_signbit & to)
	return v;
}

inline simd_double sqrt(const simd_double &vec) {
	simd_double r;
	r.v = _mmx_sqrt_pd(vec.v);
	return r;
}

inline simd_double rsqrt(const simd_double &vec) {
	return 1.0 / vec;
}

inline simd_double operator*(double d, const simd_double &other) {
	const simd_double a = d;
	return a * other;
}

inline simd_double operator/(double d, const simd_double &other) {
	const simd_double a = d;
	return a / other;
}

inline void simd_pack(simd_double *dest, double *src, int src_len, int pos) {
	for (int i = 0; i != src_len; ++i) {
		dest[i][pos] = src[i];
	}
}

inline void simd_unpack(double *dest, simd_double *src, int src_len, int pos) {
	for (int i = 0; i != src_len; ++i) {
		dest[i] = src[i][pos];
	}
}

inline simd_double max(const simd_double &a, const simd_double &b) {
	simd_double r;
	r.v = _mmx_max_pd(a.v, b.v);
	return r;
}

inline simd_double min(const simd_double &a, const simd_double &b) {
	simd_double r;
	r.v = _mmx_min_pd(a.v, b.v);
	return r;
}

inline simd_double abs(const simd_double &a) {
	return max(a, -a);
}

#endif /* TIGERGRAV_SIMD_HPP_ */
