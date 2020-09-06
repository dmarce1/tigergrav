#pragma once

#include <tigergrav/defs.hpp>
#include <tigergrav/simd.hpp>
#include <tigergrav/expansion.hpp>
#include <tigergrav/multipole.hpp>
#include <tigergrav/particle.hpp>
#include <tigergrav/options.hpp>

#include <memory>
#include <map>

struct multi_src {
	vect<pos_type> x;
	multipole<float> m;
	template<class A>
	void serialize(A &&arc, unsigned) {
		arc & m;
		arc & x;
	}

};

struct multipole_info {
	multipole<float> m;
	float r;
	vect<pos_type> x;
	std::uint64_t num_active;
	template<class A>
	void serialize(A &&arc, unsigned) {
		arc & m;
		arc & x;
		arc & r;
		arc & num_active;
	}
};

std::uint64_t gravity_PC_direct(std::vector<force> &g, const std::vector<vect<pos_type>> &x, std::vector<const multi_src*> &y);
std::uint64_t gravity_CC_direct(expansion<double>&, const vect<pos_type> &x, std::vector<const multi_src*> &y);
std::uint64_t gravity_CP_direct(expansion<double> &L, const vect<pos_type> &x, std::vector<vect<pos_type>> y);
std::uint64_t gravity_PP_ewald(std::vector<force> &g, const std::vector<vect<pos_type>> &x, std::vector<vect<pos_type>> y);
std::uint64_t gravity_PC_ewald(std::vector<force> &g, const std::vector<vect<pos_type>> &x, std::vector<const multi_src*> &y);
std::uint64_t gravity_CC_ewald(expansion<double>&, const vect<pos_type> &x, std::vector<const multi_src*> &y);
std::uint64_t gravity_CP_ewald(expansion<double> &L, const vect<pos_type> &x, std::vector<vect<pos_type>> y);

inline float separation2(const vect<pos_type> &x, const vect<pos_type> &y) {
	static const auto opts = options::get();
	static const auto ewald = opts.ewald;
	static const auto POS_INV2 = POS_INV * POS_INV;
	if (ewald) {
		std::uint64_t diff = x[0] - y[0];
		std::uint64_t sum = diff * diff;
		for (int dim = 1; dim < NDIM; dim++) {
			diff = x[dim] - y[dim];
			sum += diff * diff;
		}
		return sum * POS_INV2;
	} else {
		float sum = 0;
		for (int dim = 0; dim < NDIM; dim++) {
			const float diff = float(x[dim]) - float(y[dim]);
			sum += diff * diff;
		}
		return sum * POS_INV2;
	}
}

inline float ewald_far_separation2(const vect<pos_type> &x, const vect<pos_type> &y) {
	static const float r2 = 0.25 * 0.25;
	return std::max(r2, separation2(x, y));
}
