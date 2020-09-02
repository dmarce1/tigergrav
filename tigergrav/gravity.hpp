#pragma once

#include <tigergrav/defs.hpp>
#include <tigergrav/simd.hpp>
#include <tigergrav/expansion.hpp>
#include <tigergrav/multipole.hpp>
#include <tigergrav/particle.hpp>

#include <memory>
#include <map>

struct multi_src {
	multipole<float> m;
	vect<pos_type> x;
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


std::uint64_t gravity_PC_direct(std::vector<force> &g, const std::vector<vect<pos_type>> &x, std::vector<multi_src> &y);
std::uint64_t gravity_CC_direct(expansion<double>&, const vect<pos_type> &x, std::vector<multi_src> &y);
std::uint64_t gravity_CP_direct(expansion<double> &L, const vect<pos_type> &x, std::vector<vect<pos_type>> y);
std::uint64_t gravity_PP_ewald(std::vector<force> &g, const std::vector<vect<pos_type>> &x, std::vector<vect<pos_type>> y);
std::uint64_t gravity_PC_ewald(std::vector<force> &g, const std::vector<vect<pos_type>> &x, std::vector<multi_src> &y);
std::uint64_t gravity_CC_ewald(expansion<double>&, const vect<pos_type> &x, std::vector<multi_src> &y);
std::uint64_t gravity_CP_ewald(expansion<double> &L, const vect<pos_type> &x, std::vector<vect<pos_type>> y);


double ewald_near_separation(const vect<double> x);
double ewald_far_separation(const vect<double> x, double r);
void init_ewald();
