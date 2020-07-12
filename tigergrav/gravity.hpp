#pragma once

#include <tigergrav/defs.hpp>
#include <tigergrav/simd.hpp>
#include <tigergrav/expansion.hpp>
#include <tigergrav/multipole.hpp>
#include <tigergrav/particle.hpp>


struct multipole_info {
	multipole<ireal> m;
	vect<ireal> x;
	ireal r;
};

struct multi_src {
	multipole<ireal > m;
	vect<ireal> x;
};


std::uint64_t gravity_direct(std::vector<force> &g, const std::vector<vect<float>> &x, std::vector<vect<float>> &y);
std::uint64_t gravity_direct_multipole(std::vector<force> &g, const std::vector<vect<float>> &x, std::vector<multi_src> &y);
std::uint64_t gravity_indirect_multipole(expansion<ireal>&, const vect<ireal> &x, std::vector<multi_src> &y);
std::uint64_t gravity_ewald(std::vector<force> &g, const std::vector<vect<float>> &x, std::vector<source> &y);
std::uint64_t gravity_indirect_ewald(expansion<ireal> &L, const vect<float> &x, std::vector<source> &y);
double ewald_near_separation(const vect<double> x);
double ewald_far_separation(const vect<double> x);
void init_ewald();
