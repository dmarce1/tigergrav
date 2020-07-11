#pragma once

#include <tigergrav/expansion.hpp>
#include <tigergrav/multipole.hpp>
#include <tigergrav/particle.hpp>


struct multipole_info {
	multipole<double> m;
	vect<double> x;
	double r;
};

struct multi_src {
	multipole<double > m;
	vect<double> x;
};


std::uint64_t gravity_direct(std::vector<force> &g, const std::vector<vect<float>> &x, std::vector<vect<float>> &y);
std::uint64_t gravity_direct_multipole(std::vector<force> &g, const std::vector<vect<float>> &x, std::vector<multi_src> &y);
std::uint64_t gravity_indirect_multipole(expansion<double>&, const vect<double> &x, std::vector<multi_src> &y);
std::uint64_t gravity_ewald(std::vector<force> &g, const std::vector<vect<float>> &x, std::vector<source> &y);
std::uint64_t gravity_indirect_ewald(expansion<double> &L, const vect<float> &x, std::vector<source> &y);
double ewald_near_separation(const vect<double> x);
double ewald_far_separation(const vect<double> x);
void init_ewald();
