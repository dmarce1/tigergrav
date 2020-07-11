#pragma once

#include <tigergrav/expansion.hpp>
#include <tigergrav/multipole.hpp>
#include <tigergrav/particle.hpp>


struct multipole_info {
	multipole<float> m;
	vect<float> x;
	float r;
};

struct multi_src {
	multipole<float > m;
	vect<float> x;
};


std::uint64_t gravity_direct(std::vector<force> &g, const std::vector<vect<float>> &x, std::vector<source> &y);
std::uint64_t gravity_direct_multipole(std::vector<force> &g, const std::vector<vect<float>> &x, std::vector<multi_src> &y);
std::uint64_t gravity_ewald(std::vector<force> &g, const std::vector<vect<float>> &x, std::vector<source> &y);
float ewald_near_separation(const vect<float> x);
float ewald_far_separation(const vect<float> x);
void init_ewald();
