#pragma once


#include <tigergrav/expansion.hpp>
#include <tigergrav/multipole.hpp>
#include <tigergrav/particle.hpp>


struct mono_source {
	float m;
	vect<float> x;
};

struct multi_source {
	multipole<float> m;
	vect<float> x;
};

struct monopole {
	float m;
	vect<float> x;
	float r;
};

std::uint64_t gravity_mono_mono(std::vector<force> &g, const std::vector<vect<pos_type>> &x, std::vector<vect<pos_type>> &y, const bool do_phi);
std::uint64_t gravity_mono_multi(std::vector<force> &g, const std::vector<vect<pos_type>> &x, std::vector<multi_source> &y, const bool do_phi);
std::uint64_t gravity_multi_multi(expansion<double> &, const vect<float> &x, std::vector<multi_source> &y);
std::uint64_t gravity_multi_mono(expansion<double> &, const vect<float> &x, std::vector<vect<pos_type>> &y);
std::uint64_t gravity_ewald(std::vector<force> &g, const std::vector<vect<float>> &x, std::vector<mono_source> &y, const bool do_phi);
float ewald_near_separation(const vect<float> x);
float ewald_far_separation(const vect<float> x);
void init_ewald();
