#pragma once

#include <tigergrav/particle.hpp>

std::uint64_t gravity_direct(std::vector<force> &g, const std::vector<vect<float>> &x, std::vector<source> &y);
std::uint64_t gravity_ewald(std::vector<force> &g, const std::vector<vect<float>> &x, std::vector<source> &y);
float ewald_separation(const vect<float> x);
void init_ewald();
