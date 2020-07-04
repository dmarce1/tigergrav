#pragma once

#include <tigergrav/particle.hpp>

std::uint64_t gravity(std::vector<force> &g, const std::vector<vect<float>> &x, std::vector<source> &y);
float ewald_separation(const vect<float> x);
void init_ewald();
