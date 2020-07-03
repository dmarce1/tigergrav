#pragma once

#include <tigergrav/particle.hpp>

std::uint64_t gravity(std::vector<force>& g, const std::vector<vect<float>> &x, const std::vector<source> &y);
