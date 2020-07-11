#pragma once

#include <tigergrav/vect.hpp>

#include <vector>

struct output {
	vect<double> x;
	vect<float> v;
	vect<float> g;
	float phi;
	int rung;
};


void output_particles(const std::vector<output>& parts, const std::string filename);
