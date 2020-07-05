#pragma once

#include <tigergrav/vect.hpp>

struct output {
	vect<double> x;
	vect<float> v;
	vect<float> g;
	float phi;
	int rung;
};
