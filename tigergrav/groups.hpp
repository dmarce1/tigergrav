#pragma once

#include <tigergrav/particle.hpp>
#include <functional>

struct group {
	int N;
	float rmax;
	float rc;
	vect<double> x;
	vect<float> v;
	vect<float> dv;
	std::vector<float> radii;
	group() {
		N = 0;
		rmax = 0.0;
		x = vect<double>(0.0);
		v = vect<float>(0.0);
		dv = vect<float>(0.0);
	}
};


void groups_add_finder(std::function<bool(void)>);
bool groups_execute_finders();
void groups_reset();
void groups_output(int i);
void groups_finish1();
void groups_add_particle1(particle p);
void groups_finish2();
void groups_add_particle2(particle p);
