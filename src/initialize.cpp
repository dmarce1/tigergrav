#include <tigergrav/initialize.hpp>
#include <tigergrav/options.hpp>
#include <tigergrav/rand.hpp>

part_vect initial_particle_set(int N) {
	part_vect parts;
	parts.reserve(N);
	for (int i = 0; i < N; i++) {
		particle p;
		for (int dim = 0; dim < NDIM; dim++) {
			p.x[dim] = double_to_pos(rand1());
			p.v[dim] = 0.0;
		}
#ifndef GLOBAL_DT
		p.rung = -1;
#endif
		parts.push_back(std::move(p));
	}
	return parts;
}
