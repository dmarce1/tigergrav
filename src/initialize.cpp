#include <tigergrav/initialize.hpp>
#include <tigergrav/options.hpp>
#include <tigergrav/rand.hpp>


part_vect initial_particle_set(int N) {
	part_vect parts;
	parts.reserve(N);
	for( int i = 0; i < N; i++) {
		particle p;
		for( int dim = 0; dim < NDIM; dim++) {
			p.x[dim] = rand1();
			p.v[dim] = 0.0;
		}
		p.dt = 0.0;
		p.t = 0.0;
		parts.push_back(std::move(p));
	}
	return parts;
}
