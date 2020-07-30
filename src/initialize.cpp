#include <tigergrav/initialize.hpp>
#include <tigergrav/options.hpp>
#include <tigergrav/rand.hpp>

part_vect initial_particle_set(std::string pn, int N, int Nout) {
	part_vect parts;
	parts.reserve(N);
	srand(0);
	if (pn == "cosmos") {
		for (int i = 0; i < N; i++) {
			particle p;
			for (int dim = 0; dim < NDIM; dim++) {
				p.x[dim] = double_to_pos(rand1());
				p.v[dim] = 0.0;
			}
			parts.push_back(std::move(p));
		}
	} else if (pn == "two_body") {
		parts.resize(2);
		for (int i = 0; i < 2; i++) {
			parts[i].x = double_to_pos(vect<float>(0.5));
			parts[i].v = vect<float>(0.0);
		}
		parts[0].x[0] = double_to_pos(0.75);
		parts[0].v[1] = 1.0 / std::sqrt(2);
		parts[1].x[0] = double_to_pos(0.25);
		parts[1].v[1] = -1.0 / std::sqrt(2);
	} else {
		printf("Problem %s unknown\n", pn.c_str());
		abort();
	}
	int j = 0;
	for (auto i = parts.begin(); i != parts.end(); i++, j++) {
		i->rung = 0;
		i->flags.out = j < Nout;
	}
	return parts;
}
