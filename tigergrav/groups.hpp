#pragma once

#include <tigergrav/particle.hpp>

struct group {
	int N;
	float W;
	float T;
	vect<float> x;
	vect<float> v;

	group() {
		N = 0;
		W = 0.0;
		T = 0.0;
		x = vect<float>(0.0);
		v = vect<float>(0.0);
	}
};

struct gmember {
	vect<float> x;
	vect<float> v;
	float phi;
	std::uint64_t id;

	gmember(particle p, float phi_) {
		x = pos_to_double(p.x);
		v = p.v;
		phi = phi_;
		id = p.flags.group;
	}
	gmember() = default;

	template<class A>
	void serialize(A&& arc, unsigned) {
		arc & x;
		arc & v;
		arc & phi;
		arc & id;
	}
};

void groups_reset();
void groups_output(int i);
void groups_finish1();
void groups_add_particle1(gmember p);
