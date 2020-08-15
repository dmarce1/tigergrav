#include <tigergrav/initialize.hpp>
#include <tigergrav/options.hpp>
#include <tigergrav/rand.hpp>

#ifdef HPX_LITE
#include <hpx/hpx_lite.hpp>
#else
#include <hpx/include/components.hpp>
#endif

#include <tigergrav/options.hpp>
#include <tigergrav/load.hpp>

part_vect initial_particle_set(std::string pn, int N, int Nout) {
	const auto opts = options::get();
	part_vect parts;
	if (opts.init_file == "") {
		parts.reserve(N);
		srand(hpx::get_locality_id() * 0xA3CF98A7);
		if (pn == "cosmos") {
			for (int i = 0; i < N; i++) {
				particle p;
				for (int dim = 0; dim < NDIM; dim++) {
					p.x[dim] = double_to_pos(rand1());
					p.v[dim] = 0.0;
				}
				parts.push_back(std::move(p));
			}
		} else if (pn == "grid") {
			auto nx = std::pow(N, 1.0 / 3.0) + 1;
			if (nx * nx * nx > N) {
				nx--;
			}
			if (nx * nx * nx != N) {
				printf("Problem size must be 2^n\n");
				abort();
			}
			const auto xmax = std::numeric_limits<pos_type>::max();
			const auto dx = xmax / nx;
			int l = 0;
			parts.resize(N);
			for (int i = 0; i < nx; i++) {
				for (int j = 0; j < nx; j++) {
					for (int k = 0; k < nx; k++) {
						parts[l].x[0] = i * dx + rand1() * dx * 0.5;
						parts[l].x[1] = j * dx + rand1() * dx * 0.5;
						parts[l].x[2] = k * dx + rand1() * dx * 0.5;
						parts[l++].v = vect<float>(0.0);
					}
				}
			}
		} else if (opts.problem == "two_body") {
			parts.resize(2);
			parts[0].v = parts[1].v = vect<float>(0.0);
			parts[0].x[0] = double_to_pos(0.25);
			parts[1].x[0] = double_to_pos(0.75);
			parts[0].x[1] = double_to_pos(0.5);
			parts[1].x[1] = double_to_pos(0.5);
			parts[0].x[2] = double_to_pos(0.5);
			parts[1].x[2] = double_to_pos(0.5);
		} else {
			printf("Problem %s unknown\n", pn.c_str());
			abort();
		}
		int j = 0;
		for (auto i = parts.begin(); i != parts.end(); i++, j++) {
			i->flags.rung = 0;
			i->flags.out = j < Nout;
		}

	} else {
		parts = load_particles(opts.init_file);
		if (parts.size() != N) {
			printf("Size mismatch in initialize.cpp %li %i\n", parts.size(), N);
		}
	}
	return parts;
}
