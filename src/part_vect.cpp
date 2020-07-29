#include <tigergrav/part_vect.hpp>
#include <tigergrav/initialize.hpp>

static std::vector<particle> parts;

void part_vect_init() {
	static const auto opts = options::get();
	parts = initial_particle_set(opts.problem, opts.problem_size, opts.out_parts);
}

std::vector<particle> part_vect_read(part_iter b, part_iter e) {
	std::vector<particle> these_parts;
	these_parts.reserve(e - b);
	for (int i = b; i < e; i++) {
		these_parts.push_back(parts[i]);
	}
	return these_parts;
}

void part_vect_write(part_iter b, part_iter e, std::vector<particle> these_parts) {
	int i = b;
	for (auto this_part : these_parts) {
		parts[i] = this_part;
		i++;
	}
}

part_iter part_vect_sort(part_iter b, part_iter e, double xmid, int dim) {
	auto mid_iter = bisect(parts.begin() + b, parts.begin() + e, [dim, xmid](const particle &p) {
		return pos_to_double(p.x[dim]) < xmid;
	});
	return mid_iter - parts.begin();
}

range part_vect_range(part_iter b, part_iter e) {
	range r;
	for (int dim = 0; dim < NDIM; dim++) {
		r.max[dim] = 0.0;
		r.min[dim] = 1.0;
	}
	for (int i = b; i < e; i++) {
		const auto &p = parts[i];
		for (int dim = 0; dim < NDIM; dim++) {
			const auto x = pos_to_double(p.x[dim]);
			r.max[dim] = std::max(r.max[dim], x);
			r.min[dim] = std::min(r.min[dim], x);
		}
	}
	return r;
}
