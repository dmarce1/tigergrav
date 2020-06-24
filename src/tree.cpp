#include <tigergrav/options.hpp>
#include <tigergrav/tree.hpp>

tree_ptr tree::new_(range r, part_iter b, part_iter e) {
	return std::make_shared<tree>(r, b, e);
}

tree::tree(range r, part_iter b, part_iter e) {
	const auto &opts = options::get();
	box = r;
	part_begin = b;
	part_end = e;
	if (e - b > opts.parts_per_node) {
		leaf = false;
		float max_span = 0.0;
		int max_dim;
		for (int dim = 0; dim < NDIM; dim++) {
			const auto this_span = box.max[dim] - box.min[dim];
			if (this_span > max_span) {
				max_span = this_span;
				max_dim = dim;
			}
		}
//		printf("%li %li %e\n", e - b, max_dim, max_span);
		range boxl = box;
		range boxr = box;
		float mid = (box.max[max_dim] + box.min[max_dim]) * 0.5;
		boxl.max[max_dim] = boxr.min[max_dim] = mid;
		const auto mid_iter = bisect(b, e, [max_dim, mid](const particle &p) {
			return (float) p.x[max_dim] < mid;
		});
		children[0] = new_(boxl, b, mid_iter);
		children[1] = new_(boxr, mid_iter, e);
	} else {
		leaf = true;
	}
}
