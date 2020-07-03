#include <tigergrav/gravity.hpp>
#include <tigergrav/options.hpp>

std::uint64_t gravity(std::vector<force> &f, const std::vector<vect<float>> &x, const std::vector<source> &y) {
	std::uint64_t flop = 0;
	static const auto h = options::get().soft_len;
	static const auto h3inv = 1.0 / (h * h * h);
	static const auto h2 = h * h;
	for (int i = 0; i < x.size(); i++) {
		auto &g = f[i].g;
		auto &phi = f[i].phi;
		phi = 0.0;
		g = vect<float>(0.0);
		for (int j = 0; j < y.size(); j++) {
			const vect<float> dx = x[i] - y[j].x;                  // 3 OP
			const float m = y[j].m;
			const float r2 = dx.dot(dx);                           // 5 OP
			const float rinv = 1.0 / std::sqrt(r2 + h2);           // 3 OP
			const float rinv3 = rinv * rinv * rinv;                // 2 OP
			g -= dx * (m * rinv3);                                 // 7 OP
			phi -= m * rinv;                                       // 2 OP
			flop += 22;
		}
	}
	return flop;
}
