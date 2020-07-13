#pragma once

#include <tigergrav/defs.hpp>
#include <tigergrav/simd.hpp>
#include <tigergrav/expansion.hpp>
#include <tigergrav/multipole.hpp>
#include <tigergrav/particle.hpp>

#include <hpx/lcos/promise.hpp>

#include <memory>
#include <map>

struct multi_src {
	multipole<ireal> m;
	vect<ireal> x;
};

inline bool operator<(const const_part_set &s1, const const_part_set &s2) {
	return s1.first < s2.first;
}

using sink_vector = std::vector<vect<float>>;
using force_vector = std::vector<force>;
using sink_pair = std::pair<const sink_vector*,force_vector*>;
using source_map_type = std::map<const_part_set,std::shared_ptr<std::vector<sink_pair>>>;

struct grav_work_unit {
	source_map_type source_sets;
	std::vector<std::shared_ptr<hpx::promise<void>>> promises;
	hpx::future<void> add_work(std::vector<force> &g, const sink_vector &x, std::vector<const_part_set> &part_sets) {
		for (auto this_set : part_sets) {
			auto i = source_sets.find(this_set);
			if (i == source_sets.end()) {
				i = source_sets.insert(std::make_pair(this_set, std::make_shared<std::vector<sink_pair>>())).first;
			}
			i->second->push_back(std::make_pair(&x, &g));
		}
		promises.push_back(std::make_shared<hpx::promise<void>>());
		return promises.back()->get_future();
	}
	void compute() {
		static const auto opts = options::get();
		static const bool ewald = opts.ewald;
		static const auto h = opts.soft_len;
		static const auto m = 1.0 / opts.problem_size;


		for (auto entry : source_sets) {
			const_part_set source_set = entry.first;
			auto &sinks = *(entry.second);
			for (auto this_sink : sinks) {
				const auto &x = this_sink.first;
				auto &f = this_sink.second;
				for (auto j = source_set.first; j != source_set.second; j++) {
					const auto& x =(*this_sink.first);
					for (int i = 0; i < x.size(); i++) {
						const auto X = pos_to_double(x[i]);
						const auto Y = pos_to_double(j->x);
						const auto dX = X - Y;
						const auto r2 = dX.dot(dX);
						const auto rinv = 1.0 / std::sqrt(r2 + h*h);
						const auto rinv3 = rinv * rinv * rinv;
						(*this_sink.second)[i].g -= dX * rinv3;
						(*this_sink.second)[i].phi -= rinv;
					}
				}
			}
		}
		for( auto& p : promises) {
			p->set_value();
		}
	}
};

std::uint64_t gravity_PP(std::vector<force> &g, const std::vector<vect<float>> &x, std::vector<const_part_set> &part_sets);
std::uint64_t gravity_PC(std::vector<force> &g, const std::vector<vect<float>> &x, std::vector<multi_src> &y);
std::uint64_t gravity_CC(expansion<ireal>&, const vect<ireal> &x, std::vector<multi_src> &y);
std::uint64_t gravity_CP(expansion<ireal> &L, const vect<ireal> &x, std::vector<vect<float>> &y);
std::uint64_t gravity_PP_ewald(std::vector<force> &g, const std::vector<vect<float>> &x, std::vector<source> &y);
std::uint64_t gravity_CP_ewald(expansion<ireal> &L, const vect<float> &x, std::vector<source> &y);
double ewald_near_separation(const vect<double> x);
double ewald_far_separation(const vect<double> x);
void init_ewald();
