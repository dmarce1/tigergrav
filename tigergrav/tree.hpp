#pragma once

#include <tigergrav/options.hpp>
#include <tigergrav/particle.hpp>
#include <tigergrav/range.hpp>
#include <tigergrav/tree_id.hpp>

#include <array>
#include <memory>

class tree;

using tree_ptr = std::shared_ptr<tree>;


class tree {
	monopole mono;
	part_iter part_begin;
	part_iter part_end;
	bool leaf;
	std::array<tree_ptr, NCHILD> children;

public:
	static tree_ptr new_(range, part_iter, part_iter);
	tree(range, part_iter, part_iter);
	monopole compute_monopoles();
	monopole get_monopole() const;
	bool is_leaf() const;
	std::array<tree_ptr, NCHILD> get_children() const;
	std::vector<vect<float>> get_positions() const;
	void drift(float);
	void output(float,int);
#ifdef GLOBAL_DT
	void kick(float);
	float compute_gravity(std::vector<tree_ptr> checklist, std::vector<source> sources);
#else
	std::int8_t kick(std::vector<tree_ptr> checklist, std::vector<source> sources, std::int8_t min_rung);
#endif
};
