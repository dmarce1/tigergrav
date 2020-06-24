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

	part_iter part_begin;
	part_iter part_end;
	bool leaf;
	range box;
	std::array<tree_ptr, NCHILD> children;

public:
	static tree_ptr new_(range, part_iter, part_iter);
	tree(range, part_iter, part_iter);
};
