/*
 * gravity_cuda.hpp
 *
 *  Created on: Sep 1, 2020
 *      Author: dmarce1
 */

#ifndef TIGERGRAV_GRAVITY_CUDA_HPP_
#define TIGERGRAV_GRAVITY_CUDA_HPP_

#include <tigergrav/cuda_check.hpp>
#include <tigergrav/gravity.hpp>
#include <tigergrav/particle.hpp>
#include <tigergrav/pinned_vector.hpp>


#define EWALD_NFOUR 80
#define EWALD_NREAL 171

double cuda_reset_flop();

struct cuda_ewald_const {
	vect<float> four_indices[EWALD_NFOUR];
	vect<float> real_indices[EWALD_NREAL];
	expansion<float> periodic_parts[EWALD_NFOUR];
	expansion<float> exp_factors;
};

struct cuda_work_unit {
	std::vector<vect<pos_type>> y;
	std::vector<std::pair<part_iter, part_iter>> yiters;
	std::vector<const multi_src*> z;
	std::vector<vect<pos_type>> *xptr;
	std::vector<force> *fptr;
};

//
//
//std::uint64_t gravity_PP_direct_cuda(std::vector<force> &f, const std::vector<vect<pos_type>> &x, std::vector<std::pair<part_iter,part_iter>>& yiter, bool do_phi);
//
void cuda_copy_particle_image(part_iter part_begin, part_iter part_end, const std::vector<vect<pos_type>> &parts);

bool cuda_thread_count();

template<class T>
class pinned_vector;

void gravity_PP_direct_cuda(std::vector<cuda_work_unit>&&, const pinned_vector<vect<pos_type>>&, bool do_phi);
void gravity_CC_ewald_cuda(expansion<float> &L, const vect<pos_type> &x, std::vector<const multi_src*> &y, bool do_phi);

#endif /* TIGERGRAV_GRAVITY_CUDA_HPP_ */
