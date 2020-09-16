/*
 * gravity_cuda.hpp
 *
 *  Created on: Sep 1, 2020
 *      Author: dmarce1
 */

#ifndef TIGERGRAV_GRAVITY_CUDA_HPP_
#define TIGERGRAV_GRAVITY_CUDA_HPP_

#include <tigergrav/gravity.hpp>
#include <tigergrav/particle.hpp>

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

#include <cuda_runtime.h>

template<class T>
class pinned_vector {
	T *ptr;
	std::size_t sz;
	std::size_t cap;
	void free() {
		if (ptr) {
			cudaFreeHost(ptr);
		}
	}
	void allocate() {
		cudaMallocHost((void**) &ptr, sizeof(T) * cap);
	}
public:
	pinned_vector() {
		cap = 0;
		sz = 0;
		ptr = nullptr;
	}
	pinned_vector(std::size_t this_sz) {
		cap = this_sz;
		sz = this_sz;
		allocate();
	}
	pinned_vector& operator=(pinned_vector &&other) {
		cap = other.cap;
		sz = other.sz;
		ptr = other.ptr;
		other.cap = 0;
		other.sz = 0;
		other.ptr = nullptr;
		return *this;
	}
	pinned_vector(pinned_vector &&other) {
		(*this) = std::move(other);
	}
	void resize(std::size_t new_sz) {
		if (new_sz > cap) {
			T *new_ptr;
			cudaMallocHost((void**) &new_ptr, sizeof(T) * new_sz);
			if (sz) {
				std::memcpy(new_ptr, ptr, sz * sizeof(T));
			}
			free();
			ptr = new_ptr;
			cap = new_sz;
			sz = new_sz;
		} else {
			sz = new_sz;
		}
	}
	void reserve(std::size_t new_cap) {
		if (cap < new_cap) {
			auto old_sz = sz;
			resize(new_cap);
			sz = old_sz;
		}
	}
	void push_back(T &&v) {
		if (sz + 1 >= cap) {
			reserve(std::max(2 * cap, (std::size_t) 1));
		}
		ptr[sz] = std::move(v);
		sz++;
	}
	void push_back(const T &v) {
		if (sz + 1 >= cap) {
			reserve(std::max(2 * cap, (std::size_t) 1));
		}
		ptr[sz] = v;
		sz++;
	}
	T* data() {
		return ptr;
	}
	const T* data() const {
		return ptr;
	}
	std::size_t size() const {
		return sz;
	}
	T operator[](int i) const {
		return ptr[i];
	}
	T& operator[](int i) {
		return ptr[i];
	}
	T* begin() {
		return ptr;
	}
	T* end() {
		return ptr + sz;
	}
//	template<class I>
//	void insert(T *start, I begin, I end) {
//		printf("Insert begin\n");
//		const std::size_t rsz = (start - ptr) + (end - begin);
//		if (rsz > cap) {
//			reserve(rsz);
//		}
//		for (auto i = begin; i != end; i++) {
//			*(start + std::distance(i, begin)) = *i;
//		}
//		sz = std::max(sz, rsz);
//		printf("Insert end\n");
//	}
};

#endif /* TIGERGRAV_GRAVITY_CUDA_HPP_ */
