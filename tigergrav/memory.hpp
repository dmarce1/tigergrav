/*
 * memory.hpp
 *
 *  Created on: Sep 6, 2020
 *      Author: dmarce1
 */

#ifndef TIGERGRAV_MEMORY_HPP_
#define TIGERGRAV_MEMORY_HPP_

#include <atomic>
#include <stack>
#include <vector>

template<class T>
class memory {
	static std::atomic<int> mtx;
	static std::stack<std::vector<T>*> stack;

	void lock() const {
		while (mtx++ != 0) {
			mtx--;
		}
	}

	void unlock() const {
		mtx--;
	}
public:

	std::vector<T>* get_vector() const {
		std::vector<T>* ptr;
		lock();
		if (stack.empty()) {
			ptr = new std::vector<T>;
		} else {
			ptr = stack.top();
			stack.pop();
		}
		unlock();
		return ptr;
	}

	void trash_vector(std::vector<T> *ptr) const {
		lock();
		stack.push(ptr);
		unlock();
	}

};

template<class T>
std::atomic<int> memory<T>::mtx(0);

template<class T>
std::stack<std::vector<T>*> memory<T>::stack;

#endif /* TIGERGRAV_MEMORY_HPP_ */
