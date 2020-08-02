/*
 * mutex.cpp
 *
 *  Created on: Dec 11, 2015
 *      Author: dmarce1
 */

#include "hpx/mutex.hpp"
#include "hpx/thread.hpp"

namespace hpx {

void mutex::lock() {
	while (locked++ != 0) {
		hpx::this_thread::yield();
	}
}

void mutex::unlock() {
	locked = 0;
}

void spinlock::lock() {
	while (locked++ != 0) {
		hpx::this_thread::yield();
	}
}

void spinlock::unlock() {
	locked = 0;
}

}
