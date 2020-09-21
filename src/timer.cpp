#include <unordered_map>
#include <memory>
#include <hpx_lite/hpx/hpx_lite.hpp>

#include <tigergrav/timer.hpp>

using mutex_type = hpx::lcos::local::mutex;

std::unordered_map<std::string,std::shared_ptr<double>> timings;
mutex_type mtx;

timer_type timer_start() {
	return timer();
}


void timer_stop(timer_type tstart, const std::string& str) {
//	std::lock_guard<mutex_type> lock(mtx);
//	auto entry = timings.find(str);
//	if( entry == timings.end()) {
//		timings.insert(std::make_pair(str,std::make_shared(0.0)));
//		entry.first = str;
//		entry.second = std::make_shared<double>(0.0);
//		timings.insert(entry);
//	}
//	*entry.second += timer() - tstart;
}

void timer_display() {
//	for( const auto& i : timings) {
//		printf( "%32s %e\n", i.first, *i.second);
//	}
}
