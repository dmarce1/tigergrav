#include <tigergrav/options.hpp>
#include <fstream>
#include <iostream>
#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/async.hpp>
#include <boost/program_options.hpp>

options options::global;

HPX_PLAIN_ACTION(options::set, set_options_action);

options& options::get() {
	return global;
}

void options::set(options o) {
	global = o;
}

bool options::process_options(int argc, char *argv[]) {
//	std::thread([&]() {
	namespace po = boost::program_options;

	po::options_description command_opts("options");

	command_opts.add_options() //
	("help", "produce help message") //
	("config_file", po::value < std::string > (&config_file)->default_value(""), "configuration file") //
	("problem", po::value < std::string > (&problem)->default_value("cosmos"), "problem type") //
	("ewald", po::value<bool>(&ewald)->default_value(1), "periodic gravity boundary") //
	("parts_per_node", po::value<int>(&parts_per_node)->default_value(64), "maximum number of particles on a node") //
	("problem_size", po::value<int>(&problem_size)->default_value(4096), "number of particles") //
	("theta", po::value<float>(&theta)->default_value(0.7), "separation parameter") //
	("eta", po::value<float>(&eta)->default_value(0.2), "accuracy parameter") //
	("soft_len", po::value<float>(&soft_len)->default_value(-1), "softening parameter") //
	("dt_max", po::value<float>(&dt_max)->default_value(-1), "maximum timestep size") //
	("dt_stat", po::value<float>(&dt_stat)->default_value(-1), "statistics frequency") //
	("dt_out", po::value<float>(&dt_out)->default_value(-1), "output frequency") //
	("t_max", po::value<float>(&t_max)->default_value(1.0), "end time") //
			;

	boost::program_options::variables_map vm;
	po::store(po::parse_command_line(argc, argv, command_opts), vm);
	po::notify(vm);
	if (vm.count("help")) {
		std::cout << command_opts << "\n";
		return false;
	}
	if (!config_file.empty()) {
		std::ifstream cfg_fs { vm["config_file"].as<std::string>() };
		if (cfg_fs) {
			po::store(po::parse_config_file(cfg_fs, command_opts), vm);
		} else {
			printf("Configuration file %s not found!\n", config_file.c_str());
			return false;
		}
	}
	po::notify(vm);

	const auto loc = hpx::find_all_localities();
	const auto sz = loc.size();
	std::vector<hpx::future<void>> futs;
	if (soft_len == -1) {
		soft_len = 0.02 * std::pow(problem_size, -1.0 / 3.0);
	}
	if (problem == "two_body") {
		problem_size = 2;
	}
	if( dt_max < 0.0) {
		dt_max = t_max / 100.0;
	}
	if( dt_stat < 0.0) {
		dt_stat = dt_max;
	}
	if( dt_out < 0.0) {
		dt_out = dt_max;
	}
	set(*this);
	for (int i = 1; i < sz; i++) {
		futs.push_back(hpx::async < set_options_action > (loc[i], *this));
	}
	hpx::wait_all(futs);
#define SHOW( opt ) std::cout << std::string( #opt ) << " = " << std::to_string(opt) << '\n';
#define SHOW_STR( opt ) std::cout << std::string( #opt ) << " = " << opt << '\n';
	SHOW(dt_max);
	SHOW(dt_out);
	SHOW(dt_stat);
	SHOW(ewald);
	SHOW(eta);
	SHOW(soft_len);
	SHOW(parts_per_node);
	SHOW_STR(problem);
	SHOW(problem_size);
	SHOW(t_max);
	SHOW(theta);
	return true;
}
