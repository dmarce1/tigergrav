#include <tigergrav/options.hpp>
#include <tigergrav/cosmo.hpp>
#include <fstream>
#include <iostream>

#ifdef HPX_LITE
#include <hpx/hpx_lite.hpp>
#else
#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/async.hpp>
#endif

#include <tigergrav/load.hpp>

HPX_PLAIN_ACTION(options::set, set_options_action);
#include <boost/program_options.hpp>

options options::global;

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
	("config_file", po::value<std::string>(&config_file)->default_value(""), "configuration file") //
	("init_file", po::value<std::string>(&init_file)->default_value(""), "initialization file from N-GenIC") //
	("problem", po::value<std::string>(&problem)->default_value("cosmos"), "problem type") //
	("solver_test", po::value<bool>(&solver_test)->default_value(0), "test gravity solver") //
	("cosmic", po::value<bool>(&cosmic)->default_value(1), "Use Friedman equation") //
	("glass", po::value<bool>(&glass)->default_value(0), "Produce glass file") //
	("ewald", po::value<bool>(&ewald)->default_value(1), "periodic gravity boundary") //
	("out_parts", po::value<int>(&out_parts)->default_value(-1), "number of particles for output file") //
	("nout", po::value<int>(&nout)->default_value(64), "number of outputs") //
	("parts_per_node", po::value<int>(&parts_per_node)->default_value(64), "maximum number of particles on a node") //
	("problem_size", po::value<std::uint64_t>(&problem_size)->default_value(4096), "number of particles") //
	("theta", po::value<double>(&theta)->default_value(0.5), "separation parameter") //
	("box_size", po::value<double>(&box_size)->default_value(DEFAULT_BOX_SIZE), "size of box in centimeters") //
	("eta", po::value<double>(&eta)->default_value(0.2), "accuracy parameter") //
	("soft_len", po::value<double>(&soft_len)->default_value(-1), "softening parameter") //
	("dt_max", po::value<double>(&dt_max)->default_value(-1), "maximum timestep size") //
	("z0", po::value<double>(&z0)->default_value(50.0), "") //
	("t_max", po::value<double>(&t_max)->default_value(-1.0), "end time") //
	("m_tot", po::value<double>(&m_tot)->default_value(-1.0), "total mass") //
	("omega_lambda", po::value<double>(&omega_lambda)->default_value(0.7), "Omega_lambda parameter") //
	("omega_m", po::value<double>(&omega_m)->default_value(0.3), "Omega_m parameter (used to set mtot if mtot not set)") //
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

	if (soft_len == -1) {
		soft_len = 0.02 * std::pow(problem_size, -1.0 / 3.0);
	}
	if (out_parts < 0) {
		out_parts = problem_size;
	}
	const auto loc = hpx::find_all_localities();
	const auto sz = loc.size();
	std::vector<hpx::future<void>> futs;
	set(*this);
	if (glass) {
		cosmic = false;
	}
	if (!cosmic && t_max < 0.0) {
		t_max = 1.0;
	}
	if (init_file == "") {
		if (m_tot < 0.0) {
			const auto H0 = box_size / DEFAULT_BOX_SIZE;
			m_tot = 0.5 * omega_m * H0 * H0;
		}
		if (cosmic) {
			if (t_max < 0.0) {
				cosmos c;
				t_max = -c.advance_to_scale(1.0 / (1.0 + z0));
//		printf( "%e %e\n", t_max, c.get_scale());
			}
		}
	} else {
		io_header_1 header;
		load_header(&header, init_file);
		omega_m = header.Omega0;
		z0 = header.redshift;
		omega_lambda = header.OmegaLambda;
		const auto H0 = box_size / DEFAULT_BOX_SIZE;
		m_tot = 0.5 * omega_m * H0 * H0;
		problem_size = header.npartTotal[1] + (header.npartTotal[2] * ((std::uint64_t) 1 << (std::uint64_t) 32));
		out_parts = problem_size;
		cosmos c;
		set(*this);
		t_max = -c.advance_to_scale(1.0 / (1.0 + z0));
	}
	if (dt_max < 0.0) {
		dt_max = t_max / 64.0;
	}
	set(*this);
	for (int i = 1; i < sz; i++) {
		futs.push_back(hpx::async<set_options_action>(loc[i], *this));
	}
	hpx::wait_all(futs.begin(), futs.end());
#define SHOW( opt ) std::cout << std::string( #opt ) << " = " << std::to_string(opt) << '\n';
#define SHOW_STR( opt ) std::cout << std::string( #opt ) << " = " << opt << '\n';
	SHOW(cosmic);
	SHOW(dt_max);
	SHOW(ewald);
	SHOW(eta);
	SHOW(m_tot);
	SHOW(omega_lambda);
	SHOW(omega_m);
	SHOW(out_parts);
	SHOW(parts_per_node);
	SHOW_STR(problem);
	SHOW(problem_size);
	SHOW(soft_len);
	SHOW(solver_test);
	SHOW(t_max);
	SHOW(theta);
	SHOW(z0);
	return true;
}
