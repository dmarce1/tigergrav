#pragma once
#include <string>

class options {
public:
	std::string config_file;
	std::string problem;
	bool ewald;
	bool solver_test;
	int out_parts;
	int parts_per_node;
	int nout;
	std::uint64_t problem_size;
	bool cosmic;
	double theta;
	double eta;
	double soft_len;
	double dt_max;
	double t_max;
	double m_tot;
	double z0;
	double omega_m;
	double omega_lambda;
	std::string init_file;

	template<class Arc>
	void serialize(Arc &arc, unsigned) {
		arc & cosmic;
		arc & init_file;
		arc & nout;
		arc & omega_lambda;
		arc & omega_m;
		arc & z0;
		arc & config_file;
		arc & problem;
		arc & ewald;
		arc & solver_test;
		arc & out_parts;
		arc & parts_per_node;
		arc & problem_size;
		arc & theta;
		arc & eta;
		arc & soft_len;
		arc & dt_max;
		arc & t_max;
		arc & m_tot;
	}
	static options global;
	static options& get();
	static void set(options);
	bool process_options(int argc, char *argv[]);
};
