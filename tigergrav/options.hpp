#pragma once
#include <string>

class options {
public:
	std::string config_file;
	std::string problem;
	bool ewald;
	bool gravity;
	bool solver_test;
	int out_parts;
	int parts_per_node;
	int nout;
	std::uint64_t problem_size;
	bool cosmic;
	bool glass;
	bool groups;
	double theta;
	double eta;
	double soft_len;
	double dt_max;
	double t_max;
	double m_tot;
	double z0;
	double G;
	double H0;
	double clight;
	double omega_m;
	double omega_lambda;
	double code_to_cm;
	double code_to_cm_per_s;
	double code_to_s;
	double code_to_g;
	double link_len;
	std::string init_file;

	template<class Arc>
	void serialize(Arc &arc, unsigned) {
		arc & gravity;
		arc & groups;
		arc & link_len;
		arc & H0;
		arc & G;
		arc & clight;
		arc & code_to_cm_per_s;
		arc & code_to_cm;
		arc & code_to_s;
		arc & code_to_g;
		arc & glass;
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
