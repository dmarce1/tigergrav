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
	int problem_size;
	int solver_type;
	double theta;
	double eta;
	double soft_len;
	double dt_out;
	double dt_max;
	double t_max;

	template<class Arc>
	void serialize(Arc &arc, unsigned) {
		arc &  config_file;
		arc & problem;
		arc & ewald;
		arc & solver_test;
		arc & out_parts;
		arc & parts_per_node;
		arc & problem_size;
		arc & solver_type;
		arc & theta;
		arc & eta;
		arc & soft_len;
		arc & dt_out;
		arc & dt_max;
		arc & t_max;
	}
	static options global;
	static options& get();
	static void set(options);
	bool process_options(int argc, char *argv[]);
};
