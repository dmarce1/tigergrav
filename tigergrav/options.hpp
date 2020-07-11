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
	float theta;
	float eta;
	float soft_len;
	float dt_out;
	float dt_max;
	float t_max;



	template<class Arc>
	void serialize(Arc &arc, unsigned) {
		arc & solver_test;
		arc & solver_type;
		arc & out_parts;
		arc & ewald;
		arc & config_file;
		arc & problem;
		arc & parts_per_node;
		arc & problem_size;
		arc & theta;
		arc & eta;
		arc & soft_len;
		arc & dt_max;
		arc & t_max;
		arc & dt_out;
	}
	static options global;
	static options& get();
	static void set(options);
	bool process_options(int argc, char *argv[]);
};
