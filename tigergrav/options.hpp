#pragma once
#include <string>

class options {
public:
	std::string config_file;
	std::string problem;
	int parts_per_node;
	int problem_size;
	float theta;
	float eta;
	float h;
	float dt_max;
	float t_max;


	template<class Arc>
	void serialize(Arc &arc, unsigned) {
		arc & config_file;
		arc & problem;
		arc & parts_per_node;
		arc & problem_size;
		arc & theta;
		arc & eta;
		arc & h;
		arc & dt_max;
		arc & t_max;
	}
	static options global;
	static options& get();
	static void set(options);
	bool process_options(int argc, char *argv[]);
};
