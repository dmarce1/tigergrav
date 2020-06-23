#pragma once
#include <string>

class options {
public:
	std::string config_file;
	int parts_per_node;
	int problem_size;

	template<class Arc>
	void serialize(Arc &arc, unsigned) {
		arc & config_file;
		arc & parts_per_node;
		arc & problem_size;
	}
	static options global;
	static options& get();
	static void set(options);
	bool process_options(int argc, char *argv[]);
};
