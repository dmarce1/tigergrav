/*
 * load.cpp
 *
 *  Created on: Aug 12, 2020
 *      Author: dmarce1
 */

#include <tigergrav/options.hpp>
#include <tigergrav/rand.hpp>

#include <tigergrav/load.hpp>

void load_header(io_header_1 *header, std::string filename) {
	int4byte dummy;
	FILE *fp = fopen(filename.c_str(), "rb");
	if (!fp) {
		printf("Unable to load %s\n", filename.c_str());
		abort();
	}
	fread(&dummy, sizeof(dummy), 1, fp);
	fread(header, sizeof(*header), 1, fp);
	fread(&dummy, sizeof(dummy), 1, fp);
	printf("Reading %lli particles\n", header->npart[1]);
	printf("Z =             %e\n", header->redshift);
	printf("particle mass = %e\n", header->mass[1]);
	printf("Omega_m =       %e\n", header->Omega0);
	printf("Omega_lambda =  %e\n", header->OmegaLambda);
	printf("Hubble Param =  %e\n", header->HubbleParam);
	fread(&dummy, sizeof(dummy), 1, fp);
	fclose(fp);

}

std::vector<particle> load_particles(std::string filename) {
	int4byte dummy;
	const auto opts = options::get();
	FILE *fp = fopen(filename.c_str(), "rb");
	if (!fp) {
		printf("Unable to load %s\n", filename.c_str());
		abort();
	}
	io_header_1 header;
	fread(&dummy, sizeof(dummy), 1, fp);
	fread(&header, sizeof(header), 1, fp);
	fread(&dummy, sizeof(dummy), 1, fp);
	printf("Reading %lli particles\n", header.npart[1]);
	printf("Z =             %e\n", header.redshift);
	printf("particle mass = %e\n", header.mass[1]);
	printf("Omega_m =       %e\n", header.Omega0);
	printf("Omega_lambda =  %e\n", header.OmegaLambda);
	printf("Hubble Param =  %e\n", header.HubbleParam);
	fread(&dummy, sizeof(dummy), 1, fp);
	std::vector<particle> parts(header.npart[1]);
//	printf( "%li\n", parts.size());
	for (int i = 0; i < header.npart[1]; i++) {
			float x, y, z;
		fread(&x, sizeof(float), 1, fp);
		fread(&y, sizeof(float), 1, fp);
		fread(&z, sizeof(float), 1, fp);
		double sep = 0.5 * std::pow(header.npart[1], -1.0 / 3.0);
		x += sep;
		y += sep;
		z += sep;
		while (x > 1.0) {
			x -= 1.0;
		}
		while (y > 1.0) {
			y -= 1.0;
		}
		while (z > 1.0) {
			z -= 1.0;
		}
		parts[i].x[0] = double_to_pos(x);
		parts[i].x[1] = double_to_pos(y);
		parts[i].x[2] = double_to_pos(z);
//		printf( "%e %e %e\n", x, y, z);
}
	fread(&dummy, sizeof(dummy), 1, fp);
	fread(&dummy, sizeof(dummy), 1, fp);
	const auto c0 = 1.0 / (1.0 + header.redshift);
	for (int i = 0; i < header.npart[1]; i++) {
		float vx, vy, vz;
		fread(&vx, sizeof(float), 1, fp);
		fread(&vy, sizeof(float), 1, fp);
		fread(&vz, sizeof(float), 1, fp);
		parts[i].v[0] = vx * std::pow(c0,1.5);
		parts[i].v[1] = vy * std::pow(c0,1.5);
		parts[i].v[2] = vz * std::pow(c0,1.5);
		parts[i].flags.rung = 0;
		parts[i].flags.out = (rand1() < (float) opts.out_parts / (float) opts.problem_size) ? 1 : 0;
//		printf( "%e %e %e\n", vx, vy, vz);
	}
	fread(&dummy, sizeof(dummy), 1, fp);
	fclose(fp);
	return parts;
}
