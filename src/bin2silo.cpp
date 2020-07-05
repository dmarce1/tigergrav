/*
 * bin2silo.cpp
 *
 *  Created on: Jul 5, 2020
 *      Author: dmarce1
 */

#include <tigergrav/output.hpp>

#include <hpx/hpx_init.hpp>

#include <silo.h>

#include <fenv.h>

int hpx_main(int argc, char *argv[]) {

	std::array<std::vector<double>, NDIM> x;
	std::array<std::vector<float>, NDIM> g;
	std::array<std::vector<float>, NDIM> v;
	std::vector<float> phi;
	std::vector<int> rung;

	if (argc < 3) {
		printf("Usage: bin2silo <in> <out>\n");
		return hpx::finalize();
	}

	FILE *fp = fopen(argv[1], "rb");
	if (!fp) {
		printf("Unable to open %s\n", argv[1]);
		return hpx::finalize();
	}

	output part;
	while (fread(&part, sizeof(output), 1, fp) == 1) {
		for (int dim = 0; dim < NDIM; dim++) {
			x[dim].push_back(part.x[dim]);
			g[dim].push_back(part.g[dim]);
			v[dim].push_back(part.v[dim]);
		}
		rung.push_back(part.rung);
		phi.push_back(part.phi);
	}
	fclose(fp);
	printf( "Read %i\n", phi.size());
	DBfile *db = DBCreateReal(argv[2], DB_CLOBBER, DB_LOCAL, "Meshless", DB_HDF5);
	const int nparts = phi.size();
	double *coords[NDIM] = { x[0].data(), x[1].data(), x[2].data() };
	DBPutPointmesh(db, "points", NDIM, coords, nparts, DB_DOUBLE, NULL);
	for (int dim = 0; dim < NDIM; dim++) {
		std::string nm = std::string() + "v_" + char('x' + char(dim));
		DBPutPointvar1(db, nm.c_str(), "points", v[dim].data(), nparts, DB_FLOAT, NULL);
	}
	for (int dim = 0; dim < NDIM; dim++) {
		std::string nm = std::string() + "g_" + char('x' + char(dim));
		DBPutPointvar1(db, nm.c_str(), "points", g[dim].data(), nparts, DB_FLOAT, NULL);
	}
	DBPutPointvar1(db, "phi", "points", phi.data(), nparts, DB_FLOAT, NULL);
	DBPutPointvar1(db, "rung", "points", rung.data(), nparts, DB_INT, NULL);
	return hpx::finalize();
}

int main(int argc, char *argv[]) {

	std::vector<std::string> cfg = { "hpx.commandline.allow_unknown=1" };

	hpx::init(argc, argv, cfg);
}

