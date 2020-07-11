/*
 * bin2silo.cpp
 *
 *  Created on: Jul 5, 2020
 *      Author: dmarce1
 */

#include <tigergrav/options.hpp>
#include <tigergrav/output.hpp>

#include <hpx/hpx_init.hpp>

#include <silo.h>

#include <fenv.h>



void output_particles(const std::vector<output>& parts, const std::string filename) {
	static const auto opts = options::get();
	static const float m = 1.0 / opts.problem_size;
	std::array<std::vector<double>, NDIM> x;
	std::array<std::vector<float>, NDIM> g;
	std::array<std::vector<float>, NDIM> v;
	std::vector<float> phi;
	std::vector<int> rung;

	for( auto i = parts.begin(); i != parts.end(); i++) {
		for (int dim = 0; dim < NDIM; dim++) {
			x[dim].push_back(i->x[dim]);
			g[dim].push_back(i->g[dim]);
			v[dim].push_back(i->v[dim]);
		}
		rung.push_back(i->rung);
		phi.push_back(i->phi);
	}
	DBfile *db = DBCreateReal(filename.c_str(), DB_CLOBBER, DB_LOCAL, "Meshless", DB_HDF5);
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
}
