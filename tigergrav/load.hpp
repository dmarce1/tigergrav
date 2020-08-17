/*
 * load.hpp
 *
 *  Created on: Aug 12, 2020
 *      Author: dmarce1
 */

#ifndef TIGERGRAV_LOAD_HPP_
#define TIGERGRAV_LOAD_HPP_


#include <tigergrav/particle.hpp>


/**** header from N-GenIC*****/

typedef int int4byte;
typedef unsigned int uint4byte;

struct io_header_1 {
	uint4byte npart[6]; /*!< npart[1] gives the number of particles in the present file, other particle types are ignored */
	double mass[6]; /*!< mass[1] gives the particle mass */
	double time; /*!< time (=cosmological scale factor) of snapshot */
	double redshift; /*!< redshift of snapshot */
	int4byte flag_sfr; /*!< flags whether star formation is used (not available in L-Gadget2) */
	int4byte flag_feedback; /*!< flags whether feedback from star formation is included */
	uint4byte npartTotal[6]; /*!< npart[1] gives the total number of particles in the run. If this number exceeds 2^32, the npartTotal[2] stores
	 the result of a division of the particle number by 2^32, while npartTotal[1] holds the remainder. */
	int4byte flag_cooling; /*!< flags whether radiative cooling is included */
	int4byte num_files; /*!< determines the number of files that are used for a snapshot */
	double BoxSize; /*!< Simulation box size (in code units) */
	double Omega0; /*!< matter density */
	double OmegaLambda; /*!< vacuum energy density */
	double HubbleParam; /*!< little 'h' */
	int4byte flag_stellarage; /*!< flags whether the age of newly formed stars is recorded and saved */
	int4byte flag_metals; /*!< flags whether metal enrichment is included */
	int4byte hashtabsize; /*!< gives the size of the hashtable belonging to this snapshot file */
	char fill[84]; /*!< fills to 256 Bytes */
};

void load_header(io_header_1* header, std::string filename);
std::vector<particle> load_particles(std::string filename);




#endif /* TIGERGRAV_LOAD_HPP_ */
