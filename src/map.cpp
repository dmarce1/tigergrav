/*
 * map.cpp
 *
 *  Created on: Aug 30, 2020
 *      Author: dmarce1
 */

#include <tigergrav/cosmo.hpp>
#include <tigergrav/map.hpp>
#include <tigergrav/options.hpp>
#include <cmath>

#ifdef HPX_LITE
#include <hpx/hpx_lite.hpp>
#else
#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/async.hpp>
#endif

#include <unordered_map>

#include <silo.h>

static int NX;
static int NY;

using mutex_type = hpx::lcos::local::spinlock;

static std::vector<std::vector<std::vector<std::shared_ptr<mutex_type>>>> mtx;
static std::vector<std::vector<std::vector<double>>> pixels;

#define NT 1024
static std::array<double, NT + 1> t_table;
static std::array<double, NT + 1> dt_table;
static double tau_max;

double t_of_tau(double tau) {
	const auto dtau = tau_max / NT;
	const int i1 = std::min(tau / dtau, (double) (NT - 1));
	const int i2 = i1 + 1;
	const double w2 = (tau - i1 * dtau) / dtau;
	const double w1 = 1.0 - w2;
	const double y1 = t_table[i1];
	const double y2 = t_table[i2];
	const double k1 = dt_table[i1];
	const double k2 = dt_table[i2];
	const double a = k1 * dtau - (y2 - y1);
	const double b = -k2 * dtau + (y2 - y1);
	return w1 * y1 + w2 * y2 + w1 * w2 * (w1 * a + w2 * b);
}

void map_init() {
	const auto opts = options::get();
	cosmos cinit;
	cinit.advance_to_scale(1.0 / (1.0 + opts.z0));
	double a0 = cinit.get_scale();
	double adot0 = cinit.get_Hubble() * a0;
	tau_max = -cinit.get_tau();
	double t = 0.0;
	t_table[0] = 0.0;
	dt_table[0] = a0;
	for (int i = 1; i < NT; i++) {
//		cosmos cinit(a0, adot0, 0.0);
		const double dtau = tau_max / NT;
		double tau = i * dtau;
		t += cinit.advance_to_tau(tau - tau_max);
//		printf("%i %e %e %e\n", i, t, cinit.get_tau() + tau_max, cinit.get_scale());
		t_table[i] = t;
		dt_table[i] = cinit.get_scale();
	}
	t_table[NT] = opts.t_max;
	dt_table[NT] = 1.0;
	map_reset(0.0);
}

void map_reset(double t) {
	static const auto opts = options::get();
	const std::uint64_t N = opts.problem_size;
	NY = opts.map_res / 2;
	NX = opts.map_res;
	const double dt = opts.t_max / opts.nout;
	const double tmax = 2.0 * dt + t_of_tau(cosmo_time() + 0.5 / opts.clight);
	for (int k = pixels.size(); k * dt <= tmax; k++) {
		pixels.resize(k + 1);
		mtx.resize(k + 1);
		pixels[k].resize(NX);
		mtx[k].resize(NX);
		for (int i = 0; i < NX; i++) {
			pixels[k][i].resize(NY);
			mtx[k][i].resize(NY);
			for (int j = 0; j < NY; j++) {
				pixels[k][i][j] = 0.0;
				mtx[k][i][j] = std::make_shared<mutex_type>();
			}
		}
	}
}

void map_output(double t) {
	static const auto opts = options::get();
	const auto dt = opts.t_max / opts.nout;
	for (int k = 0; k <= (int) (t / dt); k++) {
		if (mtx[k].size()) {
			const double dx = 2.0 / NX;
			const double dy = 1.0 / NY;
			const std::string filename = "map." + std::to_string(k) + std::string(".silo");
			printf("Writing %s\n", filename.c_str());
			DBfile *db = DBCreateReal(filename.c_str(), DB_CLOBBER, DB_LOCAL, "Meshless", DB_PDB);
			const char *coord_names[] = { "x", "y", "z" };
			double xcoord[NX + 1];
			double ycoord[NY + 1];
			for (int i = 0; i <= NX; i++) {
				xcoord[i] = i * dx;
			}
			for (int j = 0; j <= NY; j++) {
				ycoord[j] = j * dy;
			}
			void *coords[2] = { xcoord, ycoord };
			int dims1[2] = { NX + 1, NY + 1 };
			int dims[2] = { NX, NY };
			DBPutQuadmesh(db, "quadmesh", coord_names, coords, dims1, 2, DB_DOUBLE, DB_COLLINEAR, NULL);
			std::vector<double> vals;
			for (int j = 0; j < NY; j++) {
				for (int i = 0; i < NX; i++) {
					vals.push_back(pixels[k][i][j]);
				}
			}
			DBPutQuadvar1(db, "I", "quadmesh", vals.data(), dims, 2, NULL, 0, DB_DOUBLE, DB_ZONECENT, NULL);
			DBClose(db);

		}
		std::vector<std::vector<std::shared_ptr<mutex_type>>>().swap(mtx[k]);
		std::vector<std::vector<double>>().swap(pixels[k]);
	}
}

void map_add_particle(const vect<double> &x, double t, double dtau) {
	const auto opts = options::get();
	const auto dx = x - vect<double>(0.5);
	const auto r = abs(dx);
	const auto dtau_r = r / opts.clight;
	const auto t0 = t_of_tau(cosmo_time() + dtau_r - dtau);
	const auto t1 = t_of_tau(cosmo_time() + dtau_r);
	const auto dt_out = opts.t_max / opts.nout;
	const int bin0 = t0 / dt_out;
	const int bin1 = t1 / dt_out;
	if (r <= 0.5 && r > 0.0) {
		if (bin1 - bin0 > 1) {
//			printf( "Skipping bins %i %i\n", bin0, bin1);
		}
		for (int bin = bin0 + 1; bin <= bin1; bin++) {
			const auto psi = std::atan2(dx[0], dx[1]);
			const auto theta = std::asin(dx[2] / r);
			double phi = 0.0;
			double f;
			constexpr double toler = 5.0e-7;
			const auto sintheta = std::sin(theta);
			if (std::abs(M_PI * (1.0 - sintheta)) < toler) {
				phi = M_PI / 2.0;
			} else if (std::abs(M_PI * (1.0 + sintheta)) < toler) {
				phi = -M_PI / 2.0;
			} else {
				do {
					f = 2.0 * phi + std::sin(2.0 * phi) - M_PI * sintheta;
					double new_phi = phi - f / (2.0 + 2.0 * cos(2.0 * phi));
					phi = std::min(std::max(new_phi, (phi - M_PI / 2.0) / 2.0), (phi + M_PI / 2.0) / 2.0);
				} while (std::abs(f) > toler);
			}
			const double map_x = psi / M_PI * std::cos(phi);
			const double map_y = 0.5 * std::sin(phi);
			const double dx = 2.0 / NX;
			const double dy = 1.0 / NY;
			int xi = (map_x + 1.0) / dx;
			int yi = (map_y + 0.5) / dy;
			double val = 1.0 / (r * r);
			if (xi >= NX) {
				printf("xi = %i\n", xi);
				xi = NX - 1;
			}
			if (yi >= NY) {
				printf("yi = %i\n", yi);
				yi = NY - 1;
			}
			if (bin >= mtx.size()) {
				printf("Timebin exceeds capacity\n", bin, mtx.size());
				abort();
			}
			if (mtx[bin].size() == 0) {
				printf("Timebin %i already empties\n", bin);
				abort();
			}
			std::lock_guard<mutex_type> lock(*mtx[bin][xi][yi]);
			pixels[bin][xi][yi] += val;
		}
	}
}

