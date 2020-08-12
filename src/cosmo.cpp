#include <tigergrav/options.hpp>
#include <tigergrav/cosmo.hpp>

#ifdef HPX_LITE
#include <hpx/hpx_lite.hpp>
#else
#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/async.hpp>
#endif
#include <cmath>

double kick_dt1[RUNG_MAX + 1] = { -1.0 };
double kick_dt2[RUNG_MAX + 1] = { -1.0 };

double drift_dt;
float this_time = 0.0;

double cosmo_kick_dt1(rung_type i) {
	return kick_dt1[i];
}

double cosmo_kick_dt2(rung_type i) {
	return kick_dt2[i];
}

double cosmo_drift_dt() {
	return drift_dt;
}

cosmos::cosmos() {
	a = 1.0;
	adot = 1.0;
	tau = 0.0;
	drift_dt = 0.0;
	kick_dt = 0.0;
}
cosmos::cosmos(double a_, double adot_, double tau_) {
	a = a_;
	adot = adot_;
	tau = tau_;
	drift_dt = 0.0;
	kick_dt = 0.0;
}
void cosmos::advance_to_time(double t0) {
//	printf( "%e\n", a);
	const auto opts = options::get();
	const auto c0 = (4.0 / 3.0 * M_PI) * opts.m_tot;
	const auto da = [](double adot) {
		return adot;
	};
	const auto dadot = [c0](double a) {
		return -c0 * 1.0 / (a * a);
	};
	const auto dtau = [](double a) {
		return 1.0 / a;
	};
	const auto ddrift_dt = [](double a) {
		return 1.0 / (a * a);
	};
	const auto dkick_dt = [](double a) {
		return 1.0 / a;
	};
	double t = 0.0;
	bool done = false;
	bool backwards = t0 < 0.0;
	const double sgn = backwards ? -1.0 : 1.0;
//	printf( "----\n");
	kick_dt = drift_dt = 0.0;
	do {
		auto dt = sgn * 1e-3 * a / adot;
	//	printf("*%e %e %e %e %e %e\n", a, adot, t, t0, dt, kick_dt);
		if (std::abs(t + dt) > std::abs(t0)) {
			done = true;
			dt = std::abs(t0 - t) * sgn;
		}
		const double dtau1 = dtau(a) * dt;
		const double da1 = da(adot) * dt;
		const double dadot1 = dadot(a) * dt;
		const double ddrift_dt1 = ddrift_dt(a) * dt;
		const double dkick_dt1 = dkick_dt(a) * dt;
		const double dtau2 = dtau(a + 0.5 * da1) * dt;
		const double da2 = da(adot + 0.5 * dadot1) * dt;
		const double dadot2 = dadot(a + 0.5 * da1) * dt;
		const double ddrift_dt2 = ddrift_dt(a + 0.5 * da1) * dt;
		const double dkick_dt2 = dkick_dt(a + 0.5 * da1) * dt;
		const double dtau3 = dtau(a + 0.5 * da2) * dt;
		const double da3 = da(adot + 0.5 * dadot2) * dt;
		const double dadot3 = dadot(a + 0.5 * da2) * dt;
		const double ddrift_dt3 = ddrift_dt(a + 0.5 * da2) * dt;
		const double dkick_dt3 = dkick_dt(a + 0.5 * da2) * dt;
		const double dtau4 = dtau(a + da3) * dt;
		const double da4 = da(adot + dadot3) * dt;
		const double dadot4 = dadot(a + da3) * dt;
		const double ddrift_dt4 = ddrift_dt(a + da3) * dt;
		const double dkick_dt4 = dkick_dt(a + da3) * dt;
		tau += (dtau1 + 2.0 * (dtau2 + dtau3) + dtau4) / 6.0;
		a += (da1 + 2.0 * (da2 + da3) + da4) / 6.0;
		adot += (dadot1 + 2.0 * (dadot2 + dadot3) + dadot4) / 6.0;
		drift_dt += (ddrift_dt1 + 2.0 * (ddrift_dt2 + ddrift_dt3) + ddrift_dt4) / 6.0;
		kick_dt += (dkick_dt1 + 2.0 * (dkick_dt2 + dkick_dt3) + dkick_dt4) / 6.0;
		t += dt;
	} while (!done);
}

double cosmos::advance_to_scale(double a0) {
	int iters = 0;
	double t = 0.0;
	do {
		const auto dt = (a0 - a) / adot / 10.0;
//		printf("%e %e %e %e\n", t, a, a0, adot);
		advance_to_time(dt);
		t += dt;
		iters++;
		if( iters > 1000) {
			printf( "Unable to find t=0 scale factor\n");
		}
	} while (std::abs(a0 / a - 1.0) > 1.0e-10);
	return t;
}

static cosmos this_cosmos;
static cosmos last_cosmos;

static std::vector<hpx::id_type> localities;
static int myid;

std::pair<double, double> cosmo_scale() {
	return std::make_pair(last_cosmos.get_scale(), this_cosmos.get_scale());
}

double cosmo_Hubble() {
	return this_cosmos.get_Hubble();
}

double cosmo_time() {
	return this_cosmos.get_tau();
}

HPX_PLAIN_ACTION(cosmo_advance);
HPX_PLAIN_ACTION(cosmo_init);

void cosmo_advance(double dt) {
	std::vector<hpx::future<void>> futs;
	if (myid == 0) {
		for (int i = 1; i < localities.size(); i++) {
			futs.push_back(hpx::async<cosmo_advance_action>(localities[i], dt));
		}
	}
	last_cosmos = this_cosmos;
	this_time += dt;
	if (dt != 0.0) {
		this_cosmos.advance_to_time(dt);
	}
	for (rung_type i = 0; i <= RUNG_MAX; i++) {
		const auto dt = rung_to_dt(i);
		if (this_time >= dt) {
			cosmos c1 = this_cosmos;
			c1.advance_to_time(-0.5 * dt);
			kick_dt1[i] = -c1.get_kick_dt();
//			printf( "%i %e %e\n", i, 0.5 * dt / cosmo_scale().second, kick_dt2[i]);
		}

		cosmos c2 = this_cosmos;
		c2.advance_to_time(+0.5 * dt);
		kick_dt2[i] = c2.get_kick_dt();
	}
	if (dt != 0.0) {
		cosmos c1 = last_cosmos;
		c1.advance_to_time(dt);
		drift_dt = c1.get_drift_dt();
	}
	hpx::wait_all(futs.begin(), futs.end());
}


void cosmo_init( double a0, double adot0) {
	std::vector<hpx::future<void>> futs;
	if (myid == 0) {
		for (int i = 1; i < localities.size(); i++) {
			futs.push_back(hpx::async<cosmo_init_action>(localities[i], a0, adot0));
		}
	}
	this_cosmos = cosmos(a0,adot0,0.0);
	last_cosmos = this_cosmos;
	localities = hpx::find_all_localities();
	myid = hpx::get_locality_id();
	hpx::wait_all(futs.begin(), futs.end());
}
