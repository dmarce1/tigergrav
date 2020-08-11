#include <tigergrav/options.hpp>
#include <tigergrav/cosmo.hpp>

#ifdef HPX_LITE
#include <hpx/hpx_lite.hpp>
#else
#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/async.hpp>
#endif
#include <cmath>

static double last_a = 1.0;
static double last_adot = 1.0;
static double a = 1.0;
static double adot = 1.0;
static double tau = 0.0;
static double t = 0.0;
static std::vector<hpx::id_type> localities;
static int myid;

std::pair<double, double> cosmo_scale() {
	return std::make_pair(last_a, a);
}

std::pair<double, double> cosmo_Hubble() {
	return std::make_pair(last_adot / last_a, adot / a);
}

double cosmo_time() {
	return tau;
}

HPX_PLAIN_ACTION(cosmo_advance);

void cosmo_advance(double dt) {
	static const auto opts = options::get();
	t += dt;
	if (a == 1.0) {
		table_tmax = 1.01 * opts.dt_max;
		table_dt = table_tmax / N;
		localities = hpx::find_all_localities();
		myid = hpx::get_locality_id();
	}
	std::vector<hpx::future<void>> futs;
	if (myid == 0) {
		for (int i = 1; i < localities.size(); i++) {
			futs.push_back(hpx::async<cosmo_advance_action>(localities[i], dt));
		}
	}
	last_a = a;
	last_adot = adot;
	static const auto c0 = (4.0 / 3.0 * M_PI) * opts.m_tot;
	const auto da = [=](double adot) {
		return adot * dt;
	};
	const auto dadot = [dt](double a) {
		return -c0 * dt / (a * a);
	};
	const auto dtau = [dt](double a) {
		return dt / a;
	};
	const double dtau1 = dtau(a);
	const double da1 = da(adot);
	const double dadot1 = dadot(a);
	const double dtau2 = dtau(a + 0.5 * da1);
	const double da2 = da(adot + 0.5 * dadot1);
	const double dadot2 = dadot(a + 0.5 * da1);
	const double dtau3 = dtau(a + 0.5 * da2);
	const double da3 = da(adot + 0.5 * dadot2);
	const double dadot3 = dadot(a + 0.5 * da2);
	const double dtau4 = dtau(a + da3);
	const double da4 = da(adot + dadot3);
	const double dadot4 = dadot(a + da3);
	tau += (dtau1 + 2.0 * (dtau2 + dtau3) + dtau4) / 6.0;
	a += (da1 + 2.0 * (da2 + da3) + da4) / 6.0;
	adot += (dadot1 + 2.0 * (dadot2 + dadot3) + dadot4) / 6.0;
	hpx::wait_all(futs.begin(), futs.end());
}
