#pragma once

#include <tigergrav/time.hpp>

class cosmos {
	double a;
	double adot;
	double tau;
	double drift_dt;
	double kick_dt;
public:
	cosmos();
	cosmos(double a_, double adot_, double tau_);
	double get_scale() const {
		return a;
	}
	double get_drift_dt() const {
		return drift_dt;
	}
	double get_Hubble() const {
		return adot / a;
	}
	double get_kick_dt() const {
		return kick_dt;
	}
	double get_tau() const {
		return tau;
	}
	cosmos& operator=( const cosmos& other) {
		drift_dt = 0.0;
		kick_dt = 0.0;
		adot = other.adot;
		a = other.a;
		tau = other.tau;
		return *this;
	}
	void advance_to_time(double t0);
	double advance_to_scale(double a0);
};

std::pair<double,double> cosmo_scale();
double cosmo_kick_dt1(rung_type);
double cosmo_kick_dt2(rung_type);
double cosmo_drift_dt();
double cosmo_Hubble();
double cosmo_time();
void cosmo_init( double, double );
void cosmo_advance(double dt);
