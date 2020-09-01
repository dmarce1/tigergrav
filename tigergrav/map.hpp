/*
 * map.hpp
 *
 *  Created on: Aug 30, 2020
 *      Author: dmarce1
 */
#pragma once



#include <tigergrav/vect.hpp>

void map_init();
void map_add_particle(const vect<double>& x, double t, double dt);
void map_reset(double t);
void map_output(double t);
