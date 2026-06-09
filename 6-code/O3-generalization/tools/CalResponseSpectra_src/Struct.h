#pragma once

#include <vector>

struct ResponseSpectra {
	std::vector<double> Sa;			// acceleration spectrum, "m/s^2" or "g", depends on the input ground motion units
	std::vector<double> Sv;			// velocity spectrum
	std::vector<double> Sd;			// displacement spectrum
	std::vector<double> pSa;		// pseudo Sa
	std::vector<double> pSv;		// pseudo Sv
	double dt;			// final timestep for calculation
};

struct GroundMotionRecord {
	std::string ground_motion_name;
	std::vector<double> time;
	std::vector<double> acceleration;
};
