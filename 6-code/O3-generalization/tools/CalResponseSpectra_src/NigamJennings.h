#pragma once

#include "BaseSolver.h"

class NigamJennings : public BaseSolver {
public:
	// constructor
	NigamJennings(double T_start, double T_end, double intervals, double damp);

	// solve
	std::vector<double> solve(double& T, double& damp, const GroundMotionRecord& ground_motion) override;

	void getMethodInfo() override;

	std::string getMethodName() override;

};