#pragma once

#include "BaseSolver.h"

class NewmarkBeta : public BaseSolver {
protected:
	const double gamma = 0.5;

public:

	double timeStepFactor;			// timestep of calculation, not the ground motion dt
	double beta;

	// constructor
	NewmarkBeta(double T_start, double T_end, double intervals, double damp, 
		double beta = 0.25, double timeStepFactor = -1.0);

	// solve
	std::vector<double> solve(double& T, double& damp, const GroundMotionRecord& ground_motion) override;

	void getMethodInfo() override;

	std::string getMethodName() override;

};