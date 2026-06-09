#pragma once

#include <vector>
#include <string>
#include <cmath>
#include "FileManager.h"

class BaseSolver {
protected:
	double m;			// mass
	double damp;		// damp

	const double pi = 3.14159265358979323846;

	ResponseSpectra response_spectra;

	double T_start;		// periods start
	double T_end;		// periods end
	double intervals;	// intervals

public:
	// constructor
	BaseSolver(double T_start, double T_end, double intervals, double damp);

	// desctructor
	virtual ~BaseSolver() = default;

	// main functions
	virtual std::vector<double> solve(double& T, double& damp, const GroundMotionRecord& ground_motion) = 0;

	void computeResponseSpectra(const GroundMotionRecord& ground_motion);

	// record the detailed information of the method.
	virtual void getMethodInfo() = 0;
	virtual std::string getMethodName() = 0;

	const double getDamp() const { return damp; }
	const ResponseSpectra& getResponseSpetra() const { return response_spectra; }
};