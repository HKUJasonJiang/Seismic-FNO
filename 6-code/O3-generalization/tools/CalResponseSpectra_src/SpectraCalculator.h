#pragma once

#include <string>
#include <vector>
#include "BaseSolver.h"

class SpectraCalculator {
private:
	double T_start;
	double T_end;
	double intervals;
	double damp;

	std::vector<double> calculatePeriods();


public:
	SpectraCalculator(double T_start, double T_end, double intervals, double damp);

	void processSingleFile(const std::string& filePath, BaseSolver* solver);

	void processMultiFiles(BaseSolver* solver, const std::string& save_mode, const std::string& print_mode);




};