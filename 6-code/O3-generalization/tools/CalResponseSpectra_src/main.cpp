#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include "BaseSolver.h"
#include "NigamJennings.h"
#include "NewmarkBeta.h"
#include "FileManager.h"
#include "SpectraCalculator.h"
#include "Utility.h"

int main() {

	double T_start = 0;
	double T_end = 6.0;
	double intervals = 0.05;
	double damp = 0.05;

	// information of this exe file.
	printHead();
	
	SpectraCalculator* spectraCalculator = new SpectraCalculator(T_start, T_end, intervals, damp);

	// Chose the solver here
	//BaseSolver* solver = new NewmarkBeta(T_start, T_end, intervals, damp);
	BaseSolver* solver = new NigamJennings(T_start, T_end, intervals, damp);

	// Process Single File
	// std::string file_path = "C:\\Users\\User\\Desktop\\knet_2567.txt";
	// spectraCalculator->processSingleFile(file_path, solver);

	// Process Multi-files
	/* 
		Args:
			save mode: "1" (1 CSV for each GMs) or "2" (5 CSVs for all GMs)
			print mode: "0" (print all infos) or else (simple print)
	*/
	std::string save_mode = "2";			// 1 or 2
	std::string print_mode = "1";			// 1 or else
	spectraCalculator->processMultiFiles(solver, save_mode, print_mode);

	delete solver;
	delete spectraCalculator;

	std::cin.get();

	return 0;
}
