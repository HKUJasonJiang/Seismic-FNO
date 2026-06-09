#include <iostream>
#include <chrono>

#include "SpectraCalculator.h"
#include "FileManager.h"
#include "BaseSolver.h"
#include "NigamJennings.h"
#include "NewmarkBeta.h"

SpectraCalculator::SpectraCalculator(double T_start, double T_end, double intervals, double damp)
	: T_start(T_start), T_end(T_end), intervals(intervals), damp(damp) {}

// single file -> single excel for: Sa, Sv, Sd, pSa, pSv
void SpectraCalculator::processSingleFile(const std::string& filePath, BaseSolver* solver) {

	// get method name
	std::string method_name = solver->getMethodName();

	// read the ground motion file
	GroundMotionRecord ground_motion = FileManager::readFile(filePath, "0");

	std::cout << "[ACTION] Using " << method_name << " to calculate response spectra..." << std::endl;

	// compute response spectra
	solver->computeResponseSpectra(ground_motion);

	// get response spectra
	ResponseSpectra response_spectra = solver->getResponseSpetra();
	
	// construct teh periods
	std::vector<double> periods = calculatePeriods();
	
	// write the file
	FileManager::writeSingleOutputFile(ground_motion.ground_motion_name, method_name, periods, response_spectra);

	std::cout << "\033[0;32m[SUCCESS] Output saved to: Results\033[0m\n" << std::endl;
}

// folder with multi-files -> all spectra
void SpectraCalculator::processMultiFiles(BaseSolver* solver, const std::string& save_mode, const std::string& print_mode) {

	std::cout << "\033[0;34m[ACTION] Reading ground motion records... May take some times...\033[0m" << std::endl;
	// read the folder
	std::vector<GroundMotionRecord> ground_motions = FileManager::readGMFolder(print_mode);

	// get method name
	std::string method_name = solver->getMethodName();

	std::cout << "\033[0;34m[ACTION] Using " << method_name << " to calculate response spectra...\033[0m" << std::endl;

	// initialize the data
	std::vector<ResponseSpectra> all_response_spectra;

	// start_time
	auto start_time_computation = std::chrono::high_resolution_clock::now();

	// calculate the response spectra
	for (const GroundMotionRecord& ground_motion : ground_motions) {

		if (print_mode == "0") {
			std::cout << "[MESSAGE] Computing " << ground_motion.ground_motion_name << "...\n";
		}
		
		solver->computeResponseSpectra(ground_motion);

		ResponseSpectra response_spectra = solver->getResponseSpetra();

		all_response_spectra.push_back(response_spectra);
	}

	// start_time
	auto end_time_computation = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> duration_c = end_time_computation - start_time_computation;

	std::cout << "\033[0;32m[SUCCESS] Computation completed.\033[0m" << std::endl;
	std::cout << "[MESSAGE] Time taken for the calculation: " << duration_c.count() << " seconds." << std::endl;

	// check if the number of results = inputs
	if (ground_motions.size() != all_response_spectra.size()) {

		std::cerr << "Error: size of ground_motions is not equal to size of response spectra (SpectraCalculator::processMultiFiles)" << std::endl;
	}

	// construct teh periods
	std::vector<double> periods = calculatePeriods();

	std::cout << "\033[0;34m[ACTION] Start to write results to the CSV files...\033[0m" << std::endl;
	// start_time
	auto start_time_save = std::chrono::high_resolution_clock::now();

	// save the data
	// mode 1: 1 CSVs for each ground motion, same as the single files
	if (save_mode == "1") {

		std::cout << "[MESSAGE] Save mode 1: 1 CSV for each groud motion.\n";

		for (size_t i = 0; i < ground_motions.size(); i++) {
			const GroundMotionRecord ground_motion = ground_motions[i];
			const ResponseSpectra response_spectra = all_response_spectra[i];

			FileManager::writeSingleOutputFile(ground_motion.ground_motion_name, method_name, periods, response_spectra);
		}
	}
	// mode 2: 5 CSVs for all ground motions, Sa, Sv, Sd, pSa, pSv
	else if (save_mode == "2") {

		std::cout << "[MESSAGE] Save mode 2: 5 CSVs (Sa, Sv, Sd, pSa, pSv) for all ground motions.\n";
		
		FileManager::writeMultiOutputFile(ground_motions, periods, all_response_spectra);
	}

	// start_time
	auto end_time_save = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> duration_s = end_time_save - start_time_save;

	std::cout << "\033[0;32m[SUCCESS] All the results have been save to 'Results' folder.\033[0m" << std::endl;
	std::cout << "[MESSAGE] Time taken for the saving: " << duration_s.count() << " seconds." << std::endl;

	// create log file
	FileManager::createLog(method_name,
						ground_motions.size(),
						solver->getDamp(),
						all_response_spectra[0].dt,
						duration_c.count(),
						duration_s.count());

	std::cout << "[MESSAGE] More details can be found in 'details.txt'.\n" 
		<< "---------------------------------------------------------------------------------------\n"
		<< "Done."
		<< std::endl;
}

// construct the periods
std::vector<double> SpectraCalculator::calculatePeriods() {

	std::vector<double> periods;

	for (double t = T_start; t <= T_end; t += intervals) {
		periods.push_back(t);
	};

	return periods;
}


