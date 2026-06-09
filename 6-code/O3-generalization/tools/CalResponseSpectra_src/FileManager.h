#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <iostream>

#include "Struct.h"

class FileManager {
public:

	// read ground motion txt (skip first row, first column = time, second column = acceleration)
	static GroundMotionRecord readFile(const std::string& filePath, const std::string& print_mode) {
		std::ifstream file(filePath);
		std::string line;
		bool firstLine = true;
		
		// using a struct to contain time and acceleration
		GroundMotionRecord ground_motion_record;

		// Extract the filename without extension
		std::string filename = filePath.substr(filePath.find_last_of("/\\") + 1);
		size_t lastDot = filename.find_last_of('.');
		if (lastDot != std::string::npos) {
			filename = filename.substr(0, lastDot);
			if (print_mode == "0") {
				std::cout << "[MESSAGE] Reading " << filename << "..." << std::endl;
			}
		}
		ground_motion_record.ground_motion_name = filename;

		while (std::getline(file, line)) {
			if (firstLine) {			// skip first line
				firstLine = false;
				continue;
			}

			std::istringstream iss(line);		// get the data
			double t, a;
			if (iss >> t >> a) {
				ground_motion_record.time.push_back(t);
				ground_motion_record.acceleration.push_back(a);
			}
		}
		return ground_motion_record;
	}

	// write response spectra to a single CSV file.
	static void writeSingleOutputFile(const std::string& ground_motion_name,
									  const std::string& method_name,	
									  const std::vector<double>& periods, 
									  const ResponseSpectra& response_spectra) {

		// create the results directory
		FileManager::createResultsDir();

		// construct the full file path
		std::string file_path = "Results/" + ground_motion_name + "_response_spectra.csv";

		// open file
		std::ofstream file(file_path);

		// write the header
		file << "Grouond Motion" << "," << ground_motion_name << "," 
			<< "calculation dt" << "," << response_spectra.dt << "," 
			<< method_name << "\n";

		file << "Period (s), Sa, Sd, Sv, pSa, pSv\n";

		// write the data
		for (size_t i = 0; i < periods.size(); i++) {
			file << periods[i] << "," << response_spectra.Sa[i] << "," 
				<< response_spectra.Sd[i] << "," << response_spectra.Sv[i]<< "," 
				<< response_spectra.pSa[i] << "," << response_spectra.pSv[i] << "\n";
		}
	}

	// read folder "GMs"
	static std::vector<GroundMotionRecord> readGMFolder(const std::string& print_mode) {

		std::vector<GroundMotionRecord> ground_motion_records;
		std::string folderPath = "./GMs";

		// check the directory
		if (!std::filesystem::exists(folderPath) || !std::filesystem::is_directory(folderPath)) {
			std::cerr << "Error: Folder " << folderPath << " does not exist or is not a directory.\n" 
				<< "Please create a 'GMs' folder and put your ground motion records (txt files) inside it to start.\n" 
				<< "Please also note that the txt files should have header, and 1st column is time, 2nd column is acceleration." 
				<< std::endl;
			return ground_motion_records;
		}

		// loop for all txt files, and add it to ground motion records
		for (const auto& file : std::filesystem::directory_iterator(folderPath)) {
			if (file.path().extension() == ".txt") {
				try {
					GroundMotionRecord record = readFile(file.path().string(), print_mode);
					ground_motion_records.push_back(record);
				}
				// if file is not valid or can't be read
				catch (const std::exception& e) {
					std::cerr << "Error reading file " << file.path() << ": " << e.what() << std::endl;
				}
			}
		}

		if (ground_motion_records.empty()) {
			std::cerr << "Warning: No TXT files found in folder " << folderPath << "." << std::endl;
		}

		return ground_motion_records;
	}

	// write to 5 CSVs, Sa, Sv, Sd, pSa, pSv
	static void writeMultiOutputFile(const std::vector<GroundMotionRecord>& ground_motion_records,
									 const std::vector<double>& periods, 
									 const std::vector<ResponseSpectra>& all_response_spectra) {
		
		// create the results directory
		FileManager::createResultsDir();

		// file names
		std::vector<std::string> file_names = { "Sa.csv", "Sv.csv", "Sd.csv", "pSa.csv", "pSv.csv" };

		// open files
		std::vector<std::ofstream> files;
		for (const auto& file_name : file_names) {
			files.emplace_back("Results/" + file_name);
		}

		// write header
		for (size_t i = 0; i < files.size(); i++) {
			files[i] << "Period (s)";
			for (const auto& ground_motion_record : ground_motion_records) {
				files[i] << "," << ground_motion_record.ground_motion_name;
			}
			files[i] << "\n";
		}

		// write data
		for (size_t j = 0; j < periods.size(); j++) {
			files[0] << periods[j];
			files[1] << periods[j];
			files[2] << periods[j];
			files[3] << periods[j];
			files[4] << periods[j];

			for (size_t i = 0; i < all_response_spectra.size(); i++) {
				files[0] << "," << all_response_spectra[i].Sa[j];
				files[1] << "," << all_response_spectra[i].Sv[j];
				files[2] << "," << all_response_spectra[i].Sd[j];
				files[3] << "," << all_response_spectra[i].pSa[j];
				files[4] << "," << all_response_spectra[i].pSv[j];
			}
			for (auto& file : files) {
				file << "\n";
			}
		}
		
		// close the files
		for (auto& file : files) {
			file.close();
		}
	}
	
	// clear and create results folder to save
	static void createResultsDir() {

		const std::filesystem::path resultsDir = "Results";

		if (std::filesystem::exists(resultsDir)) {
			
			std::cout << "\033[0;33m[WARNING] Results folder is already exist, clear and recreating...\033[0m\n";
			std::filesystem::remove_all(resultsDir);
		}

		std::filesystem::create_directory(resultsDir);
	}

	// write log file
	static void createLog(const std::string& method_name,
						  const int num_GMs,
						  const double& damp,
						  const double dt,
						  const double& calculation_time,
						  const double& saving_time) {

		std::string log_file_name = "./Results/details.txt";
		std::ofstream log_file(log_file_name);

		if (log_file.is_open()) {
			log_file << "Method: " << method_name << "\n";
			log_file << "Number of GMs: " << num_GMs << "\n";
			log_file << "Damp: " << damp << "\n";
			log_file << "dt (calculation) (s): " << dt << "\n";
			log_file << "Calculation Time (s): " << calculation_time << "\n";
			log_file << "Saving Time (s): " << saving_time << "\n";
			log_file.close();
		}
		else {
			std::cerr << "Error: Unable to open file for writing: " << log_file_name << std::endl;
		}

	}
};
