#include <fstream>

#include "FileManager.h"

std::vector<double> FileManager::time;
std::vector<double> FileManager::acceleration;


void FileManager::readFile(const std::string& filePath) {
	
	std::ifstream file(filePath);
	std::string line;
	bool firstLine = true;				// skip first line







}

