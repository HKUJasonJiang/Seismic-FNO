#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "FileManager.h"
#include "NigamJennings.h"
#include "NewmarkBeta.h"

namespace fs = std::filesystem;

struct Args {
    std::string gm_dir;
    std::string output_dir;
    std::string method = "NigamJennings";
    double t_start = 0.0;
    double t_end = 6.0;
    double interval = 0.05;
    double damp = 0.05;
    bool write_all = false;
};

void printUsage() {
    std::cout
        << "Usage: CalResponseSpectra_batch --gm_dir DIR --output_dir DIR [options]\n"
        << "Options:\n"
        << "  --method NigamJennings|NewmarkBeta   Default: NigamJennings\n"
        << "  --t_start VALUE                     Default: 0.0\n"
        << "  --t_end VALUE                       Default: 6.0\n"
        << "  --interval VALUE                    Default: 0.05\n"
        << "  --damp VALUE                        Default: 0.05\n"
        << "  --write_all                         Also write Sa/Sv/Sd/pSv; pSa is always written\n";
}

Args parseArgs(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        auto requireValue = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) {
                throw std::invalid_argument("Missing value for " + name);
            }
            return argv[++i];
        };

        if (key == "--gm_dir") {
            args.gm_dir = requireValue(key);
        } else if (key == "--output_dir") {
            args.output_dir = requireValue(key);
        } else if (key == "--method") {
            args.method = requireValue(key);
        } else if (key == "--t_start") {
            args.t_start = std::stod(requireValue(key));
        } else if (key == "--t_end") {
            args.t_end = std::stod(requireValue(key));
        } else if (key == "--interval") {
            args.interval = std::stod(requireValue(key));
        } else if (key == "--damp") {
            args.damp = std::stod(requireValue(key));
        } else if (key == "--write_all") {
            args.write_all = true;
        } else if (key == "--help" || key == "-h") {
            printUsage();
            std::exit(0);
        } else {
            throw std::invalid_argument("Unknown argument: " + key);
        }
    }

    if (args.gm_dir.empty() || args.output_dir.empty()) {
        printUsage();
        throw std::invalid_argument("--gm_dir and --output_dir are required");
    }
    return args;
}

std::vector<double> makePeriods(double t_start, double t_end, double interval) {
    std::vector<double> periods;
    for (double t = t_start; t <= t_end + interval * 1e-9; t += interval) {
        periods.push_back(t);
    }
    return periods;
}

std::vector<fs::path> listTxtFiles(const fs::path& gm_dir) {
    if (!fs::exists(gm_dir) || !fs::is_directory(gm_dir)) {
        throw std::invalid_argument("GM directory does not exist: " + gm_dir.string());
    }

    std::vector<fs::path> files;
    for (const auto& entry : fs::directory_iterator(gm_dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".txt") {
            files.push_back(entry.path());
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}

std::unique_ptr<BaseSolver> makeSolver(const Args& args) {
    if (args.method == "NigamJennings") {
        return std::make_unique<NigamJennings>(args.t_start, args.t_end, args.interval, args.damp);
    }
    if (args.method == "NewmarkBeta") {
        return std::make_unique<NewmarkBeta>(args.t_start, args.t_end, args.interval, args.damp);
    }
    throw std::invalid_argument("Unsupported method: " + args.method);
}

void writeMatrix(
    const fs::path& out_path,
    const std::vector<double>& periods,
    const std::vector<std::string>& names,
    const std::vector<std::vector<double>>& values
) {
    std::ofstream file(out_path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open output file: " + out_path.string());
    }
    file << std::setprecision(10);
    file << "Period (s)";
    for (const auto& name : names) {
        file << "," << name;
    }
    file << "\n";
    for (size_t p = 0; p < periods.size(); ++p) {
        file << periods[p];
        for (size_t i = 0; i < values.size(); ++i) {
            file << "," << values[i][p];
        }
        file << "\n";
    }
}

int main(int argc, char** argv) {
    try {
        Args args = parseArgs(argc, argv);
        fs::create_directories(args.output_dir);

        std::vector<fs::path> gm_files = listTxtFiles(args.gm_dir);
        if (gm_files.empty()) {
            throw std::runtime_error("No .txt files found in " + args.gm_dir);
        }

        std::vector<double> periods = makePeriods(args.t_start, args.t_end, args.interval);
        std::vector<std::string> names;
        std::vector<std::vector<double>> pSa_values;
        std::vector<std::vector<double>> Sa_values;
        std::vector<std::vector<double>> Sv_values;
        std::vector<std::vector<double>> Sd_values;
        std::vector<std::vector<double>> pSv_values;

        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < gm_files.size(); ++i) {
            if ((i + 1) % 100 == 0 || i == 0 || i + 1 == gm_files.size()) {
                std::cout << "[O3] " << (i + 1) << "/" << gm_files.size()
                          << " " << gm_files[i].filename().string() << std::endl;
            }

            GroundMotionRecord gm = FileManager::readFile(gm_files[i].string(), "1");
            auto solver = makeSolver(args);
            solver->computeResponseSpectra(gm);
            const ResponseSpectra& spectra = solver->getResponseSpetra();

            names.push_back(gm.ground_motion_name);
            pSa_values.push_back(spectra.pSa);
            if (args.write_all) {
                Sa_values.push_back(spectra.Sa);
                Sv_values.push_back(spectra.Sv);
                Sd_values.push_back(spectra.Sd);
                pSv_values.push_back(spectra.pSv);
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed_s = std::chrono::duration<double>(end - start).count();

        writeMatrix(fs::path(args.output_dir) / "pSa.csv", periods, names, pSa_values);
        if (args.write_all) {
            writeMatrix(fs::path(args.output_dir) / "Sa.csv", periods, names, Sa_values);
            writeMatrix(fs::path(args.output_dir) / "Sv.csv", periods, names, Sv_values);
            writeMatrix(fs::path(args.output_dir) / "Sd.csv", periods, names, Sd_values);
            writeMatrix(fs::path(args.output_dir) / "pSv.csv", periods, names, pSv_values);
        }

        std::ofstream details(fs::path(args.output_dir) / "details.txt");
        details << "Method: " << args.method << "\n";
        details << "Number of GMs: " << gm_files.size() << "\n";
        details << "Damp: " << args.damp << "\n";
        details << "T_start: " << args.t_start << "\n";
        details << "T_end: " << args.t_end << "\n";
        details << "Interval: " << args.interval << "\n";
        details << "Calculation Time (s): " << elapsed_s << "\n";
        details << "GM directory: " << args.gm_dir << "\n";

        std::cout << "[O3] Wrote response spectra to " << args.output_dir
                  << " in " << elapsed_s << " seconds." << std::endl;
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "[ERROR] " << exc.what() << std::endl;
        return 1;
    }
}
