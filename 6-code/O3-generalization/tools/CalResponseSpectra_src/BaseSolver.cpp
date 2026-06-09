#include <stdexcept>
#include <cmath>
#include <algorithm>

#include "BaseSolver.h"

BaseSolver::BaseSolver(double T_start, double T_end, double intervals, double damp)
	: T_start(T_start), T_end(T_end), intervals(intervals), damp(damp), m(1.0) {

	// prevent input period range error
	if (T_start < 0 || T_end <= 0 || T_start >= T_end || intervals <= 0) {
		throw std::invalid_argument("Invalid period range or intervals. Plz check again.");
	}

	// prevent damp and dt error
	if (damp <= 0 ) {
		throw std::invalid_argument("Invalid damp. Plz check again.");
	}

	// initialize the spectra vector
	size_t num_periods = static_cast<size_t>((T_end - T_start) / intervals) + 1;
	response_spectra.Sa.resize(num_periods, 0.0);
	response_spectra.Sv.resize(num_periods, 0.0);
	response_spectra.Sd.resize(num_periods, 0.0);
	response_spectra.pSa.resize(num_periods, 0.0);
	response_spectra.pSv.resize(num_periods, 0.0);
}

void BaseSolver::computeResponseSpectra(const GroundMotionRecord& ground_motion) {

	size_t index = 0;
	bool dt_assigned = false;			// Flag to ensure 'dt' is assigned only once

	for (double T = T_start; T <= T_end; T += intervals) {

		// when T = 0, the value of Sa is equal to PGA of the ground motion.
		if (T == 0.0) {
			double max_acceleration = *std::max_element(ground_motion.acceleration.begin(), ground_motion.acceleration.end(),
							[](double a, double b) { return std::abs(a) < std::abs(b); });
			response_spectra.Sa[index] = std::abs(max_acceleration);
		}

		// core function
		else {
			std::vector<double> results = solve(T, damp, ground_motion);

			if (results.size() != 6) {
				throw std::runtime_error("Solve function return a vector size is not 5. Plz check solve function.");
			}

			response_spectra.Sa[index] = results[0];
			response_spectra.Sv[index] = results[1];
			response_spectra.Sd[index] = results[2];
			response_spectra.pSa[index] = results[3];
			response_spectra.pSv[index] = results[4];

			if (!dt_assigned) {
				response_spectra.dt = results[5];
				dt_assigned = true;
			}
		}

		index++;
	}
}