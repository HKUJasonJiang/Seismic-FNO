#include <iostream>

#include "NigamJennings.h"


// constructor
NigamJennings::NigamJennings(double T_start, double T_end, double intervals, double damp)
	: BaseSolver(T_start, T_end, intervals, damp) {}

std::vector<double> NigamJennings::solve(double& T, double& damp, const GroundMotionRecord& ground_motion) {
	
	// number of time steps
	size_t num_steps = ground_motion.time.size();

	double dt = ground_motion.time[1] - ground_motion.time[0];

	// constants
	double w = 2.0 * pi / T;
	double h = damp;
	double exp_factor = std::exp(-h * w * dt);		// e^(-h * w * dt)
	double h_d = std::sqrt(1 - h * h);				// sqrt(1 - h^2)
	double w_d = w * h_d;							// w * sqrt(1 - h^2)
	double sin_term = std::sin(w_d * dt);			// sin(w' * dt)
	double cos_term = std::cos(w_d * dt);			// cos(w' * dt)

	// coefficients for state transition matrix
	double a_11 = exp_factor * (h / h_d * sin_term + cos_term);
	double a_12 = exp_factor / w_d * sin_term;
	double a_21 = -w / h_d * exp_factor * sin_term;
	double a_22 = exp_factor * (cos_term - h / h_d * sin_term);

	// coefficients for forcing function
	double b_hwt = (2 * h * h - 1) / (w * w * dt);	// (2h^2 - 1)/(w^2 * dt)
	double b_hwt2 = 2 * h / (w * w * w * dt);		// 2h/w^3 * dt

	double b_11 = exp_factor * ((b_hwt + h / w) * sin_term / w_d + (b_hwt2 + 1 / (w * w)) * cos_term) - b_hwt2;
	double b_12 = -exp_factor * ((b_hwt * sin_term / w_d) + b_hwt2 * cos_term) - 1 / (w * w) + b_hwt2;
	double b_21 = exp_factor * ((b_hwt + h / w) * (cos_term - h / h_d * sin_term)
		- (b_hwt2 + 1 / (w * w)) * (w_d * sin_term + h * w * cos_term)) + 1 / (w * w * dt);
	double b_22 = -exp_factor * (b_hwt * (cos_term - h / h_d * sin_term)
		- (b_hwt2 * (w_d * sin_term + h * w * cos_term))) - 1 / (w * w * dt);

	// initialize state variables
	double velocity = -ground_motion.acceleration[0] * dt;
	double displacement = 0.0;

	// initialize a mini value for update
	double max_abs_acceleration = -std::numeric_limits<double>::infinity();
	double max_abs_velocity = -std::numeric_limits<double>::infinity();
	double max_abs_displacement = -std::numeric_limits<double>::infinity();

	// time integration loop
	for (size_t step = 1; step < num_steps; step++) {
		double previous_displacement = displacement;
		double previous_velocity = velocity;
		double current_acceleration = ground_motion.acceleration[step];
		double previous_acceleration = ground_motion.acceleration[step - 1];

		// update displacement and velocity
		displacement = a_11 * previous_displacement + a_12 * previous_velocity
			+ b_11 * previous_acceleration + b_12 * current_acceleration;
		velocity = a_21 * previous_displacement + a_22 * previous_velocity
			+ b_21 * previous_acceleration + b_22 * current_acceleration;

		// compute absolute acceleration
		double abs_acceleration = -2.0 * h * w * velocity - w * w * displacement;

		// update maximum response values
		max_abs_acceleration = std::max(max_abs_acceleration, std::abs(abs_acceleration));
		max_abs_velocity = std::max(max_abs_velocity, std::abs(velocity));
		max_abs_displacement = std::max(max_abs_displacement, std::abs(displacement));
	}

	// compute pseudo-acceleration and pseudo-velocity
	double pseudo_acceleration = w * w * max_abs_displacement;
	double pseudo_velocity = w * max_abs_displacement;

	if (max_abs_acceleration <= 0 || max_abs_velocity <= 0 || max_abs_displacement <= 0
		|| pseudo_acceleration <= 0 || pseudo_velocity <= 0) {
		throw std::invalid_argument("Nigam-Jennings::Spectra results have 0. Plz Check the calculation.");
	}

	return { max_abs_acceleration, max_abs_velocity, max_abs_displacement,
		pseudo_acceleration, pseudo_velocity , dt};
}

void NigamJennings::getMethodInfo() {

	std::cout << "Nigam-Jennings Method: " << "\n";
	std::cout << "Provide precise solution, only rely on the 'omega' and 'damp', same as SeismoSignal results." << "\n";
}

std::string NigamJennings::getMethodName() {
	return "NigamJennings";
}