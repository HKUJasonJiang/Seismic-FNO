#include <iostream>
#include <limits>

#include "NewmarkBeta.h"

NewmarkBeta::NewmarkBeta(double T_start, double T_end, double intervals, double damp,
	double beta, double timeStepFactor)
	: BaseSolver(T_start, T_end, intervals, damp) {

	this->beta = beta;				// default is linear assumption.
	this->timeStepFactor = timeStepFactor;
}

std::vector<double> NewmarkBeta::solve(double& T, double& damp, const GroundMotionRecord& ground_motion) {

	// number of time steps
	size_t num_steps = ground_motion.time.size();

	double dt = ground_motion.time[1] - ground_motion.time[0];

	if (timeStepFactor != -1.0) {
		dt *= timeStepFactor;
	}

	// constants
	double w = 2.0 * pi / T;
	double h = damp;
	double k = 4 * pi * pi * m / (T * T);
	double c = 2 * h * m * w;

	// coefficient
	double alpha_0 = 1 / (beta * dt * dt);
	double alpha_1 = gamma / (beta * dt);
	double alpha_2 = 1 / (beta * dt);
	double alpha_3 = 1 / (2 * beta) - 1;
	double alpha_4 = gamma / beta - 1;
	double alpha_5 = dt * (gamma / (2 * beta) - 1);
	double alpha_6 = dt * (1 - gamma);
	double alpha_7 = gamma * dt;

	// effective stiffness
	double K_eff = k + alpha_0 * m + alpha_1 * c;

	// initialize state varibles
	std::vector<double> d(num_steps, 0.0), v(num_steps, 0.0), a(num_steps, 0.0);

	// initialize a mini value for update
	double max_abs_acceleration = -std::numeric_limits<double>::infinity();
	double max_abs_velocity = -std::numeric_limits<double>::infinity();
	double max_abs_displacement = -std::numeric_limits<double>::infinity();

	// calculation
	for (size_t i = 1; i < num_steps; i++) {
		double f = -m * ground_motion.acceleration[i] + m * (alpha_0 * d[i - 1] + alpha_2 * v[i - 1] + alpha_3 * a[i - 1])
			+ c * (alpha_1 * d[i - 1] + alpha_4 * v[i - 1] + alpha_5 * a[i - 1]);

		d[i] = f / K_eff;
		a[i] = alpha_0 * (d[i] - d[i - 1]) - alpha_2 * v[i - 1] - alpha_3 * a[i - 1];
		v[i] = v[i - 1] + alpha_6 * a[i - 1] + alpha_7 * a[i];

		max_abs_acceleration = std::max(max_abs_acceleration, std::abs(a[i]));
		max_abs_velocity = std::max(max_abs_velocity, std::abs(v[i]));
		max_abs_displacement = std::max(max_abs_displacement, std::abs(d[i]));
	}

	// compute pseudo-acceleration and pseudo-velocity
	double pseudo_acceleration = w * w * max_abs_displacement;
	double pseudo_velocity = w * max_abs_displacement;

	if (max_abs_acceleration <= 0 || max_abs_velocity <= 0 || max_abs_displacement <= 0
		|| pseudo_acceleration <= 0 || pseudo_velocity <= 0) {
		throw std::invalid_argument("Newmark-Beta::Spectra results have 0. Plz Check the calculation.");
	}

	return { max_abs_acceleration, max_abs_velocity, max_abs_displacement,
		pseudo_acceleration, pseudo_velocity, dt };
}

void NewmarkBeta::getMethodInfo() {

	std::cout << "Newmark-Beta Method: " << "\n";
	std::cout << "* Averange acceleration: beta = 0.25 (default, stable)" << "\n";
	std::cout << "* Linear acceleration: beta = 1/6 (not stable)" << "\n";
	std::cout << "* Jump acceleration: beta = 1/8 (not stable)" << "\n";
	std::cout << "Calculation 'dt' can be define manually: this->timeStep" << "\n";
}

std::string NewmarkBeta::getMethodName() {
	return "Newmark-Beta";
}