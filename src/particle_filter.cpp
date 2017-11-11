/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

// global engine gen
static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 10;

	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_t(theta, std[2]);

	for (int i=0; i<num_particles; i++) { // initialize individual particles
		Particle p;
		p.id = i;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_t(gen);
		p.weight = 1.0;
		particles.push_back(p);
		weights.push_back(1);
	}
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	for (int i=0; i<num_particles; i++) {
		
		if (fabs(yaw_rate) < 1E-4) {
			particles[i].x += velocity * delta_t * cos(particles[i].theta);
			particles[i].y += velocity * delta_t * sin(particles[i].theta);
		}
		else {
			particles[i].x += (velocity/yaw_rate)*(sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
			particles[i].y += (velocity/yaw_rate)*(cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
			particles[i].theta += yaw_rate * delta_t;
		}


		normal_distribution<double> dist_x(particles[i].x, std_pos[0]);
		normal_distribution<double> dist_y(particles[i].y, std_pos[1]);
		normal_distribution<double> dist_t(particles[i].theta, std_pos[2]);

		particles[i].x = dist_x(gen);
		particles[i].y = dist_y(gen);
		particles[i].theta = dist_t(gen);
		
		// theta angle normalization
		while (particles[i].theta >= 2.*M_PI) particles[i].theta -= 2.*M_PI;
		while (particles[i].theta <= 0) particles[i].theta += 2.*M_PI;
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	// each particle with the observations
	for (int i=0; i<num_particles; i++) {
		Particle p;
		p.id = i;
		p.x = particles[i].x;
		p.y = particles[i].y;
		p.theta = particles[i].theta;
		p.weight = 1.0;

		// 
		vector<int> assoc;
		vector<double> sense_x;
		vector<double> sense_y;

		// translate vehicle coordinate into map coordinates
		vector<LandmarkObs> map_predictions;
		for (int j=0; j<observations.size(); j++) {
			LandmarkObs obs, maptr;
			obs = observations[j];

			maptr.id = 0;
			maptr.x = p.x + (obs.x * cos(p.theta) - obs.y * sin(p.theta));
			maptr.y = p.y + (obs.x * sin(p.theta) + obs.y * cos(p.theta));
			map_predictions.push_back(maptr);
		}

		// each observation, calculate gaussian weight and then feed it into particle weight
		for (int j=0; j<map_predictions.size(); j++) {
			double distance = 1E+6;
			double dist_check = 0.0;
			for (int k=0; k<map_landmarks.landmark_list.size(); k++) {
				dist_check = dist(map_predictions[j].x, map_predictions[j].y, map_landmarks.landmark_list[k].x_f, map_landmarks.landmark_list[k].y_f);
				if (dist_check < distance) {
					distance = dist_check;
					map_predictions[j].id = k;
				}
			}
			double e_x = ((pow((map_predictions[j].x - map_landmarks.landmark_list[map_predictions[j].id].x_f),2)/pow(std_landmark[0],2)));
			double e_y = ((pow((map_predictions[j].y - map_landmarks.landmark_list[map_predictions[j].id].y_f),2)/pow(std_landmark[1],2)));
			double norm = 1/(2.*M_PI*std_landmark[0]*std_landmark[0]);

			p.weight *= norm * exp(-(e_x + e_y));

			assoc.push_back(map_landmarks.landmark_list[map_predictions[j].id].id_i);

			double map_x = map_predictions[j].x;
			double map_y = map_predictions[j].y;
			sense_x.push_back(map_x);
			sense_y.push_back(map_y);
		}
		particles[i] = SetAssociations(p, assoc, sense_x, sense_y);
		weights[i] = p.weight;
	}
		
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	// ----------
	
	discrete_distribution<int> dist_p(weights.begin(), weights.end());

	vector<Particle> new_particles;
	for (int i=0; i<num_particles; i++) {
		Particle p = particles[dist_p(gen)];
		new_particles.push_back(p);
		new_particles[i].id = i;
	}
	particles = new_particles;


}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
