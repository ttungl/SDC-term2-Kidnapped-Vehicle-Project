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

// declare a random engine to be used across multiple and various method calls
static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	num_particles = 101;

	// normal distribution for sensor noise 
	normal_distribution<double> n_x_init(0, std[0]);
	normal_distribution<double> n_y_init(0, std[1]);
	normal_distribution<double> n_theta_init(0, std[2]);

	// initialize particles
	for (int i=0; i<num_particles; i++){
		Particle p;
		p.id = i;
		p.x = x;
		p.y = y;
		p.theta = theta;
		p.weight = 1.0;
		
		// gaussian noise
		p.x += n_x_init(gen);
		p.y += n_y_init(gen);
		p.theta += n_theta_init(gen);

		// current particles
		particles.push_back(p); 

	}

	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	// normal distribution for sensor noise 
	normal_distribution<double> n_x(0, std_pos[0]);
	normal_distribution<double> n_y(0, std_pos[1]);
	normal_distribution<double> n_theta(0, std_pos[2]);

	// assign for each particle
	for(int i=0; i< num_particles; i++){
		// calculate new state
		if(fabs(yaw_rate) > 0.00001){
			// x,y,theta
			particles[i].x += n_x(gen) + velocity / yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
			particles[i].y += n_y(gen) + velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
			particles[i].theta += n_theta(gen) + yaw_rate * delta_t;
			 
		} else { 
			// x,y
			particles[i].x += n_x(gen) + velocity * delta_t * cos(particles[i].theta);
			particles[i].y += n_y(gen) + velocity * delta_t * sin(particles[i].theta);

		}
		// particles[i].theta += n_theta(gen) + yaw_rate * delta_t;
		// add gaussian noise
		// particles[i].x += n_x(gen);
		// particles[i].y += n_y(gen);
		// particles[i].theta += n_theta(gen);
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	for (unsigned int i = 0; i< observations.size(); i++){
		// current observation
		LandmarkObs obs = observations[i];

		// initialize min distance to max possible
		double min_dist = numeric_limits<double>::max();

		int map_id = -1; // init landmark id from map to be associated to the observation.

		for (unsigned int j =0; j< predicted.size(); j++){
			// current observation
			LandmarkObs p = predicted[j];

			// distance between predicted and observation
			double cur_dist = dist(obs.x, obs.y, p.x, p.y);

			// find the nearest neighbors of current observation and predicted landmark
			if (cur_dist < min_dist){
				min_dist = cur_dist;
				map_id = p.id;
			}
		}

		// update observation's id
		observations[i].id = map_id;
	}
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

	// each particle
	for (int i=0; i<num_particles; i++){
		// get particles coordinates
		double p_x = particles[i].x;
		double p_y = particles[i].y;
		double p_theta = particles[i].theta;

		// create a vector for holding map's predicted landmark locations within sensor range of the particle.
		vector<LandmarkObs> predictions;

		// each landmark of the map
		for(unsigned int j=0; j<map_landmarks.landmark_list.size(); j++){
			// get landmark coordinates
			float lm_x = map_landmarks.landmark_list[j].x_f; 
			float lm_y = map_landmarks.landmark_list[j].y_f; 
			int lm_id = map_landmarks.landmark_list[j].id_i; 

			// update for landmarks within the sensor range of the particle.
			if(fabs(lm_x - p_x) <= sensor_range && fabs(lm_y - p_y) <= sensor_range){
				predictions.push_back(LandmarkObs{lm_id, lm_x, lm_y});
			}
		}

		// for(auto lm : map_landmarks.landmark_list){
		// 	LandmarkObs pred;
		// 	pred.x = lm.x_f;
		// 	pred.y = lm.y_f;
		// 	pred.id = lm.id_i;
		// 	auto dx = pred.x - p_x;
		// 	auto dy = pred.y - p_y; 
		// 	if(dx*dx + dy*dy <= sensor_range*sensor_range)
		// 		predictions.push_back(pred);
		// }

		// observations transformed from car coordinates to map coordinates.
		vector<LandmarkObs> transformed_obs;
		
		// for(unsigned int j = 0; j< observations.size(); j++){
		// 	double t_x = cos(p_theta) * observations[j].x - sin(p_theta) * observations[j].y + p_x;
		// 	double t_y = sin(p_theta) * observations[j].x + cos(p_theta) * observations[j].y + p_y;
		// 	transformed_obs.push_back(LandmarkObs{observations[j].id, t_x, t_y}); 
		// }

		for(auto obs : observations){
			double t_x = cos(p_theta) * obs.x - sin(p_theta) * obs.y + p_x;
			double t_y = sin(p_theta) * obs.x + cos(p_theta) * obs.y + p_y;
			transformed_obs.push_back(LandmarkObs{obs.id, t_x, t_y});
		}

		// data association for predicted and transformed observations of the particle.
		dataAssociation(predictions, transformed_obs);

		// particles[i].weight = 1.0;

		for(unsigned int j =0; j<transformed_obs.size(); j++){
			double obs_x, obs_y, p_x, p_y;
			obs_x = transformed_obs[j].x;
			obs_y = transformed_obs[j].y;

			int associated_prediction = transformed_obs[j].id;

			// get coordinates
			for (unsigned int k=0; k<predictions.size(); k++){
				if(predictions[k].id==associated_prediction){
					p_x = predictions[k].x;
					p_y = predictions[k].y;
				}
			}

			// calculate the weight with multivariate gaussian.
			double s_x = std_landmark[0];
			double s_y = std_landmark[1];
			double obs_w = (1/(2*M_PI*s_x*s_y))*exp(-(pow(p_x-obs_x,2)/(2*pow(s_x,2)) + (pow(p_y - obs_y, 2)/(2*pow(s_y, 2)))));

			// total weight
			particles[j].weight *= obs_w;

		}

		

	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	// ----------
	// vector<Particle> new_particles; 
	// vector<double> weights; // get current weights
	// for(int i=0; i< num_particles; i++){
	// 	weights.push_back(particles[i].weight);
	// }

	// // random sampling
	// uniform_int_distribution<int> uni_int_dist(0, num_particles-1);
	// auto index = uni_int_dist(gen);

	// double max_weight = *max_element(weights.begin(), weights.end());

	// // uniform distribution
	// uniform_real_distribution<double> uni_real_dist(0.0, max_weight);

	// double beta = 0.0;

	// // resampling
	// for (int i=0; i<num_particles; i++){
	// 	beta += uni_real_dist(gen) * 2.0;
	// 	while (beta > weights[index]){
	// 		beta -= weights[index];
	// 		index = (index+1)%num_particles;
	// 	}
	// 	new_particles.push_back(particles[index]);
	// }

	// particles = new_particles;
	// ----------

	discrete_distribution<int> dist(weights.begin(), weights.end());
	vector<Particle> new_particles;

	for (unsigned int i=0; i<num_particles; i++){
		auto index = dist(gen);
		new_particles.push_back(move(particles[index]));
	}
	particles = move(new_particles);
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
