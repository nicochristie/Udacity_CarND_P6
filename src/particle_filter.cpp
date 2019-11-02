/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"
#include "map.h"

using std::string;
using std::vector;
using namespace std;

#define EPS 0.001

std::default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  if (is_initialized) return;

  num_particles = 50;  // TODO: Set the number of particles
  
  normal_distribution<double> dist_x(0, std[0]);
  normal_distribution<double> dist_y(0, std[1]);
  normal_distribution<double> dist_theta(0, std[2]);
  weights.resize(num_particles); // For resample()
  
  for (int i = 0; i < num_particles; i++) 
  {
    Particle particle;
    particle.id = i;
    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);
    particle.weight = 1.0;    
    particles.push_back(particle);
    
    weights[i] = 0.8;
  }

  is_initialized = true;  
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  
  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[2]);

  for (int i = 0; i < num_particles; i++)
  {
    Particle p = particles[i];
    
    if (fabs(yaw_rate) < EPS)
    {
      p.x += velocity * delta_t * cos(p.theta);
      p.y += velocity * delta_t * sin(p.theta);
      // assume Theta is not changing
    }
    else
    {
      p.x += (velocity/yaw_rate) * (sin(p.theta + yaw_rate*delta_t) - sin(p.theta));
      p.y += (velocity/yaw_rate) * (cos(p.theta) - cos(p.theta + yaw_rate*delta_t));
      p.theta += yaw_rate * delta_t;
    }

    // Add some noise
    p.x += dist_x(gen);
    p.y += dist_y(gen);
    p.theta += dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  
  for (unsigned int i = 0; i < observations.size(); i++) 
  {
    double id = -1; // target prediction id for this observation
    double minDist = numeric_limits<double>::max();
    
    for (unsigned int j = 0; j < predicted.size(); j++)
    {
      double d = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
      if (d < minDist) { minDist = d; id = predicted[j].id; }
    }
    
    //if (id != -1) ...?
      observations[i].id = id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

  // save constant values for weight calculation
  double K = 1 / (2 * M_PI * std_landmark[0] * std_landmark[1]);
  double ux_s = std_landmark[0] * std_landmark[0]; // mu X squared
  double uy_s = std_landmark[1] * std_landmark[1]; // mu Y squared
  //std::cout << "K: " << K << " - ux_s: " << ux_s << " - uy_s: " << uy_s << std::endl;
  
  for (int i = 0; i < num_particles; i++)
  {
    Particle p = particles[i];
    p.weight = 1.0;
    
    // Transform observations into map coordinates
    vector<LandmarkObs> transformedObservations;
    for(unsigned int j = 0; j < observations.size(); j++) 
    {
      LandmarkObs to, o = observations[j];
      to.id = o.id;
      to.x = cos(p.theta)*o.x - sin(p.theta)*o.y + p.x;
      to.y = sin(p.theta)*o.x + cos(p.theta)*o.y + p.y;
      transformedObservations.push_back(to);
    }
    
    // Keep only those Landmarks in range of sensor
    vector<LandmarkObs> inRangeLandmarks;
    for(unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++)
    {
      Map::single_landmark_s l = map_landmarks.landmark_list[j];
      double d = dist(p.x, p.y, l.x_f, l.y_f);

      if ( d <= sensor_range )
        inRangeLandmarks.push_back(LandmarkObs{ l.id_i, l.x_f, l.y_f });
    }
    
    // Associate Data
    dataAssociation(inRangeLandmarks, transformedObservations);
    
    // Update weights
    for (unsigned int j = 0; j < transformedObservations.size(); j++)
    {
      int matchIndex;
      LandmarkObs o = transformedObservations[j];
      
      for (unsigned int k = 0; k < inRangeLandmarks.size(); k++)
      {
        LandmarkObs l = inRangeLandmarks[k];
        if (l.id == o.id) { matchIndex = k; break; }
      }
      
      LandmarkObs l = inRangeLandmarks[matchIndex];
      
      double dx = o.x - l.x, dy = o.y - l.y;
      p.weight *= K * exp( -0.5 * (dx*dx/ux_s + dy*dy/uy_s) );
      if (p.weight == 0) p.weight += EPS;
      weights[i] = p.weight;

      //std::cout << "  ___________________________________________________________" << std::endl;
      //std::cout << "  o.id: " << o.id << ", match index: " << matchIndex << " | inRangeLandmarks[matchIndex].id: " << l.id << std::endl;
      //std::cout << "  dx: " << dx << ", dy: " << dy << std::endl;
      //std::cout << "Particle weight: " << p.weight << std::endl;
    }
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

  double maxWeight = numeric_limits<double>::min();
  for (int i = 0; i < num_particles; i++)
  {
    weights[i] = particles[i].weight;
    if (particles[i].weight > maxWeight)
      maxWeight = particles[i].weight;
  }
  
  vector<Particle> newParticles(num_particles);
  uniform_real_distribution<double> distBeta(0.0, maxWeight);
  uniform_int_distribution<int> distIndex(0, num_particles - 1);

  // Spin the wheel
  double beta = 0.0;
  int index = distIndex(gen);
  for (int i = 0; i < num_particles; i++)
  {
    beta += distBeta(gen);
    while (beta > weights[index])
    {
      beta -= weights[index++];
      index = index % num_particles;
    }
    newParticles[i] = particles[index];
  }
  
  particles = newParticles; // Replace new sampling
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}