/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huangs
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <string>
#include <vector>

#include "helper_functions.h"

// Fixed seed for reproducible results
static std::mt19937 gen{42U};

void ParticleFilter::init(const double x, const double y, const double theta, const double std_dev[])
{
  /**
   * Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * Add random Gaussian noise to each particle.
   */
  if (!is_initialized_)
  {
    // Initialize Gaussian Random Variables
    x_grv_ = std::normal_distribution<double>(0.0, std_dev[0]);
    y_grv_ = std::normal_distribution<double>(0.0, std_dev[1]);
    theta_grv_ = std::normal_distribution<double>(0.0, std_dev[2]);

    for (auto &particle : particles_)
    {
      particle.x = x_grv_(gen) + x;
      particle.y = y_grv_(gen) + y;
      particle.theta = theta_grv_(gen) + theta;
      particle.weight = 1.0;
    }
    is_initialized_ = true;
  }
}

void ParticleFilter::prediction(const double delta_t, const double velocity, const double yaw_rate)
{
  /**
   * Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  for (auto &particle : particles_)
  {
    if (fabs(yaw_rate) < EPSILON)
    {
      particle.x += velocity * delta_t * cos(particle.theta);
      particle.y += velocity * delta_t * sin(particle.theta);
    }
    else
    {
      particle.x += velocity * (sin(particle.theta + yaw_rate * delta_t) - sin(particle.theta)) / yaw_rate;
      particle.y += velocity * (cos(particle.theta) - cos(particle.theta + yaw_rate * delta_t)) / yaw_rate;
      particle.theta += yaw_rate * delta_t;
    }
    // Noise
    particle.x += x_grv_(gen);
    particle.y += y_grv_(gen);
    particle.theta += theta_grv_(gen);
  }
}

void ParticleFilter::dataAssociation(const std::vector<LandmarkObs> &predicted_measurements,
                                     std::vector<LandmarkObs> &observations)
{
  /**
   * Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  for (auto &observation : observations)
  {
    auto minimum_distance = std::numeric_limits<double>::max();
    for (const auto &prediction : predicted_measurements)
    {
      auto distance = dist(prediction.x, prediction.y, observation.x, observation.y);
      if (distance < minimum_distance)
      {
        observation.id = prediction.id;
        minimum_distance = distance;
      }
    }
  }
}

void ParticleFilter::updateWeights(const double sensor_range, const double std_landmark[],
                                   const std::vector<LandmarkObs> &observations,
                                   const Map &map_landmarks)
{
  /**
   * Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You wi of elementll need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  const double normalization = 1.0 / (2.0 * M_PI * std_landmark[0] * std_landmark[1]);
  double total_weight = 0.0;

  for (auto &particle : particles_)
  {
    std::cout << "new particle" << std::endl;
    auto transformed_observations(observations);
    // Transform to map coordinates
    for (auto &observation : transformed_observations)
    {
      auto x_obs = observation.x;
      auto y_obs = observation.y;
      observation.x = particle.x + (cos(particle.theta) * x_obs - sin(particle.theta) * y_obs);
      observation.y = particle.y + (sin(particle.theta) * x_obs + cos(particle.theta) * y_obs);
    }

    // Filter landmarks according to sensor range
    std::vector<LandmarkObs> visible_landmarks;
    for (const auto &landmark : map_landmarks.landmark_list)
    {
      if (dist(particle.x, particle.y, landmark.x_f, landmark.y_f) < sensor_range)
      {
        visible_landmarks.push_back(LandmarkObs{landmark.id_i, landmark.x_f, landmark.y_f});
      }
    }

    // Associate
    dataAssociation(visible_landmarks, transformed_observations);

    // Compute Weights
    particle.weight = 1.0;
    for (const auto &observation : transformed_observations)
    {
      for (const auto &landmark : visible_landmarks)
      {
        if (landmark.id == observation.id)
        {
          double exponent = -0.5 * dist2(observation.x, observation.y, landmark.x, landmark.y, std_landmark);
          particle.weight *= normalization * std::exp(exponent);
        }
      }
    }
    total_weight += particle.weight;
  }

  // Normalize
  for (uint8_t i = 0; i < particles_.size(); i++)
  {
    particles_[i].weight /= total_weight;
    weights_[i] = particles_[i].weight;
  }
}

void ParticleFilter::resample()
{
  /**
   * Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

  std::array<Particle, NUMBER_OF_PARTICLES> resampled_particles;
  std::discrete_distribution<> random_index(weights_.begin(), weights_.end());

  for (auto &new_particle : resampled_particles)
  {
    new_particle = particles_[random_index(gen)];
  }
  particles_ = resampled_particles;
}

void ParticleFilter::SetAssociations(Particle &particle,
                                     const std::vector<int> &associations,
                                     const std::vector<double> &sense_x,
                                     const std::vector<double> &sense_y)
{
  // particle: the particle to which assign each listed association,
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

std::string ParticleFilter::getAssociations(Particle best)
{
  std::vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  std::string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}

std::string ParticleFilter::getSenseCoord(Particle best, std::string coord)
{
  std::vector<double> v;

  if (coord == "X")
  {
    v = best.sense_x;
  }
  else
  {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  std::string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}