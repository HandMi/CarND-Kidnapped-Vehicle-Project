/**
 * particle_filter.h
 * 2D particle filter class.
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#ifndef PARTICLE_FILTER_H_
#define PARTICLE_FILTER_H_

#include <array>
#include <random>
#include <string>
#include "helper_functions.h"

#define EPSILON 0.0001
#define NUMBER_OF_PARTICLES 100U

struct Particle
{
  double x;
  double y;
  double theta;
  double weight;
  std::vector<int> associations;
  std::vector<double> sense_x;
  std::vector<double> sense_y;
};

class ParticleFilter
{
public:
  // Constructor
  // @param num_particles Number of particles
  ParticleFilter() : is_initialized_(false) {}

  // Destructor
  ~ParticleFilter() {}

  /**
   * init Initializes particle filter by initializing particles to Gaussian
   *   distribution around first position and all the weights to 1.
   * @param x Initial x position [m] (simulated estimate from GPS)
   * @param y Initial y position [m]
   * @param theta Initial orientation [rad]
   * @param std_dev[] Array of dimension 3 [standard deviation of x [m], 
   *   standard deviation of y [m], standard deviation of yaw [rad]]
   */
  void init(const double x, const double y, const double theta, const double std_dev[]);

  /**
   * prediction Predicts the state for the next time step
   *   using the process model.
   * @param delta_t Time between time step t and t+1 in measurements [s]
   * @param velocity Velocity of car from t to t+1 [m/s]
   * @param yaw_rate Yaw rate of car from t to t+1 [rad/s]
   */
  void prediction(const double delta_t, const double velocity, const double yaw_rate);

  /**
   * dataAssociation Finds which observations correspond to which landmarks 
   *   (likely by using a nearest-neighbors data association).
   * @param predicted_measurements Vector of predicted landmark observations
   * @param observations Vector of landmark observations
   */
  void dataAssociation(const std::vector<LandmarkObs> &predicted_measurements,
                       std::vector<LandmarkObs> &observations);

  /**
   * updateWeights Updates the weights for each particle based on the likelihood
   *   of the observed measurements. 
   * @param sensor_range Range [m] of sensor
   * @param std_landmark[] Array of dimension 2
   *   [Landmark measurement uncertainty [x [m], y [m]]]
   * @param observations Vector of landmark observations
   * @param map Map class containing map landmarks
   */
  void updateWeights(const double sensor_range, const double std_landmark[],
                     const std::vector<LandmarkObs> &observations,
                     const Map &map_landmarks);

  /**
   * resample Resamples from the updated set of particles to form
   *   the new set of particles.
   */
  void resample();

  /**
   * Set a particles list of associations, along with the associations'
   *   calculated world x,y coordinates
   * This can be a very useful debugging tool to make sure transformations 
   *   are correct and assocations correctly connected
   */
  void SetAssociations(Particle &particle, const std::vector<int> &associations,
                       const std::vector<double> &sense_x,
                       const std::vector<double> &sense_y);

  /**
   * initialized Returns whether particle filter is initialized yet or not.
   */
  const bool initialized() const
  {
    return is_initialized_;
  }

  /**
   * Used for obtaining debugging information related to particles.
   */
  std::string getAssociations(Particle best);
  std::string getSenseCoord(Particle best, std::string coord);

  // Set of current particles
  std::array<Particle, NUMBER_OF_PARTICLES> particles_;

private:
  // Flag, if filter is initialized
  bool is_initialized_;

  // Vector of weights of all particles
  std::array<double, NUMBER_OF_PARTICLES> weights_;
  // Gaussian Random Variables
  std::normal_distribution<double> x_grv_;
  std::normal_distribution<double> y_grv_;
  std::normal_distribution<double> theta_grv_;
};

#endif // PARTICLE_FILTER_H_