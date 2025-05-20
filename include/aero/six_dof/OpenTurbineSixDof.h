#ifndef OPENTURBINESIXDOF_H
#define OPENTURBINESIXDOF_H

#include "yaml-cpp/yaml.h"

#include <array>
#include <memory>

#include <stk_mesh/base/BulkData.hpp>

#include "FieldTypeDef.h"

namespace sierra {

namespace nalu {

enum BodyType {
  Point=1,
  NumberOfBodyTypes
};

struct Tether
{
  std::array<double,3> fairlead_position = {0.0, 0.0, 0.0};
  std::array<double,3> anchor_position = {0.0, 0.0, 0.0};
  double stiffness{0.0};
  double initial_length{0.0};
};

struct PointMass 
{
  std::array<double,9> moments_of_inertia = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  std::array<double,3> center_of_mass = {0.0, 0.0, 0.0};
  double mass{0.0};
  std::vector<Tether> tethers; 
  std::vector<std::string> forcing_surfaces;
  std::vector<std::string> moving_mesh_blocks;
};

class OpenTurbineSixDof
{
public:
  OpenTurbineSixDof(const YAML::Node&);
  virtual ~OpenTurbineSixDof() = default;

  void setup(double dtNalu, std::shared_ptr<stk::mesh::BulkData> bulk);

  void initialize(int restartFreqNalu, double curTime);

  void map_displacements(double, bool);

  void predict_struct_states();

  void predict_struct_timestep(const double curTime);

  void advance_struct_timestep(const double curTime);

  void compute_div_mesh_velocity();

  void map_loads(const int tStep, const double curTime);

private:
  OpenTurbineSixDof() = delete;
  OpenTurbineSixDof(const OpenTurbineSixDof&) = delete;

  void load_point(const YAML::Node&);

  void load(const YAML::Node&);

  void get_displacements(double);

  void compute_mapping();

  void send_loads(const double curTime);
  void timer_start(std::pair<double, double>& timer);
  void timer_stop(std::pair<double, double>& timer);

  std::shared_ptr<stk::mesh::BulkData> bulk_;

  bool enable_calc_loads_;

  int tStep_{0}; // Time step count

  double dt_{-1.0}; // Store nalu-wind step

  int writeFreq_{
    30}; // Frequency to write line loads and deflections to netcdf file

  int number_of_bodies_{0};

  std::vector<PointMass> point_bodies_;

};

} // namespace nalu

} // namespace sierra

#endif 
