#ifndef OPENTURBINESIXDOF_H
#define OPENTURBINESIXDOF_H

#include <array>
#include <memory>

#include "yaml-cpp/yaml.h"

#include <stk_mesh/base/BulkData.hpp>

#include <interfaces/cfd/interface.hpp>
#include <interfaces/cfd/interface_builder.hpp>
#include <interfaces/cfd/interface_input.hpp>

#include "FieldTypeDef.h"
#include "aero/fsi/CalcLoads.h"
#include "aero/fsi/MapLoad.h"

namespace sierra {

namespace nalu {

struct Tether
{
  std::array<double,3> fairlead_position = {0.0, 0.0, 0.0};
  std::array<double,3> anchor_position = {0.0, 0.0, 0.0};
  double stiffness{0.0};
  double initial_length{0.0};
};
struct PointMass 
{
  std::shared_ptr<openturbine::interfaces::cfd::Interface> openturbine_interface = nullptr;
  std::array<double,9> moments_of_inertia = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  std::array<double,3> center_of_mass = {0.0, 0.0, 0.0};
  double mass{0.0};
  std::vector<Tether> tethers; 
  std::vector<std::string> forcing_surface_names;
  std::vector<std::string> moving_mesh_block_names;
  stk::mesh::PartVector forcing_surfaces;
  stk::mesh::PartVector moving_mesh_blocks;
  std::string restart_file_name = "point.restart";
  GenericFieldType* total_force = nullptr;
  std::shared_ptr<stk::mesh::BulkData> bulk = nullptr;
  std::shared_ptr<CalcLoads> calc_loads = nullptr;
};

class OpenTurbineSixDof
{
public:
  OpenTurbineSixDof(const YAML::Node&);
  virtual ~OpenTurbineSixDof() = default;

  void setup(double dtNalu, std::shared_ptr<stk::mesh::BulkData> bulk);

  void initialize(int restartFreqNalu, double curTime);

  void map_displacements(double, bool);

  void advance_struct_timestep();

  void map_loads();

  const stk::mesh::PartVector
  get_mesh_blocks() {
    stk::mesh::PartVector all_mesh_blocks;
    for (auto && point : point_bodies_) {
      for (auto && block : point.moving_mesh_blocks) {
        all_mesh_blocks.push_back(block);
      }
    }
    return all_mesh_blocks;
  }

private:
  OpenTurbineSixDof() = delete;
  OpenTurbineSixDof(const OpenTurbineSixDof&) = delete;

  void map_displacements_point(PointMass &point, bool updateCur);

  void setup_point(PointMass &point, const double dtNalu, std::shared_ptr<stk::mesh::BulkData> bulk);

  void map_loads_point(PointMass &point);

  void load_point(const YAML::Node&);

  void load(const YAML::Node&);

  void send_loads(const double curTime);
  void timer_start(std::pair<double, double>& timer);
  void timer_stop(std::pair<double, double>& timer);

  std::shared_ptr<stk::mesh::BulkData> bulk_;

  bool enable_calc_loads_;

  int tStep_{0}; // Time step count

  double dt_{-1.0}; // Store nalu-wind step

  std::array<double, 3> gravity_ = {0.0, 0.0, 0.0};

  int writeFreq_{
    30}; // Frequency to write line loads and deflections to netcdf file

  int number_of_bodies_{0};

  int restart_frequency_{0};

  std::vector<PointMass> point_bodies_;
  std::vector<openturbine::interfaces::cfd::Interface> point_body_interfaces_;
};

} // namespace nalu

} // namespace sierra

#endif 
