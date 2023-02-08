#ifndef OPENFASTFSI_H
#define OPENFASTFSI_H

#include "aero/fsi/FSIturbine.h"
#include "OpenFAST.H"
#include "yaml-cpp/yaml.h"

#include <array>
#include <memory>

namespace sierra {

namespace nalu {

class OpenfastFSI
{
public:
  OpenfastFSI(const YAML::Node&);
  virtual ~OpenfastFSI() = default;

  void setup(double dtNalu, std::shared_ptr<stk::mesh::BulkData> bulk);

  void initialize(int restartFreqNalu, double curTime);

  void map_displacements(double);

  void predict_struct_states();

  void predict_struct_timestep(const double curTime);

  void advance_struct_timestep(const double curTime);

  void compute_div_mesh_velocity();

  int get_nTurbinesGlob() { return FAST.get_nTurbinesGlob(); }

  fsiTurbine* get_fsiTurbineData(int iTurb)
  {
    return fsiTurbineData_[iTurb].get();
  }

  bool get_meshmotion() { return mesh_motion_; }

  void map_loads(const int tStep, const double curTime);

  void set_rotational_displacement(
    std::array<double, 3> axis, double omega, double curTime);
  void end_openfast();

private:
  OpenfastFSI() = delete;
  OpenfastFSI(const OpenfastFSI&) = delete;

  void load(const YAML::Node&);

  void get_displacements(double);

  void compute_mapping();

  void send_loads(const double curTime);

  std::shared_ptr<stk::mesh::BulkData> bulk_;

  std::vector<std::string> partNames_;

  // Data for coupling to Openfast

  fast::OpenFAST FAST;

  fast::fastInputs fi;

  std::vector<std::unique_ptr<fsiTurbine>> fsiTurbineData_;

  bool mesh_motion_;

  bool enable_calc_loads_{false};

  int tStep_{0}; // Time step count

  int writeFreq_{
    30}; // Frequency to write line loads and deflections to netcdf file

  void read_turbine_data(int iTurb, fast::fastInputs& fi, YAML::Node turbNode);

  void bcast_turbine_params(int iTurb);

  void read_inputs(fast::fastInputs& fi, YAML::Node& ofNode);
};

} // namespace nalu

} // namespace sierra

#endif /* OPENFASTFSI_H */
