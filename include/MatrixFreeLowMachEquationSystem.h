// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef MatrixFreeLowMachEquationSystem_h
#define MatrixFreeLowMachEquationSystem_h

#include "EquationSystem.h"
#include "Kokkos_Array.hpp"

#include "stk_mesh/base/Selector.hpp"

#include <iosfwd>
#include <map>
#include <memory>
#include <vector>

namespace YAML {
class Node;
}

namespace stk {
class MetaData;
class Part;
struct topology;
} // namespace stk

namespace sierra {
namespace nalu {

class EquationSystems;
class TpetraLinearSystem;

namespace matrix_free {
class LowMachEquationUpdate;
}

class MatrixFreeLowMachEquationSystem final : public EquationSystem
{
public:
  static constexpr int dim = 3;
  MatrixFreeLowMachEquationSystem(EquationSystems&);
  virtual ~MatrixFreeLowMachEquationSystem();
  void initialize() final;
  virtual void register_nodal_fields(const stk::mesh::PartVector& part_vec);
  void register_interior_algorithm(stk::mesh::Part*) final;
  double provide_norm() const final;
  double provide_scaled_norm() const final;
  void solve_and_update() final;
  void reinitialize_linear_system() final;
  void predict_state() final;

  void register_initial_condition_fcn(
    stk::mesh::Part*,
    const std::map<std::string, std::string>&,
    const std::map<std::string, std::vector<double>>&) final;

  virtual void register_wall_bc(
    stk::mesh::Part*,
    const stk::topology&,
    const WallBoundaryConditionData&) final;

  virtual void register_inflow_bc(
    stk::mesh::Part*,
    const stk::topology&,
    const InflowBoundaryConditionData&) final
  {
    throw std::runtime_error("inflow not implemented for matrix free");
  }

  virtual void register_open_bc(
    stk::mesh::Part*,
    const stk::topology&,
    const OpenBoundaryConditionData&) final
  {
    throw std::runtime_error("open not implemented for matrix free");
  }

  virtual void register_symmetry_bc(
    stk::mesh::Part*,
    const stk::topology&,
    const SymmetryBoundaryConditionData&) final
  {
    throw std::runtime_error("symmetry not implemented for matrix free");
  }

  virtual void register_abltop_bc(
    stk::mesh::Part*,
    const stk::topology&,
    const ABLTopBoundaryConditionData&) final
  {
    throw std::runtime_error("abltop not implemented for matrix free");
  }

  virtual void
  register_non_conformal_bc(stk::mesh::Part*, const stk::topology&) final
  {
    throw std::runtime_error("nonconformal not implemented for matrix free");
  }

  virtual void register_overset_bc()
  {
    throw std::runtime_error("overset not implemented for matrix free");
  }

  virtual void register_surface_pp_algorithm(
    const PostProcessingData&, stk::mesh::PartVector&) final
  {
    throw std::runtime_error(
      "surface post-processing not implemented for matrix free");
  }

  void compute_filter_scale() const;
  void compute_body_force() const;

private:
  struct names
  {
    static constexpr auto density = "density";
    static constexpr auto velocity = "velocity";
    static constexpr auto velocity_bc = "velocity_bc";
    static constexpr auto pressure = "pressure";
    static constexpr auto viscosity = "viscosity";
    static constexpr auto scaled_filter_length = "scaled_filter_length";
    static constexpr auto dpdx_tmp = "dpdx_tmp";
    static constexpr auto dpdx = "dpdx";
    static constexpr auto body_force = "body_force";
    static constexpr auto tpetra_gid = "tpet_global_id";
  };

  std::ostream& log();
  void validate_matrix_free_linear_solver_config();
  void check_solver_configuration(std::string, std::string);
  void copy_pressure_grad();
  void compute_provisional_velocity(Kokkos::Array<double, 3> gammas);
  void correct_velocity(double proj_time_scale);
  void initialize_solve_and_update();
  void sync_field_on_periodic_nodes(std::string name, int len) const;
  void setup_and_compute_continuity_preconditioner();
  void compute_courant_reynolds();
  void check_part_is_valid(const stk::mesh::PartVector&);
  void register_copy_state_algorithm(
    std::string, int dim, const stk::mesh::PartVector&);

  std::string get_muelu_xml_file_name();

  const int polynomial_order_{1};
  stk::mesh::MetaData& meta_;
  stk::mesh::Selector interior_selector_;
  stk::mesh::Selector wall_selector_;
  std::unique_ptr<matrix_free::LowMachEquationUpdate> update_;
  std::unique_ptr<TpetraLinearSystem> precond_linsys_;
  bool initialized_{false};
};

} // namespace nalu
} // namespace sierra
#endif
