// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef MatrixFreeHeatCondEquationSystem_h
#define MatrixFreeHeatCondEquationSystem_h

#include "EquationSystem.h"
#include "matrix_free/EquationUpdate.h"

#include "Kokkos_Array.hpp"

#include "stk_mesh/base/Ngp.hpp"
#include "stk_mesh/base/CoordinateSystems.hpp"

#include "Realm.h"

#include "Tpetra_Map.hpp"

namespace stk {
struct topology;
}

namespace sierra {
namespace nalu {

class MatrixFreeHeatCondEquationSystem final : public EquationSystem
{
public:
  static constexpr int dim = 3;
  MatrixFreeHeatCondEquationSystem(EquationSystems& equationSystems);
  virtual ~MatrixFreeHeatCondEquationSystem();

  void initialize() final;
  virtual void register_nodal_fields(const stk::mesh::PartVector& part_vec);
  void register_interior_algorithm(stk::mesh::Part* part) final;
  void register_wall_bc(
    stk::mesh::Part* part,
    const stk::topology& partTopo,
    const WallBoundaryConditionData& wallBCData) final;

  double provide_norm() const final;
  double provide_scaled_norm() const final;
  void solve_and_update() final;
  void reinitialize_linear_system() final;
  void predict_state() final;
  void load(const YAML::Node& node) final { EquationSystem::load(node); }

private:
  struct names
  {
    static constexpr auto temperature = "temperature";
    static constexpr auto delta = "tTmp";
    static constexpr auto nalu_gid = "nalu_global_id";
    static constexpr auto tpetra_gid = "tpet_global_id";
    static constexpr auto qbc = "temperature_bc";
    static constexpr auto flux = "heat_flux_bc";
    static constexpr auto volume_weight = "volumetric_heat_capacity";
    static constexpr auto thermal_conductivity = "thermal_conductivity";
    static constexpr auto density = "density";
    static constexpr auto specific_heat = "specific_heat";
    static constexpr auto dtdx = "dtdx";
  };

  void initialize_solve_and_update();
  void sync_field_on_periodic_nodes(std::string name, int len) const;
  void compute_volumetric_heat_capacity() const;

  const int polynomial_order_{1};
  stk::mesh::MetaData& meta_;

  stk::mesh::Selector interior_selector_;
  stk::mesh::Selector dirichlet_selector_;
  stk::mesh::Selector flux_selector_;

  std::unique_ptr<matrix_free::EquationUpdate> update_;
  std::unique_ptr<matrix_free::GradientUpdate> grad_;

  bool initialized_{false};
};

} // namespace nalu
} // namespace sierra
#endif
