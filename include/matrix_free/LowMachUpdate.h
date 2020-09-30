// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef LOWMACH_SOLUTION_UPDATE_H
#define LOWMACH_SOLUTION_UPDATE_H

#include "matrix_free/ContinuitySolutionUpdate.h"
#include "matrix_free/EquationUpdate.h"
#include "matrix_free/GradientSolutionUpdate.h"
#include "matrix_free/KokkosViewTypes.h"
#include "matrix_free/LinSysInfo.h"
#include "matrix_free/LowMachGatheredFieldManager.h"
#include "matrix_free/MomentumSolutionUpdate.h"
#include "matrix_free/StkToTpetraMap.h"

#include "stk_mesh/base/Selector.hpp"
#include "stk_mesh/base/NgpField.hpp"

#include "Teuchos_RCP.hpp"

#include "Kokkos_Array.hpp"

#include "Teuchos_ParameterList.hpp"
#include "Tpetra_CrsMatrix_fwd.hpp"
#include "Tpetra_Export_fwd.hpp"

#include <iosfwd>

namespace stk {
namespace mesh {
class BulkData;
} // namespace mesh
} // namespace stk

namespace sierra {
namespace nalu {
namespace matrix_free {

template <int p>
class LowMachPostProcessP : public LowMachPostProcess
{
public:
  LowMachPostProcessP(LowMachGatheredFieldManager<p>& gather) : gather_(gather)
  {
  }
  Kokkos::Array<double, 2>
  compute_local_courant_reynolds_numbers(double dt) const;

private:
  LowMachGatheredFieldManager<p>& gather_;
};

template <int p>
class LowMachUpdate : public LowMachEquationUpdate
{
public:
  LowMachUpdate(
    stk::mesh::BulkData&,
    Teuchos::ParameterList,
    Teuchos::ParameterList,
    Teuchos::ParameterList,
    stk::mesh::Selector,
    stk::mesh::Selector,
    Kokkos::View<gid_type*>);

  LowMachUpdate(
    stk::mesh::BulkData& bulk_in,
    Teuchos::ParameterList params_mom,
    Teuchos::ParameterList params_cont,
    Teuchos::ParameterList params_grad,
    stk::mesh::Selector active,
    stk::mesh::Selector dirichlet_wall,
    const Tpetra::Map<>& owned,
    const Tpetra::Map<>& owned_and_shared,
    Kokkos::View<const lid_type*> elids);

  // u^* -> p -> mdot -> Gp -> proj(u^*) LOOP
  void initialize();
  void swap_states();
  void predict_state();
  void update_provisional_velocity(
    Kokkos::Array<double, 3>, stk::mesh::NgpField<double>&);
  void update_pressure(double, stk::mesh::NgpField<double>&);
  void update_pressure_gradient(stk::mesh::NgpField<double>&);
  void project_velocity(
    double proj_time_scale,
    stk::mesh::NgpField<double> rho,
    stk::mesh::NgpField<double> gp,
    stk::mesh::NgpField<double> gp_star,
    stk::mesh::NgpField<double>& u);

  void gather_velocity();
  void gather_pressure();
  void gather_grad_p();
  void update_transport_coefficients(GradTurbModel model);
  void update_advection_metric(double dt);

  double provide_norm() const { return residual_norm_[VEL]; };
  double provide_scaled_norm() const
  {
    return residual_norm_[VEL] /
           std::max(
             std::numeric_limits<double>::epsilon(), initial_residual_[VEL]);
  }

  void velocity_banner(std::string, std::ostream&) const;
  void pressure_banner(std::string, std::ostream&) const;
  void grad_p_banner(std::string, std::ostream&) const;

  void update_gathered_fields();
  void compute_momentum_preconditioner(double gamma);
  void compute_gradient_preconditioner();
  void create_continuity_preconditioner(
    const stk::mesh::NgpField<double>& coords,
    Tpetra::CrsMatrix<>& mat,
    std::string xmlname = "milestone.xml");

  const LowMachPostProcess& post_processor() const { return post_process_; }

private:
  stk::mesh::BulkData& bulk_;
  const stk::mesh::Selector active_;
  const stk::mesh::Selector dirichlet_;
  const StkToTpetraMaps linsys_;
  const Tpetra::Export<> exporter_;
  const const_elem_offset_view<p> offsets_;
  const const_face_offset_view<p> exposed_face_offsets_;
  const const_node_offset_view dirichlet_offsets_;

  LowMachGatheredFieldManager<p> field_gather_;
  LowMachPostProcessP<p> post_process_;

  MomentumSolutionUpdate<p> momentum_update_;
  ContinuitySolutionUpdate<p> continuity_update_;
  GradientSolutionUpdate<p> gradient_update_;

  Teuchos::ParameterList muelu_params{};

  enum { VEL = 0, CONT = 1, GP = 2 };
  mutable Kokkos::Array<double, 3> initial_residual_{{-1, -1, -1}};
  mutable Kokkos::Array<double, 3> residual_norm_{{0, 0, 0}};
};

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
#endif
