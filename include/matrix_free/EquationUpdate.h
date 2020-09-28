// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef EQUATION_UPDATE_H
#define EQUATION_UPDATE_H

#include "matrix_free/PolynomialOrders.h"
#include "matrix_free/LowMachInfo.h"

#include "Kokkos_Array.hpp"
#include "stk_mesh/base/NgpField.hpp"
#include "stk_mesh/base/Part.hpp"

#include <memory>

#include "Tpetra_CrsMatrix_fwd.hpp"

namespace sierra {
namespace nalu {
namespace matrix_free {

class EquationUpdate
{
public:
  using update_type = EquationUpdate;
  virtual ~EquationUpdate() = default;
  virtual void initialize() = 0;
  virtual void swap_states() = 0;
  virtual void predict_state() = 0;
  virtual void compute_preconditioner(double = -1) = 0;
  virtual void
  compute_update(Kokkos::Array<double, 3>, stk::mesh::NgpField<double>&) = 0;
  virtual void update_solution_fields() = 0;
  virtual double provide_norm() const = 0;
  virtual double provide_scaled_norm() const = 0;
  virtual void banner(std::string, std::ostream&) const = 0;
};

class GradientUpdate
{
public:
  using update_type = GradientUpdate;

  virtual ~GradientUpdate() = default;
  virtual void gradient(
    const stk::mesh::NgpField<double>&, stk::mesh::NgpField<double>&) = 0;
  virtual void banner(std::string, std::ostream&) const = 0;
  virtual void reset_initial_residual() = 0;
};

class LowMachPostProcess
{
public:
  virtual ~LowMachPostProcess() = default;
  virtual Kokkos::Array<double, 2>
  compute_local_courant_reynolds_numbers(double dt) const = 0;
};

class LowMachEquationUpdate
{
public:
  using update_type = LowMachEquationUpdate;
  virtual ~LowMachEquationUpdate() = default;
  virtual void initialize() = 0;
  virtual void swap_states() = 0;
  virtual void predict_state() = 0;

  virtual void update_provisional_velocity(
    Kokkos::Array<double, 3>, stk::mesh::NgpField<double>&) = 0;
  virtual void update_pressure(double, stk::mesh::NgpField<double>&) = 0;
  virtual void update_pressure_gradient(stk::mesh::NgpField<double>&) = 0;

  virtual void project_velocity(
    double,
    stk::mesh::NgpField<double>,
    stk::mesh::NgpField<double>,
    stk::mesh::NgpField<double>,
    stk::mesh::NgpField<double>&) = 0;

  virtual void gather_velocity() = 0;
  virtual void gather_pressure() = 0;
  virtual void gather_grad_p() = 0;
  virtual void update_transport_coefficients(GradTurbModel update) = 0;
  virtual void update_advection_metric(double dt) = 0;

  virtual double provide_norm() const = 0;
  virtual double provide_scaled_norm() const = 0;

  virtual void velocity_banner(std::string, std::ostream&) const = 0;
  virtual void pressure_banner(std::string, std::ostream&) const = 0;
  virtual void grad_p_banner(std::string, std::ostream&) const = 0;

  virtual void compute_gradient_preconditioner() = 0;
  virtual void compute_momentum_preconditioner(double gamma) = 0;
  virtual void create_continuity_preconditioner(
    const stk::mesh::NgpField<double>& coords,
    Tpetra::CrsMatrix<>& mat,
    std::string xmlname = "milestone.xml") = 0;

  virtual const LowMachPostProcess& post_processor() const = 0;
};

template <template <int> class PhysicsUpdate, typename... Args>
std::unique_ptr<typename PhysicsUpdate<inst::P1>::update_type>
make_updater(int p, Args&&... args)
{
  switch (p) {
  case inst::P2:
    return std::unique_ptr<PhysicsUpdate<inst::P2>>(
      new PhysicsUpdate<inst::P2>(std::forward<Args>(args)...));
  case inst::P3:
    return std::unique_ptr<PhysicsUpdate<inst::P3>>(
      new PhysicsUpdate<inst::P3>(std::forward<Args>(args)...));
  case inst::P4:
    return std::unique_ptr<PhysicsUpdate<inst::P4>>(
      new PhysicsUpdate<inst::P4>(std::forward<Args>(args)...));
  default:
    return std::unique_ptr<PhysicsUpdate<inst::P1>>(
      new PhysicsUpdate<inst::P1>(std::forward<Args>(args)...));
  }
}

inline bool
part_is_valid_for_matrix_free(int order, const stk::mesh::Part& part)
{
  if (
    part.topology() == stk::topology::HEX_8 ||
    part.topology() == stk::topology::QUAD_4) {
    return order == 1;
  }

  if (
    part.topology() == stk::topology::HEX_27 ||
    part.topology() == stk::topology::QUAD_9) {
    return order == 2;
  }

  if (part.topology().is_superelement()) {
    return order == floor(std::cbrt(part.topology().num_nodes() + 1) - 1);
  }

  if (part.topology().is_superface()) {
    return order == floor(std::sqrt(part.topology().num_nodes() + 1) - 1);
  }

  for (const auto* subpart : part.subsets()) {
    if (subpart == nullptr) {
      return false;
    }
    return part_is_valid_for_matrix_free(order, *subpart);
  }
  return false;
}

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
#endif
