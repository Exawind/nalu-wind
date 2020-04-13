// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef CONDUCTION_UPDATE_H
#define CONDUCTION_UPDATE_H

#include "matrix_free/EquationUpdate.h"
#include "matrix_free/ConductionGatheredFieldManager.h"
#include "matrix_free/ConductionSolutionUpdate.h"
#include "matrix_free/KokkosFramework.h"
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Ngp.hpp>

namespace sierra {
namespace nalu {
namespace matrix_free {

template <int p>
class ConductionUpdate final : public EquationUpdate
{
public:
  ConductionUpdate(
    stk::mesh::BulkData&,
    Teuchos::ParameterList,
    stk::mesh::Selector active,
    stk::mesh::Selector dirichlet,
    stk::mesh::Selector flux,
    stk::mesh::Selector replicas = {});

  void initialize() final;
  void swap_states() final;
  void predict_state() final;
  void compute_preconditioner(double projected_dt) final;
  void compute_update(
    Kokkos::Array<double, 3>, stk::mesh::NgpField<double>& delta) final;
  void update_solution_fields() final;
  double provide_norm() const final { return residual_norm_; };
  double provide_scaled_norm() const final { return scaled_residual_norm_; }
  void banner(std::string name, std::ostream& stream) const final;

private:
  stk::mesh::BulkData& bulk_;
  const stk::mesh::MetaData& meta_;
  stk::mesh::Selector active_;

  ConductionSolutionUpdate<p> field_update_;
  ConductionGatheredFieldManager<p> field_gather_;

  double initial_residual_{-1};
  double residual_norm_{0};
  double scaled_residual_norm_{0};
};

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
#endif
