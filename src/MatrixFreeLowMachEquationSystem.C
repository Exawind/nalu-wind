// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "MatrixFreeLowMachEquationSystem.h"

#include "matrix_free/EquationUpdate.h"
#include "matrix_free/LocalDualNodalVolume.h"
#include "matrix_free/LowMachUpdate.h"
#include "matrix_free/MaxCourantReynolds.h"
#include "matrix_free/SparsifiedEdgeLaplacian.h"
#include "matrix_free/LocalDualNodalVolume.h"

#include "AuxFunctionAlgorithm.h"
#include "ConstantAuxFunction.h"
#include "CopyFieldAlgorithm.h"
#include "Enums.h"
#include "EquationSystems.h"
#include "FieldTypeDef.h"
#include "LinearSolverConfig.h"
#include "LinearSolvers.h"
#include "NaluEnv.h"
#include "NaluParsing.h"
#include "Realm.h"
#include "Simulation.h"
#include "SolutionOptions.h"
#include "TimeIntegrator.h"
#include "TpetraLinearSystem.h"
#include "user_functions/TaylorGreenPressureAuxFunction.h"
#include "user_functions/TaylorGreenVelocityAuxFunction.h"
#include "user_functions/SinProfileChannelFlowVelocityAuxFunction.h"
#include "utils/StkHelpers.h"

#include "Kokkos_Core.hpp"
#include "Teuchos_RCP.hpp"

#include "stk_mesh/base/Field.hpp"
#include "stk_mesh/base/FieldState.hpp"
#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/Ngp.hpp"
#include "stk_mesh/base/NgpField.hpp"
#include "stk_mesh/base/NgpFieldParallel.hpp"
#include "stk_mesh/base/NgpProfilingBlock.hpp"
#include "stk_mesh/base/Part.hpp"
#include "stk_mesh/base/Types.hpp"
#include "stk_util/parallel/ParallelReduce.hpp"
#include "stk_util/util/ReportHandler.hpp"

#include <string>
#include <utility>
#include <iomanip>
#include <ostream>
#include <stdexcept>

namespace sierra {
namespace nalu {

class Algorithm;

MatrixFreeLowMachEquationSystem::MatrixFreeLowMachEquationSystem(
  EquationSystems& eqSystems)
  : EquationSystem(eqSystems, "LowMachEQS", "temperature"),
    polynomial_order_(realm_.polynomial_order()),
    meta_(realm_.meta_data())
{
  realm_.push_equation_to_systems(this);
  ThrowRequireMsg(
    realm_.spatialDimension_ == dim,
    "Only 3D supported for matrix free heat conduction");
  ThrowRequireMsg(realm_.matrixFree_, "Only matrix free supported");
  realm_.hasFluids_ = true;
}

MatrixFreeLowMachEquationSystem::~MatrixFreeLowMachEquationSystem() = default;

void
MatrixFreeLowMachEquationSystem::check_solver_configuration(
  std::string field_name, std::string avail_precond)
{
  const auto solver_config_map =
    realm_.root()->linearSolvers_->solverTpetraConfig_;
  const auto it = solver_config_map.find(field_name);
  if (it == solver_config_map.end()) {
    throw std::runtime_error("Must specify a " + field_name + " solver");
  } else {
    // setting the preconditioner is not an option yet, so
    // check that either the preconditioner matches what
    // will actually be used, or is left blank/default
    const auto precond_type = it->second->preconditioner_name();
    if (!(precond_type == avail_precond || precond_type == "default")) {
      throw std::runtime_error(
        "Only " + avail_precond + " is supported for " + field_name);
    }
  }
}

void
MatrixFreeLowMachEquationSystem::validate_matrix_free_linear_solver_config()
{
  check_solver_configuration(
    equationSystems_.get_solver_block_name(names::velocity), "jacobi");
  check_solver_configuration(
    equationSystems_.get_solver_block_name(names::pressure), "muelu");
  check_solver_configuration(
    equationSystems_.get_solver_block_name(names::dpdx), "jacobi");
}

void
MatrixFreeLowMachEquationSystem::check_part_is_valid(
  const stk::mesh::Part* part)
{
  ThrowRequire(part);
  ThrowRequireMsg(
    matrix_free::part_is_valid_for_matrix_free(polynomial_order_, *part),
    "part " + part->name() + " has invalid topology " +
      part->topology().name() + ". Only hex8/hex27 supported");
}

void
MatrixFreeLowMachEquationSystem::register_copy_state_algorithm(
  std::string name, int length, stk::mesh::Part& part)
{
  if (!realm_.restarted_simulation()) {
    auto* field = meta_.get_field(stk::topology::NODE_RANK, name);
    auto copy = new CopyFieldAlgorithm(
      realm_, &part, field->field_state(stk::mesh::StateNP1),
      field->field_state(stk::mesh::StateN), 0, length,
      stk::topology::NODE_RANK);
    copyStateAlg_.push_back(copy);
  }
}

void
MatrixFreeLowMachEquationSystem::register_nodal_fields(stk::mesh::Part* part)
{
  check_part_is_valid(part);
  ThrowRequire(realm_.number_of_states() == 3);
  constexpr int one_state = 1;
  constexpr int three_states = 3;

  register_scalar_nodal_field_on_part(
    meta_, names::density, *part, three_states);
  realm_.augment_restart_variable_list(names::density);
  realm_.augment_property_map(
    DENSITY_ID,
    meta_.get_field<ScalarFieldType>(stk::topology::NODE_RANK, names::density));
  register_copy_state_algorithm(names::density, 1, *part);

  register_vector_nodal_field_on_part(
    meta_, names::velocity, *part, three_states, {{0, 0, 0}});
  realm_.augment_restart_variable_list(names::velocity);
  register_copy_state_algorithm(names::velocity, dim, *part);

  register_scalar_nodal_field_on_part(
    meta_, names::viscosity, *part, one_state);
  realm_.augment_property_map(
    VISCOSITY_ID, meta_.get_field<ScalarFieldType>(
                    stk::topology::NODE_RANK, names::viscosity));

  register_scalar_nodal_field_on_part(
    meta_, names::pressure, *part, one_state, 0);
  realm_.augment_restart_variable_list(names::pressure);

  register_scalar_nodal_field_on_part(
    meta_, names::scaled_filter_length, *part, one_state, 0);
  register_vector_nodal_field_on_part(
    meta_, names::dpdx_tmp, *part, one_state, {{0, 0, 0}});
  register_vector_nodal_field_on_part(
    meta_, names::dpdx, *part, one_state, {{0, 0, 0}});
  register_vector_nodal_field_on_part(
    meta_, names::body_force, *part, one_state, {{0, 0, 0}});
}

void
MatrixFreeLowMachEquationSystem::register_interior_algorithm(
  stk::mesh::Part* part)
{
  check_part_is_valid(part);
  interior_selector_ |= *part;
}

void
MatrixFreeLowMachEquationSystem::register_wall_bc(
  stk::mesh::Part* part,
  const stk::topology&,
  const WallBoundaryConditionData& bc)
{
  check_part_is_valid(part);

  auto data = bc.userData_;
  ThrowRequireMsg(
    !(data.wallFunctionApproach_ || data.ablWallFunctionApproach_),
    "Wall function not implemented");

  constexpr int one_state = 1;
  register_vector_nodal_field_on_part(
    meta_, names::velocity_bc, *part, one_state,
    {{data.u_.ux_, data.u_.uy_, data.u_.uz_}});

  auto velocity_name = std::string(names::velocity);
  auto bc_data_type = get_bc_data_type(data, velocity_name);
  ThrowRequireMsg(bc_data_type != FUNCTION_UD, "No user functions yet enabled");

  auto* bc_field =
    meta_.get_field(stk::topology::NODE_RANK, names::velocity_bc);
  auto* u_field = meta_.get_field(stk::topology::NODE_RANK, names::velocity)
                    ->field_state(stk::mesh::StateNP1);

  auto ux = data.u_;
  auto* theAuxFunc = new ConstantAuxFunction(0, dim, {ux.ux_, ux.uy_, ux.uz_});
  auto* auxAlg = new AuxFunctionAlgorithm(
    realm_, part, bc_field, theAuxFunc, stk::topology::NODE_RANK);

  realm_.initCondAlg_.push_back(auxAlg);

  CopyFieldAlgorithm* theCopyAlg = new CopyFieldAlgorithm(
    realm_, part, bc_field, u_field, 0, dim, stk::topology::NODE_RANK);
  bcDataMapAlg_.push_back(theCopyAlg);

  wall_selector_ |= *part;
}

void
MatrixFreeLowMachEquationSystem::compute_filter_scale() const
{
  {
    stk::mesh::ProfilingBlock pf("compute_filter_scale");
    auto coords = get_node_field(meta_, realm_.get_coordinates_name());
    coords.sync_to_device();
    // compute the dual node volume first, then overwrite the field
    auto dnv = get_node_field(meta_, names::scaled_filter_length);
    matrix_free::local_dual_nodal_volume(
      realm_.polynomial_order(), realm_.ngp_mesh(), interior_selector_, coords,
      dnv);

    stk::mesh::parallel_sum<double>(realm_.bulk_data(), {&dnv}, false);
    if (realm_.hasPeriodic_) {
      realm_.periodic_field_update(
        meta_.get_field(stk::topology::NODE_RANK, names::scaled_filter_length),
        1);
    }
    dnv.sync_to_device();
  }

  {
    stk::mesh::ProfilingBlock pf("compute filter scale from dnv");
    auto filter_scale = get_node_field(meta_, names::scaled_filter_length);

    double scaling = 0;
    switch (realm_.get_turbulence_model()) {
    case TurbulenceModel::LAMINAR:
      scaling = 0;
      break;
    case TurbulenceModel::SMAGORINSKY:
      scaling = realm_.get_turb_model_constant(TM_cmuCs);
      break;
    case TurbulenceModel::WALE:
      scaling = realm_.get_turb_model_constant(TM_Cw);
      break;
    default:
      throw std::runtime_error("invalid turbulence model for matrix free");
    }

    stk::mesh::for_each_entity_run(
      realm_.ngp_mesh(), stk::topology::NODE_RANK, interior_selector_,
      KOKKOS_LAMBDA(stk::mesh::FastMeshIndex mi) {
        NGP_ThrowAssert(filter_scale.get(mi, 0) > 0);
        filter_scale.get(mi, 0) =
          scaling * stk::math::cbrt(filter_scale.get(mi, 0));
      });
    filter_scale.modify_on_device();
  }
}

void
MatrixFreeLowMachEquationSystem::register_initial_condition_fcn(
  stk::mesh::Part* part,
  const std::map<std::string, std::string>& names,
  const std::map<std::string, std::vector<double>>&)
{
  check_part_is_valid(part);

  auto it = names.find(names::velocity);
  if (it != names.end()) {
    ThrowRequireMsg(
      (it->second == "TaylorGreen" || it->second == "SinProfileChannelFlow"),
      "Only TaylorGreen/SinProfileChannelFlow currently implemented for "
      "matrix-free");

    if (it->second == "TaylorGreen") {
      auto* velocity_field = meta_.get_field<VectorFieldType>(
        stk::topology::NODE_RANK, names::velocity);
      ThrowRequire(velocity_field);
      auto* vel_func = new TaylorGreenVelocityAuxFunction(0, dim);
      auto* vel_aux_alg = new AuxFunctionAlgorithm(
        realm_, part, velocity_field, vel_func, stk::topology::NODE_RANK);
      realm_.initCondAlg_.push_back(vel_aux_alg);
    } else if (it->second == "SinProfileChannelFlow") {
      auto* velocity_field = meta_.get_field<VectorFieldType>(
        stk::topology::NODE_RANK, names::velocity);
      ThrowRequire(velocity_field);
      auto* vel_func = new SinProfileChannelFlowVelocityAuxFunction(0, dim);
      auto* vel_aux_alg = new AuxFunctionAlgorithm(
        realm_, part, velocity_field, vel_func, stk::topology::NODE_RANK);
      realm_.initCondAlg_.push_back(vel_aux_alg);
    }

    if (it->second == "TaylorGreen") {
      auto pressure_field = meta_.get_field<ScalarFieldType>(
        stk::topology::NODE_RANK, names::pressure);
      ThrowRequire(pressure_field);
      auto* pressure_func = new TaylorGreenPressureAuxFunction();
      auto* pressure_aux_alg = new AuxFunctionAlgorithm(
        realm_, part, pressure_field, pressure_func, stk::topology::NODE_RANK);
      realm_.initCondAlg_.push_back(pressure_aux_alg);
    }
  }
}

void
MatrixFreeLowMachEquationSystem::initialize()
{
  stk::mesh::ProfilingBlock pf("MatrixFreeLowMachEquationSystem::initialize");
  validate_matrix_free_linear_solver_config();
  compute_filter_scale();
  compute_body_force();
  {
    stk::mesh::ProfilingBlock pfinner("create linsys");

    // this is unused but required by TpetraLinearSystem
    std::string solverName =
      realm_.equationSystems_.get_solver_block_name(names::pressure);
    auto* solver = realm_.root()->linearSolvers_->create_solver(
      solverName, realm_.name(), EQ_CONTINUITY);

    precond_linsys_ = std::unique_ptr<TpetraLinearSystem>(
      new TpetraLinearSystem(realm_, 1, this, solver));
    precond_linsys_->buildSparsifiedEdgeElemToNodeGraph(interior_selector_);
    precond_linsys_->finalizeLinearSystem();
  }

  {
    stk::mesh::ProfilingBlock pfinner("create update");
    auto blank = Teuchos::ParameterList{};
    update_ = matrix_free::make_updater<matrix_free::LowMachUpdate>(
      polynomial_order_, realm_.bulk_data(),
      realm_.solver_parameters(names::velocity),
      realm_.solver_parameters(names::pressure),
      realm_.solver_parameters(names::dpdx), interior_selector_, wall_selector_,
      *precond_linsys_->getOwnedRowsMap(),
      *precond_linsys_->getOwnedAndSharedRowsMap(),
      precond_linsys_->getRowLIDs());
  }
}

void
MatrixFreeLowMachEquationSystem::reinitialize_linear_system()
{
  initialized_ = false;
  initialize();
}

void
MatrixFreeLowMachEquationSystem::predict_state()
{
  update_->predict_state();
  update_->swap_states();
}

double
MatrixFreeLowMachEquationSystem::provide_norm() const
{
  return update_->provide_norm();
}

double
MatrixFreeLowMachEquationSystem::provide_scaled_norm() const
{
  return update_->provide_scaled_norm();
}

void
MatrixFreeLowMachEquationSystem::sync_field_on_periodic_nodes(
  std::string name, int len) const
{
  stk::mesh::ProfilingBlock pf("sync periodic nodes");
  if (realm_.hasPeriodic_) {
    const bool doCommunication = false;
    realm_.periodic_delta_solution_update(
      meta_.get_field(stk::topology::NODE_RANK, name), len, doCommunication);
  }
}

namespace {

Kokkos::Array<double, 3>
compute_scaled_gammas(const TimeIntegrator& ti)
{
  ThrowRequire(ti.get_time_step() > 0);
  ThrowRequire(ti.get_gamma1() > 0);
  return Kokkos::Array<double, 3>{
    {ti.get_gamma1() / ti.get_time_step(), ti.get_gamma2() / ti.get_time_step(),
     ti.get_gamma3() / ti.get_time_step()}};
}

double
compute_projected_timescale(const TimeIntegrator& ti)
{
  ThrowRequire(ti.get_time_step() > 0);
  ThrowRequire(ti.get_gamma1() > 0);
  return ti.get_time_step() / ti.get_gamma1();
}

void
nonlinear_iteration_banner(
  int k, int max_k, std::string name, std::ostream& stream)
{
  stream << " " << k + 1 << "/" << max_k << std::setw(15) << std::right << name
         << std::endl;
}

void
copy_field(
  const stk::mesh::NgpMesh& mesh,
  const stk::mesh::Selector& active,
  stk::mesh::NgpField<double> dst,
  stk::mesh::NgpField<double> src)
{
  src.sync_to_device();
  stk::mesh::for_each_entity_run(
    mesh, stk::topology::NODE_RANK, active,
    KOKKOS_LAMBDA(stk::mesh::FastMeshIndex mi) {
      const int len = dst.get_num_components_per_entity(mi);
      for (int d = 0; d < len; ++d) {
        dst.get(mi, d) = src.get(mi, d);
      };
    });
  dst.modify_on_device();
}

class ScopeTimer
{
public:
  ScopeTimer(double& timer) : timer_(timer), t0_(NaluEnv::self().nalu_time()) {}
  ~ScopeTimer() { timer_ += NaluEnv::self().nalu_time() - t0_; }

private:
  double& timer_;
  const double t0_;
};

} // namespace

std::string
MatrixFreeLowMachEquationSystem::get_muelu_xml_file_name()
{
  const auto solver_config_map =
    realm_.root()->linearSolvers_->solverTpetraConfig_;
  const auto block_name =
    equationSystems_.get_solver_block_name(names::pressure);
  auto it = solver_config_map.find(block_name);
  ThrowRequire(it != solver_config_map.end());
  auto precond_params = it->second->paramsPrecond();
  ThrowRequire(precond_params);
  ThrowRequire(precond_params->isParameter("xml parameter file"));
  return precond_params->get<std::string>("xml parameter file");
}

void
MatrixFreeLowMachEquationSystem::setup_and_compute_continuity_preconditioner()
{
  stk::mesh::ProfilingBlock pf("setup_continuity_preconditioner");

  auto device_mat = std::make_unique<matrix_free::NoAuraDeviceMatrix>(
    precond_linsys_->getMaxOwnedRowId(), precond_linsys_->getOwnedLocalMatrix(),
    precond_linsys_->getSharedNotOwnedLocalMatrix(),
    precond_linsys_->getRowLIDs(), precond_linsys_->getColLIDs());

  auto coords = get_node_field(meta_, realm_.get_coordinates_name());
  coords.sync_to_device();

  {
    stk::mesh::ProfilingBlock pfinner("fill sparsified laplacian");
    ScopeTimer{timerPrecond_};
    precond_linsys_->zeroSystem();
    matrix_free::assemble_sparsified_edge_laplacian(
      polynomial_order_, realm_.ngp_mesh(), interior_selector_, coords,
      *device_mat);
    precond_linsys_->loadComplete();
  }

  auto xml_name = get_muelu_xml_file_name();
  device_mat.reset();

  update_->create_continuity_preconditioner(
    coords, *precond_linsys_->getOwnedMatrix(), xml_name);
}

void
MatrixFreeLowMachEquationSystem::initialize_solve_and_update()
{
  stk::mesh::ProfilingBlock pf("initialize");
  if (initialized_) {
    return;
  }

  update_->initialize();

  setup_and_compute_continuity_preconditioner();

  // the preconditioner isn't needed anymore
  // so we can get rid of it.  This isn't a good idea
  // if we want to support mesh motion here
  precond_linsys_.reset();

  update_->compute_gradient_preconditioner();

  update_->gather_pressure();
  update_->gather_grad_p();
  update_->update_pressure_gradient(get_node_field(meta_, names::dpdx));
  sync_field_on_periodic_nodes(names::dpdx, 3);
  update_->grad_p_banner(names::dpdx, log());
  update_->gather_grad_p();
  update_->gather_velocity();
  correct_velocity(compute_projected_timescale(*realm_.timeIntegrator_));
  initialized_ = true;
}

void
MatrixFreeLowMachEquationSystem::compute_provisional_velocity(
  Kokkos::Array<double, 3> gammas)
{
  stk::mesh::ProfilingBlock pf(
    "MatrixFreeLowMachEquationSystem::compute_provisional_velocity");
  {
    stk::mesh::ProfilingBlock pfinner(
      "velocity solve, periodic sync, and banner");
    ScopeTimer st{timerSolve_};
    update_->update_provisional_velocity(
      gammas, get_node_field(meta_, names::velocity));
    sync_field_on_periodic_nodes(names::velocity, 3);
    update_->velocity_banner(names::velocity, log());
  }

  {
    stk::mesh::ProfilingBlock pfinner("update velocity element field");
    ScopeTimer st{timerAssemble_};
    update_->gather_velocity();
  }
}

void
MatrixFreeLowMachEquationSystem::correct_velocity(double proj_time_scale)
{
  stk::mesh::ProfilingBlock pf(
    "MatrixFreeLowMachEquationSystem::correct_velocity");
  {
    stk::mesh::ProfilingBlock pf("pressure solve, periodic sync, and banner");
    ScopeTimer st{timerSolve_};
    update_->update_pressure(
      proj_time_scale, get_node_field(meta_, names::pressure));
    sync_field_on_periodic_nodes(names::pressure, 1);
    update_->pressure_banner(names::pressure, log());
  }

  {
    stk::mesh::ProfilingBlock pfinner("update mdot with updated pressure");
    ScopeTimer st{timerAssemble_};
    update_->gather_pressure();
    update_->update_advection_metric(proj_time_scale);
  }

  {
    stk::mesh::ProfilingBlock pfinner("save off old grad p before updating");
    ScopeTimer st{timerMisc_};
    copy_pressure_grad();
  }

  {
    stk::mesh::ProfilingBlock pfinner(
      "gradient solve, periodic sync, and banner");
    ScopeTimer st{timerSolve_};
    update_->update_pressure_gradient(get_node_field(meta_, names::dpdx));
    sync_field_on_periodic_nodes(names::dpdx, 3);
    update_->grad_p_banner(names::dpdx, log());
  }

  {
    stk::mesh::ProfilingBlock pfinner("update grad p element field");
    ScopeTimer st{timerAssemble_};
    update_->gather_grad_p();
  }

  {
    stk::mesh::ProfilingBlock pfinner("project velocity");
    ScopeTimer st{timerMisc_};
    update_->project_velocity(
      proj_time_scale, get_node_field(meta_, names::density),
      get_node_field(meta_, names::dpdx_tmp),
      get_node_field(meta_, names::dpdx),
      get_node_field(meta_, names::velocity));
  }

  {
    stk::mesh::ProfilingBlock pfinner("update velocity element field");
    ScopeTimer st{timerAssemble_};
    update_->gather_velocity();
  }
}

std::ostream&
MatrixFreeLowMachEquationSystem::log()
{
  return NaluEnv::self().naluOutputP0();
}

void
MatrixFreeLowMachEquationSystem::copy_pressure_grad()
{
  auto dpdx_tmp = get_node_field(meta_, names::dpdx_tmp);
  auto dpdx = get_node_field(meta_, names::dpdx);
  copy_field(realm_.ngp_mesh(), interior_selector_, dpdx_tmp, dpdx);
}

void
MatrixFreeLowMachEquationSystem::compute_courant_reynolds()
{
  stk::mesh::ProfilingBlock pf(
    "MatrixFreeLowMachEquationSystem::compute_courant_reynolds");

  const auto& pp = update_->post_processor();
  auto comm = realm_.bulk_data().parallel();
  auto l_cflre =
    pp.compute_local_courant_reynolds_numbers(realm_.get_time_step());

  {
    stk::mesh::ProfilingBlock pfinner("all reduce");
    Kokkos::Array<double, 2> g_cflre;
    stk::all_reduce_max(comm, l_cflre.data(), g_cflre.data(), 2);
    realm_.maxCourant_ = g_cflre[0];
    realm_.maxReynolds_ = g_cflre[1];
  }
}

void
MatrixFreeLowMachEquationSystem::compute_body_force() const
{
  auto force = get_node_field(meta_, names::body_force);
  Kokkos::Array<double, 3> constant_force{{0, 0, 0}};

  // todo: coriolis, etc.
  {
    const auto it = realm_.solutionOptions_->srcTermParamMap_.find("momentum");
    if (it != realm_.solutionOptions_->srcTermParamMap_.end()) {
      const auto& force_vec = it->second;
      ThrowRequireMsg(force_vec.size() == 3u, "Only 3d body force");

      for (int d = 0; d < 3; ++d) {
        constant_force[d] = force_vec[d];
      }
    }
  }

  stk::mesh::for_each_entity_run(
    realm_.ngp_mesh(), stk::topology::NODE_RANK, interior_selector_,
    KOKKOS_LAMBDA(stk::mesh::FastMeshIndex mi) {
      for (int d = 0; d < 3; ++d) {
        force.get(mi, d) = constant_force[d];
      };
    });
  force.modify_on_device();
}

namespace {

matrix_free::GradTurbModel
gradient_turbulence_model(TurbulenceModel all_models)
{
  switch (all_models) {
  case TurbulenceModel::LAMINAR:
    return matrix_free::GradTurbModel::LAM;
  case TurbulenceModel::SMAGORINSKY:
    return matrix_free::GradTurbModel::SMAG;
  case TurbulenceModel::WALE:
    return matrix_free::GradTurbModel::WALE;
  default:
    throw std::runtime_error("Invalid turbulence model for matrix-free");
    return matrix_free::GradTurbModel::LAM;
  }
}
} // namespace

void
MatrixFreeLowMachEquationSystem::solve_and_update()
{
  stk::mesh::ProfilingBlock pf(
    "MatrixFreeLowMachEquationSystem::solve_and_update");
  {
    ScopeTimer st{timerInit_};
    initialize_solve_and_update();
  }

  const auto gammas = compute_scaled_gammas(*realm_.timeIntegrator_);
  const auto proj_time_scale =
    compute_projected_timescale(*realm_.timeIntegrator_);

  {
    ScopeTimer st{timerAssemble_};
    update_->gather_velocity();
  }

  for (int k = 0; k < maxIterations_; ++k) {
    nonlinear_iteration_banner(k, maxIterations_, userSuppliedName_, log());
    const auto gradient_model =
      gradient_turbulence_model(realm_.get_turbulence_model());
    update_->update_transport_coefficients(gradient_model);
    update_->compute_momentum_preconditioner(gammas[0]);
    compute_provisional_velocity(gammas);
    correct_velocity(proj_time_scale);
  }
  compute_courant_reynolds();
}

} // namespace nalu
} // namespace sierra
