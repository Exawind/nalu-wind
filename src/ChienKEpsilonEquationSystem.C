// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <ChienKEpsilonEquationSystem.h>
#include <AlgorithmDriver.h>
#include <FieldFunctions.h>
#include <master_element/MasterElement.h>
#include <master_element/MasterElementRepo.h>
#include <NaluEnv.h>
#include <NaluParsing.h>
#include <TotalDissipationRateEquationSystem.h>
#include <SolutionOptions.h>
#include <TurbKineticEnergyEquationSystem.h>
#include <Realm.h>

// ngp
#include "FieldTypeDef.h"
#include "ngp_algorithms/GeometryAlgDriver.h"
#include "ngp_algorithms/WallFuncGeometryAlg.h"
#include "ngp_utils/NgpLoopUtils.h"
#include "ngp_utils/NgpFieldUtils.h"
#include "ngp_utils/NgpFieldBLAS.h"

// stk_util
#include <stk_util/parallel/Parallel.hpp>
#include "utils/StkHelpers.h"

// stk_mesh/base/fem
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldParallel.hpp>

#include <stk_mesh/base/MetaData.hpp>

// stk_io
#include <stk_io/IossBridge.hpp>

// basic c++
#include <cmath>
#include <vector>
#include <iomanip>

// ngp
#include "ngp_utils/NgpFieldBLAS.h"

namespace sierra {
namespace nalu {

//==========================================================================
// Class Definition
//==========================================================================
// ChienKEpsilonEquationSystem - manage SST
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
ChienKEpsilonEquationSystem::ChienKEpsilonEquationSystem(
  EquationSystems& eqSystems)
  : EquationSystem(eqSystems, "ChienKEpsilonWrap"),
    tkeEqSys_(NULL),
    tdrEqSys_(NULL),
    tke_(NULL),
    tdr_(NULL),
    minDistanceToWall_(NULL),
    dplus_(NULL),
    isInit_(true),
    resetAMSAverages_(realm_.solutionOptions_->resetAMSAverages_)
{
  // push back EQ to manager
  realm_.push_equation_to_systems(this);

  tkeEqSys_ = new TurbKineticEnergyEquationSystem(eqSystems);
  tdrEqSys_ = new TotalDissipationRateEquationSystem(eqSystems);
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
ChienKEpsilonEquationSystem::~ChienKEpsilonEquationSystem() {}

void
ChienKEpsilonEquationSystem::load(const YAML::Node& node)
{
  EquationSystem::load(node);

  if (realm_.query_for_overset()) {
    tkeEqSys_->decoupledOverset_ = decoupledOverset_;
    tkeEqSys_->numOversetIters_ = numOversetIters_;
    tdrEqSys_->decoupledOverset_ = decoupledOverset_;
    tdrEqSys_->numOversetIters_ = numOversetIters_;
  }
}

//--------------------------------------------------------------------------
//-------- initialize ------------------------------------------------------
//--------------------------------------------------------------------------
void
ChienKEpsilonEquationSystem::initialize()
{
  // let equation systems that are owned some information
  tkeEqSys_->convergenceTolerance_ = convergenceTolerance_;
  tdrEqSys_->convergenceTolerance_ = convergenceTolerance_;
}

//--------------------------------------------------------------------------
//-------- register_nodal_fields -------------------------------------------
//--------------------------------------------------------------------------
void
ChienKEpsilonEquationSystem::register_nodal_fields(
  const stk::mesh::PartVector& part_vec)
{

  stk::mesh::MetaData& meta_data = realm_.meta_data();
  const int numStates = realm_.number_of_states();
  stk::mesh::Selector selector = stk::mesh::selectUnion(part_vec);

  // re-register tke and tdr for convenience
  tke_ = &(meta_data.declare_field<double>(
    stk::topology::NODE_RANK, "turbulent_ke", numStates));
  stk::mesh::put_field_on_mesh(*tke_, selector, nullptr);
  tdr_ = &(meta_data.declare_field<double>(
    stk::topology::NODE_RANK, "total_dissipation_rate", numStates));
  stk::mesh::put_field_on_mesh(*tdr_, selector, nullptr);

  // SST parameters that everyone needs
  minDistanceToWall_ = &(meta_data.declare_field<double>(
    stk::topology::NODE_RANK, "minimum_distance_to_wall"));
  stk::mesh::put_field_on_mesh(*minDistanceToWall_, selector, nullptr);
  dplus_ = &(meta_data.declare_field<double>(
    stk::topology::NODE_RANK, "dplus_wall_function"));
  stk::mesh::put_field_on_mesh(*dplus_, selector, nullptr);

  // add to restart field
  realm_.augment_restart_variable_list("minimum_distance_to_wall");
  realm_.augment_restart_variable_list("dplus_wall_function");
}

//--------------------------------------------------------------------------
//-------- register_interior_algorithm -------------------------------------
//--------------------------------------------------------------------------
void
ChienKEpsilonEquationSystem::register_interior_algorithm(
  stk::mesh::Part* /*part*/)
{
  // nothing to do here...
}

//--------------------------------------------------------------------------
//-------- register_wall_bc ------------------------------------------------
//--------------------------------------------------------------------------
void
ChienKEpsilonEquationSystem::register_wall_bc(
  stk::mesh::Part* part,
  const stk::topology& partTopo,
  const WallBoundaryConditionData& wallBCData)
{
  // determine if using RANS for ABL
  WallUserData userData = wallBCData.userData_;

  // push mesh part
  wallBcPart_.push_back(part);

  auto& meta = realm_.meta_data();
  auto& assembledWallArea = meta.declare_field<double>(
    stk::topology::NODE_RANK, "assembled_wall_area_wf");
  stk::mesh::put_field_on_mesh(assembledWallArea, *part, nullptr);
  auto& assembledWallNormDist = meta.declare_field<double>(
    stk::topology::NODE_RANK, "assembled_wall_normal_distance");
  stk::mesh::put_field_on_mesh(assembledWallNormDist, *part, nullptr);
  auto& wallNormDistBip =
    meta.declare_field<double>(meta.side_rank(), "wall_normal_distance_bip");
  auto* meFC = MasterElementRepo::get_surface_master_element_on_host(partTopo);
  const int numScsBip = meFC->num_integration_points();
  stk::mesh::put_field_on_mesh(wallNormDistBip, *part, numScsBip, nullptr);
}

//--------------------------------------------------------------------------
//-------- solve_and_update ------------------------------------------------
//--------------------------------------------------------------------------
void
ChienKEpsilonEquationSystem::solve_and_update()
{
  // wrap timing
  // SST_FIXME: deal with timers; all on misc for SSTEqs double timeA, timeB;
  if (isInit_) {
    // compute projected nodal gradients
    tkeEqSys_->compute_projected_nodal_gradient();
    tdrEqSys_->assemble_nodal_gradient();
    clip_min_distance_to_wall();
    compute_dplus_function();

    isInit_ = false;
  } else if (realm_.has_mesh_motion()) {
    if (realm_.currentNonlinearIteration_ == 1)
      clip_min_distance_to_wall();
    compute_dplus_function();
  }

  // SST effective viscosity for k and omega
  tkeEqSys_->compute_effective_diff_flux_coeff();
  tdrEqSys_->compute_effective_diff_flux_coeff();

  // start the iteration loop
  for (int k = 0; k < maxIterations_; ++k) {

    NaluEnv::self().naluOutputP0()
      << " " << k + 1 << "/" << maxIterations_ << std::setw(20) << std::right
      << name_ << std::endl;

    for (int oi = 0; oi < numOversetIters_; ++oi) {
      // tke and tdr assemble, load_complete and solve; Jacobi iteration
      tkeEqSys_->assemble_and_solve(tkeEqSys_->kTmp_);
      tdrEqSys_->assemble_and_solve(tdrEqSys_->eTmp_);

      update_and_clip();

      if (decoupledOverset_ && realm_.hasOverset_) {
        realm_.overset_field_update(tkeEqSys_->tke_, 1, 1);
        realm_.overset_field_update(tdrEqSys_->tdr_, 1, 1);
      }
    }
    // compute projected nodal gradients
    tkeEqSys_->compute_projected_nodal_gradient();
    tdrEqSys_->assemble_nodal_gradient();
  }
}

/** Perform sanity checks on TKE/TDR fields
 */
void
ChienKEpsilonEquationSystem::initial_work()
{
  const auto& meshInfo = realm_.mesh_info();
  const auto& meta = meshInfo.meta();
  const auto& ngpMesh = meshInfo.ngp_mesh();
  const auto& fieldMgr = meshInfo.ngp_field_manager();

  auto& tkeNp1 = fieldMgr.get_field<double>(tke_->mesh_meta_data_ordinal());
  auto& tdrNp1 = fieldMgr.get_field<double>(tdr_->mesh_meta_data_ordinal());
  const stk::mesh::Selector sel =
    (meta.locally_owned_part() | meta.globally_shared_part()) &
    stk::mesh::selectField(*tdr_);
  clip_ke(ngpMesh, sel, tkeNp1, tdrNp1);
}

void
ChienKEpsilonEquationSystem::post_external_data_transfer_work()
{
  const auto& meshInfo = realm_.mesh_info();
  const auto& meta = meshInfo.meta();
  const auto& ngpMesh = meshInfo.ngp_mesh();
  const auto& fieldMgr = meshInfo.ngp_field_manager();

  auto& tkeNp1 = fieldMgr.get_field<double>(tke_->mesh_meta_data_ordinal());
  auto& tdrNp1 = fieldMgr.get_field<double>(tdr_->mesh_meta_data_ordinal());

  const stk::mesh::Selector owned_and_shared =
    (meta.locally_owned_part() | meta.globally_shared_part());
  auto interior_sel = owned_and_shared & stk::mesh::selectField(*tdr_);
  clip_ke(ngpMesh, interior_sel, tkeNp1, tdrNp1);

  auto tdrBCField = meta.get_field<double>(stk::topology::NODE_RANK, "tdr_bc");
  auto tkeBCField = meta.get_field<double>(stk::topology::NODE_RANK, "tke_bc");
  if (tdrBCField != nullptr) {
    ThrowRequire(tkeBCField);
    auto bc_sel = owned_and_shared & stk::mesh::selectField(*tdrBCField);
    auto ngpTkeBC =
      fieldMgr.get_field<double>(tkeBCField->mesh_meta_data_ordinal());
    auto ngpTdrBC =
      fieldMgr.get_field<double>(tdrBCField->mesh_meta_data_ordinal());
    clip_ke(ngpMesh, bc_sel, ngpTkeBC, ngpTdrBC);
  }
}

/** Update solution but ensure that TKE and TDR are greater than zero
 */
void
ChienKEpsilonEquationSystem::update_and_clip()
{
  using MeshIndex = nalu_ngp::NGPMeshTraits<>::MeshIndex;

  const auto& meshInfo = realm_.mesh_info();
  const auto& meta = meshInfo.meta();
  const auto& ngpMesh = meshInfo.ngp_mesh();
  const auto& fieldMgr = meshInfo.ngp_field_manager();

  auto& tkeNp1 = fieldMgr.get_field<double>(tke_->mesh_meta_data_ordinal());
  auto& tdrNp1 = fieldMgr.get_field<double>(tdr_->mesh_meta_data_ordinal());
  const auto& kTmp =
    fieldMgr.get_field<double>(tkeEqSys_->kTmp_->mesh_meta_data_ordinal());
  const auto& eTmp =
    fieldMgr.get_field<double>(tdrEqSys_->eTmp_->mesh_meta_data_ordinal());

  auto* turbViscosity =
    meta.get_field<double>(stk::topology::NODE_RANK, "turbulent_viscosity");

  const stk::mesh::Selector sel =
    (meta.locally_owned_part() | meta.globally_shared_part()) &
    stk::mesh::selectField(*turbViscosity);

  // Bring class variables to local scope for lambda capture
  const double tkeMinVal = tkeMinValue_;
  const double tdrMinVal = tdrMinValue_;

  tkeNp1.sync_to_device();
  tdrNp1.sync_to_device();

  nalu_ngp::run_entity_algorithm(
    "KE::update_and_clip", ngpMesh, stk::topology::NODE_RANK, sel,
    KOKKOS_LAMBDA(const MeshIndex& mi) {
      const double tkeNew = tkeNp1.get(mi, 0) + kTmp.get(mi, 0);
      const double tdrNew = tdrNp1.get(mi, 0) + eTmp.get(mi, 0);

      tkeNp1.get(mi, 0) = (tkeNew < 0.0) ? tkeMinVal : tkeNew;
      tdrNp1.get(mi, 0) = stk::math::max(tdrNew, tdrMinVal);
    });

  tkeNp1.modify_on_device();
  tdrNp1.modify_on_device();
}

void
ChienKEpsilonEquationSystem::clip_ke(
  const stk::mesh::NgpMesh& ngpMesh,
  const stk::mesh::Selector& sel,
  stk::mesh::NgpField<double>& tke,
  stk::mesh::NgpField<double>& tdr)
{
  tke.sync_to_device();
  tdr.sync_to_device();

  // Bring class variables to local scope for lambda capture
  const double tkeMinVal = tkeMinValue_;
  const double tdrMinVal = tdrMinValue_;

  nalu_ngp::run_entity_algorithm(
    "KE::clip", ngpMesh, stk::topology::NODE_RANK, sel,
    KOKKOS_LAMBDA(const nalu_ngp::NGPMeshTraits<>::MeshIndex& mi) {
      const double tkeNew = tke.get(mi, 0);
      const double tdrNew = tdr.get(mi, 0);

      tke.get(mi, 0) = (tkeNew < 0.0) ? tkeMinVal : tkeNew;
      tdr.get(mi, 0) = stk::math::max(tdrNew, tdrMinVal);
    });
  tke.modify_on_device();
  tdr.modify_on_device();
}

//--------------------------------------------------------------------------
//-------- clip_min_distance_to_wall ---------------------------------------
//--------------------------------------------------------------------------
void
ChienKEpsilonEquationSystem::clip_min_distance_to_wall()
{
  using MeshIndex = nalu_ngp::NGPMeshTraits<>::MeshIndex;
  const auto& meshInfo = realm_.mesh_info();
  const auto& ngpMesh = meshInfo.ngp_mesh();
  const auto& meta = meshInfo.meta();
  const auto& fieldMgr = meshInfo.ngp_field_manager();

  auto& ndtw =
    fieldMgr.get_field<double>(minDistanceToWall_->mesh_meta_data_ordinal());
  const auto& wallNormDist =
    nalu_ngp::get_ngp_field(meshInfo, "assembled_wall_normal_distance");

  const stk::mesh::Selector sel =
    (meta.locally_owned_part() | meta.globally_shared_part()) &
    stk::mesh::selectUnion(wallBcPart_);

  ndtw.sync_to_device();

  nalu_ngp::run_entity_algorithm(
    "SST::clip_ndtw", ngpMesh, stk::topology::NODE_RANK, sel,
    KOKKOS_LAMBDA(const MeshIndex& mi) {
      const double minD = ndtw.get(mi, 0);

      ndtw.get(mi, 0) = stk::math::max(minD, wallNormDist.get(mi, 0));
    });
  ndtw.modify_on_device();

  stk::mesh::parallel_max(realm_.bulk_data(), {minDistanceToWall_});
  if (realm_.hasPeriodic_) {
    realm_.periodic_field_max(minDistanceToWall_, 1);
  }
}

/** Compute non-local function of distance to wall
 */
void
ChienKEpsilonEquationSystem::compute_dplus_function()
{
  using MeshIndex = nalu_ngp::NGPMeshTraits<>::MeshIndex;

  const auto& meshInfo = realm_.mesh_info();
  const auto& meta = meshInfo.meta();
  const auto& ngpMesh = meshInfo.ngp_mesh();
  const auto& fieldMgr = meshInfo.ngp_field_manager();

  const double utau = realm_.get_turb_model_constant(TM_utau);

  const auto& density = nalu_ngp::get_ngp_field(meshInfo, "density");
  const auto& viscosity = nalu_ngp::get_ngp_field(meshInfo, "viscosity");
  const auto& ndtw =
    fieldMgr.get_field<double>(minDistanceToWall_->mesh_meta_data_ordinal());
  auto& dplus = fieldMgr.get_field<double>(dplus_->mesh_meta_data_ordinal());

  const stk::mesh::Selector sel =
    (meta.locally_owned_part() | meta.globally_shared_part()) &
    (stk::mesh::selectField(*dplus_));

  dplus.sync_to_device();

  nalu_ngp::run_entity_algorithm(
    "KE::compute_dplus_function", ngpMesh, stk::topology::NODE_RANK, sel,
    KOKKOS_LAMBDA(const MeshIndex& mi) {
      const double rho = density.get(mi, 0);
      const double mu = viscosity.get(mi, 0);
      const double minD = ndtw.get(mi, 0);

      dplus.get(mi, 0) = minD * rho * utau / mu;
    });

  dplus.modify_on_device();
}

void
ChienKEpsilonEquationSystem::post_iter_work()
{
  // nothing to do here ...
}

} // namespace nalu
} // namespace sierra
