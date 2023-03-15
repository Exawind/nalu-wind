// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <WilcoxKOmegaEquationSystem.h>
#include <AlgorithmDriver.h>
#include <FieldFunctions.h>
#include <master_element/MasterElement.h>
#include <master_element/MasterElementRepo.h>
#include <NaluEnv.h>
#include <NaluParsing.h>
#include <SpecificDissipationRateEquationSystem.h>
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
// WilcoxKOmegaEquationSystem - manage KOmega
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
WilcoxKOmegaEquationSystem::WilcoxKOmegaEquationSystem(
  EquationSystems& eqSystems)
  : EquationSystem(eqSystems, "WilcoxKOmegaWrap"),
    tkeEqSys_(NULL),
    sdrEqSys_(NULL),
    tke_(NULL),
    sdr_(NULL),
    minDistanceToWall_(NULL),
    isInit_(true),
    resetAMSAverages_(realm_.solutionOptions_->resetAMSAverages_)
{
  // push back EQ to manager
  realm_.push_equation_to_systems(this);

  tkeEqSys_ = new TurbKineticEnergyEquationSystem(eqSystems);
  sdrEqSys_ = new SpecificDissipationRateEquationSystem(eqSystems);
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
WilcoxKOmegaEquationSystem::~WilcoxKOmegaEquationSystem() {}

void
WilcoxKOmegaEquationSystem::load(const YAML::Node& node)
{
  EquationSystem::load(node);

  if (realm_.query_for_overset()) {
    tkeEqSys_->decoupledOverset_ = decoupledOverset_;
    tkeEqSys_->numOversetIters_ = numOversetIters_;
    sdrEqSys_->decoupledOverset_ = decoupledOverset_;
    sdrEqSys_->numOversetIters_ = numOversetIters_;
  }
}

//--------------------------------------------------------------------------
//-------- initialize ------------------------------------------------------
//--------------------------------------------------------------------------
void
WilcoxKOmegaEquationSystem::initialize()
{
  // let equation systems that are owned some information
  tkeEqSys_->convergenceTolerance_ = convergenceTolerance_;
  sdrEqSys_->convergenceTolerance_ = convergenceTolerance_;
}

//--------------------------------------------------------------------------
//-------- register_nodal_fields -------------------------------------------
//--------------------------------------------------------------------------
void
WilcoxKOmegaEquationSystem::register_nodal_fields(stk::mesh::Part* part)
{

  stk::mesh::MetaData& meta_data = realm_.meta_data();
  const int numStates = realm_.number_of_states();

  // re-register tke and sdr for convenience
  tke_ = &(meta_data.declare_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "turbulent_ke", numStates));
  stk::mesh::put_field_on_mesh(*tke_, *part, nullptr);
  sdr_ = &(meta_data.declare_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "specific_dissipation_rate", numStates));
  stk::mesh::put_field_on_mesh(*sdr_, *part, nullptr);

  // SST parameters that everyone needs
  minDistanceToWall_ = &(meta_data.declare_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "minimum_distance_to_wall"));
  stk::mesh::put_field_on_mesh(*minDistanceToWall_, *part, nullptr);

  // add to restart field
  realm_.augment_restart_variable_list("minimum_distance_to_wall");
}

//--------------------------------------------------------------------------
//-------- register_interior_algorithm -------------------------------------
//--------------------------------------------------------------------------
void
WilcoxKOmegaEquationSystem::register_interior_algorithm(
  stk::mesh::Part* /*part*/)
{
  // nothing to do here...
}

//--------------------------------------------------------------------------
//-------- register_wall_bc ------------------------------------------------
//--------------------------------------------------------------------------
void
WilcoxKOmegaEquationSystem::register_wall_bc(
  stk::mesh::Part* part,
  const stk::topology& partTopo,
  const WallBoundaryConditionData& wallBCData)
{
  // determine if using RANS for ABL
  WallUserData userData = wallBCData.userData_;

  // push mesh part
  wallBcPart_.push_back(part);

  auto& meta = realm_.meta_data();
  auto& assembledWallArea = meta.declare_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "assembled_wall_area_wf");
  stk::mesh::put_field_on_mesh(assembledWallArea, *part, nullptr);
  auto& assembledWallNormDist = meta.declare_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "assembled_wall_normal_distance");
  stk::mesh::put_field_on_mesh(assembledWallNormDist, *part, nullptr);
  auto& wallNormDistBip = meta.declare_field<ScalarFieldType>(
    meta.side_rank(), "wall_normal_distance_bip");
  auto* meFC = MasterElementRepo::get_surface_master_element_on_host(partTopo);
  const int numScsBip = meFC->num_integration_points();
  stk::mesh::put_field_on_mesh(wallNormDistBip, *part, numScsBip, nullptr);
}

//--------------------------------------------------------------------------
//-------- solve_and_update ------------------------------------------------
//--------------------------------------------------------------------------
void
WilcoxKOmegaEquationSystem::solve_and_update()
{
  // wrap timing
  // SST_FIXME: deal with timers; all on misc for SSTEqs double timeA, timeB;
  if (isInit_) {
    // compute projected nodal gradients
    tkeEqSys_->compute_projected_nodal_gradient();
    sdrEqSys_->assemble_nodal_gradient();
    clip_min_distance_to_wall();

    isInit_ = false;
  } else if (realm_.has_mesh_motion()) {
    if (realm_.currentNonlinearIteration_ == 1)
      clip_min_distance_to_wall();
  }

  // SST effective viscosity for k and omega
  tkeEqSys_->compute_effective_diff_flux_coeff();
  sdrEqSys_->compute_effective_diff_flux_coeff();

  tkeEqSys_->compute_wall_model_parameters();
  sdrEqSys_->compute_wall_model_parameters();

  // start the iteration loop
  for (int k = 0; k < maxIterations_; ++k) {

    NaluEnv::self().naluOutputP0()
      << " " << k + 1 << "/" << maxIterations_ << std::setw(15) << std::right
      << name_ << std::endl;

    for (int oi = 0; oi < numOversetIters_; ++oi) {
      // tke and sdr assemble, load_complete and solve; Jacobi iteration
      tkeEqSys_->assemble_and_solve(tkeEqSys_->kTmp_);
      sdrEqSys_->assemble_and_solve(sdrEqSys_->wTmp_);

      update_and_clip();

      if (decoupledOverset_ && realm_.hasOverset_) {
        realm_.overset_field_update(tkeEqSys_->tke_, 1, 1);
        realm_.overset_field_update(sdrEqSys_->sdr_, 1, 1);
      }
    }
    // compute projected nodal gradients
    tkeEqSys_->compute_projected_nodal_gradient();
    sdrEqSys_->assemble_nodal_gradient();
  }
}

/** Perform sanity checks on TKE/TDR fields
 */
void
WilcoxKOmegaEquationSystem::initial_work()
{
  const auto& meshInfo = realm_.mesh_info();
  const auto& meta = meshInfo.meta();
  const auto& ngpMesh = meshInfo.ngp_mesh();
  const auto& fieldMgr = meshInfo.ngp_field_manager();

  auto& tkeNp1 = fieldMgr.get_field<double>(tke_->mesh_meta_data_ordinal());
  auto& sdrNp1 = fieldMgr.get_field<double>(sdr_->mesh_meta_data_ordinal());
  const stk::mesh::Selector sel =
    (meta.locally_owned_part() | meta.globally_shared_part()) &
    stk::mesh::selectField(*sdr_);
  clip_ko(ngpMesh, sel, tkeNp1, sdrNp1);
}

void
WilcoxKOmegaEquationSystem::post_external_data_transfer_work()
{
  const auto& meshInfo = realm_.mesh_info();
  const auto& meta = meshInfo.meta();
  const auto& ngpMesh = meshInfo.ngp_mesh();
  const auto& fieldMgr = meshInfo.ngp_field_manager();

  auto& tkeNp1 = fieldMgr.get_field<double>(tke_->mesh_meta_data_ordinal());
  auto& sdrNp1 = fieldMgr.get_field<double>(sdr_->mesh_meta_data_ordinal());

  const stk::mesh::Selector owned_and_shared =
    (meta.locally_owned_part() | meta.globally_shared_part());
  auto interior_sel = owned_and_shared & stk::mesh::selectField(*sdr_);
  clip_ko(ngpMesh, interior_sel, tkeNp1, sdrNp1);

  auto sdrBCField =
    meta.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "sdr_bc");
  auto tkeBCField =
    meta.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "tke_bc");
  if (sdrBCField != nullptr) {
    ThrowRequire(tkeBCField);
    auto bc_sel = owned_and_shared & stk::mesh::selectField(*sdrBCField);
    auto ngpTkeBC =
      fieldMgr.get_field<double>(tkeBCField->mesh_meta_data_ordinal());
    auto ngpTdrBC =
      fieldMgr.get_field<double>(sdrBCField->mesh_meta_data_ordinal());
    clip_ko(ngpMesh, bc_sel, ngpTkeBC, ngpTdrBC);
  }
}

/** Update solution but ensure that TKE and TDR are greater than zero
 */
void
WilcoxKOmegaEquationSystem::update_and_clip()
{
  using MeshIndex = nalu_ngp::NGPMeshTraits<>::MeshIndex;

  const auto& meshInfo = realm_.mesh_info();
  const auto& meta = meshInfo.meta();
  const auto& ngpMesh = meshInfo.ngp_mesh();
  const auto& fieldMgr = meshInfo.ngp_field_manager();

  auto& tkeNp1 = fieldMgr.get_field<double>(tke_->mesh_meta_data_ordinal());
  auto& sdrNp1 = fieldMgr.get_field<double>(sdr_->mesh_meta_data_ordinal());
  const auto& kTmp =
    fieldMgr.get_field<double>(tkeEqSys_->kTmp_->mesh_meta_data_ordinal());
  const auto& wTmp =
    fieldMgr.get_field<double>(sdrEqSys_->wTmp_->mesh_meta_data_ordinal());

  auto* turbViscosity = meta.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "turbulent_viscosity");

  const stk::mesh::Selector sel =
    (meta.locally_owned_part() | meta.globally_shared_part()) &
    stk::mesh::selectField(*turbViscosity);

  // Bring class variables to local scope for lambda capture
  const double tkeMinVal = tkeMinValue_;
  const double sdrMinVal = sdrMinValue_;

  tkeNp1.sync_to_device();
  sdrNp1.sync_to_device();

  nalu_ngp::run_entity_algorithm(
    "KE::update_and_clip", ngpMesh, stk::topology::NODE_RANK, sel,
    KOKKOS_LAMBDA(const MeshIndex& mi) {
      const double tkeNew = tkeNp1.get(mi, 0) + kTmp.get(mi, 0);
      const double sdrNew = sdrNp1.get(mi, 0) + wTmp.get(mi, 0);

      tkeNp1.get(mi, 0) = (tkeNew < 0.0) ? tkeMinVal : tkeNew;
      sdrNp1.get(mi, 0) = stk::math::max(sdrNew, sdrMinVal);
    });

  tkeNp1.modify_on_device();
  sdrNp1.modify_on_device();
}

void
WilcoxKOmegaEquationSystem::clip_ko(
  const stk::mesh::NgpMesh& ngpMesh,
  const stk::mesh::Selector& sel,
  stk::mesh::NgpField<double>& tke,
  stk::mesh::NgpField<double>& sdr)
{
  tke.sync_to_device();
  sdr.sync_to_device();

  // Bring class variables to local scope for lambda capture
  const double tkeMinVal = tkeMinValue_;
  const double sdrMinVal = sdrMinValue_;

  nalu_ngp::run_entity_algorithm(
    "KE::clip", ngpMesh, stk::topology::NODE_RANK, sel,
    KOKKOS_LAMBDA(const nalu_ngp::NGPMeshTraits<>::MeshIndex& mi) {
      const double tkeNew = tke.get(mi, 0);
      const double sdrNew = sdr.get(mi, 0);

      tke.get(mi, 0) = (tkeNew < 0.0) ? tkeMinVal : tkeNew;
      sdr.get(mi, 0) = stk::math::max(sdrNew, sdrMinVal);
    });
  tke.modify_on_device();
  sdr.modify_on_device();
}

//--------------------------------------------------------------------------
//-------- clip_min_distance_to_wall ---------------------------------------
//--------------------------------------------------------------------------
void
WilcoxKOmegaEquationSystem::clip_min_distance_to_wall()
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

void
WilcoxKOmegaEquationSystem::post_iter_work()
{
  // nothing to do here ...
}

} // namespace nalu
} // namespace sierra
