// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <ShearStressTransportEquationSystem.h>
#include <AlgorithmDriver.h>
#include <ComputeSSTMaxLengthScaleElemAlgorithm.h>
#include <FieldFunctions.h>
#include <master_element/MasterElement.h>
#include <master_element/MasterElementRepo.h>
#include <NaluEnv.h>
#include <NaluParsing.h>
#include <SpecificDissipationRateEquationSystem.h>
#include <GammaEquationSystem.h>
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
// ShearStressTransportEquationSystem - manage SST
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
ShearStressTransportEquationSystem::ShearStressTransportEquationSystem(
  EquationSystems& eqSystems)
  : EquationSystem(eqSystems, "ShearStressTransportWrap"),
    tkeEqSys_(NULL),
    sdrEqSys_(NULL),
    gammaEqSys_(NULL),
    tke_(NULL),
    sdr_(NULL),
    gamma_(NULL),
    minDistanceToWall_(NULL),
    fOneBlending_(NULL),
    maxLengthScale_(NULL),
    isInit_(true),
    sstMaxLengthScaleAlgDriver_(NULL),
    resetAMSAverages_(realm_.solutionOptions_->resetAMSAverages_)
{
  // push back EQ to manager
  realm_.push_equation_to_systems(this);

  tkeEqSys_ = new TurbKineticEnergyEquationSystem(eqSystems);
  sdrEqSys_ = new SpecificDissipationRateEquationSystem(eqSystems);
  if (realm_.solutionOptions_->gammaEqActive_)
    gammaEqSys_ = new GammaEquationSystem(eqSystems);
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
ShearStressTransportEquationSystem::~ShearStressTransportEquationSystem()
{
  if (NULL != sstMaxLengthScaleAlgDriver_)
    delete sstMaxLengthScaleAlgDriver_;
}

void
ShearStressTransportEquationSystem::load(const YAML::Node& node)
{
  EquationSystem::load(node);

  if (realm_.query_for_overset()) {
    tkeEqSys_->decoupledOverset_ = decoupledOverset_;
    tkeEqSys_->numOversetIters_ = numOversetIters_;
    sdrEqSys_->decoupledOverset_ = decoupledOverset_;
    sdrEqSys_->numOversetIters_ = numOversetIters_;
    if (realm_.solutionOptions_->gammaEqActive_) {
      gammaEqSys_->decoupledOverset_ = decoupledOverset_;
      gammaEqSys_->numOversetIters_ = numOversetIters_;
    }
  }
}

//--------------------------------------------------------------------------
//-------- initialize ------------------------------------------------------
//--------------------------------------------------------------------------
void
ShearStressTransportEquationSystem::initialize()
{
  // let equation systems that are owned some information
  tkeEqSys_->convergenceTolerance_ = convergenceTolerance_;
  sdrEqSys_->convergenceTolerance_ = convergenceTolerance_;
  if (realm_.solutionOptions_->gammaEqActive_)
    gammaEqSys_->convergenceTolerance_ = convergenceTolerance_;
}

//--------------------------------------------------------------------------
//-------- register_nodal_fields -------------------------------------------
//--------------------------------------------------------------------------
void
ShearStressTransportEquationSystem::register_nodal_fields(
  const stk::mesh::PartVector& part_vec)
{

  stk::mesh::MetaData& meta_data = realm_.meta_data();
  const int numStates = realm_.number_of_states();
  stk::mesh::Selector selector = stk::mesh::selectUnion(part_vec);

  // re-register tke and sdr for convenience
  tke_ = &(meta_data.declare_field<double>(
    stk::topology::NODE_RANK, "turbulent_ke", numStates));
  stk::mesh::put_field_on_mesh(*tke_, selector, nullptr);
  sdr_ = &(meta_data.declare_field<double>(
    stk::topology::NODE_RANK, "specific_dissipation_rate", numStates));
  stk::mesh::put_field_on_mesh(*sdr_, selector, nullptr);
  if (realm_.solutionOptions_->gammaEqActive_) {
    gamma_ = &(meta_data.declare_field<double>(
      stk::topology::NODE_RANK, "gamma_transition", numStates));
    stk::mesh::put_field_on_mesh(*gamma_, selector, nullptr);
  }

  // SST parameters that everyone needs
  minDistanceToWall_ = &(meta_data.declare_field<double>(
    stk::topology::NODE_RANK, "minimum_distance_to_wall"));
  stk::mesh::put_field_on_mesh(*minDistanceToWall_, selector, nullptr);
  fOneBlending_ = &(meta_data.declare_field<double>(
    stk::topology::NODE_RANK, "sst_f_one_blending"));
  stk::mesh::put_field_on_mesh(*fOneBlending_, selector, nullptr);

  // DES model
  if (
    (TurbulenceModel::SST_DES == realm_.solutionOptions_->turbulenceModel_) ||
    (TurbulenceModel::SST_IDDES == realm_.solutionOptions_->turbulenceModel_)) {
    maxLengthScale_ = &(meta_data.declare_field<double>(
      stk::topology::NODE_RANK, "sst_max_length_scale"));
    stk::mesh::put_field_on_mesh(*maxLengthScale_, selector, nullptr);
  }

  // add to restart field
  realm_.augment_restart_variable_list("minimum_distance_to_wall");
  realm_.augment_restart_variable_list("sst_f_one_blending");
}

//--------------------------------------------------------------------------
//-------- register_interior_algorithm -------------------------------------
//--------------------------------------------------------------------------
void
ShearStressTransportEquationSystem::register_interior_algorithm(
  stk::mesh::Part* part)
{

  // types of algorithms
  const AlgorithmType algType = INTERIOR;
  if (
    (TurbulenceModel::SST_DES == realm_.solutionOptions_->turbulenceModel_) ||
    (TurbulenceModel::SST_IDDES == realm_.solutionOptions_->turbulenceModel_)) {

    if (NULL == sstMaxLengthScaleAlgDriver_)
      sstMaxLengthScaleAlgDriver_ = new AlgorithmDriver(realm_);

    // create edge algorithm
    std::map<AlgorithmType, Algorithm*>::iterator it =
      sstMaxLengthScaleAlgDriver_->algMap_.find(algType);

    if (it == sstMaxLengthScaleAlgDriver_->algMap_.end()) {
      ComputeSSTMaxLengthScaleElemAlgorithm* theAlg =
        new ComputeSSTMaxLengthScaleElemAlgorithm(realm_, part);
      sstMaxLengthScaleAlgDriver_->algMap_[algType] = theAlg;
    } else {
      it->second->partVec_.push_back(part);
    }
  }
}

//--------------------------------------------------------------------------
//-------- register_wall_bc ------------------------------------------------
//--------------------------------------------------------------------------
void
ShearStressTransportEquationSystem::register_wall_bc(
  stk::mesh::Part* part,
  const stk::topology& partTopo,
  const WallBoundaryConditionData& wallBCData)
{
  // determine if using RANS for ABL
  WallUserData userData = wallBCData.userData_;
  bool RANSAblBcApproach = userData.RANSAblBcApproach_;

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

  RoughnessHeight rough = userData.z0_;
  double z0 = rough.z0_;
  realm_.geometryAlgDriver_->register_wall_func_algorithm<WallFuncGeometryAlg>(
    sierra::nalu::WALL, part, get_elem_topo(realm_, *part), "sst_geometry_wall",
    RANSAblBcApproach, z0);
}

//--------------------------------------------------------------------------
//-------- solve_and_update ------------------------------------------------
//--------------------------------------------------------------------------
void
ShearStressTransportEquationSystem::solve_and_update()
{
  // wrap timing
  // SST_FIXME: deal with timers; all on misc for SSTEqs double timeA, timeB;
  if (isInit_) {
    // compute projected nodal gradients
    tkeEqSys_->compute_projected_nodal_gradient();
    sdrEqSys_->assemble_nodal_gradient();
    if (realm_.solutionOptions_->gammaEqActive_)
      gammaEqSys_->assemble_nodal_gradient();
    clip_min_distance_to_wall();

    // deal with DES option
    if (
      (TurbulenceModel::SST_DES == realm_.solutionOptions_->turbulenceModel_) ||
      (TurbulenceModel::SST_IDDES == realm_.solutionOptions_->turbulenceModel_))
      sstMaxLengthScaleAlgDriver_->execute();

    isInit_ = false;
  } else if (realm_.has_mesh_motion()) {
    if (realm_.currentNonlinearIteration_ == 1)
      clip_min_distance_to_wall();

    if (
      (TurbulenceModel::SST_DES == realm_.solutionOptions_->turbulenceModel_) ||
      (TurbulenceModel::SST_IDDES == realm_.solutionOptions_->turbulenceModel_))
      sstMaxLengthScaleAlgDriver_->execute();
  }

  // compute blending for SST model
  compute_f_one_blending();

  // SST effective viscosity for k and omega
  tkeEqSys_->compute_effective_diff_flux_coeff();
  sdrEqSys_->compute_effective_diff_flux_coeff();
  if (realm_.solutionOptions_->gammaEqActive_)
    gammaEqSys_->compute_effective_diff_flux_coeff();

  // wall values
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
      if (realm_.solutionOptions_->gammaEqActive_)
        gammaEqSys_->assemble_and_solve(gammaEqSys_->gamTmp_);

      update_and_clip();
      if (realm_.solutionOptions_->gammaEqActive_)
        update_and_clip_gamma();

      if (decoupledOverset_ && realm_.hasOverset_) {
        realm_.overset_field_update(tkeEqSys_->tke_, 1, 1);
        realm_.overset_field_update(sdrEqSys_->sdr_, 1, 1);
        if (realm_.solutionOptions_->gammaEqActive_)
          realm_.overset_field_update(gammaEqSys_->gamma_, 1, 1);
      }
    }
    // compute projected nodal gradients
    tkeEqSys_->compute_projected_nodal_gradient();
    sdrEqSys_->assemble_nodal_gradient();
    if (realm_.solutionOptions_->gammaEqActive_)
      gammaEqSys_->assemble_nodal_gradient();
  }
}

/** Perform sanity checks on TKE/SDR fields
 */
void
ShearStressTransportEquationSystem::initial_work()
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
  clip_sst(ngpMesh, sel, tkeNp1, sdrNp1);

  if (realm_.solutionOptions_->gammaEqActive_) {
    auto& gammaNp1 =
      fieldMgr.get_field<double>(gamma_->mesh_meta_data_ordinal());
    clip_sst_gamma(ngpMesh, sel, gammaNp1);
  }
}

void
ShearStressTransportEquationSystem::post_external_data_transfer_work()
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
  clip_sst(ngpMesh, interior_sel, tkeNp1, sdrNp1);
  if (realm_.solutionOptions_->gammaEqActive_) {
    auto& gammaNp1 =
      fieldMgr.get_field<double>(gamma_->mesh_meta_data_ordinal());
    clip_sst_gamma(ngpMesh, interior_sel, gammaNp1);
  }

  auto sdrBCField = meta.get_field<double>(stk::topology::NODE_RANK, "sdr_bc");
  auto tkeBCField = meta.get_field<double>(stk::topology::NODE_RANK, "tke_bc");

  if (sdrBCField != nullptr) {
    STK_ThrowRequire(tkeBCField);
    auto bc_sel = owned_and_shared & stk::mesh::selectField(*sdrBCField);
    auto ngpTkeBC =
      fieldMgr.get_field<double>(tkeBCField->mesh_meta_data_ordinal());
    auto ngpSdrBC =
      fieldMgr.get_field<double>(sdrBCField->mesh_meta_data_ordinal());

    clip_sst(ngpMesh, bc_sel, ngpTkeBC, ngpSdrBC);

    if (realm_.solutionOptions_->gammaEqActive_) {
      auto gammaBCField =
        meta.get_field<double>(stk::topology::NODE_RANK, "gamma_bc");

      auto ngpGammaBC =
        fieldMgr.get_field<double>(gammaBCField->mesh_meta_data_ordinal());

      clip_sst_gamma(ngpMesh, bc_sel, ngpGammaBC);
    }
  }
}

/** Update solution but ensure that TKE and SDR are greater than zero
 */
void
ShearStressTransportEquationSystem::update_and_clip()
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

  auto* turbViscosity =
    meta.get_field<double>(stk::topology::NODE_RANK, "turbulent_viscosity");

  const stk::mesh::Selector sel =
    (meta.locally_owned_part() | meta.globally_shared_part()) &
    stk::mesh::selectField(*turbViscosity);

  // Bring class variables to local scope for lambda capture
  const double tkeMinVal = tkeMinValue_;
  const double sdrMinVal = sdrMinValue_;

  tkeNp1.sync_to_device();
  sdrNp1.sync_to_device();

  nalu_ngp::run_entity_algorithm(
    "SST::update_and_clip", ngpMesh, stk::topology::NODE_RANK, sel,
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
ShearStressTransportEquationSystem::update_and_clip_gamma()
{
  using MeshIndex = nalu_ngp::NGPMeshTraits<>::MeshIndex;

  const auto& meshInfo = realm_.mesh_info();
  const auto& meta = meshInfo.meta();
  const auto& ngpMesh = meshInfo.ngp_mesh();
  const auto& fieldMgr = meshInfo.ngp_field_manager();

  auto& gammaNp1 = fieldMgr.get_field<double>(gamma_->mesh_meta_data_ordinal());
  const auto& gamTmp =
    fieldMgr.get_field<double>(gammaEqSys_->gamTmp_->mesh_meta_data_ordinal());
  auto* turbViscosity =
    meta.get_field<double>(stk::topology::NODE_RANK, "turbulent_viscosity");

  const stk::mesh::Selector sel =
    (meta.locally_owned_part() | meta.globally_shared_part()) &
    stk::mesh::selectField(*turbViscosity);
  const double gammaMinVal = gammaMinValue_;
  const double gammaMaxVal = gammaMaxValue_;

  gammaNp1.sync_to_device();

  nalu_ngp::run_entity_algorithm(
    "SST_GammaEQActive::update_and_clip", ngpMesh, stk::topology::NODE_RANK,
    sel, KOKKOS_LAMBDA(const MeshIndex& mi) {
      const double gammaNew = gammaNp1.get(mi, 0) + gamTmp.get(mi, 0);
      gammaNp1.get(mi, 0) =
        stk::math::min(stk::math::max(gammaNew, gammaMinVal), gammaMaxVal);
    });

  gammaNp1.modify_on_device();
}

void
ShearStressTransportEquationSystem::clip_sst(
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
    "SST::clip", ngpMesh, stk::topology::NODE_RANK, sel,
    KOKKOS_LAMBDA(const nalu_ngp::NGPMeshTraits<>::MeshIndex& mi) {
      const double tkeNew = tke.get(mi, 0);
      const double sdrNew = sdr.get(mi, 0);

      tke.get(mi, 0) = (tkeNew < 0.0) ? tkeMinVal : tkeNew;
      sdr.get(mi, 0) = stk::math::max(sdrNew, sdrMinVal);
    });
  tke.modify_on_device();
  sdr.modify_on_device();
}

void
ShearStressTransportEquationSystem::clip_sst_gamma(
  const stk::mesh::NgpMesh& ngpMesh,
  const stk::mesh::Selector& sel,
  stk::mesh::NgpField<double>& gamma)
{
  gamma.sync_to_device();
  const double gammaMinVal = gammaMinValue_;
  const double gammaMaxVal = gammaMaxValue_;

  nalu_ngp::run_entity_algorithm(
    "SST::clip", ngpMesh, stk::topology::NODE_RANK, sel,
    KOKKOS_LAMBDA(const nalu_ngp::NGPMeshTraits<>::MeshIndex& mi) {
      const double gammaNew = gamma.get(mi, 0);
      gamma.get(mi, 0) =
        stk::math::min(stk::math::max(gammaNew, gammaMinVal), gammaMaxVal);
    });
  gamma.modify_on_device();
}

//--------------------------------------------------------------------------
//-------- clip_min_distance_to_wall ---------------------------------------
//--------------------------------------------------------------------------
void
ShearStressTransportEquationSystem::clip_min_distance_to_wall()
{
  using MeshIndex = nalu_ngp::NGPMeshTraits<>::MeshIndex;
  const auto& meshInfo = realm_.mesh_info();
  const auto& ngpMesh = meshInfo.ngp_mesh();
  const auto& meta = meshInfo.meta();
  const auto& fieldMgr = meshInfo.ngp_field_manager();

  if (wallBcPart_.empty())
    return;

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
  if (realm_.hasPeriodic_)
    realm_.periodic_field_max(minDistanceToWall_, 1);
}

/** Compute f1 field with parameters appropriate for 2003 SST implementation
 */
void
ShearStressTransportEquationSystem::compute_f_one_blending()
{
  using MeshIndex = nalu_ngp::NGPMeshTraits<>::MeshIndex;

  const auto& meshInfo = realm_.mesh_info();
  const auto& meta = meshInfo.meta();
  const auto& ngpMesh = meshInfo.ngp_mesh();
  const auto& fieldMgr = meshInfo.ngp_field_manager();

  const int ndim = meta.spatial_dimension();
  const double betaStar = realm_.get_turb_model_constant(TM_betaStar);
  const double sigmaWTwo = realm_.get_turb_model_constant(TM_sigmaWTwo);
  const double CDkwClip = 1.0e-10; // 2003 SST

  const auto& tkeNp1 =
    fieldMgr.get_field<double>(tke_->mesh_meta_data_ordinal());
  const auto& sdrNp1 =
    fieldMgr.get_field<double>(sdr_->mesh_meta_data_ordinal());
  const auto& density = nalu_ngp::get_ngp_field(meshInfo, "density");
  const auto& viscosity = nalu_ngp::get_ngp_field(meshInfo, "viscosity");
  const auto& dkdx =
    fieldMgr.get_field<double>(tkeEqSys_->dkdx_->mesh_meta_data_ordinal());
  const auto& dwdx =
    fieldMgr.get_field<double>(sdrEqSys_->dwdx_->mesh_meta_data_ordinal());
  const auto& ndtw =
    fieldMgr.get_field<double>(minDistanceToWall_->mesh_meta_data_ordinal());
  auto& fOneBlend =
    fieldMgr.get_field<double>(fOneBlending_->mesh_meta_data_ordinal());

  const stk::mesh::Selector sel =
    (meta.locally_owned_part() | meta.globally_shared_part()) &
    (stk::mesh::selectField(*fOneBlending_));

  fOneBlend.sync_to_device();

  nalu_ngp::run_entity_algorithm(
    "SST::compute_fone_blending", ngpMesh, stk::topology::NODE_RANK, sel,
    KOKKOS_LAMBDA(const MeshIndex& mi) {
      const double tke = tkeNp1.get(mi, 0);
      const double sdr = sdrNp1.get(mi, 0);
      const double rho = density.get(mi, 0);
      const double mu = viscosity.get(mi, 0);
      const double minD = ndtw.get(mi, 0);

      // cross diffusion
      double crossdiff = 0.0;
      for (int d = 0; d < ndim; ++d)
        crossdiff += dkdx.get(mi, d) * dwdx.get(mi, d);

      const double minDistSq = minD * minD;
      const double turbDiss = stk::math::sqrt(tke) / betaStar / sdr / minD;
      const double lamDiss = 500.0 * mu / rho / sdr / minDistSq;
      const double CDkw =
        stk::math::max(2.0 * rho * sigmaWTwo * crossdiff / sdr, CDkwClip);

      const double fArgOne = stk::math::min(
        stk::math::max(turbDiss, lamDiss),
        4.0 * rho * sigmaWTwo * tke / CDkw / minDistSq);

      fOneBlend.get(mi, 0) =
        stk::math::tanh(fArgOne * fArgOne * fArgOne * fArgOne);

      // Modifications of f1 blending function for the transition model
      if (realm_.solutionOptions_->gammaEqActive_) {
        const double f1Orig = fOneBlend.get(mi, 0);
        const double ry = rho * minD * stk::math::sqrt(tke)/mu;
        const double arg = ry / 120.0;
        const double f3 = stk::math::exp(-arg*arg*arg*arg*arg*arg*arg*arg); // original
        //const double f3 = std::exp(-std::pow(arg,8)); // new
        fOneBlend.get(mi, 0) = stk::math::max(f1Orig, f3);
      }

    });

  fOneBlend.modify_on_device();
}

void
ShearStressTransportEquationSystem::pre_iter_work()
{
  const auto turbModel = realm_.solutionOptions_->turbulenceModel_;
  if (turbModel == TurbulenceModel::SST_IDDES) {
    const auto& fieldMgr = realm_.ngp_field_manager();
    const auto& meta = realm_.meta_data();

    auto ngpIddesRans = fieldMgr.get_field<double>(
      get_field_ordinal(meta, "iddes_rans_indicator"));
    ngpIddesRans.sync_to_device();
  }
}

void
ShearStressTransportEquationSystem::post_iter_work()
{
  const auto turbModel = realm_.solutionOptions_->turbulenceModel_;
  if (turbModel == TurbulenceModel::SST_IDDES) {
    const auto& fieldMgr = realm_.ngp_field_manager();
    const auto& meta = realm_.meta_data();
    auto& bulk = realm_.bulk_data();

    auto ngpIddesRans = fieldMgr.get_field<double>(
      get_field_ordinal(meta, "iddes_rans_indicator"));
    ngpIddesRans.modify_on_device();
    ngpIddesRans.sync_to_host();

    ScalarFieldType* iddesRansInd =
      meta.get_field<double>(stk::topology::NODE_RANK, "iddes_rans_indicator");

    stk::mesh::copy_owned_to_shared(bulk, {iddesRansInd});
    if (realm_.hasPeriodic_) {
      realm_.periodic_delta_solution_update(iddesRansInd, 1);
    }
  }
}

} // namespace nalu
} // namespace sierra
