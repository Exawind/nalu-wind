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
#include <master_element/MasterElementFactory.h>
#include <NaluEnv.h>
#include <SpecificDissipationRateEquationSystem.h>
#include <SolutionOptions.h>
#include <TurbKineticEnergyEquationSystem.h>
#include <Realm.h>

// ngp
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

namespace sierra{
namespace nalu{

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
    tke_(NULL),
    sdr_(NULL),
    minDistanceToWall_(NULL),
    fOneBlending_(NULL),
    maxLengthScale_(NULL),
    isInit_(true),
    sstMaxLengthScaleAlgDriver_(NULL),
    resetTAMSAverages_(realm_.solutionOptions_->resetTAMSAverages_)
{
  // push back EQ to manager
  realm_.push_equation_to_systems(this);

  // create momentum and pressure
  tkeEqSys_= new TurbKineticEnergyEquationSystem(eqSystems);
  sdrEqSys_ = new SpecificDissipationRateEquationSystem(eqSystems);
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
ShearStressTransportEquationSystem::~ShearStressTransportEquationSystem()
{
  if ( NULL != sstMaxLengthScaleAlgDriver_ )
    delete sstMaxLengthScaleAlgDriver_;
}

void ShearStressTransportEquationSystem::load(const YAML::Node& node)
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
ShearStressTransportEquationSystem::initialize()
{
  // let equation systems that are owned some information
  tkeEqSys_->convergenceTolerance_ = convergenceTolerance_;
  sdrEqSys_->convergenceTolerance_ = convergenceTolerance_;
}

//--------------------------------------------------------------------------
//-------- register_nodal_fields -------------------------------------------
//--------------------------------------------------------------------------
void
ShearStressTransportEquationSystem::register_nodal_fields(
  stk::mesh::Part *part)
{

  stk::mesh::MetaData &meta_data = realm_.meta_data();
  const int numStates = realm_.number_of_states();

  // re-register tke and sdr for convenience
  tke_ =  &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "turbulent_ke", numStates));
  stk::mesh::put_field_on_mesh(*tke_, *part, nullptr);
  sdr_ =  &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "specific_dissipation_rate", numStates));
  stk::mesh::put_field_on_mesh(*sdr_, *part, nullptr);

  // SST parameters that everyone needs
  minDistanceToWall_ =  &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "minimum_distance_to_wall"));
  stk::mesh::put_field_on_mesh(*minDistanceToWall_, *part, nullptr);
  fOneBlending_ =  &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "sst_f_one_blending"));
  stk::mesh::put_field_on_mesh(*fOneBlending_, *part, nullptr);
  
  // DES model
  if ( SST_DES == realm_.solutionOptions_->turbulenceModel_ ) {
    maxLengthScale_ =  &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "sst_max_length_scale"));
    stk::mesh::put_field_on_mesh(*maxLengthScale_, *part, nullptr);
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
  stk::mesh::Part *part)
{

  // types of algorithms
  const AlgorithmType algType = INTERIOR;
  
  if ( SST_DES == realm_.solutionOptions_->turbulenceModel_ ) {

    if ( NULL == sstMaxLengthScaleAlgDriver_ )
      sstMaxLengthScaleAlgDriver_ = new AlgorithmDriver(realm_);

    // create edge algorithm
    std::map<AlgorithmType, Algorithm *>::iterator it =
      sstMaxLengthScaleAlgDriver_->algMap_.find(algType);

    if ( it == sstMaxLengthScaleAlgDriver_->algMap_.end() ) {
      ComputeSSTMaxLengthScaleElemAlgorithm *theAlg
        = new ComputeSSTMaxLengthScaleElemAlgorithm(realm_, part);
      sstMaxLengthScaleAlgDriver_->algMap_[algType] = theAlg;
    }
    else {
      it->second->partVec_.push_back(part);
    }
  }
}

//--------------------------------------------------------------------------
//-------- register_wall_bc ------------------------------------------------
//--------------------------------------------------------------------------
void
ShearStressTransportEquationSystem::register_wall_bc(
  stk::mesh::Part *part,
  const stk::topology &partTopo,
  const WallBoundaryConditionData &/*wallBCData*/)
{
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
  auto* meFC = MasterElementRepo::get_surface_master_element(partTopo);
  const int numScsBip = meFC->num_integration_points();
  stk::mesh::put_field_on_mesh(wallNormDistBip, *part, numScsBip, nullptr);

  realm_.geometryAlgDriver_->register_wall_func_algorithm<WallFuncGeometryAlg>(
    sierra::nalu::WALL, part, get_elem_topo(realm_, *part), "sst_geometry_wall");
}

//--------------------------------------------------------------------------
//-------- solve_and_update ------------------------------------------------
//--------------------------------------------------------------------------
void
ShearStressTransportEquationSystem::solve_and_update()
{
  // wrap timing
  // SST_FIXME: deal with timers; all on misc for SSTEqs double timeA, timeB;
  if ( isInit_ ) {
    // compute projected nodal gradients
    tkeEqSys_->compute_projected_nodal_gradient();
    sdrEqSys_->assemble_nodal_gradient();
    clip_min_distance_to_wall();

    // deal with DES option
    if ( SST_DES == realm_.solutionOptions_->turbulenceModel_ )
      sstMaxLengthScaleAlgDriver_->execute();

    isInit_ = false;
  } else if (realm_.has_mesh_motion()) {
    if (realm_.currentNonlinearIteration_ == 1)
      clip_min_distance_to_wall();

    if (SST_DES == realm_.solutionOptions_->turbulenceModel_)
      sstMaxLengthScaleAlgDriver_->execute();
  }

  // compute blending for SST model
  compute_f_one_blending();

  // SST effective viscosity for k and omega
  tkeEqSys_->compute_effective_diff_flux_coeff();
  sdrEqSys_->compute_effective_diff_flux_coeff();

  // wall values
  tkeEqSys_->compute_wall_model_parameters();
  sdrEqSys_->compute_wall_model_parameters();

  // start the iteration loop
  for ( int k = 0; k < maxIterations_; ++k ) {

    NaluEnv::self().naluOutputP0() << " " << k+1 << "/" << maxIterations_
                    << std::setw(15) << std::right << name_ << std::endl;

    for (int oi=0; oi < numOversetIters_; ++oi) {
      // tke and sdr assemble, load_complete and solve; Jacobi iteration
      tkeEqSys_->assemble_and_solve(tkeEqSys_->kTmp_);
      sdrEqSys_->assemble_and_solve(sdrEqSys_->wTmp_);

      update_and_clip();

      if (decoupledOverset_ && realm_.hasOverset_) {
        realm_.overset_orphan_node_field_update(tkeEqSys_->tke_, 1, 1);
        realm_.overset_orphan_node_field_update(sdrEqSys_->sdr_, 1, 1);
      }
    }
    // compute projected nodal gradients
    tkeEqSys_->compute_projected_nodal_gradient();
    sdrEqSys_->assemble_nodal_gradient();
  }

}

/** Perform sanity checks on TKE/SDR fields
 */
void
ShearStressTransportEquationSystem::initial_work()
{
  using MeshIndex = nalu_ngp::NGPMeshTraits<>::MeshIndex;

  const auto& meshInfo = realm_.mesh_info();
  const auto& meta = meshInfo.meta();
  const auto& ngpMesh = meshInfo.ngp_mesh();
  const auto& fieldMgr = meshInfo.ngp_field_manager();

  auto& tkeNp1 = fieldMgr.get_field<double>(
    tke_->mesh_meta_data_ordinal());
  auto& sdrNp1 = fieldMgr.get_field<double>(
    sdr_->mesh_meta_data_ordinal());
  const auto& density = nalu_ngp::get_ngp_field(meshInfo, "density");
  const auto& viscosity = nalu_ngp::get_ngp_field(meshInfo, "viscosity");

  const stk::mesh::Selector sel =
    (meta.locally_owned_part() | meta.globally_shared_part())
    & stk::mesh::selectField(*sdr_);

  const double clipValue = 1.0e-8;
  nalu_ngp::run_entity_algorithm(
    "SST::update_and_clip", ngpMesh, stk::topology::NODE_RANK, sel,
    KOKKOS_LAMBDA(const MeshIndex& mi) {
      const double tkeNew = tkeNp1.get(mi, 0);
      const double sdrNew = sdrNp1.get(mi, 0);

      if ((tkeNew >= 0.0) && (sdrNew >= 0.0)) {
        tkeNp1.get(mi, 0) = tkeNew;
        sdrNp1.get(mi, 0) = sdrNew;
      } else if ((tkeNew < 0.0) && (sdrNew < 0.0)) {
        // both negative; set TKE to small value, tvisc to molecular visc and use
        // Prandtl/Kolm for SDR
        tkeNp1.get(mi, 0) = clipValue;
        sdrNp1.get(mi, 0) = density.get(mi, 0) * clipValue / viscosity.get(mi, 0);
      } else if (tkeNew < 0.0) {
        // only TKE is off; reset turbulent viscosity to molecular vis and
        // compute new TKE based on SDR and tvisc
        sdrNp1.get(mi, 0) = sdrNew;
        tkeNp1.get(mi, 0) = viscosity.get(mi, 0) * sdrNew / density.get(mi, 0);
      } else {
        // Only SDR is off; reset turbulent viscosity to molecular visc and
        // compute new SDR based on others
        tkeNp1.get(mi, 0) = tkeNew;
        sdrNp1.get(mi, 0) = density.get(mi, 0) * tkeNew / viscosity.get(mi, 0);
      }
    });

  tkeNp1.modify_on_device();
  sdrNp1.modify_on_device();
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

  auto& tkeNp1 = fieldMgr.get_field<double>(
    tke_->mesh_meta_data_ordinal());
  auto& sdrNp1 = fieldMgr.get_field<double>(
    sdr_->mesh_meta_data_ordinal());
  const auto& kTmp = fieldMgr.get_field<double>(
    tkeEqSys_->kTmp_->mesh_meta_data_ordinal());
  const auto& wTmp = fieldMgr.get_field<double>(
    sdrEqSys_->wTmp_->mesh_meta_data_ordinal());
  const auto& density = nalu_ngp::get_ngp_field(meshInfo, "density");
  const auto& viscosity = nalu_ngp::get_ngp_field(meshInfo, "viscosity");
  auto& turbVisc = nalu_ngp::get_ngp_field(meshInfo, "turbulent_viscosity");

  auto* turbViscosity = meta.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "turbulent_viscosity");

  const stk::mesh::Selector sel =
    (meta.locally_owned_part() | meta.globally_shared_part())
    & stk::mesh::selectField(*turbViscosity);

  const double clipValue = 1.0e-8;
  nalu_ngp::run_entity_algorithm(
    "SST::update_and_clip", ngpMesh, stk::topology::NODE_RANK, sel,
    KOKKOS_LAMBDA(const MeshIndex& mi) {
      const double tkeNew = tkeNp1.get(mi, 0) + kTmp.get(mi, 0);
      const double sdrNew = sdrNp1.get(mi, 0) + wTmp.get(mi, 0);

      if ((tkeNew >= 0.0) && (sdrNew >= 0.0)) {
        tkeNp1.get(mi, 0) = tkeNew;
        sdrNp1.get(mi, 0) = sdrNew;
      } else if ((tkeNew < 0.0) && (sdrNew < 0.0)) {
        // both negative; set TKE to small value, tvisc to molecular visc and use
        // Prandtl/Kolm for SDR
        tkeNp1.get(mi, 0) = clipValue;
        turbVisc.get(mi, 0) = viscosity.get(mi, 0);
        sdrNp1.get(mi, 0) = density.get(mi, 0) * clipValue / viscosity.get(mi, 0);
      } else if (tkeNew < 0.0) {
        // only TKE is off; reset turbulent viscosity to molecular vis and
        // compute new TKE based on SDR and tvisc
        turbVisc.get(mi, 0) = viscosity.get(mi, 0);
        sdrNp1.get(mi, 0) = sdrNew;
        tkeNp1.get(mi, 0) = viscosity.get(mi, 0) * sdrNew / density.get(mi, 0);
      } else {
        // Only SDR is off; reset turbulent viscosity to molecular visc and
        // compute new SDR based on others
        turbVisc.get(mi, 0) = viscosity.get(mi, 0);
        tkeNp1.get(mi, 0) = tkeNew;
        sdrNp1.get(mi, 0) = density.get(mi, 0) * tkeNew / viscosity.get(mi, 0);
      }
    });

  tkeNp1.modify_on_device();
  sdrNp1.modify_on_device();
  turbVisc.modify_on_device();
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

  auto& ndtw = fieldMgr.get_field<double>(
    minDistanceToWall_->mesh_meta_data_ordinal());
  const auto& wallNormDist = nalu_ngp::get_ngp_field(
    meshInfo, "assembled_wall_normal_distance");

  const stk::mesh::Selector sel = (meta.locally_owned_part() | meta.globally_shared_part())
    & stk::mesh::selectUnion(wallBcPart_);

  nalu_ngp::run_entity_algorithm(
    "SST::clip_ndtw", ngpMesh, stk::topology::NODE_RANK, sel,
    KOKKOS_LAMBDA(const MeshIndex& mi) {
      const double minD = ndtw.get(mi, 0);

      ndtw.get(mi, 0) = stk::math::max(minD, wallNormDist.get(mi, 0));
    });

   stk::mesh::parallel_max(realm_.bulk_data(), {minDistanceToWall_});
   if (realm_.hasPeriodic_) {
     realm_.periodic_field_max(minDistanceToWall_, 1);
   }
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

  const auto& tkeNp1 = fieldMgr.get_field<double>(
    tke_->mesh_meta_data_ordinal());
  const auto& sdrNp1 = fieldMgr.get_field<double>(
    sdr_->mesh_meta_data_ordinal());
  const auto& density = nalu_ngp::get_ngp_field(meshInfo, "density");
  const auto& viscosity = nalu_ngp::get_ngp_field(meshInfo, "viscosity");
  const auto& dkdx = fieldMgr.get_field<double>(
    tkeEqSys_->dkdx_->mesh_meta_data_ordinal());
  const auto& dwdx = fieldMgr.get_field<double>(
    sdrEqSys_->dwdx_->mesh_meta_data_ordinal());
  const auto& ndtw = fieldMgr.get_field<double>(
    minDistanceToWall_->mesh_meta_data_ordinal());
  auto& fOneBlend = fieldMgr.get_field<double>(
    fOneBlending_->mesh_meta_data_ordinal());

  const stk::mesh::Selector sel =
    (meta.locally_owned_part() | meta.globally_shared_part())
    & (stk::mesh::selectField(*fOneBlending_));

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
      for (int d=0; d < ndim; ++d)
        crossdiff += dkdx.get(mi, d) * dwdx.get(mi, d);

      const double minDistSq = minD * minD;
      const double turbDiss = stk::math::sqrt(tke) / betaStar / sdr / minD;
      const double lamDiss = 500.0 * mu / rho / sdr / minDistSq;
      const double CDkw = stk::math::max(
        2.0 * rho * sigmaWTwo * crossdiff / sdr, CDkwClip);

      const double fArgOne = stk::math::min(
        stk::math::max(turbDiss, lamDiss),
        4.0 * rho * sigmaWTwo * tke / CDkw / minDistSq);

      fOneBlend.get(mi, 0) = stk::math::tanh(fArgOne * fArgOne * fArgOne * fArgOne);
    });

  fOneBlend.modify_on_device();
}

} // namespace nalu
} // namespace Sierra
