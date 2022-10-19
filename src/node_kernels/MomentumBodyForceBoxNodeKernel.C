// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "node_kernels/MomentumBodyForceBoxNodeKernel.h"
#include "Realm.h"
#include "SolutionOptions.h"
#include "LowMachEquationSystem.h"
#include "NaluEnv.h"
#include "master_element/MasterElementFactory.h"
#include "TimeIntegrator.h"

#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/Types.hpp"
#include "stk_mesh/base/Selector.hpp"
#include "utils/StkHelpers.h"
#include <stk_util/parallel/ParallelReduce.hpp>

#include "ngp_utils/NgpLoopUtils.h"

#include <fstream>
#include <iomanip>

namespace sierra {
namespace nalu {

MomentumBodyForceBoxNodeKernel::MomentumBodyForceBoxNodeKernel(
  Realm& realm,
  const std::vector<double>& forces,
  const std::vector<double>& box)
  : NGPNodeKernel<MomentumBodyForceBoxNodeKernel>(),
    nDim_(realm.meta_data().spatial_dimension()),
    coordinatesID_(get_field_ordinal(
      realm.meta_data(), realm.solutionOptions_->get_coordinates_name())),
    dualNodalVolumeID_(
      get_field_ordinal(realm.meta_data(), "dual_nodal_volume")),
    outputFileName_(realm.solutionOptions_->dynamicBodyForceOutFile_),
    dynamic_(realm.solutionOptions_->dynamicBodyForceBox_),
    w_(16)
{
  for (int i = 0; i < nDim_; ++i)
    forceVector_[i] = forces[i];
  for (int i = 0; i < nDim_; ++i) {
    lo_[i] = box[i];
    hi_[i] = box[nDim_ + i];
  }

  if (dynamic_) {

    // Define the parts
    mdotPart_ = realm.meta_data().get_part(
      realm.solutionOptions_->dynamicBodyForceVelTarget_);

    // Register the exposed area vector
    geometryAlgDriver_ = realm.geometryAlgDriver_.get();
    const AlgorithmType algType = BOUNDARY;
    const auto& subParts = mdotPart_->subsets();
    for (auto* part : subParts) {
      const auto topo = part->topology();
      MasterElement* meFC = MasterElementRepo::get_surface_master_element(topo);
      const int numScsIp = meFC->num_integration_points();
      GenericFieldType* exposedAreaVec_ =
        &(realm.meta_data().declare_field<GenericFieldType>(
          static_cast<stk::topology::rank_t>(realm.meta_data().side_rank()),
          "exposed_area_vector"));
      stk::mesh::put_field_on_mesh(
        *exposedAreaVec_, *part, nDim_ * numScsIp, nullptr);
      geometryAlgDriver_->register_face_algorithm<GeometryBoundaryAlg>(
        algType, part, "geometry");
    }
    exposedAreaVecID_ = get_field_ordinal(
      realm.meta_data(), "exposed_area_vector", realm.meta_data().side_rank());

    // Register the mdot inflow algorithm
    for (auto* eqsys : realm.equationSystems_.equationSystemVector_) {
      if (eqsys->name_ == "ContinuityEQS") {
        auto* ceq = dynamic_cast<ContinuityEquationSystem*>(eqsys);
        mdotAlgDriver_ = ceq->mdotAlgDriver_.get();
        const AlgorithmType algType = INFLOW;
        const bool useShifted =
          !ceq->elementContinuityEqs_ ? true : realm.get_cvfem_shifted_mdot();
        for (auto* part : subParts) {
          mdotAlgDriver_->register_face_algorithm<MdotInflowAlg>(
            algType, part, "body_force_inflow", *mdotAlgDriver_, useShifted);
        }
      }
    }

    if (NaluEnv::self().parallel_rank() == 0) {
      std::ofstream myfile;
      myfile.open(outputFileName_.c_str());
      myfile << std::setw(w_) << "Time" << std::setw(w_) << "mdot"
             << std::setw(w_) << "drag" << std::setw(w_) << "Fx"
             << std::setw(w_) << "Fy" << std::setw(w_) << "Fz" << std::endl;
      myfile.close();
    }
  }
}

void
MomentumBodyForceBoxNodeKernel::setup(Realm& realm)
{
  const auto& fieldMgr = realm.ngp_field_manager();
  coordinates_ = fieldMgr.get_field<double>(coordinatesID_);
  dualNodalVolume_ = fieldMgr.get_field<double>(dualNodalVolumeID_);

  if ((dynamic_) && (realm.currentNonlinearIteration_ == 1)) {

    const double currentTime = realm.get_current_time();
    const double dt = realm.get_time_step();
    const double mdot = -mdotAlgDriver_->mdot_inflow();

    // Compute area of the mdot sideset
    using MeshIndex = nalu_ngp::NGPMeshTraits<stk::mesh::NgpMesh>::MeshIndex;
    const auto& ngpMesh = realm.ngp_mesh();
    auto areaVec = fieldMgr.get_field<double>(exposedAreaVecID_);
    const auto& subParts = mdotPart_->subsets();
    double l_mdotArea = 0.0;
    for (auto* part : subParts) {
      const auto topo = part->topology();
      MasterElement* meFC = MasterElementRepo::get_surface_master_element(topo);
      const int numScsIp = meFC->num_integration_points();
      const std::string algName = "compute_mdot_area_" + std::to_string(topo);
      const stk::mesh::Selector sel = realm.meta_data().locally_owned_part() &
                                      stk::mesh::Selector(*mdotPart_);
      const auto ndim = nDim_;

      double ma = 0.0;
      nalu_ngp::run_entity_par_reduce(
        algName, ngpMesh, realm.meta_data().side_rank(), sel,
        KOKKOS_LAMBDA(const MeshIndex& mi, double& sum) {
          for (int ip = 0; ip < numScsIp; ++ip) {
            const int offSetAveraVec = ip * ndim;
            double aMag = 0.0;
            for (int j = 0; j < ndim; ++j) {
              aMag += areaVec.get(mi, offSetAveraVec + j) *
                      areaVec.get(mi, offSetAveraVec + j);
            }
            aMag = stk::math::sqrt(aMag);
            sum += aMag;
          }
        },
        ma);
      l_mdotArea += ma;
    }
    double mdotArea = 0.0;
    stk::all_reduce_sum(
      NaluEnv::self().parallel_comm(), &l_mdotArea, &mdotArea, 1);

    // Compute drag
    pressureForceID_ = get_field_ordinal(realm.meta_data(), "pressure_force");
    viscousForceID_ = get_field_ordinal(realm.meta_data(), "viscous_force");
    auto pForce = fieldMgr.get_field<double>(pressureForceID_);
    auto vForce = fieldMgr.get_field<double>(viscousForceID_);
    double l_drag = 0.0;
    const std::string algName = "compute_drag";
    const auto& dragTarget =
      realm.solutionOptions_->dynamicBodyForceDragTarget_;
    stk::mesh::PartVector dragPartVec;
    for (const auto& dt : dragTarget) {
      stk::mesh::Part* targetPart =
        realm.bulk_data().mesh_meta_data().get_part(dt);
      dragPartVec.push_back(targetPart);
    }
    const stk::mesh::Selector sel = realm.meta_data().locally_owned_part() &
                                    stk::mesh::selectUnion(dragPartVec);
    nalu_ngp::run_entity_par_reduce(
      algName, ngpMesh, stk::topology::NODE_RANK, sel,
      KOKKOS_LAMBDA(const MeshIndex& mi, double& total_force_x) {
        total_force_x += pForce.get(mi, 0) + vForce.get(mi, 0);
      },
      l_drag);
    double drag = 0.0;
    stk::all_reduce_sum(NaluEnv::self().parallel_comm(), &l_drag, &drag, 1);

    // Compute forcing
    const auto& uRef = realm.solutionOptions_->dynamicBodyForceVelReference_;
    const auto& forcingDir = realm.solutionOptions_->dynamicBodyForceDir_;
    const auto& densityRef =
      realm.solutionOptions_->dynamicBodyForceDenReference_;
    const auto& volume = geometryAlgDriver_->total_volume();

    for (int d = 0; d < NodeKernelTraits::NDimMax; d++) {
      forceVector_[d] =
        d == forcingDir
          ? drag / volume -
              (mdot - densityRef * uRef * mdotArea) / (mdotArea * dt)
          : 0.0;
    }

    if (NaluEnv::self().parallel_rank() == 0) {
      std::ofstream myfile;
      myfile.open(outputFileName_.c_str(), std::ios_base::app);
      myfile << std::setprecision(6) << std::setw(w_) << currentTime
             << std::setw(w_) << mdot << std::setw(w_) << drag << std::setw(w_)
             << forceVector_[0] << std::setw(w_) << forceVector_[1]
             << std::setw(w_) << forceVector_[2] << std::endl;
      myfile.close();
    }
  }
}

void
MomentumBodyForceBoxNodeKernel::execute(
  NodeKernelTraits::LhsType&,
  NodeKernelTraits::RhsType& rhs,
  const stk::mesh::FastMeshIndex& node)
{

  bool is_inside = true;
  for (int i = 0; i < nDim_; ++i) {
    if (!((lo_[i] <= coordinates_.get(node, i)) &&
          (coordinates_.get(node, i) <= hi_[i]))) {
      is_inside = false;
      break;
    }
  }

  if (is_inside) {
    const NodeKernelTraits::DblType dualVolume = dualNodalVolume_.get(node, 0);
    for (int i = 0; i < nDim_; ++i)
      rhs(i) += dualVolume * forceVector_[i];
  }
}

} // namespace nalu
} // namespace sierra
