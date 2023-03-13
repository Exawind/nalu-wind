// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <iomanip>

#include "ngp_algorithms/MdotAlgDriver.h"
#include "ngp_utils/NgpLoopUtils.h"
#include "ngp_utils/NgpFieldOps.h"
#include "Realm.h"
#include "SolutionOptions.h"
#include "master_element/MasterElement.h"
#include "master_element/MasterElementRepo.h"
#include "utils/StkHelpers.h"

#include "stk_mesh/base/Field.hpp"
#include "stk_mesh/base/FieldParallel.hpp"
#include "stk_mesh/base/FieldBLAS.hpp"
#include "stk_mesh/base/MetaData.hpp"

namespace sierra {
namespace nalu {

namespace {
inline void
simd_add(const DoubleType& inVal, double& outVal)
{
  for (int i = 0; i < simdLen; ++i)
    outVal += stk::simd::get_data(inVal, i);
}

} // namespace

MdotAlgDriver::MdotAlgDriver(Realm& realm, const bool elemContinuityEqs)
  : NgpAlgDriver(realm), elemContinuityEqs_(elemContinuityEqs)
{
}

void
MdotAlgDriver::add_density_accumulation(const DoubleType& rhoAcc)
{
  simd_add(rhoAcc, rhoAccum_);
}

void
MdotAlgDriver::add_inflow_mdot(const DoubleType& inflow)
{
  simd_add(inflow, mdotInflow_);
}

void
MdotAlgDriver::add_open_mdot(const DoubleType& outflow)
{
  simd_add(outflow, mdotOpen_);
}

void
MdotAlgDriver::add_density_accumulation(const double& rhoAcc)
{
  rhoAccum_ += rhoAcc;
}

void
MdotAlgDriver::add_inflow_mdot(const double& inflow)
{
  mdotInflow_ += inflow;
}

void
MdotAlgDriver::add_open_mdot(const double& outflow)
{
  mdotOpen_ += outflow;
}

void
MdotAlgDriver::add_open_mdot_post(const double& outflow)
{
  mdotOpenPost_ += outflow;
}

void
MdotAlgDriver::pre_work()
{
  // Reset variables that algorithms will accumulate
  rhoAccum_ = 0.0;
  mdotInflow_ = 0.0;
  mdotOpen_ = 0.0;
  mdotOpenPost_ = 0.0;
  mdotOpenCorrection_ = 0.0;

  // Assume that the presence of "open_mass_flow_rate" means there is at least
  // one open BC sideset
  auto* openMassFlowRate = realm_.meta_data().get_field<GenericFieldType>(
    realm_.meta_data().side_rank(), "open_mass_flow_rate");
  hasOpenBC_ = !(openMassFlowRate == nullptr);

  if (isInit_ && hasOpenBC_) {
    // Compute the number of integration points across all the open boundary
    // sidesets. The global sum is used to determine the net correction added to
    // the mdot value at each integration point if user sets
    // activate_open_mdot_correction to true. See post_work for more details
    const stk::mesh::Selector sel = realm_.meta_data().locally_owned_part() &
                                    stk::mesh::selectField(*openMassFlowRate) &
                                    !(realm_.get_inactive_selector());

    unsigned numIp = 0;
    const auto& bkts =
      realm_.bulk_data().get_buckets(realm_.meta_data().side_rank(), sel);
    if (elemContinuityEqs_) {
      for (const auto* b : bkts) {
        auto* meFC =
          MasterElementRepo::get_surface_master_element_on_host(b->topology());
        numIp += (b->size() * meFC->num_integration_points());
      }
    } else {
      for (const auto* b : bkts) {
        numIp += (b->size() * b->topology().num_nodes());
      }
    }

    unsigned g_numIp = 0;
    stk::all_reduce_sum(NaluEnv::self().parallel_comm(), &numIp, &g_numIp, 1);
    mdotOpenIpCount_ = g_numIp;
    isInit_ = false;
  }
}

void
MdotAlgDriver::post_work()
{
  double local[3] = {rhoAccum_, mdotInflow_, mdotOpen_};
  double global[3] = {0.0, 0.0, 0.0};
  stk::all_reduce_sum(NaluEnv::self().parallel_comm(), local, global, 3);

  rhoAccum_ = global[0];
  mdotInflow_ = global[1];
  mdotOpen_ = global[2];

  if (realm_.solutionOptions_->activateOpenMdotCorrection_ && hasOpenBC_) {
    mdotOpenCorrection_ =
      (rhoAccum_ + mdotInflow_ + mdotOpen_) / mdotOpenIpCount_;

    for (auto& kv : correctOpenMdotAlgs_)
      kv.second->execute();

    double gPost = 0.0;
    stk::all_reduce_sum(
      NaluEnv::self().parallel_comm(), &mdotOpenPost_, &gPost, 1);
    mdotOpenPost_ = gPost;
  }

  // TODO: Remove these from SolutionOptions. Here to assist during transition
  // phase
  realm_.solutionOptions_->mdotAlgOpenCorrection_ = mdotOpenCorrection_;
}

void
MdotAlgDriver::provide_output()
{
  const double totalMassClosure = (rhoAccum_ + mdotInflow_ + mdotOpen_);

  NaluEnv::self().naluOutputP0()
    << "Mass Balance Review:"
    << "\nDensity accumulation: " << std::setprecision(16) << rhoAccum_
    << "\nIntegrated inflow:    " << std::setprecision(16) << mdotInflow_
    << "\nIntegrated open:      " << std::setprecision(16) << mdotOpen_
    << "\nTotal mass closure:   " << std::setprecision(16) << totalMassClosure
    << std::endl;

  if (realm_.solutionOptions_->activateOpenMdotCorrection_) {
    NaluEnv::self().naluOutputP0()
      << "A mass correction of: " << mdotOpenCorrection_
      << " occurred on: " << mdotOpenIpCount_ << " boundary integration points."
      << std::endl
      << "Post-corrected integrated open: " << std::setprecision(16)
      << mdotOpenPost_ << std::endl;
  }
}

void
MdotAlgDriver::register_open_mdot_corrector_alg(
  AlgorithmType algType, stk::mesh::Part* part, const std::string& algSuffix)
{
  const auto topo = part->topology();
  const std::string entityName = "face_" + topo.name();

  const std::string algName = unique_name(algType, entityName, algSuffix);

  const auto it = correctOpenMdotAlgs_.find(algName);
  if (it == correctOpenMdotAlgs_.end()) {
    correctOpenMdotAlgs_[algName].reset(
      nalu_ngp::create_face_algorithm<Algorithm, MdotOpenCorrectorAlg>(
        topo, realm_, part, *this));
  } else {
    it->second->partVec_.push_back(part);
  }
}

} // namespace nalu
} // namespace sierra
