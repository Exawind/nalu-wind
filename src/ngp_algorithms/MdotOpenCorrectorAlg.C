// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "ngp_algorithms/MdotOpenCorrectorAlg.h"
#include "BuildTemplates.h"
#include "ngp_algorithms/MdotAlgDriver.h"
#include "ngp_utils/NgpLoopUtils.h"
#include "ngp_utils/NgpFieldOps.h"
#include "Realm.h"
#include "utils/StkHelpers.h"

namespace sierra {
namespace nalu {

template <typename BcAlgTraits>
MdotOpenCorrectorAlg<BcAlgTraits>::MdotOpenCorrectorAlg(
  Realm& realm, stk::mesh::Part* part, MdotAlgDriver& algDriver)
  : Algorithm(realm, part),
    mdotDriver_(algDriver),
    openMassFlowRate_(get_field_ordinal(
      realm_.meta_data(),
      "open_mass_flow_rate",
      realm_.meta_data().side_rank()))
{}

template<typename BcAlgTraits>
void MdotOpenCorrectorAlg<BcAlgTraits>::execute()
{
  using MeshIndex = nalu_ngp::NGPMeshTraits<ngp::Mesh>::MeshIndex;

  const auto& ngpMesh = realm_.ngp_mesh();
  const auto& fieldMgr = realm_.ngp_field_manager();
  auto openMdot = fieldMgr.template get_field<double>(openMassFlowRate_);
  const double mdotCorr = mdotDriver_.mdot_open_correction();

  const stk::mesh::Selector sel = realm_.meta_data().locally_owned_part()
    & stk::mesh::selectUnion(partVec_);

  double mdotSum = 0.0;
  const std::string algName = "correct_open_mdot_" + std::to_string(BcAlgTraits::topo_);
  nalu_ngp::run_entity_par_reduce(
    algName, ngpMesh, realm_.meta_data().side_rank(), sel,
    KOKKOS_LAMBDA(const MeshIndex& mi, double& pSum) {
      for (int ip=0; ip < BcAlgTraits::numFaceIp_; ++ip) {
        openMdot.get(mi, ip) -= mdotCorr;
        pSum += openMdot.get(mi, ip);
      }
    }, mdotSum);

  mdotDriver_.add_open_mdot_post(mdotSum);
  openMdot.modify_on_device();
}

INSTANTIATE_KERNEL_FACE(MdotOpenCorrectorAlg)

}  // nalu
}  // sierra
