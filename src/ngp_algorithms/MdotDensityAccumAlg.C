// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "ngp_algorithms/MdotDensityAccumAlg.h"
#include "BuildTemplates.h"
#include "master_element/MasterElement.h"
#include "master_element/MasterElementRepo.h"
#include "ngp_algorithms/ViewHelper.h"
#include "ngp_algorithms/MdotAlgDriver.h"
#include "ngp_utils/NgpLoopUtils.h"
#include "ngp_utils/NgpFieldOps.h"
#include "ngp_utils/NgpReduceUtils.h"
#include "Realm.h"
#include "ScratchViews.h"
#include "SolutionOptions.h"
#include "utils/StkHelpers.h"
#include "stk_mesh/base/NgpMesh.hpp"

namespace sierra {
namespace nalu {

template <typename AlgTraits>
MdotDensityAccumAlg<AlgTraits>::MdotDensityAccumAlg(
  Realm& realm,
  stk::mesh::Part* part,
  MdotAlgDriver& mdotDriver,
  bool lumpedMass)
  : Algorithm(realm, part),
    mdotDriver_(mdotDriver),
    elemData_(realm.meta_data()),
    rhoNp1_(
      get_field_ordinal(realm_.meta_data(), "density", stk::mesh::StateNP1)),
    rhoN_(get_field_ordinal(realm_.meta_data(), "density", stk::mesh::StateN)),
    rhoNm1_(get_field_ordinal(
      realm_.meta_data(),
      "density",
      realm_.number_of_states() == 2 ? stk::mesh::StateN
                                     : stk::mesh::StateNM1)),
    meSCV_(
      MasterElementRepo::get_volume_master_element_on_dev(AlgTraits::topo_)),
    lumpedMass_(lumpedMass)
{
  elemData_.add_cvfem_volume_me(meSCV_);

  const auto coordID = get_field_ordinal(
    realm_.meta_data(), realm_.solutionOptions_->get_coordinates_name());
  elemData_.add_coordinates_field(
    coordID, AlgTraits::nDim_, CURRENT_COORDINATES);
  elemData_.add_gathered_nodal_field(rhoNp1_, 1);
  elemData_.add_gathered_nodal_field(rhoN_, 1);
  elemData_.add_gathered_nodal_field(rhoNm1_, 1);

  elemData_.add_master_element_call(SCV_VOLUME, CURRENT_COORDINATES);
  const auto shp_fcn_type = lumpedMass_ ? SCV_SHIFTED_SHAPE_FCN : SCV_SHAPE_FCN;
  elemData_.add_master_element_call(shp_fcn_type, CURRENT_COORDINATES);
}

template <typename AlgTraits>
void
MdotDensityAccumAlg<AlgTraits>::execute()
{
  using ElemSimdDataType =
    sierra::nalu::nalu_ngp::ElemSimdData<stk::mesh::NgpMesh>;

  const auto& meshInfo = realm_.mesh_info();

  const unsigned rhoNp1ID = rhoNp1_;
  const unsigned rhoNID = rhoN_;
  const unsigned rhoNm1ID = rhoNm1_;
  const bool lumpedMass = lumpedMass_;

  const DoubleType dt = realm_.get_time_step();
  const DoubleType gamma1 = realm_.get_gamma1();
  const DoubleType gamma2 = realm_.get_gamma2();
  const DoubleType gamma3 = realm_.get_gamma3();

  const std::string algName =
    "mdot_density_accum_" + std::to_string(AlgTraits::topo_);
  DoubleType rhoAcc = 0.0;
  Kokkos::Sum<DoubleType> mdotReducer(rhoAcc);

  const stk::mesh::Selector sel =
    stk::mesh::selectField(
      *realm_.meta_data().template get_field<ScalarFieldType>(
        stk::topology::NODE_RANK, "density")) &
    !(realm_.get_inactive_selector());

  nalu_ngp::run_elem_par_reduce(
    algName, meshInfo, stk::topology::ELEM_RANK, elemData_, sel,
    KOKKOS_LAMBDA(ElemSimdDataType & edata, DoubleType & acc) {
      auto& scrViews = edata.simdScrView;
      auto& densityNp1 = scrViews.get_scratch_view_1D(rhoNp1ID);
      auto& densityN = scrViews.get_scratch_view_1D(rhoNID);
      auto& densityNm1 = scrViews.get_scratch_view_1D(rhoNm1ID);

      const auto& meViews = scrViews.get_me_views(CURRENT_COORDINATES);
      const auto& v_scv_vol = meViews.scv_volume;
      const auto& v_shape_fcn =
        lumpedMass ? meViews.scv_shifted_shape_fcn : meViews.scv_shape_fcn;

      for (int ip = 0; ip < AlgTraits::numScvIp_; ++ip) {
        DoubleType rhoNm1 = 0.0;
        DoubleType rhoN = 0.0;
        DoubleType rhoNp1 = 0.0;

        for (int ic = 0; ic < AlgTraits::nodesPerElement_; ++ic) {
          const DoubleType r = v_shape_fcn(ip, ic);
          rhoNm1 += r * densityNm1(ic);
          rhoN += r * densityN(ic);
          rhoNp1 += r * densityNp1(ic);
        }

        acc += (gamma1 * rhoNp1 + gamma2 * rhoN + gamma3 * rhoNm1) / dt *
               v_scv_vol(ip);
      }
    },
    mdotReducer);

  mdotDriver_.add_density_accumulation(rhoAcc);
}

INSTANTIATE_KERNEL(MdotDensityAccumAlg)

} // namespace nalu
} // namespace sierra
