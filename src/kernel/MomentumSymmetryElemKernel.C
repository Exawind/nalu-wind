// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "kernel/MomentumSymmetryElemKernel.h"
#include "master_element/MasterElement.h"
#include "master_element/MasterElementFactory.h"
#include "SolutionOptions.h"

// template and scratch space
#include "BuildTemplates.h"
#include "ScratchViews.h"
#include "utils/StkHelpers.h"

// stk_mesh/base/fem
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Field.hpp>

namespace sierra {
namespace nalu {
namespace {
template <typename BcAlgTraits, typename T>
void
get_shape_fcn(T& vf_shape_function, MasterElement* meFC_dev)
{
  auto dev_shape_function = Kokkos::create_mirror_view(vf_shape_function);
  Kokkos::parallel_for(
    "get_shape_fcn_data", DeviceRangePolicy(0, 1), KOKKOS_LAMBDA(int) {
      SharedMemView<DoubleType**, DeviceShmem> ShmemView(
        dev_shape_function.data(), BcAlgTraits::numFaceIp_,
        BcAlgTraits::nodesPerFace_);
      meFC_dev->shape_fcn<>(ShmemView);
    });
  Kokkos::deep_copy(vf_shape_function, dev_shape_function);
}
} // namespace

template <typename BcAlgTraits>
MomentumSymmetryElemKernel<BcAlgTraits>::MomentumSymmetryElemKernel(
  const stk::mesh::MetaData& metaData,
  const SolutionOptions& solnOpts,
  VectorFieldType* velocity,
  ScalarFieldType* viscosity,
  ElemDataRequests& faceDataPreReqs,
  ElemDataRequests& elemDataPreReqs)
  : Kernel(),
    viscosity_(viscosity->mesh_meta_data_ordinal()),
    velocityNp1_(
      velocity->field_of_state(stk::mesh::StateNP1).mesh_meta_data_ordinal()),
    coordinates_(get_field_ordinal(metaData, solnOpts.get_coordinates_name())),
    exposedAreaVec_(
      get_field_ordinal(metaData, "exposed_area_vector", metaData.side_rank())),
    includeDivU_(solnOpts.includeDivU_),
    meSCS_(MasterElementRepo::get_surface_master_element_on_host(
      BcAlgTraits::elemTopo_)),
    penaltyFactor_(solnOpts.symmetryBcPenaltyFactor_)
{
  auto* meFC = MasterElementRepo::get_surface_master_element_on_host(
    BcAlgTraits::faceTopo_);
  auto* meFC_dev = MasterElementRepo::get_surface_master_element_on_dev(
    BcAlgTraits::faceTopo_);
  faceDataPreReqs.add_cvfem_face_me(meFC);
  elemDataPreReqs.add_cvfem_surface_me(meSCS_);
  faceDataPreReqs.add_gathered_nodal_field(viscosity_, 1);
  faceDataPreReqs.add_gathered_nodal_field(velocityNp1_, BcAlgTraits::nDim_);
  faceDataPreReqs.add_face_field(
    exposedAreaVec_, BcAlgTraits::numFaceIp_, BcAlgTraits::nDim_);
  elemDataPreReqs.add_coordinates_field(
    coordinates_, BcAlgTraits::nDim_, CURRENT_COORDINATES);
  elemDataPreReqs.add_gathered_nodal_field(velocityNp1_, BcAlgTraits::nDim_);

  if (solnOpts.get_shifted_grad_op(velocity->name())) {
    elemDataPreReqs.add_master_element_call(
      SCS_SHIFTED_FACE_GRAD_OP, CURRENT_COORDINATES);
  } else {
    elemDataPreReqs.add_master_element_call(
      SCS_FACE_GRAD_OP, CURRENT_COORDINATES);
  }
  get_shape_fcn<BcAlgTraits>(vf_shape_function_, meFC_dev);
}

template <typename BcAlgTraits>
MomentumSymmetryElemKernel<BcAlgTraits>::~MomentumSymmetryElemKernel()
{
}

template <int n, typename ScalarU, typename ScalarV>
KOKKOS_FORCEINLINE_FUNCTION DoubleType
ddot(const ScalarU* u, const ScalarV* v)
{
  DoubleType result = 0;
  for (int d = 0; d < n; ++d) {
    result += u[d] * v[d];
  }
  return result;
}

template <typename BcAlgTraits>
void
MomentumSymmetryElemKernel<BcAlgTraits>::execute(
  SharedMemView<DoubleType**>& lhs,
  SharedMemView<DoubleType*>& rhs,
  ScratchViews<DoubleType>& faceScratchViews,
  ScratchViews<DoubleType>& elemScratchViews,
  int elemFaceOrdinal)
{
  constexpr int dim = BcAlgTraits::nDim_;
  const int* face_node_ordinals = meSCS_->side_node_ordinals(elemFaceOrdinal);

  auto& face_mu = faceScratchViews.get_scratch_view_1D(viscosity_);
  auto& face_velocity = faceScratchViews.get_scratch_view_2D(velocityNp1_);
  auto& face_areav = faceScratchViews.get_scratch_view_2D(exposedAreaVec_);
  auto& dndx = elemScratchViews.get_me_views(CURRENT_COORDINATES).dndx_fc_scs;
  auto& elem_velocity = elemScratchViews.get_scratch_view_2D(velocityNp1_);

  for (int ip = 0; ip < BcAlgTraits::numFaceIp_; ++ip) {
    const int nn = meSCS_->ipNodeMap(elemFaceOrdinal)[ip];

    NALU_ALIGNED Kokkos::Array<DoubleType, 3> uIp = {{0, 0, 0}};
    DoubleType viscIp = 0.;
    for (int n = 0; n < BcAlgTraits::nodesPerFace_; ++n) {
      const auto r = vf_shape_function_(ip, n);
      viscIp += r * face_mu(n);
      for (int d = 0; d < dim; ++d) {
        uIp[d] += r * face_velocity(n, d);
      }
    }
    const auto* areavIp = &face_areav(ip, 0);
    DoubleType areaWeightedInverseLengthScale = 0;
    for (int n = 0; n < BcAlgTraits::nodesPerFace_; ++n) {
      areaWeightedInverseLengthScale +=
        ddot<dim>(&dndx(ip, face_node_ordinals[n], 0), areavIp);
    }

    const auto inv_amag = 1.0 / stk::math::sqrt(ddot<dim>(areavIp, areavIp));
    const auto un = ddot<dim>(uIp.data(), areavIp) * inv_amag;

    const auto penaltyFac =
      -penaltyFactor_ * viscIp * areaWeightedInverseLengthScale;
    for (int dj = 0; dj < dim; ++dj) {
      const int indexR = nn * dim + dj;
      for (int di = 0; di < dim; ++di) {
        for (int n = 0; n < BcAlgTraits::nodesPerFace_; ++n) {
          lhs(indexR, face_node_ordinals[n] * dim + di) +=
            -vf_shape_function_(ip, n) * penaltyFac * areavIp[dj] *
            areavIp[di] * inv_amag * inv_amag;
        }
      }
      rhs(indexR) -= -penaltyFac * un * areavIp[dj] * inv_amag;
    }

    NALU_ALIGNED Kokkos::Array<Kokkos::Array<DoubleType, 3>, 3> viscStressIp;
    for (int dj = 0; dj < dim; ++dj) {
      for (int di = 0; di < dim; ++di) {
        viscStressIp[dj][di] = 0;
      }
    }

    for (int n = 0; n < BcAlgTraits::nodesPerElement_; ++n) {
      for (int dj = 0; dj < dim; ++dj) {
        for (int di = 0; di < dim; ++di) {
          viscStressIp[dj][di] +=
            viscIp * (dndx(ip, n, dj) * elem_velocity(n, di) +
                      dndx(ip, n, di) * elem_velocity(n, dj));
        }
      }
    }

    DoubleType divuIp = 0.;
    for (int n = 0; n < BcAlgTraits::nodesPerElement_; ++n) {
      for (int dj = 0; dj < dim; ++dj) {
        divuIp += dndx(ip, n, dj) * elem_velocity(n, dj);
      }
    }
    for (int dj = 0; dj < dim; ++dj) {
      viscStressIp[dj][dj] -= 2.0 / dim * viscIp * divuIp * includeDivU_;
    }

    DoubleType faceNormalViscFlux = 0;
    for (int d = 0; d < dim; ++d) {
      faceNormalViscFlux -=
        ddot<dim>(&viscStressIp[d][0], areavIp) * areavIp[d] * inv_amag;
    }

    for (int dj = 0; dj < dim; ++dj) {
      for (int n = 0; n < BcAlgTraits::nodesPerElement_; ++n) {
        const auto fac = -2 * viscIp *
                         (ddot<dim>(&dndx(ip, n, 0), areavIp) * areavIp[dj] *
                            inv_amag * inv_amag -
                          dndx(ip, n, dj) * includeDivU_ / dim);
        for (int di = 0; di < dim; ++di) {
          lhs(nn * dim + dj, n * dim + di) += fac * areavIp[di];
        }
      }
      rhs(nn * dim + dj) -= faceNormalViscFlux * areavIp[dj] * inv_amag;
    }
  }
}

INSTANTIATE_KERNEL_FACE_ELEMENT(MomentumSymmetryElemKernel)

} // namespace nalu
} // namespace sierra
