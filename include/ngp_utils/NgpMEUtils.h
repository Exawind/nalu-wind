// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef NGPMEUTILS_H
#define NGPMEUTILS_H

/** \file
 *  \brief Utility functions to extract ME info from device instances
 */

#include "master_element/MasterElement.h"

namespace sierra {
namespace nalu {

/** Types of MasterElements that can be registered with ElemDataRequests
 */
enum struct METype {
  SCV = 0, //!< CVFEM Volume
  SCS,     //!< CVFEM Surface
  FACE,    //!< CVFEM Face
  FEM      //!< FEM
};

/** Type of element data requested by the algorithm in ScratchViews
 */
enum struct ElemReqType { ELEM = 0, FACE, FACE_ELEM };

template <typename DataReqType>
MasterElement*
get_me_instance(const DataReqType& dataReq, const METype meType)
{
  MasterElement* me = nullptr;

  switch (meType) {
  case METype::SCV:
    me = dataReq.get_cvfem_volume_me();
    break;

  case METype::SCS:
    me = dataReq.get_cvfem_surface_me();
    break;

  case METype::FACE:
    me = dataReq.get_cvfem_face_me();
    break;

  case METype::FEM:
    me = dataReq.get_fem_volume_me();
    break;
  }

  return me;
}

template <typename DataReqType>
int
nodes_per_entity(const DataReqType& dataReq, const METype meType)
{
  auto* me = get_me_instance(dataReq, meType);

  // Return immediately if ME is not registered
  if (me == nullptr)
    return 0;

  Kokkos::View<int*, sierra::nalu::MemSpace> npe("npe", 1);
  Kokkos::parallel_for(
    "get_nodes_per_element", DeviceRangePolicy(0, 1),
    KOKKOS_LAMBDA(const int i) { npe(i) = me->nodesPerElement_; });
  Kokkos::View<int*, sierra::nalu::MemSpace>::HostMirror npe_host("npe", 1);
  Kokkos::deep_copy(npe_host, npe);
  return npe_host(0);
}

template <typename DataReqType>
int
num_integration_points(const DataReqType& dataReq, const METype meType)
{
  auto* me = get_me_instance(dataReq, meType);

  // Return immediately if ME is not registered
  if (me == nullptr)
    return 0;

  Kokkos::View<int*, sierra::nalu::MemSpace> nips("nips", 1);
  Kokkos::parallel_for(
    "get_num_integration_points", DeviceRangePolicy(0, 1),
    KOKKOS_LAMBDA(const int i) { nips(i) = me->num_integration_points(); });
  Kokkos::View<int*, sierra::nalu::MemSpace>::HostMirror nips_host("nips", 1);
  Kokkos::deep_copy(nips_host, nips);
  return nips_host(0);
}

template <typename DataReqType>
int
nodes_per_entity(const DataReqType& dataReq)
{
  auto* meFC = dataReq.get_cvfem_face_me();
  auto* meSCS = dataReq.get_cvfem_surface_me();
  auto* meSCV = dataReq.get_cvfem_volume_me();
  auto* meFEM = dataReq.get_fem_volume_me();

  int npe =
    (meSCS != nullptr)
      ? nodes_per_entity(dataReq, METype::SCS)
      : (meSCV != nullptr)
          ? nodes_per_entity(dataReq, METype::SCV)
          : (meFEM != nullptr)
              ? nodes_per_entity(dataReq, METype::FEM)
              : (meFC != nullptr) ? nodes_per_entity(dataReq, METype::FACE) : 0;

  return npe;
}

} // namespace nalu
} // namespace sierra

#endif /* NGPMEUTILS_H */
