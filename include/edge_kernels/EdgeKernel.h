// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#ifndef EDGEKERNEL_H
#define EDGEKERNEL_H

/** \file
 *  \brief Kernel-style implementation of Edge algorithms
 */

#include "KokkosInterface.h"
#include "NGPInstance.h"
#include "ElemDataRequests.h"
#include "ElemDataRequestsGPU.h"
#include "ScratchViews.h"
#include "SharedMemData.h"

#include "stk_mesh/base/Ngp.hpp"
#include "stk_mesh/base/Entity.hpp"
#include "stk_mesh/base/Types.hpp"

namespace sierra {
namespace nalu {

class Realm;

/** Traits for use with Edge Algorithms/Kernels
 */
struct EdgeKernelTraits
{
  static constexpr int NDimMax = 3;
  using DblType = double;
  using ShmemType = DeviceShmem;
  using ShmemDataType = SharedMemData_Edge<DeviceTeamHandleType, ShmemType>;
  using RhsType = SharedMemView<DblType*, ShmemType>;
  using LhsType = SharedMemView<DblType**, ShmemType>;
};

class EdgeKernel
{
public:
  KOKKOS_DEFAULTED_FUNCTION
  EdgeKernel() = default;

  KOKKOS_DEFAULTED_FUNCTION
  virtual ~EdgeKernel() = default;

  virtual EdgeKernel* create_on_device() = 0;

  virtual void free_on_device() = 0;

  virtual void setup(Realm&) = 0;

  KOKKOS_FUNCTION
  virtual void execute(
    EdgeKernelTraits::ShmemDataType&,
    const stk::mesh::FastMeshIndex&,
    const stk::mesh::FastMeshIndex&,
    const stk::mesh::FastMeshIndex&) = 0;
};

template <typename T>
class NGPEdgeKernel : public EdgeKernel
{
public:
  KOKKOS_DEFAULTED_FUNCTION
  NGPEdgeKernel() = default;

  KOKKOS_DEFAULTED_FUNCTION
  virtual ~NGPEdgeKernel() = default;

  virtual EdgeKernel* create_on_device() final
  {
    free_on_device();
    deviceCopy_ = nalu_ngp::create<T>(*dynamic_cast<T*>(this));
    return deviceCopy_;
  }

  virtual void free_on_device() final
  {
    if (deviceCopy_ != nullptr) {
      nalu_ngp::destroy<T>(dynamic_cast<T*>(deviceCopy_));
      deviceCopy_ = nullptr;
    }
  }

  T* device_copy() const { return deviceCopy_; }

private:
  T* deviceCopy_{nullptr};
};

} // namespace nalu
} // namespace sierra

#endif /* EDGEKERNEL_H */
