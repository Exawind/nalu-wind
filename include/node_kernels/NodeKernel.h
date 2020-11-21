// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#ifndef NODEKERNEL_H
#define NODEKERNEL_H

#include "KokkosInterface.h"
#include "NGPInstance.h"

#include "stk_mesh/base/Ngp.hpp"
#include "stk_mesh/base/Entity.hpp"
#include "stk_mesh/base/Types.hpp"

namespace sierra {
namespace nalu {

class Realm;

struct NodeKernelTraits
{
  static constexpr int NDimMax = 3;
  using DblType = double;
  using ShmemType = DeviceShmem;
  using RhsType = SharedMemView<DblType*, ShmemType>;
  using LhsType = SharedMemView<DblType**, ShmemType>;
};

class NodeKernel
{
public:
  KOKKOS_DEFAULTED_FUNCTION
  NodeKernel() = default;

  KOKKOS_DEFAULTED_FUNCTION
  virtual ~NodeKernel() = default;

  virtual NodeKernel* create_on_device() = 0;

  virtual void free_on_device() = 0;

  virtual void setup(Realm&) = 0;

  KOKKOS_FUNCTION
  virtual void execute(
    NodeKernelTraits::LhsType&,
    NodeKernelTraits::RhsType&,
    const stk::mesh::FastMeshIndex&) = 0;
};

template<typename T>
class NGPNodeKernel : public NodeKernel
{
public:
  KOKKOS_DEFAULTED_FUNCTION
  NGPNodeKernel() = default;

  KOKKOS_DEFAULTED_FUNCTION
  virtual ~NGPNodeKernel() = default;

  virtual NodeKernel* create_on_device() final
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

}  // nalu
}  // sierra


#endif /* NODEKERNEL_H */
