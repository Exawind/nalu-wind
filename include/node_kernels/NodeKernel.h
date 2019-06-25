/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef NODEKERNEL_H
#define NODEKERNEL_H

#include "KokkosInterface.h"
#include "NGPInstance.h"

#include "stk_ngp/Ngp.hpp"
#include "stk_mesh/base/Entity.hpp"

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
  KOKKOS_FORCEINLINE_FUNCTION
  NodeKernel() = default;

  KOKKOS_FUNCTION
  virtual ~NodeKernel() {}

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
  KOKKOS_FORCEINLINE_FUNCTION
  NGPNodeKernel() = default;

  KOKKOS_FUNCTION
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
