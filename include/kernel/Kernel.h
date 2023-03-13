// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef KERNEL_H
#define KERNEL_H

#include "KokkosInterface.h"
#include "SimdInterface.h"
#include "ScratchViews.h"
#include "AlgTraits.h"
#include "NGPInstance.h"

#include <Kokkos_Core.hpp>

#include <stk_mesh/base/Entity.hpp>

#include <array>

#include <master_element/MasterElementRepo.h>

namespace sierra {
namespace nalu {

class TimeIntegrator;
class SolutionOptions;

/** Base class for computational kernels in Nalu
 *
 * A kernel represents an atomic unit of computation applied over a given set of
 * nodes, elements, etc. using STK and Kokkos looping constructs.
 */
class Kernel
{
public:
  KOKKOS_DEFAULTED_FUNCTION
  Kernel() = default;

  virtual ~Kernel() = default;

  virtual Kernel* create_on_device() { return this; }

  virtual void free_on_device() {}

  /** Perform pre-timestep work for the computational kernel
   */
  virtual void setup(const TimeIntegrator&) {}

  /** Execute the kernel within a Kokkos loop and populate the LHS and RHS for
   *  the linear solve
   */
  virtual void execute(
    SharedMemView<DoubleType**>& /* lhs */,
    SharedMemView<DoubleType*>& /* rhs */,
    ScratchViews<DoubleType>& /* scratchViews */)
  {
  }

  /** Special execute for face-element kernels
   *
   */
  virtual void execute(
    SharedMemView<DoubleType**>& /* lhs */,
    SharedMemView<DoubleType*>& /* rhs */,
    ScratchViews<DoubleType>& /* faceScratchViews */,
    ScratchViews<DoubleType>& /* elemScratchViews */,
    int /* elemFaceOrdinal */)
  {
  }

  /** Execute for NGP Kernels
   *
   */
  KOKKOS_FUNCTION
  virtual void execute(
    SharedMemView<DoubleType**, DeviceShmem>&,
    SharedMemView<DoubleType*, DeviceShmem>&,
    ScratchViews<double, DeviceTeamHandleType, DeviceShmem>&)
  {
  }

#if defined(KOKKOS_ENABLE_GPU)
  KOKKOS_FUNCTION
  virtual void execute(
    SharedMemView<DoubleType**, DeviceShmem>&,
    SharedMemView<DoubleType*, DeviceShmem>&,
    ScratchViews<DoubleType, DeviceTeamHandleType, DeviceShmem>&)
  {
  }

  KOKKOS_FUNCTION
  virtual void execute(
    SharedMemView<DoubleType**, DeviceShmem>& /* lhs */,
    SharedMemView<DoubleType*, DeviceShmem>& /* rhs */,
    ScratchViews<
      DoubleType,
      DeviceTeamHandleType,
      DeviceShmem>& /* faceScratchViews */,
    ScratchViews<
      DoubleType,
      DeviceTeamHandleType,
      DeviceShmem>& /* elemScratchViews */,
    int /* elemFaceOrdinal */)
  {
  }
#endif
};

/** Kernel that can be transferred to a device
 *
 */
template <typename T>
class NGPKernel : public Kernel
{
public:
  KOKKOS_DEFAULTED_FUNCTION
  NGPKernel() = default;

  // Implementation note
  //
  // The destructor does not free the deviceCopy_ instance. This is done to
  // eliminate the warnings issued when compiling with nvcc for GPU builds.
  // Instead the `deviceCopy_` is freed by explicitly calling `free_on_device`
  // from sierra::nalu::Algorithm::~Algorithm() before freeing the host pointers
  // stored in `activeKernels_`
  KOKKOS_DEFAULTED_FUNCTION virtual ~NGPKernel() = default;

  virtual Kernel* create_on_device() final
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

protected:
  T* deviceCopy_{nullptr};
};

} // namespace nalu
} // namespace sierra

#endif /* KERNEL_H */
