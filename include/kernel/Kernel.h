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

#include <stk_mesh/base/Entity.hpp>

#include <array>

#include <master_element/MasterElementFactory.h>


namespace sierra {
namespace nalu {

class TimeIntegrator;
class SolutionOptions;

template<typename AlgTraits, typename LambdaFunction, typename ViewType>
void get_scv_shape_fn_data(LambdaFunction lambdaFunction, ViewType& shape_fn_view)
{
  static_assert(ViewType::Rank == 2u, "2D View");
  ThrowRequireMsg(shape_fn_view.extent_int(0) == AlgTraits::numScvIp_, "Inconsistent number of scv ips");
  ThrowRequireMsg(shape_fn_view.extent_int(1) == AlgTraits::nodesPerElement_, "Inconsistent number of of nodes");

  double tmp_data[AlgTraits::numScvIp_*AlgTraits::nodesPerElement_];
  lambdaFunction(tmp_data);

  DoubleType* data = &shape_fn_view(0,0);
  for(int i=0; i<AlgTraits::numScvIp_*AlgTraits::nodesPerElement_; ++i) {
    data[i] = tmp_data[i];
  }
}

template<typename AlgTraits, typename LambdaFunction, typename ViewType>
void get_scs_shape_fn_data(LambdaFunction lambdaFunction, ViewType& shape_fn_view)
{
  static_assert(ViewType::Rank == 2u, "2D View");
  ThrowRequireMsg(shape_fn_view.extent_int(0) == AlgTraits::numScsIp_, "Inconsistent number of scs ips");
  ThrowRequireMsg(shape_fn_view.extent_int(1) == AlgTraits::nodesPerElement_, "Inconsistent number of of nodes");

  double tmp_data[AlgTraits::numScsIp_*AlgTraits::nodesPerElement_];
  lambdaFunction(tmp_data);

  DoubleType* data = &shape_fn_view(0,0);
  for(int i=0; i<AlgTraits::numScsIp_*AlgTraits::nodesPerElement_; ++i) {
    data[i] = tmp_data[i];
  }
}

template<typename AlgTraits, typename LambdaFunction, typename ViewType>
void get_fem_shape_fn_data(LambdaFunction lambdaFunction, ViewType& shape_fn_view)
{
  static_assert(ViewType::Rank == 2u, "2D View");
  ThrowRequireMsg(shape_fn_view.extent_int(0) == AlgTraits::numGp_, "Inconsistent number of Gauss points");
  ThrowRequireMsg(shape_fn_view.extent_int(1) == AlgTraits::nodesPerElement_, "Inconsistent number of of nodes");

  double tmp_data[AlgTraits::numGp_*AlgTraits::nodesPerElement_];
  lambdaFunction(tmp_data);

  DoubleType* data = &shape_fn_view(0,0);
  for(int i=0; i<AlgTraits::numGp_*AlgTraits::nodesPerElement_; ++i) {
    data[i] = tmp_data[i];
  }
}

template<typename BcAlgTraits, typename LambdaFunction, typename ViewType>
void get_face_shape_fn_data(LambdaFunction lambdaFunction, ViewType& shape_fn_view)
{
  static_assert(ViewType::Rank == 2u, "2D View");
  ThrowRequireMsg(shape_fn_view.extent_int(0) == BcAlgTraits::numFaceIp_, "Inconsistent number of face ips");
  ThrowRequireMsg(shape_fn_view.extent_int(1) == BcAlgTraits::nodesPerFace_, "Inconsistent number of of nodes");

  double tmp_data[BcAlgTraits::numFaceIp_*BcAlgTraits::nodesPerFace_];
  lambdaFunction(tmp_data);

  DoubleType* data = &shape_fn_view(0,0);
  for(int i=0; i<BcAlgTraits::numFaceIp_*BcAlgTraits::nodesPerFace_; ++i) {
    data[i] = tmp_data[i];
  }
}

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
    SharedMemView<DoubleType**> & /* lhs */,
    SharedMemView<DoubleType*> & /* rhs */,
    ScratchViews<DoubleType> & /* scratchViews */)
  {}


  /** Special execute for face-element kernels
   *
   */
  virtual void execute(
    SharedMemView<DoubleType**> & /* lhs */,
    SharedMemView<DoubleType*> & /* rhs */,
    ScratchViews<DoubleType> & /* faceScratchViews */,
    ScratchViews<DoubleType> & /* elemScratchViews */,
    int /* elemFaceOrdinal */)
  {}

  /** Execute for NGP Kernels
   *
   */
  KOKKOS_FUNCTION
  virtual void execute(
    SharedMemView<DoubleType**, DeviceShmem>&,
    SharedMemView<DoubleType*, DeviceShmem>&,
    ScratchViews<double, DeviceTeamHandleType, DeviceShmem>&)
  {}

#ifdef KOKKOS_ENABLE_CUDA
  KOKKOS_FUNCTION
  virtual void execute(
    SharedMemView<DoubleType**, DeviceShmem>&,
    SharedMemView<DoubleType*, DeviceShmem>&,
    ScratchViews<DoubleType, DeviceTeamHandleType, DeviceShmem>&)
  {}

  KOKKOS_FUNCTION
  virtual void execute(
    SharedMemView<DoubleType**,DeviceShmem> & /* lhs */,
    SharedMemView<DoubleType*,DeviceShmem> & /* rhs */,
    ScratchViews<DoubleType, DeviceTeamHandleType, DeviceShmem> & /* faceScratchViews */,
    ScratchViews<DoubleType, DeviceTeamHandleType, DeviceShmem> & /* elemScratchViews */,
    int /* elemFaceOrdinal */)
  {}
#endif
};

/** Kernel that can be transferred to a device
 *
 */
template<typename T>
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

}  // nalu
}  // sierra

#endif /* KERNEL_H */
