// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef ACTUATORTYPES_H_
#define ACTUATORTYPES_H_

#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>

namespace sierra {
namespace nalu {

#ifndef ACTUATOR_LAMBDA
#define ACTUATOR_LAMBDA [=]
#endif

#ifdef KOKKOS_ENABLE_CUDA
using ActuatorMemSpace = Kokkos::CudaSpace;
using ActuatorExecutionSpace = Kokkos::DefaultExecutionSpace;
#else
using ActuatorMemSpace = Kokkos::HostSpace;
using ActuatorExecutionSpace = Kokkos::DefaultHostExecutionSpace;
#endif
using ActuatorMemLayout = Kokkos::LayoutRight;
using ActuatorFixedMemSpace = Kokkos::HostSpace;
using ActuatorFixedMemLayout = Kokkos::LayoutRight;
using ActuatorFixedExecutionSpace = Kokkos::DefaultHostExecutionSpace;

// DUAL VIEWS
using ActScalarIntDv =
  Kokkos::DualView<int*, ActuatorMemLayout, ActuatorMemSpace>;
using ActScalarU64Dv =
  Kokkos::DualView<uint64_t*, ActuatorMemLayout, ActuatorMemSpace>;
using ActScalarDblDv =
  Kokkos::DualView<double*, ActuatorMemLayout, ActuatorMemSpace>;
using ActVectorDblDv =
  Kokkos::DualView<double* [3], ActuatorMemLayout, ActuatorMemSpace>;
using ActTensorDblDv =
  Kokkos::DualView<double* [9], ActuatorMemLayout, ActuatorMemSpace>;
using Act2DArrayDblDv =
  Kokkos::DualView<double**, ActuatorMemLayout, ActuatorMemSpace>;

// VIEWS
using ActScalarInt = Kokkos::View<int*, ActuatorMemLayout, ActuatorMemSpace>;
using ActScalarU64 =
  Kokkos::View<uint64_t*, ActuatorMemLayout, ActuatorMemSpace>;
using ActScalarDbl = Kokkos::View<double*, ActuatorMemLayout, ActuatorMemSpace>;
using ActVectorDbl =
  Kokkos::View<double* [3], ActuatorMemLayout, ActuatorMemSpace>;
using ActTensorDbl =
  Kokkos::View<double* [9], ActuatorMemLayout, ActuatorMemSpace>;
using Act2DArrayDbl =
  Kokkos::View<double**, ActuatorMemLayout, ActuatorMemSpace>;

// VIEWS FIXED TO HOST
using ActFixRangePolicy = Kokkos::RangePolicy<ActuatorFixedExecutionSpace>;
using ActFixScalarInt =
  Kokkos::View<int*, ActuatorFixedMemLayout, ActuatorFixedMemSpace>;
using ActFixScalarDbl =
  Kokkos::View<double*, ActuatorFixedMemLayout, ActuatorFixedMemSpace>;
using ActFixVectorDbl =
  Kokkos::View<double* [3], ActuatorFixedMemLayout, ActuatorFixedMemSpace>;
using ActFixElemIds =
  Kokkos::View<uint64_t*, ActuatorFixedMemLayout, ActuatorFixedMemSpace>;
using ActFixScalarBool =
  Kokkos::View<bool*, ActuatorFixedMemLayout, ActuatorFixedMemSpace>;
using ActFixArrayInt =
  Kokkos::View<int**, ActuatorFixedMemLayout, ActuatorFixedMemSpace>;
using ActFixTensorDbl =
  Kokkos::View<double* [9], ActuatorFixedMemLayout, ActuatorFixedMemSpace>;

template <typename memory_space>
struct ActDualViewHelper
{
  template <typename T>
  inline auto get_local_view(T dualView) const
    -> decltype(dualView.template view<memory_space>())
  {
    dualView.template sync<memory_space>();
    return dualView.template view<memory_space>();
  }

  template <typename T>
  inline void touch_dual_view(T dualView)
  {
    dualView.template sync<memory_space>();
    dualView.template modify<memory_space>();
  }

  template <typename T>
  inline void sync(T dualView)
  {
    dualView.template sync<memory_space>();
  }
  // TODO create view in this space
  // TODO create range policy
};

} // namespace nalu
} // namespace sierra

#endif // INCLUDE_ACTUATOR_ACTUATORTYPES_H_
