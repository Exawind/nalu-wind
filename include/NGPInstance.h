// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef NGPINSTANCE_H
#define NGPINSTANCE_H

#include "KokkosInterface.h"

#include <type_traits>

namespace sierra {
namespace nalu {

namespace nalu_ngp {

template <class T>
inline T*
create()
{
  const std::string debuggingName(typeid(T).name());
  T* obj = kokkos_malloc_on_device<T>(debuggingName);

  Kokkos::parallel_for(
    debuggingName, DeviceRangePolicy(0, 1),
    KOKKOS_LAMBDA(const int) { new (obj) T(); });
  return obj;
}

template <class T>
inline T*
create(const T& hostObj)
{
  const std::string debuggingName(typeid(T).name());
  T* obj = kokkos_malloc_on_device<T>(debuggingName);

  // Create local copy for capture on device
  const T hostCopy(hostObj);
  Kokkos::parallel_for(
    debuggingName, DeviceRangePolicy(0, 1),
    KOKKOS_LAMBDA(const int) { new (obj) T(hostCopy); });
  return obj;
}

template <class T, class... Args>
inline T*
create(Args&&... args)
{
  const std::string debuggingName(typeid(T).name());
  T* obj = kokkos_malloc_on_device<T>(debuggingName);

  // CUDA lambda cannot capture packed parameter
  const T hostObj(std::forward<Args>(args)...);
  Kokkos::parallel_for(
    debuggingName, DeviceRangePolicy(0, 1),
    KOKKOS_LAMBDA(const int) { new (obj) T(hostObj); });
  return obj;
}

template <typename T>
inline void
destroy(T* obj)
{
  // Return immediately if object is a null pointer
  if (obj == nullptr)
    return;

  const std::string debuggingName(typeid(T).name());
  Kokkos::parallel_for(
    debuggingName, DeviceRangePolicy(0, 1),
    KOKKOS_LAMBDA(const int) { obj->~T(); });
  kokkos_free_on_device(obj);
}

/** Wrapper object to hold device pointers within a Kokkos::View
 *
 *  The struct does not own the pointer and will not perform any cleanup within
 *  its destructor.
 */
template <typename T>
struct NGPCopyHolder
{
  KOKKOS_DEFAULTED_FUNCTION
  NGPCopyHolder() = default;

  KOKKOS_DEFAULTED_FUNCTION
  ~NGPCopyHolder() = default;

  NGPCopyHolder(T* instance) : deviceInstance_(instance) {}

  KOKKOS_FUNCTION
  operator T*() const { return deviceInstance_; }

private:
  T* deviceInstance_{nullptr};
};

/** Create a Kokkos::View of instances that can be copied over to the device
 *
 *  Given a vector of host pointers to instances, this function will create the
 *  associated device instance, wrap it in NGPCopyHolder instance and return a
 *  Kokkos::View of the wrapped objects that is safe to be transferred to the
 *  device.
 */
template <typename T, typename Container>
Kokkos::View<NGPCopyHolder<T>*, Kokkos::LayoutRight, MemSpace>
create_ngp_view(const Container& hostVec)
{
  using NGPInfo = NGPCopyHolder<T>;
  using NGPInfoView = Kokkos::View<NGPInfo*, Kokkos::LayoutRight, MemSpace>;

  const size_t numObjects = hostVec.size();
  const std::string clsName(typeid(T).name());
  const std::string debuggingName = "NGP" + clsName + "View";
  NGPInfoView ngpVec(debuggingName, numObjects);

  typename NGPInfoView::HostMirror hostNgpView =
    Kokkos::create_mirror_view(ngpVec);

  for (size_t i = 0; i < numObjects; ++i)
    hostNgpView(i) = NGPInfo(hostVec[i]->create_on_device());

  Kokkos::deep_copy(ngpVec, hostNgpView);

  return ngpVec;
}

} // namespace nalu_ngp

} // namespace nalu
} // namespace sierra

#endif /* NGPINSTANCE_H */
