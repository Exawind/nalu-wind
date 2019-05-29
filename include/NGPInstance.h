/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef NGPINSTANCE_H
#define NGPINSTANCE_H

#include "KokkosInterface.h"

#include <type_traits>

namespace sierra {
namespace nalu {

namespace nalu_ngp {

template<class T>
inline T* create()
{
  const std::string debuggingName(typeid(T).name());
  T* obj = kokkos_malloc_on_device<T>(debuggingName);

  Kokkos::parallel_for(debuggingName, 1, KOKKOS_LAMBDA(const int) {
      new (obj) T();
    });
  return obj;
}

template<class T>
inline T* create(const T& hostObj)
{
  const std::string debuggingName(typeid(T).name());
  T* obj = kokkos_malloc_on_device<T>(debuggingName);

  // Create local copy for capture on device
  const T hostCopy(hostObj);
  Kokkos::parallel_for(debuggingName, 1, KOKKOS_LAMBDA(const int) {
      new (obj) T(hostCopy);
    });
  return obj;
}

template<class T, class... Args>
inline T* create(Args&&... args)
{
  const std::string debuggingName(typeid(T).name());
  T* obj = kokkos_malloc_on_device<T>(debuggingName);

  // CUDA lambda cannot capture packed parameter
  const T hostObj(std::forward<Args>(args)...);
  Kokkos::parallel_for(debuggingName, 1, KOKKOS_LAMBDA(const int) {
      new (obj) T(hostObj);
    });
  return obj;
}

template<typename T>
inline void destroy(T* obj)
{
  // Return immediately if object is a null pointer
  if (obj == nullptr) return;

  const std::string debuggingName(typeid(T).name());
  Kokkos::parallel_for(debuggingName, 1, KOKKOS_LAMBDA(const int) {
      obj->~T();
    });
  Kokkos::kokkos_free(obj);
}

/** Wrapper object to hold device pointers within a Kokkos::View
 *
 *  The struct does not own the pointer and will not perform any cleanup within
 *  its destructor.
 */
template<typename T>
struct NGPCopyHolder
{
  KOKKOS_INLINE_FUNCTION
  NGPCopyHolder() = default;

  KOKKOS_FUNCTION
  ~NGPCopyHolder() = default;

  NGPCopyHolder(T* instance)
    : deviceInstance_(instance)
  {}

  KOKKOS_FUNCTION
  operator T*() const
  { return deviceInstance_; }

private:
  T* deviceInstance_{nullptr};
};

} // nalu_ngp

}  // nalu
}  // sierra


#endif /* NGPINSTANCE_H */
