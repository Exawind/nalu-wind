#ifndef CREATEDEVICEEXPRESSION_H
#define CREATEDEVICEEXPRESSION_H

#include <type_traits>

#include <KokkosInterface.h>

namespace sierra {
namespace nalu {

template <typename T>
inline 
T* create_device_expression(const T & rhs)
{
  const std::string debuggingName(typeid(T).name());
  T* t = kokkos_malloc_on_device<T>(debuggingName);
  // Bring rhs into local scope for capture to device.
  const T RHS(rhs);
  kokkos_parallel_for(debuggingName, 1, [&] (const int /* i */) {
    new (t) T(RHS); 
  });
  return t;
}

template <typename T>
inline 
T* create_device_expression()
{
  const std::string debuggingName(typeid(T).name());
  T* t = kokkos_malloc_on_device<T>(debuggingName);
  kokkos_parallel_for(debuggingName, 1, [&] (const int /* i */) {
    new (t) T(); 
  });
  return t;
}
} // namespace nalu
} // namespace sierra

#endif /* CREATEDEVICEEXPRESSION_H */
