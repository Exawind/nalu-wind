// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef POLYNOMIAL_ORDER_H
#define POLYNOMIAL_ORDER_H

#include <type_traits>

#ifndef NALU_POLYNOMIAL_ORDER1
#define NALU_POLYNOMIAL_ORDER1 1
#endif

#ifndef NALU_POLYNOMIAL_ORDER2
#define NALU_POLYNOMIAL_ORDER2 2
#endif

#ifndef NALU_POLYNOMIAL_ORDER3
#define NALU_POLYNOMIAL_ORDER3 3
#endif

#ifndef NALU_POLYNOMIAL_ORDER4
#define NALU_POLYNOMIAL_ORDER4 4
#endif

namespace sierra {
namespace nalu {
namespace matrix_free {
namespace inst {
enum {
  P1 = NALU_POLYNOMIAL_ORDER1,
  P2 = NALU_POLYNOMIAL_ORDER2,
  P3 = NALU_POLYNOMIAL_ORDER3,
  P4 = NALU_POLYNOMIAL_ORDER4
};
}

#define INSTANTIATE_TYPE(type, Name)                                           \
  template type Name<inst::P1>;                                                \
  template type Name<inst::P2>;                                                \
  template type Name<inst::P3>;                                                \
  template type Name<inst::P4>

#define INSTANTIATE_POLYCLASS(ClassName) INSTANTIATE_TYPE(class, ClassName)
#define INSTANTIATE_POLYSTRUCT(ClassName) INSTANTIATE_TYPE(struct, ClassName)

#define P_INVOKEABLE(func)                                                     \
  template <int p, typename... Args>                                           \
  auto func(Args&&... args)                                                    \
    ->decltype(impl::func##_t<p>::invoke(std::forward<Args>(args)...))         \
  {                                                                            \
    return impl::func##_t<p>::invoke(std::forward<Args>(args)...);             \
  }

} // namespace matrix_free
} // namespace nalu
} // namespace sierra

#endif
