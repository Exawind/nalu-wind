#ifndef LINEAR_VOLUME_H
#define LINEAR_VOLUME_H

#include "matrix_free/PolynomialOrders.h"
#include "matrix_free/KokkosFramework.h"
#include "matrix_free/LocalArray.h"

namespace sierra {
namespace nalu {
namespace matrix_free {
namespace geom {

namespace impl {

template <int p>
struct volume_metric_t
{
  static scalar_view<p> invoke(
    const const_scalar_view<p> alpha, const const_vector_view<p> coordinates);
};

} // namespace impl
P_INVOKEABLE(volume_metric)

} // namespace geom
} // namespace matrix_free
} // namespace nalu
} // namespace sierra

#endif
