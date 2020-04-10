#ifndef LINEAR_DIFFUSION_METRIC_H
#define LINEAR_DIFFUSION_METRIC_H

#include "matrix_free/PolynomialOrders.h"
#include "matrix_free/KokkosFramework.h"
#include "matrix_free/LocalArray.h"

namespace sierra {
namespace nalu {
namespace matrix_free {
namespace geom {

namespace impl {
template <int p>
struct diffusion_metric_t
{
  static scs_vector_view<p> invoke(
    const const_scalar_view<p> alpha, const const_vector_view<p> coordinates);
};
} // namespace impl
P_INVOKEABLE(diffusion_metric)

} // namespace geom
} // namespace matrix_free
} // namespace nalu
} // namespace sierra

#endif
