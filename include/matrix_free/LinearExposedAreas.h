#ifndef LINEAR_EXPOSED_AREA_H
#define LINEAR_EXPOSED_AREA_H

#include "matrix_free/PolynomialOrders.h"
#include "matrix_free/KokkosFramework.h"
#include "matrix_free/LocalArray.h"

namespace sierra {
namespace nalu {
namespace matrix_free {
namespace geom {

namespace impl {

template <int p>
struct exposed_areas_t
{
  static face_vector_view<p>
  invoke(const const_face_vector_view<p> coordinates);
};

} // namespace impl
P_INVOKEABLE(exposed_areas)

} // namespace geom
} // namespace matrix_free
} // namespace nalu
} // namespace sierra

#endif
