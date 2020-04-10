#ifndef EQUATION_UPDATE_H
#define EQUATION_UPDATE_H

#include "matrix_free/PolynomialOrders.h"
#include "Kokkos_Array.hpp"

#include "stk_mesh/base/Ngp.hpp"
#include "stk_mesh/base/Part.hpp"

#include <memory>

namespace sierra {
namespace nalu {
namespace matrix_free {

class EquationUpdate
{
public:
  virtual ~EquationUpdate() = default;
  virtual void initialize() = 0;
  virtual void swap_states() = 0;
  virtual void predict_state() = 0;
  virtual void compute_preconditioner(double scaling = -1) = 0;
  virtual void compute_update(
    Kokkos::Array<double, 3> gammas, stk::mesh::NgpField<double>& delta) = 0;
  virtual void update_solution_fields() = 0;
  virtual double provide_norm() const = 0;
  virtual double provide_scaled_norm() const = 0;
  virtual void banner(std::string, std::ostream&) const = 0;
};

inline bool
part_is_valid_for_matrix_free(int order, const stk::mesh::Part& part)
{
  if (
    part.topology() == stk::topology::HEX_8 ||
    part.topology() == stk::topology::QUAD_4) {
    return order == 1;
  }

  if (
    part.topology() == stk::topology::HEX_27 ||
    part.topology() == stk::topology::QUAD_9) {
    return order == 2;
  }

  if (part.topology().is_superelement()) {
    return order == floor(std::cbrt(part.topology().num_nodes() + 1) - 1);
  }

  if (part.topology().is_superface()) {
    return order == floor(std::sqrt(part.topology().num_nodes() + 1) - 1);
  }

  for (const auto* subpart : part.subsets()) {
    if (subpart == nullptr) {
      return false;
    }
    return part_is_valid_for_matrix_free(order, *subpart);
  }
  return false;
}

template <template <int> class PhysicsUpdate, typename... Args>
std::unique_ptr<EquationUpdate>
make_equation_update(int p, Args&&... args)
{
  switch (p) {
  case inst::P2:
    return std::unique_ptr<PhysicsUpdate<inst::P2>>(
      new PhysicsUpdate<inst::P2>(std::forward<Args>(args)...));
  case inst::P3:
    return std::unique_ptr<PhysicsUpdate<inst::P3>>(
      new PhysicsUpdate<inst::P3>(std::forward<Args>(args)...));
  case inst::P4:
    return std::unique_ptr<PhysicsUpdate<inst::P4>>(
      new PhysicsUpdate<inst::P4>(std::forward<Args>(args)...));
  default:
    return std::unique_ptr<PhysicsUpdate<inst::P1>>(
      new PhysicsUpdate<inst::P1>(std::forward<Args>(args)...));
  }
}

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
#endif
