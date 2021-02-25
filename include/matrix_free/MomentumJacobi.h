#ifndef Momentum_JACOBI_H
#define Momentum_JACOBI_H

#include "matrix_free/KokkosViewTypes.h"

#include "Kokkos_Array.hpp"
#include "Teuchos_BLAS_types.hpp"

#include "Teuchos_RCP.hpp"
#include "Tpetra_Export.hpp"
#include "Tpetra_MultiVector.hpp"
#include "Tpetra_Operator.hpp"

namespace sierra {
namespace nalu {
namespace matrix_free {

template <int p>
class MomentumJacobiOperator final : public Tpetra::Operator<>
{
public:
  static constexpr int num_vectors = 3;
  using mv_type = Tpetra::MultiVector<>;
  using map_type = Tpetra::Map<>;
  using base_operator_type = Tpetra::Operator<>;
  using export_type = Tpetra::Export<>;

  MomentumJacobiOperator(
    const_elem_offset_view<p> elem_offsets_in,
    const export_type& exporter,
    int num_sweeps = 1);

  void apply(
    const mv_type& ownedSolution,
    mv_type& ownedRHS,
    Teuchos::ETransp trans = Teuchos::NO_TRANS,
    double alpha = 1.0,
    double beta = 0.0) const final;

  void compute_diagonal(
    double gamma,
    const_scalar_view<p> vol,
    const_scs_scalar_view<p> adv,
    const_scs_vector_view<p> diff);
  mv_type& get_inverse_diagonal() { return owned_diagonal; }
  void set_linear_operator(Teuchos::RCP<const Tpetra::Operator<>>);

  void set_dirichlet_nodes(const_node_offset_view dirichlet)
  {
    dirichlet_bc_active_ = dirichlet.extent(0) > 0;
    dirichlet_bc_offsets_ = dirichlet;
  }

  Teuchos::RCP<const map_type> getDomainMap() const final
  {
    return exporter.getTargetMap();
  }
  Teuchos::RCP<const map_type> getRangeMap() const final
  {
    return exporter.getTargetMap();
  }

private:
  const const_elem_offset_view<p> elem_offsets;
  const export_type& exporter;
  const int num_sweeps;
  mv_type owned_diagonal;
  mv_type owned_and_shared_diagonal;
  mutable mv_type cached_mv;

  bool dirichlet_bc_active_{false};
  const_node_offset_view dirichlet_bc_offsets_;

  Teuchos::RCP<const base_operator_type> op;
};

} // namespace matrix_free
} // namespace nalu
} // namespace sierra

#endif
