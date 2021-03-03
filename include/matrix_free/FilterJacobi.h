#ifndef FILTER_JACOBI_H
#define FILTER_JACOBI_H

#include "matrix_free/KokkosViewTypes.h"

#include "Teuchos_BLAS_types.hpp"
#include "Teuchos_RCP.hpp"
#include "Tpetra_Export.hpp"
#include "Tpetra_MultiVector.hpp"
#include "Tpetra_Operator.hpp"
#include "Tpetra_Map.hpp"

namespace sierra {
namespace nalu {
namespace matrix_free {

template <int p>
class FilterJacobiOperator final : public Tpetra::Operator<>
{
public:
  static constexpr int num_vectors = 3;
  using mv_type = Tpetra::MultiVector<>;
  using map_type = Tpetra::Map<>;
  using base_operator_type = Tpetra::Operator<>;
  using export_type = Tpetra::Export<>;

  FilterJacobiOperator(
    const_elem_offset_view<p> elem_offsets_in,
    const export_type& exporter,
    int num_sweeps = 1);

  void apply(
    const mv_type& ownedSolution,
    mv_type& ownedRHS,
    Teuchos::ETransp trans = Teuchos::NO_TRANS,
    double alpha = 1.0,
    double beta = 0.0) const final;

  void compute_diagonal(const_scalar_view<p> vols_in);
  mv_type& get_inverse_diagonal() { return owned_diagonal; }
  void set_linear_operator(Teuchos::RCP<const Tpetra::Operator<>>);

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
  mv_type shared_diagonal;
  mutable mv_type cached_mv;

  Teuchos::RCP<const base_operator_type> op;
};

} // namespace matrix_free
} // namespace nalu
} // namespace sierra

#endif
