#include "matrix_free/FilterJacobi.h"
#include "matrix_free/StkSimdConnectivityMap.h"
#include "matrix_free/StkToTpetraMap.h"
#include "matrix_free/StkSimdGatheredElementData.h"
#include "matrix_free/MakeRCP.h"

#include "matrix_free/LinearVolume.h"

#include "matrix_free/StkGradientFixture.h"

#include "math.h"
#include "stdlib.h"

#include "Kokkos_Array.hpp"
#include "Kokkos_View.hpp"
#include "Teuchos_ArrayView.hpp"
#include "Teuchos_DefaultMpiComm.hpp"
#include "Teuchos_OrdinalTraits.hpp"
#include "Teuchos_Ptr.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_RCPDecl.hpp"
#include "Tpetra_ConfigDefs.hpp"
#include "Tpetra_Map_decl.hpp"
#include "Tpetra_MultiVector_decl.hpp"
#include <algorithm>
#include <random>
#include "stk_mesh/base/Bucket.hpp"
#include "stk_mesh/base/FieldState.hpp"
#include "stk_mesh/base/Types.hpp"
#include "stk_util/parallel/Parallel.hpp"

#include "gtest/gtest.h"
#include "mpi.h"
#include "stk_mesh/base/BulkData.hpp"
#include "stk_mesh/base/Field.hpp"
#include "stk_mesh/base/FieldBase.hpp"
#include "stk_mesh/base/MetaData.hpp"
#include "stk_topology/topology.hpp"
#include "stk_mesh/base/GetNgpField.hpp"

namespace sierra {
namespace nalu {
namespace matrix_free {

class FilterJacobiFixture : public GradientFixture
{
protected:
  static constexpr int nodes_per_elem = (order + 1) * (order + 1) * (order + 1);
  static constexpr int nx = 3;
  static constexpr double scale = nx;

  FilterJacobiFixture()
    : GradientFixture(nx, scale),
      owned_map(make_owned_row_map(mesh(), meta.universal_part())),
      owned_and_shared_map(make_owned_and_shared_row_map(
        mesh(), meta.universal_part(), gid_field_ngp)),
      exporter(
        Teuchos::rcpFromRef(owned_and_shared_map),
        Teuchos::rcpFromRef(owned_map)),
      owned_lhs(Teuchos::rcpFromRef(owned_map), 1),
      owned_rhs(Teuchos::rcpFromRef(owned_map), 1),
      owned_and_shared_lhs(Teuchos::rcpFromRef(owned_and_shared_map), 1),
      owned_and_shared_rhs(Teuchos::rcpFromRef(owned_and_shared_map), 1),
      elid(make_stk_lid_to_tpetra_lid_map(
        mesh(),
        meta.universal_part(),
        gid_field_ngp,
        owned_and_shared_map.getLocalMap())),
      conn(stk_connectivity_map<order>(mesh(), meta.universal_part())),
      offsets(create_offset_map<order>(mesh(), active(), elid))
  {
  }

  const Tpetra::Map<> owned_map;
  const Tpetra::Map<> owned_and_shared_map;
  const Tpetra::Export<> exporter;
  Tpetra::MultiVector<> owned_lhs;
  Tpetra::MultiVector<> owned_rhs;
  Tpetra::MultiVector<> owned_and_shared_lhs;
  Tpetra::MultiVector<> owned_and_shared_rhs;

  const const_entity_row_view_type elid;
  elem_mesh_index_view<order> conn;
  elem_offset_view<order> offsets;
};

TEST_F(FilterJacobiFixture, jacobi_operator_is_stricly_positive_for_mass)
{
  vector_view<order> elem_coords("elem_coords", conn.extent(0));
  auto coord_ngp = stk::mesh::get_updated_ngp_field<double>(coordinate_field());
  stk_simd_vector_field_gather<order>(conn, coord_ngp, elem_coords);
  const auto vols = geom::volume_metric<order>(elem_coords);

  FilterJacobiOperator<order> prec_op(offsets, exporter);
  prec_op.compute_diagonal(vols);

  auto& result = prec_op.get_inverse_diagonal();
  result.sync_host();
  auto view_h = result.getLocalViewHost();
  for (size_t k = 0u; k < result.getLocalLength(); ++k) {
    ASSERT_TRUE(std::isfinite(stk::simd::get_data(view_h(k, 0), 0)));
    ASSERT_GT(view_h(k, 0), 1.0e-2);
  }
}
} // namespace matrix_free
} // namespace nalu
} // namespace sierra