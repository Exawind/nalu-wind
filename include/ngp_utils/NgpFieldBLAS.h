// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef NGPFIELDBLAS_H
#define NGPFIELDBLAS_H

#include "ngp_utils/NgpTypes.h"
#include "ngp_utils/NgpLoopUtils.h"
#include "ngp_utils/NgpFieldUtils.h"
#include "stk_mesh/base/NgpMesh.hpp"

namespace sierra {
namespace nalu {
namespace nalu_ngp {

/** Operation: `y = alpha * x + beta * y`
 *
 *  @param ngpMesh Instance of stk::mesh::NgpMesh
 *  @param sel Selector where the operation is applied
 */
template <typename Mesh, typename FieldType, typename ScalarType>
inline void
field_axpby(
  const Mesh& ngpMesh,
  const stk::mesh::Selector& sel,
  const ScalarType alpha,
  const FieldType& xField,
  const ScalarType beta,
  FieldType& yField,
  const unsigned numComponents = 1,
  const stk::topology::rank_t rank = stk::topology::NODE_RANK)
{
  static_assert(
    std::is_same<typename FieldType::value_type, ScalarType>::value,
    "Mismatch on field data types");

  using Traits = NGPMeshTraits<Mesh>;
  using MeshIndex = typename Traits::MeshIndex;

  yField.sync_to_device();

  nalu_ngp::run_entity_algorithm(
    "ngp_field_axpby", ngpMesh, rank, sel, KOKKOS_LAMBDA(const MeshIndex& mi) {
      for (unsigned d = 0; d < numComponents; ++d)
        yField.get(mi, d) =
          alpha * xField.get(mi, d) + beta * yField.get(mi, d);
    });

  // Indicate modification on device for future synchronization
  yField.modify_on_device();
}

template <typename Mesh, typename FieldType>
inline void
field_copy(
  const Mesh& ngpMesh,
  const stk::mesh::Selector& sel,
  FieldType& dest,
  const FieldType& src,
  const unsigned start,
  const unsigned end,
  const stk::topology::rank_t rank = stk::topology::NODE_RANK)
{
  using Traits = NGPMeshTraits<Mesh>;
  using MeshIndex = typename Traits::MeshIndex;

  nalu_ngp::run_entity_algorithm(
    "ngp_field_copy", ngpMesh, rank, sel, KOKKOS_LAMBDA(const MeshIndex& mi) {
      for (unsigned d = start; d < end; ++d)
        dest.get(mi, d) = src.get(mi, d);
    });

  // Indicate modification on device for future synchronization
  dest.modify_on_device();
}

template <typename Mesh, typename FieldType>
inline void
field_copy(
  const Mesh& ngpMesh,
  const stk::mesh::Selector& sel,
  FieldType& dest,
  const FieldType& src,
  const unsigned numComponents = 1,
  const stk::topology::rank_t rank = stk::topology::NODE_RANK)
{
  field_copy(ngpMesh, sel, dest, src, 0, numComponents, rank);
}

template <typename MeshInfoType>
inline void
field_copy(
  const MeshInfoType& meshInfo,
  stk::mesh::FieldBase& dest,
  const stk::mesh::FieldBase& src,
  const unsigned numComponents = 1,
  const stk::topology::rank_t rank = stk::topology::NODE_RANK)
{
  const stk::mesh::Selector sel = stk::mesh::selectField(dest);
  const auto& fieldMgr = meshInfo.ngp_field_manager();
  const auto& srcField =
    fieldMgr.template get_field<double>(src.mesh_meta_data_ordinal());
  auto& destField =
    fieldMgr.template get_field<double>(dest.mesh_meta_data_ordinal());
  field_copy(
    meshInfo.ngp_mesh(), sel, destField, srcField, 0, numComponents, rank);
}

template <typename MeshInfoType, typename ScalarType>
inline void
field_axpby(
  const MeshInfoType& meshInfo,
  const stk::mesh::Selector& sel,
  const ScalarType alpha,
  const stk::mesh::FieldBase& xField,
  const ScalarType beta,
  stk::mesh::FieldBase& yField,
  const unsigned numComponents,
  const stk::topology::rank_t rank = stk::topology::NODE_RANK)
{
  const auto& fieldMgr = meshInfo.ngp_field_manager();
  const auto ngpXfield =
    fieldMgr.template get_field<ScalarType>(xField.mesh_meta_data_ordinal());
  auto ngpYfield =
    fieldMgr.template get_field<ScalarType>(yField.mesh_meta_data_ordinal());

  field_axpby(
    meshInfo.ngp_mesh(), sel, alpha, ngpXfield, beta, ngpYfield, numComponents,
    rank);
}

template <typename MeshInfoType>
inline void
scalar_field_axpby(
  const MeshInfoType& meshInfo,
  const stk::mesh::Selector& sel,
  const double alpha,
  const stk::mesh::FieldBase& xField,
  const double beta,
  stk::mesh::FieldBase& yField,
  const stk::topology::rank_t rank = stk::topology::NODE_RANK)
{
  constexpr unsigned nComp = 1;
  field_axpby(meshInfo, sel, alpha, xField, beta, yField, nComp, rank);
}

template <typename MeshInfoType>
inline void
vector_field_axpby(
  const MeshInfoType& meshInfo,
  const stk::mesh::Selector& sel,
  const double alpha,
  const VectorFieldType& xField,
  const double beta,
  VectorFieldType& yField,
  const stk::topology::rank_t rank = stk::topology::NODE_RANK)
{
  constexpr unsigned nComp = 3;
  field_axpby(meshInfo, sel, alpha, xField, beta, yField, nComp, rank);
}

} // namespace nalu_ngp
} // namespace nalu
} // namespace sierra

#endif /* NGPFIELDBLAS_H */
