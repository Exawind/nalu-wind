#ifndef KOKKOS_FRAMEWORK_H
#define KOKKOS_FRAMEWORK_H

#include "Kokkos_Core.hpp"

#include "stk_mesh/base/Ngp.hpp"
#include "stk_mesh/base/Types.hpp"
#include "stk_simd/Simd.hpp"

#include <type_traits>

namespace sierra {
namespace nalu {
namespace matrix_free {

using exec_space = Kokkos::DefaultExecutionSpace;

#ifndef USE_STK_SIMD_NONE
template <typename ExecSpace>
struct ExecTraits
{
  using data_type = stk::simd::Double;
  using memory_traits = Kokkos::MemoryTraits<Kokkos::Restrict>;
  using memory_space = typename ExecSpace::memory_space;
  using layout = Kokkos::LayoutRight;
  static constexpr int alignment = alignof(data_type);
  static constexpr int simd_len = stk::simd::ndoubles;
  static constexpr bool force_atomic =
    std::is_same<exec_space, Kokkos::Serial>::value;
};
#else
template <typename ExecSpace>
struct ExecTraits
{
  using data_type = double;
  using memory_traits = Kokkos::MemoryTraits<Kokkos::Restrict>;
  using memory_space = typename ExecSpace::memory_space;
  using layout = Kokkos::LayoutRight;
  static constexpr int alignment = alignof(double);
  static constexpr int simd_len = 1;
  static constexpr bool force_atomic =
    std::is_same<exec_space, Kokkos::Serial>::value;
};
#endif

#ifdef KOKKOS_ENABLE_CUDA
template <>
struct ExecTraits<Kokkos::Cuda>
{
  using data_type = double;
  using memory_traits =
    Kokkos::MemoryTraits<Kokkos::Restrict | Kokkos::Aligned>;
  using memory_space = typename Kokkos::Cuda::memory_space;
  using layout = Kokkos::LayoutLeft;
  static constexpr int alignment = 32;
  static constexpr int simd_len = 1;
  static constexpr bool force_atomic = true;
};
#endif

using ftype = typename ExecTraits<exec_space>::data_type;
static constexpr bool force_atomic = ExecTraits<exec_space>::force_atomic;
static constexpr int simd_len = ExecTraits<exec_space>::simd_len;
static constexpr int alignment = ExecTraits<exec_space>::alignment;

using entity_row_view_type = Kokkos::View<
  int*,
  typename ExecTraits<exec_space>::layout,
  typename ExecTraits<exec_space>::memory_space,
  typename ExecTraits<exec_space>::memory_traits>;

using const_entity_row_view_type = Kokkos::View<
  const int*,
  typename ExecTraits<exec_space>::layout,
  typename ExecTraits<exec_space>::memory_space,
  typename ExecTraits<exec_space>::memory_traits>;

using ra_entity_row_view_type = Kokkos::View<
  const int*,
  typename ExecTraits<exec_space>::layout,
  typename ExecTraits<exec_space>::memory_space,
  Kokkos::MemoryTraits<
    Kokkos::Restrict | Kokkos::Aligned | Kokkos::RandomAccess>>;

using mesh_index_row_view_type = Kokkos::View<
  stk::mesh::FastMeshIndex*,
  typename ExecTraits<exec_space>::layout,
  typename ExecTraits<exec_space>::memory_space,
  typename ExecTraits<exec_space>::memory_traits>;

using const_mesh_index_row_view_type = Kokkos::View<
  const stk::mesh::FastMeshIndex*,
  typename ExecTraits<exec_space>::layout,
  typename ExecTraits<exec_space>::memory_space,
  typename ExecTraits<exec_space>::memory_traits>;

enum class FieldType {
  NODAL_SCALAR,
  NODAL_VECTOR,
  SCS_VECTOR,
  FACE_SCALAR,
  FACE_VECTOR,
  ENTITY,
  MESH_INDEX,
  FACE_MESH_INDEX,
  NODE_OFFSET,
  FACE_OFFSET,
  ELEM_OFFSET,
};

template <int p, FieldType>
struct GlobalArrayTypeSelector
{
};

template <int p>
struct GlobalArrayTypeSelector<p, FieldType::NODAL_SCALAR>
{
  static constexpr int n = p + 1;
  using type = ftype* [n][n][n];
  using const_type = const ftype* [n][n][n];
};

template <int p>
struct GlobalArrayTypeSelector<p, FieldType::FACE_SCALAR>
{
  static constexpr int n = p + 1;
  using type = ftype* [n][n];
  using const_type = const ftype* [n][n];
};

template <int p>
struct GlobalArrayTypeSelector<p, FieldType::NODAL_VECTOR>
{
  static constexpr int n = p + 1;
  using type = ftype* [n][n][n][3];
  using const_type = const ftype* [n][n][n][3];
};

template <int p>
struct GlobalArrayTypeSelector<p, FieldType::FACE_VECTOR>
{
  static constexpr int n = p + 1;
  using type = ftype* [n][n][3];
  using const_type = const ftype* [n][n][3];
};

template <int p>
struct GlobalArrayTypeSelector<p, FieldType::SCS_VECTOR>
{
  static constexpr int n = p + 1;
  using type = ftype* [3][p][n][n][3];
  using const_type = const ftype* [3][p][n][n][3];
};

template <int p>
struct GlobalArrayTypeSelector<p, FieldType::ENTITY>
{
  static constexpr int n = p + 1;
  using type = stk::mesh::Entity* [simd_len][n][n][n];
  using const_type = const stk::mesh::Entity* [simd_len][n][n][n];
};

template <int p>
struct GlobalArrayTypeSelector<p, FieldType::MESH_INDEX>
{
  static constexpr int n = p + 1;
  using type = stk::mesh::FastMeshIndex* [n][n][n][simd_len];
  using const_type = const stk::mesh::FastMeshIndex* [n][n][n][simd_len];
};

template <int p>
struct GlobalArrayTypeSelector<p, FieldType::FACE_MESH_INDEX>
{
  static constexpr int n = p + 1;
  using type = stk::mesh::FastMeshIndex* [n][n][simd_len];
  using const_type = const stk::mesh::FastMeshIndex* [n][n][simd_len];
};

template <int p>
struct GlobalArrayTypeSelector<p, FieldType::NODE_OFFSET>
{
  using type = int* [simd_len];
  using const_type = const int* [simd_len];
};

template <int p>
struct GlobalArrayTypeSelector<p, FieldType::ELEM_OFFSET>
{
  static constexpr int n = p + 1;
  using type = int* [n][n][n][simd_len];
  using const_type = const int* [n][n][n][simd_len];
};

template <int p>
struct GlobalArrayTypeSelector<p, FieldType::FACE_OFFSET>
{
  static constexpr int n = p + 1;
  using type = int* [n][n][simd_len];
  using const_type = const int* [n][n][simd_len];
};

template <int p, FieldType field_type>
using ViewArrayType = typename GlobalArrayTypeSelector<p, field_type>::type;

template <int p, FieldType field_type>
using ConstViewArrayType =
  typename GlobalArrayTypeSelector<p, field_type>::const_type;

template <int p, FieldType type, typename ExecSpace>
using view_type = Kokkos::View<
  ViewArrayType<p, type>,
  typename ExecTraits<ExecSpace>::layout,
  typename ExecTraits<ExecSpace>::memory_space,
  typename ExecTraits<ExecSpace>::memory_traits>;

template <int p, FieldType type, typename ExecSpace>
using const_view_type = Kokkos::View<
  ConstViewArrayType<p, type>,
  typename ExecTraits<ExecSpace>::layout,
  typename ExecTraits<ExecSpace>::memory_space,
  typename ExecTraits<ExecSpace>::memory_traits>;

using node_offset_view = Kokkos::View<
  int* [simd_len],
  typename ExecTraits<exec_space>::layout,
  typename ExecTraits<exec_space>::memory_space,
  typename ExecTraits<exec_space>::memory_traits>;

using const_node_offset_view = Kokkos::View<
  const int* [simd_len],
  typename ExecTraits<exec_space>::layout,
  typename ExecTraits<exec_space>::memory_space,
  typename ExecTraits<exec_space>::memory_traits>;

using node_mesh_index_view = Kokkos::View<
  stk::mesh::FastMeshIndex* [simd_len],
  typename ExecTraits<exec_space>::layout,
  typename ExecTraits<exec_space>::memory_space,
  typename ExecTraits<exec_space>::memory_traits>;

using const_node_mesh_index_view = Kokkos::View<
  const stk::mesh::FastMeshIndex* [simd_len],
  typename ExecTraits<exec_space>::layout,
  typename ExecTraits<exec_space>::memory_space,
  typename ExecTraits<exec_space>::memory_traits>;

using node_scalar_view = Kokkos::View<
  ftype*,
  typename ExecTraits<exec_space>::layout,
  typename ExecTraits<exec_space>::memory_space,
  typename ExecTraits<exec_space>::memory_traits>;

using const_node_scalar_view = Kokkos::View<
  const ftype*,
  typename ExecTraits<exec_space>::layout,
  typename ExecTraits<exec_space>::memory_space,
  typename ExecTraits<exec_space>::memory_traits>;

template <int p>
using scalar_view = view_type<p, FieldType::NODAL_SCALAR, exec_space>;
template <int p>
using const_scalar_view =
  const_view_type<p, FieldType::NODAL_SCALAR, exec_space>;

template <int p>
using face_scalar_view = view_type<p, FieldType::FACE_SCALAR, exec_space>;
template <int p>
using const_face_scalar_view =
  const_view_type<p, FieldType::FACE_SCALAR, exec_space>;

template <int p>
using face_vector_view = view_type<p, FieldType::FACE_VECTOR, exec_space>;
template <int p>
using const_face_vector_view =
  const_view_type<p, FieldType::FACE_VECTOR, exec_space>;

template <int p>
using vector_view = view_type<p, FieldType::NODAL_VECTOR, exec_space>;
template <int p>
using const_vector_view =
  const_view_type<p, FieldType::NODAL_VECTOR, exec_space>;

template <int p>
using scs_vector_view = view_type<p, FieldType::SCS_VECTOR, exec_space>;
template <int p>
using const_scs_vector_view =
  const_view_type<p, FieldType::SCS_VECTOR, exec_space>;

template <int p>
using elem_entity_view = view_type<p, FieldType::ENTITY, exec_space>;
template <int p>
using const_elem_entity_view =
  const_view_type<p, FieldType::ENTITY, exec_space>;

template <int p>
using elem_mesh_index_view = view_type<p, FieldType::MESH_INDEX, exec_space>;
template <int p>
using const_elem_mesh_index_view =
  const_view_type<p, FieldType::MESH_INDEX, exec_space>;

template <int p>
using face_mesh_index_view =
  view_type<p, FieldType::FACE_MESH_INDEX, exec_space>;
template <int p>
using const_face_mesh_index_view =
  const_view_type<p, FieldType::FACE_MESH_INDEX, exec_space>;

template <int p>
using elem_offset_view = view_type<p, FieldType::ELEM_OFFSET, exec_space>;
template <int p>
using const_elem_offset_view =
  const_view_type<p, FieldType::ELEM_OFFSET, exec_space>;

template <int p>
using face_offset_view = view_type<p, FieldType::FACE_OFFSET, exec_space>;
template <int p>
using const_face_offset_view =
  const_view_type<p, FieldType::FACE_OFFSET, exec_space>;
} // namespace matrix_free
} // namespace nalu
} // namespace sierra

#endif
