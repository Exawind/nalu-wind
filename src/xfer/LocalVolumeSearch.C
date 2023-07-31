#include "xfer/LocalVolumeSearch.h"
#include "master_element/MasterElement.h"
#include "master_element/MasterElementRepo.h"
#include "AlgTraits.h"

#include "stk_mesh/base/BulkData.hpp"
#include "stk_mesh/base/Field.hpp"

#include "stk_search/BoundingBox.hpp"
#include "stk_search/IdentProc.hpp"
#include "stk_search/SearchMethod.hpp"
#include "stk_mesh/base/MetaData.hpp"
#include "stk_search/CoarseSearch.hpp"

#include "mpi.h"

namespace sierra {
namespace nalu {

namespace {

constexpr int dim = 3;
using vector_field_type = stk::mesh::Field<double, stk::mesh::Cartesian3d>;
using sphere_t = LocalVolumeSearchData::sphere_t;
using box_t = LocalVolumeSearchData::box_t;
using ident_t = LocalVolumeSearchData::ident_t;

auto
as_search_point(const std::array<double, dim>& x)
{
  return stk::search::Point<double>(x[0], x[1], x[2]);
}

auto
as_search_sphere(const std::array<double, dim>& x, double rad)
{
  return stk::search::Sphere<double>(as_search_point(x), rad);
}

auto
as_search_box(
  const stk::mesh::BulkData& bulk,
  const stk::mesh::Entity elem,
  const vector_field_type& coord_field)
{
  constexpr auto max_double = std::numeric_limits<double>::max();
  auto min_box = as_search_point({max_double, max_double, max_double});

  constexpr auto min_double = std::numeric_limits<double>::lowest();
  auto max_box = as_search_point({min_double, min_double, min_double});

  const int nnodes = static_cast<int>(bulk.num_nodes(elem));
  const auto* nodes = bulk.begin_nodes(elem);
  for (int n = 0; n < nnodes; ++n) {
    const auto* coord = stk::mesh::field_data(coord_field, nodes[n]);
    for (int j = 0; j < dim; ++j) {
      min_box[j] = std::min(min_box[j], coord[j]);
      max_box[j] = std::max(max_box[j], coord[j]);
    }
  }
  return stk::search::Box<double>(min_box, max_box);
}

void
fill_search_points(
  const std::vector<std::array<double, dim>>& points,
  double tol,
  std::vector<std::pair<sphere_t, ident_t>>& search_points)
{
  for (size_t j = 0; j < points.size(); ++j) {
    search_points[j] = std::make_pair(
      as_search_sphere(points[j], tol), ident_t(stk::mesh::EntityId(j), 0));
  }
}

void
fill_search_boxes(
  const stk::mesh::BulkData& bulk,
  const stk::mesh::Selector& active,
  const vector_field_type& coord_field,
  std::vector<std::pair<box_t, ident_t>>& box_list)
{
  box_list.clear();
  const auto& buckets = bulk.get_buckets(stk::topology::ELEM_RANK, active);
  for (const auto* ib : buckets) {
    for (auto elem : *ib) {
      box_list.emplace_back(
        as_search_box(bulk, elem, coord_field),
        ident_t(bulk.identifier(elem), 0));
    }
  }
}

double
determine_tolerance(const std::vector<std::pair<box_t, ident_t>>& search_boxes)
{
  double min_element_diameter = std::numeric_limits<double>::max();
  for (const auto& box_pair : search_boxes) {
    const auto& box = box_pair.first;
    const auto dx = box.get_x_max() - box.get_x_min();
    const auto dy = box.get_y_max() - box.get_y_min();
    const auto dz = box.get_z_max() - box.get_z_min();
    min_element_diameter =
      std::min(std::sqrt(dx * dx + dy * dy + dz * dz), min_element_diameter);
  }
  return min_element_diameter / 10;
}

void
local_coarse_search(
  const stk::mesh::BulkData& bulk,
  const stk::mesh::Selector& active,
  const vector_field_type& coord_field,
  const std::vector<std::array<double, dim>>& points,
  LocalVolumeSearchData& data)
{
  fill_search_boxes(bulk, active, coord_field, data.search_boxes);
  fill_search_points(
    points, determine_tolerance(data.search_boxes), data.search_points);

  data.search_matches.clear();
  stk::search::coarse_search(
    data.search_points, data.search_boxes, stk::search::SearchMethod::KDTREE,
    MPI_COMM_SELF, data.search_matches);
}

MasterElement&
master_element(const stk::mesh::BulkData& bulk, stk::mesh::Entity elem)
{
  return *MasterElementRepo::get_surface_master_element_on_host(
    bulk.bucket(elem).topology());
}

template <int nnodes>
auto
elem_gather(
  const stk::mesh::BulkData& bulk,
  stk::mesh::Entity elem,
  const vector_field_type& u)
{
  const auto* nodes = bulk.begin_nodes(elem);
  std::array<double, nnodes * dim> xc;
  for (size_t n = 0; n < bulk.num_nodes(elem); ++n) {
    const auto* xptr = stk::mesh::field_data(u, nodes[n]);
    for (int d = 0; d < dim; ++d) {
      xc[nnodes * d + n] = xptr[d];
    }
  }
  return xc;
}

template <int nnodes>
auto
extrapolated_elem_gather(
  const stk::mesh::BulkData& bulk,
  stk::mesh::Entity elem,
  const vector_field_type& unm1,
  const vector_field_type& u,
  double dtratio)
{
  const auto* nodes = bulk.begin_nodes(elem);
  std::array<double, nnodes * dim> xc;
  for (size_t n = 0; n < bulk.num_nodes(elem); ++n) {
    const auto* xold_ptr = stk::mesh::field_data(unm1, nodes[n]);
    const auto* xptr = stk::mesh::field_data(u, nodes[n]);
    for (int d = 0; d < dim; ++d) {
      xc[nnodes * d + n] = (1 + dtratio) * xptr[d] - dtratio * xold_ptr[d];
    }
  }
  return xc;
}

template <typename AlgTraits>
auto
compute_local_coordinates_t(
  const stk::mesh::BulkData& bulk,
  const vector_field_type& coord_field,
  stk::mesh::Entity elem,
  std::array<double, dim> point)
{
  const auto nodal_x =
    elem_gather<AlgTraits::nodesPerElement_>(bulk, elem, coord_field);

  auto& me = master_element(bulk, elem);
  std::array<double, dim> elem_x{};
  const auto dist = me.isInElement(nodal_x.data(), point.data(), elem_x.data());
  return std::make_pair(elem_x, dist);
}

auto
compute_local_coordinates(
  const stk::mesh::BulkData& bulk,
  const vector_field_type& coord_field,
  stk::mesh::Entity elem,
  std::array<double, dim> point)
{
  const stk::topology::topology_t topo = bulk.bucket(elem).topology();
  switch (topo) {
  case stk::topology::TET_4:
    return compute_local_coordinates_t<AlgTraitsTet4>(
      bulk, coord_field, elem, point);
  case stk::topology::WEDGE_6:
    return compute_local_coordinates_t<AlgTraitsWed6>(
      bulk, coord_field, elem, point);
  case stk::topology::HEX_8:
    return compute_local_coordinates_t<AlgTraitsHex8>(
      bulk, coord_field, elem, point);
  default: {
    STK_ThrowRequire(topo == stk::topology::PYRAMID_5);
    return compute_local_coordinates_t<AlgTraitsPyr5>(
      bulk, coord_field, elem, point);
  }
  }
}

template <typename AlgTraits>
auto
interpolate_field_t(
  const stk::mesh::BulkData& bulk,
  stk::mesh::Entity elem,
  const vector_field_type& field,
  const std::array<double, dim>& x)
{
  auto& me = master_element(bulk, elem);

  constexpr int nn = AlgTraits::nodesPerElement_;
  const auto& nodal_field = elem_gather<nn>(bulk, elem, field);
  std::array<double, dim> vel;
  me.interpolatePoint(dim, x.data(), nodal_field.data(), vel.data());
  return vel;
}

template <typename AlgTraits>
auto
interpolate_field_t(
  const stk::mesh::BulkData& bulk,
  stk::mesh::Entity elem,
  const vector_field_type& field_prev,
  const vector_field_type& field,
  const std::array<double, dim>& x,
  double dtratio)
{
  auto& me = master_element(bulk, elem);

  constexpr int nn = AlgTraits::nodesPerElement_;
  const auto& nodal_field =
    extrapolated_elem_gather<nn>(bulk, elem, field_prev, field, dtratio);
  std::array<double, dim> vel;
  me.interpolatePoint(dim, x.data(), nodal_field.data(), vel.data());
  return vel;
}

auto
interpolate_field(
  const stk::mesh::BulkData& bulk,
  stk::mesh::Entity elem,
  const vector_field_type& field_prev,
  const vector_field_type& field,
  const std::array<double, dim>& x,
  double dtratio)
{
  const stk::topology::topology_t topo = bulk.bucket(elem).topology();
  switch (topo) {
  case stk::topology::TET_4:
    return interpolate_field_t<AlgTraitsTet4>(
      bulk, elem, field_prev, field, x, dtratio);
  case stk::topology::WEDGE_6:
    return interpolate_field_t<AlgTraitsWed6>(
      bulk, elem, field_prev, field, x, dtratio);
  case stk::topology::HEX_8:
    return interpolate_field_t<AlgTraitsHex8>(
      bulk, elem, field_prev, field, x, dtratio);
  default: {
    STK_ThrowRequire(topo == stk::topology::PYRAMID_5);
    return interpolate_field_t<AlgTraitsPyr5>(
      bulk, elem, field_prev, field, x, dtratio);
  }
  }
}

} // namespace

LocalVolumeSearchData::LocalVolumeSearchData(
  const stk::mesh::BulkData& bulk, const stk::mesh::Selector& sel, int npoints)
  : search_points(npoints),
    interpolated_values(npoints),
    dist(npoints),
    ownership(npoints)
{
  const auto& elem_buckets = bulk.get_buckets(stk::topology::ELEM_RANK, sel);
  int elem_count = 0;
  for (const auto* ib : elem_buckets) {
    elem_count += ib->size();
  }
  search_boxes.reserve(elem_count);

  const auto& node_buckets = bulk.get_buckets(stk::topology::NODE_RANK, sel);
  int max_connections = -1;
  for (const auto* ib : node_buckets) {
    for (const auto& node : *ib) {
      max_connections =
        std::max(max_connections, static_cast<int>(bulk.num_elements(node)));
    }
  }
  search_matches.reserve(2 * max_connections * npoints);
}

void
local_field_interpolation(
  const stk::mesh::BulkData& bulk,
  const stk::mesh::Selector& active,
  const std::vector<std::array<double, 3>>& points,
  const stk::mesh::Field<double, stk::mesh::Cartesian3d>& x_field,
  const stk::mesh::Field<double, stk::mesh::Cartesian3d>& field_prev,
  const stk::mesh::Field<double, stk::mesh::Cartesian3d>& field,
  double dtratio,
  LocalVolumeSearchData& data)
{
  local_coarse_search(bulk, active, x_field, points, data);
  std::fill(
    data.interpolated_values.begin(), data.interpolated_values.end(),
    std::array<double, dim>{0, 0, 0});
  std::fill(
    data.dist.begin(), data.dist.end(), std::numeric_limits<double>::max());
  std::fill(data.ownership.begin(), data.ownership.end(), 0);

  for (const auto& match : data.search_matches) {
    auto point_id = match.first.id();
    auto point = points[point_id];
    auto elem = bulk.get_entity(stk::topology::ELEM_RANK, match.second.id());
    const auto& x_dist = compute_local_coordinates(bulk, x_field, elem, point);
    if (x_dist.second < data.dist[point_id]) {
      data.dist.at(point_id) = x_dist.second;
      data.interpolated_values.at(point_id) =
        interpolate_field(bulk, elem, field_prev, field, x_dist.first, dtratio);
      data.ownership.at(point_id) = 1;
    }
  }
}

} // namespace nalu
} // namespace sierra
