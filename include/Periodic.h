/*--------------------------------------------------------------------*/
/*    Copyright 2002 - 2008, 2010, 2011 National Technology &         */
/*    Engineering Solutions of Sandia, LLC (NTESS). Under the terms   */
/*    of Contract DE-NA0003525 with NTESS, there is a                 */
/*    non-exclusive license for use of this work by or on behalf      */
/*    of the U.S. Government.  Export of this program may require     */
/*    a license from the United States Government.                    */
/*--------------------------------------------------------------------*/

#pragma once

#include "KokkosInterface.h"

#include "Kokkos_Core.hpp"
#include "stk_mesh/base/Entity.hpp"
#include "stk_mesh/base/Selector.hpp"
#include "stk_mesh/base/Ghosting.hpp"

#include <vector>

namespace sierra::nalu {

struct PeriodicBCData
{
  stk::mesh::Part* part_a{nullptr};
  stk::mesh::Part* part_b{nullptr};
  double search_tol{1e-6};
  bool rotational{false};
};

} // namespace sierra::nalu

namespace sierra::nalu::periodic {

inline constexpr auto ghosting_name = "periodic_ghosting";

struct TranslationMapping
{
  Kokkos::
    View<const Kokkos::pair<stk::mesh::Entity, stk::mesh::Entity>*, DeviceSpace>
      local_a_pairs;
  Kokkos::
    View<const Kokkos::pair<stk::mesh::Entity, stk::mesh::Entity>*, DeviceSpace>
      local_b_pairs;
  using map_t = decltype(local_a_pairs);

  stk::mesh::Selector selector_a;
  stk::mesh::Selector selector_b;
};

std::optional<TranslationMapping>
make_translation_mapping(
  const std::vector<PeriodicBCData>& data,
  stk::mesh::BulkData& mesh,
  const stk::mesh::Selector& active_not_aura,
  const std::string& coordinates_name);

stk::mesh::Ghosting& get_ghosting(const stk::mesh::BulkData& bulk);
stk::mesh::Selector get_ghosting_selector(const stk::mesh::BulkData& bulk);

void sync(
  const DeviceSpace& exec,
  const stk::mesh::BulkData& bulk,
  stk::mesh::FieldBase& field,
  bool local = false);

void sum(
  const DeviceSpace& exec,
  const stk::mesh::BulkData& bulk,
  stk::mesh::FieldBase& field,
  bool local = false);

void max(
  const DeviceSpace& exec,
  const stk::mesh::BulkData& bulk,
  stk::mesh::FieldBase& field,
  bool local = false);

void set_periodic_on_mesh(
  stk::mesh::BulkData& bulk,
  const std::optional<periodic::TranslationMapping>& periodic);

} // namespace sierra::nalu::periodic
