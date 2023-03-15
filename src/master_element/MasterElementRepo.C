// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "master_element/MasterElementRepo.h"
#include "master_element/MasterElement.h"

#include "master_element/Hex8CVFEM.h"
#include "master_element/Tet4CVFEM.h"
#include "master_element/Pyr5CVFEM.h"
#include "master_element/Wed6CVFEM.h"
#include "master_element/Quad43DCVFEM.h"
#include "master_element/Quad42DCVFEM.h"
#include "master_element/Tri32DCVFEM.h"
#include "master_element/Edge22DCVFEM.h"
#include "master_element/Tri33DCVFEM.h"

#include "NaluEnv.h"
#include "utils/CreateDeviceExpression.h"

#include <stk_util/util/ReportHandler.hpp>
#include <stk_topology/topology.hpp>

#include <cmath>
#include <iostream>
#include <memory>

namespace sierra {
namespace nalu {

namespace {

template <typename AlgTraits, typename ME>
std::pair<stk::topology, MasterElement*>
pair_dev()
{
  return {AlgTraits::topo_, create_device_expression<ME>()};
}

template <typename AlgTraits, typename ME>
std::pair<stk::topology, MasterElement*>
pair_host()
{
  return {AlgTraits::topo_, new ME()};
}

template <typename AlgTraits>
std::pair<stk::topology, MasterElement*>
surface_pair_dev()
{
  return pair_dev<AlgTraits, typename AlgTraits::masterElementScs_>();
}

template <typename AlgTraits>
std::pair<stk::topology, MasterElement*>
surface_pair_host()
{
  return pair_host<AlgTraits, typename AlgTraits::masterElementScs_>();
}

template <typename AlgTraits>
std::pair<stk::topology, MasterElement*>
volume_pair_dev()
{
  return pair_dev<AlgTraits, typename AlgTraits::masterElementScv_>();
}

template <typename AlgTraits>
std::pair<stk::topology, MasterElement*>
volume_pair_host()
{
  return pair_host<AlgTraits, typename AlgTraits::masterElementScv_>();
}

const std::map<stk::topology, MasterElement*>&
surface_topo_to_me_dev()
{
  static const std::map<stk::topology, MasterElement*> surface_me_map = {
    surface_pair_dev<AlgTraitsHex8>(),
    surface_pair_dev<AlgTraitsTet4>(),
    surface_pair_dev<AlgTraitsPyr5>(),
    surface_pair_dev<AlgTraitsWed6>(),
    surface_pair_dev<AlgTraitsQuad4>(),
    surface_pair_dev<AlgTraitsTri3>(),
    surface_pair_dev<AlgTraitsQuad4_2D>(),
    surface_pair_dev<AlgTraitsTri3_2D>(),
    surface_pair_dev<AlgTraitsEdge_2D>(),
    surface_pair_dev<AlgTraitsShellQuad4>(),
    surface_pair_dev<AlgTraitsShellTri3>(),
    surface_pair_dev<AlgTraitsBeam_2D>()};
  return surface_me_map;
}
const std::map<stk::topology, MasterElement*>&
surface_topo_to_me_host()
{
  static const std::map<stk::topology, MasterElement*> surface_me_map = {
    surface_pair_host<AlgTraitsHex8>(),
    surface_pair_host<AlgTraitsTet4>(),
    surface_pair_host<AlgTraitsPyr5>(),
    surface_pair_host<AlgTraitsWed6>(),
    surface_pair_host<AlgTraitsQuad4>(),
    surface_pair_host<AlgTraitsTri3>(),
    surface_pair_host<AlgTraitsQuad4_2D>(),
    surface_pair_host<AlgTraitsTri3_2D>(),
    surface_pair_host<AlgTraitsEdge_2D>(),
    surface_pair_host<AlgTraitsShellQuad4>(),
    surface_pair_host<AlgTraitsShellTri3>(),
    surface_pair_host<AlgTraitsBeam_2D>()};
  return surface_me_map;
}

const std::map<stk::topology, MasterElement*>&
volume_topo_to_me_dev()
{
  static const std::map<stk::topology, MasterElement*> volume_me_map = {
    volume_pair_dev<AlgTraitsHex8>(),     volume_pair_dev<AlgTraitsTet4>(),
    volume_pair_dev<AlgTraitsPyr5>(),     volume_pair_dev<AlgTraitsWed6>(),
    volume_pair_dev<AlgTraitsQuad4_2D>(), volume_pair_dev<AlgTraitsTri3_2D>(),
  };
  return volume_me_map;
}

const std::map<stk::topology, MasterElement*>&
volume_topo_to_me_host()
{
  static const std::map<stk::topology, MasterElement*> volume_me_map = {
    volume_pair_host<AlgTraitsHex8>(),     volume_pair_host<AlgTraitsTet4>(),
    volume_pair_host<AlgTraitsPyr5>(),     volume_pair_host<AlgTraitsWed6>(),
    volume_pair_host<AlgTraitsQuad4_2D>(), volume_pair_host<AlgTraitsTri3_2D>(),
  };
  return volume_me_map;
}

std::map<MasterElement*, stk::topology>
map_inverse(std::map<stk::topology, MasterElement*> M)
{
  std::map<MasterElement*, stk::topology> I;
  for (auto P : M)
    I[P.second] = P.first;
  return I;
}

const std::map<MasterElement*, stk::topology>&
surface_me_to_topo_dev()
{
  static const std::map<MasterElement*, stk::topology> surface_topo_map(
    map_inverse(surface_topo_to_me_dev()));
  return surface_topo_map;
}

const std::map<MasterElement*, stk::topology>&
surface_me_to_topo_host()
{
  static const std::map<MasterElement*, stk::topology> surface_topo_map(
    map_inverse(surface_topo_to_me_host()));
  return surface_topo_map;
}

const std::map<MasterElement*, stk::topology>&
volume_me_to_topo_dev()
{
  static const std::map<MasterElement*, stk::topology> volume_topo_map(
    map_inverse(volume_topo_to_me_dev()));
  return volume_topo_map;
}

const std::map<MasterElement*, stk::topology>&
volume_me_to_topo_host()
{
  static const std::map<MasterElement*, stk::topology> volume_topo_map(
    map_inverse(volume_topo_to_me_host()));
  return volume_topo_map;
}

stk::topology
find_topo(
  MasterElement* host_ptr,
  const std::map<MasterElement*, stk::topology>& host_map,
  const std::map<MasterElement*, stk::topology>& dev_map)
{
  stk::topology theTopo;
  if (const auto it = host_map.find(host_ptr); it != host_map.end()) {
    theTopo = it->second;
  } else if (const auto it = dev_map.find(host_ptr); it != dev_map.end()) {
    theTopo = it->second;
  } else {
    NaluEnv::self().naluOutputP0()
      << " Host Master Element pointer could not be converted to device "
         "pointer."
      << " The pointer was not found in the master element database:"
      << host_ptr << std::endl;
    ThrowRequire(host_map.find(host_ptr) != dev_map.end());
  }
  return theTopo;
}

MasterElement*
find_me(
  const stk::topology& theTopo,
  const std::map<stk::topology, MasterElement*>& me_map)
{
  auto it = me_map.find(theTopo);
  if (it == me_map.end()) {
    NaluEnv::self().naluOutputP0()
      << " Topology not supported: The topology, " << theTopo.name()
      << ", was not found in the map of supported topologies." << std::endl
      << " There are " << me_map.size()
      << " supported topologies:" << std::endl;
    for (const auto& v : me_map)
      NaluEnv::self().naluOutputP0() << v.first.name() << std::endl;
    NaluEnv::self().naluOutputP0()
      << " Add topology to MasterElementRepo::find_me()" << std::endl;
    ThrowRequire(it != me_map.end());
  }
  MasterElement* theElem = it->second;
  return theElem;
}

} // namespace

MasterElement*
MasterElementRepo::get_surface_master_element_on_dev(
  const stk::topology& theTopo)
{
  MasterElement* theElem = find_me(theTopo, surface_topo_to_me_dev());
  return theElem;
}

MasterElement*
MasterElementRepo::get_surface_master_element_on_host(
  const stk::topology& theTopo)
{
  MasterElement* theElem = find_me(theTopo, surface_topo_to_me_host());
  return theElem;
}

MasterElement*
MasterElementRepo::get_volume_master_element_on_dev(
  const stk::topology& theTopo)
{
  MasterElement* theElem = find_me(theTopo, volume_topo_to_me_dev());
  return theElem;
}

MasterElement*
MasterElementRepo::get_volume_master_element_on_host(
  const stk::topology& theTopo)
{
  MasterElement* theElem = find_me(theTopo, volume_topo_to_me_host());
  return theElem;
}

MasterElement*
MasterElementRepo::get_surface_dev_ptr_from_host_ptr(MasterElement* host_ptr)
{
  if (!host_ptr)
    return nullptr;
  const stk::topology theTopo =
    find_topo(host_ptr, surface_me_to_topo_host(), surface_me_to_topo_dev());
  return get_surface_master_element_on_dev(theTopo);
}

MasterElement*
MasterElementRepo::get_volume_dev_ptr_from_host_ptr(MasterElement* host_ptr)
{
  if (!host_ptr)
    return nullptr;
  const stk::topology theTopo =
    find_topo(host_ptr, volume_me_to_topo_host(), volume_me_to_topo_dev());
  return get_volume_master_element_on_dev(theTopo);
}

void
MasterElementRepo::clear()
{
  for (auto val : surface_topo_to_me_host())
    delete val.second;
  for (auto val : surface_topo_to_me_dev())
    kokkos_free_on_device(val.second);
  for (auto val : volume_topo_to_me_host())
    delete val.second;
  for (auto val : volume_topo_to_me_dev())
    kokkos_free_on_device(val.second);
}
} // namespace nalu
} // namespace sierra
