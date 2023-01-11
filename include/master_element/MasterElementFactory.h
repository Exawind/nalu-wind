// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef MasterElementFactory_h
#define MasterElementFactory_h

#include <string>
#include <map>
#include <memory>
#include <type_traits>
#include <stk_util/util/ReportHandler.hpp>

#include "AlgTraits.h"
#include "utils/CreateDeviceExpression.h"

namespace stk {
struct topology;
}

namespace sierra {
namespace nalu {
class MasterElement;
struct MasterElementRepo
{
public:
  static MasterElement*
  get_surface_master_element(const stk::topology& theTopo);

  static MasterElement* get_volume_master_element(const stk::topology& theTopo);

  template <typename AlgTraitsr>
  static MasterElement* get_volume_master_element();

  template <typename AlgTraits>
  static MasterElement* get_surface_master_element();
  static MasterElement*
  get_surface_master_element_on_dev(const stk::topology& theTopo);

  static void clear();

private:
  MasterElementRepo() = default;
  static std::map<stk::topology, std::unique_ptr<MasterElement>> surfaceMeMap_;
  static std::map<stk::topology, std::unique_ptr<MasterElement>> volumeMeMap_;
  static std::map<stk::topology, MasterElement*>& volumeMeMapDev();
  static std::map<stk::topology, MasterElement*>& surfaceMeMapDev();

public:
  template <typename AlgTraits, typename ME>
  static MasterElement*
  get_master_element(std::map<stk::topology, MasterElement*>& meMapDev);
};

template <typename AlgTraits, typename ME>
MasterElement*
MasterElementRepo::get_master_element(
  std::map<stk::topology, MasterElement*>& meMap)
{
  const stk::topology theTopo = AlgTraits::topo_;
  // FIXME: ETI this
  ThrowRequire(!theTopo.is_super_topology());
  if (meMap.find(theTopo) == meMap.end()) {
    const std::string& allocname = "ME_alloc_" + theTopo.name();
    const std::string& placementname = "ME_new_" + theTopo.name();
    ME* MEinstance = kokkos_malloc_on_device<ME>(allocname);
    ThrowRequire(MEinstance != nullptr);
    Kokkos::parallel_for(
      placementname, DeviceRangePolicy(0, 1),
      KOKKOS_LAMBDA(const int) { new (MEinstance) ME(); });
    meMap[theTopo] = MEinstance;
  }
  return meMap.at(theTopo);
}

template <typename AlgTraits>
MasterElement*
MasterElementRepo::get_volume_master_element()
{
  return get_master_element<AlgTraits, typename AlgTraits::masterElementScv_>(
    volumeMeMapDev());
}

template <typename AlgTraits>
MasterElement*
MasterElementRepo::get_surface_master_element()
{
  return get_master_element<AlgTraits, typename AlgTraits::masterElementScs_>(
    surfaceMeMapDev());
}

} // namespace nalu
} // namespace sierra

#endif
