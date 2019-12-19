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
#include "master_element/Quad43DCVFEM.h"
#include "utils/CreateDeviceExpression.h"

namespace stk { struct topology; }

namespace sierra{
namespace nalu{
  class MasterElement;
  struct MasterElementRepo
  {
  public:
    static MasterElement*
    get_surface_master_element(const stk::topology& theTopo);

    static MasterElement*
    get_volume_master_element(const stk::topology& theTopo);

    template <typename AlgTraitsr>
    static MasterElement* get_volume_master_element();

    template <typename AlgTraits>
    static MasterElement* get_surface_master_element();

    static void clear();
  private:
    MasterElementRepo() = default;
    static std::map<stk::topology, std::unique_ptr<MasterElement>> surfaceMeMap_;
    static std::map<stk::topology, std::unique_ptr<MasterElement>> volumeMeMap_;
    static std::map<stk::topology, MasterElement*> &volumeMeMapDev();
    static std::map<stk::topology, MasterElement*> &surfaceMeMapDev();

    template<typename AlgTraits, typename ME>
    static MasterElement* get_master_element(
      std::map<stk::topology, MasterElement*> &meMapDev
    );
  };

  template <typename AlgTraits, typename ME>
  MasterElement*
  MasterElementRepo::get_master_element(
    std::map<stk::topology, MasterElement*>& meMap)
  {
    const stk::topology theTopo = AlgTraits::topo_;
    //FIXME: ETI this
    ThrowRequire(!theTopo.is_super_topology());
    if (meMap.find(theTopo) == meMap.end()) {
      meMap[theTopo] = static_cast<MasterElement*>(sierra::nalu::create_device_expression<ME>());
    }
    MasterElement* theElem = meMap.at(theTopo);
    return theElem;
  }

  template <typename AlgTraits>
  MasterElement* MasterElementRepo::get_volume_master_element()
  {
    return get_master_element<AlgTraits, typename AlgTraits::masterElementScv_>(volumeMeMapDev());
  }

  template <typename AlgTraits>
  MasterElement* MasterElementRepo::get_surface_master_element()
  {
    return get_master_element<AlgTraits, typename AlgTraits::masterElementScs_>(surfaceMeMapDev());
  }


} // namespace nalu
} // namespace sierra

#endif
