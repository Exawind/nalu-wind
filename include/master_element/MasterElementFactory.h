/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef MasterElementFactory_h
#define MasterElementFactory_h

#include <string>
#include <map>
#include <memory>
#include <type_traits>
#include <stk_util/util/ReportHandler.hpp>

#include "AlgTraits.h"
#include "utils/CreateDeviceExpression.h"

namespace stk { struct topology; }

namespace sierra{
namespace nalu{
  class MasterElement;

  struct MasterElementRepo
  {
  public:
    static MasterElement*
    get_surface_master_element(
      const stk::topology& theTopo,
      int dimension = 0,
      std::string quadType = "GaussLegendre");

    static MasterElement*
    get_volume_master_element(
      const stk::topology& theTopo,
      int dimension = 0,
      std::string quadType = "GaussLegendre");

    template <
      typename AlgTraits,
      typename std::enable_if<!AlgTraits::isSuperTopo, AlgTraits>::type* = nullptr>
    static MasterElement* get_volume_master_element();

    template <
      typename AlgTraits,
      typename std::enable_if<AlgTraits::isSuperTopo, AlgTraits>::type* = nullptr>
    static MasterElement* get_volume_master_element();

    template <
      typename AlgTraits,
      typename std::enable_if<!AlgTraits::isSuperTopo, AlgTraits>::type* = nullptr>
    static MasterElement* get_surface_master_element();

    template <
      typename AlgTraits,
      typename std::enable_if<AlgTraits::isSuperTopo, AlgTraits>::type* = nullptr>
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

    ThrowRequire(!theTopo.is_super_topology());
    if (meMap.find(theTopo) == meMap.end()) {
      meMap[theTopo] = sierra::nalu::create_device_expression<ME>();
    }
    MasterElement* theElem = meMap.at(theTopo);
    return theElem;
  }

  template <
    typename AlgTraits,
    typename std::enable_if<!AlgTraits::isSuperTopo, AlgTraits>::type*>
  MasterElement* MasterElementRepo::get_volume_master_element()
  {
    return get_master_element<AlgTraits, typename AlgTraits::masterElementScv_>(volumeMeMapDev());
  }

  template <
    typename AlgTraits,
    typename std::enable_if<AlgTraits::isSuperTopo, AlgTraits>::type*>
  MasterElement* MasterElementRepo::get_volume_master_element()
  {
#ifndef KOKKOS_ENABLE_CUDA
    return get_volume_master_element(AlgTraits::topo_);
#else
    return nullptr;
#endif
  }

  template <
    typename AlgTraits,
    typename std::enable_if<!AlgTraits::isSuperTopo, AlgTraits>::type*>
  MasterElement* MasterElementRepo::get_surface_master_element()
  {
    return get_master_element<AlgTraits, typename AlgTraits::masterElementScs_>(surfaceMeMapDev());
  }

  template <
    typename AlgTraits,
    typename std::enable_if<AlgTraits::isSuperTopo, AlgTraits>::type*>
  MasterElement* MasterElementRepo::get_surface_master_element()
  {
#ifndef KOKKOS_ENABLE_CUDA
    return get_surface_master_element(AlgTraits::topo_);
#else
    return nullptr;
#endif
  }

} // namespace nalu
} // namespace sierra

#endif
