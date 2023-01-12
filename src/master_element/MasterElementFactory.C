// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "master_element/MasterElementFactory.h"
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

MasterElement*
MasterElementRepo::get_surface_master_element_on_dev(
  const stk::topology& theTopo)
{
  switch (theTopo.value()) {
  case stk::topology::HEX_8:
    return get_surface_master_element<AlgTraitsHex8>();
  case stk::topology::TET_4:
    return get_surface_master_element<AlgTraitsTet4>();
  case stk::topology::PYRAMID_5:
    return get_surface_master_element<AlgTraitsPyr5>();
  case stk::topology::WEDGE_6:
    return get_surface_master_element<AlgTraitsWed6>();
  case stk::topology::QUAD_4:
    return get_surface_master_element<AlgTraitsQuad4>();
  case stk::topology::TRI_3:
    return get_surface_master_element<AlgTraitsTri3>();
  case stk::topology::QUAD_4_2D:
    return get_surface_master_element<AlgTraitsQuad4_2D>();
  case stk::topology::TRI_3_2D:
    return get_surface_master_element<AlgTraitsTri3_2D>();
  case stk::topology::LINE_2:
    return get_surface_master_element<AlgTraitsEdge_2D>();
  case stk::topology::SHELL_QUAD_4:
    NaluEnv::self().naluOutputP0()
      << "SHELL_QUAD_4 only supported for io surface transfer applications"
      << std::endl;
    return get_surface_master_element<AlgTraitsQuad4>();

  case stk::topology::SHELL_TRI_3:
    NaluEnv::self().naluOutputP0()
      << "SHELL_TRI_3 only supported for io surface transfer applications"
      << std::endl;
    return get_surface_master_element<AlgTraitsTri3>();

  case stk::topology::BEAM_2:
    NaluEnv::self().naluOutputP0()
      << "BEAM_2 is only supported for io surface transfer applications"
      << std::endl;
    return get_surface_master_element<AlgTraitsEdge_2D>();

  default:
    NaluEnv::self().naluOutputP0()
      << "sorry, we only support hex8, tet4, pyr5, wed6,"
         " quad42d, quad3d, tri2d, tri3d and edge2d surface elements"
      << std::endl;
    NaluEnv::self().naluOutputP0()
      << "your type is " << theTopo.value() << std::endl;
    break;
  }
  return nullptr;
}

std::unique_ptr<MasterElement>
create_surface_master_element(stk::topology topo)
{
  switch (topo.value()) {

  case stk::topology::HEX_8:
    return std::make_unique<HexSCS>();

  case stk::topology::TET_4:
    return std::make_unique<TetSCS>();

  case stk::topology::PYRAMID_5:
    return std::make_unique<PyrSCS>();

  case stk::topology::WEDGE_6:
    return std::make_unique<WedSCS>();

  case stk::topology::QUAD_4:
    return std::make_unique<Quad3DSCS>();

  case stk::topology::TRI_3:
    return std::make_unique<Tri3DSCS>();

  case stk::topology::QUAD_4_2D:
    return std::make_unique<Quad42DSCS>();

  case stk::topology::TRI_3_2D:
    return std::make_unique<Tri32DSCS>();

  case stk::topology::LINE_2:
    return std::make_unique<Edge2DSCS>();

  case stk::topology::SHELL_QUAD_4:
    NaluEnv::self().naluOutputP0()
      << "SHELL_QUAD_4 only supported for io surface transfer applications"
      << std::endl;
    return std::make_unique<Quad3DSCS>();

  case stk::topology::SHELL_TRI_3:
    NaluEnv::self().naluOutputP0()
      << "SHELL_TRI_3 only supported for io surface transfer applications"
      << std::endl;
    return std::make_unique<Tri3DSCS>();

  case stk::topology::BEAM_2:
    NaluEnv::self().naluOutputP0()
      << "BEAM_2 is only supported for io surface transfer applications"
      << std::endl;
    return std::make_unique<Edge2DSCS>();

  default:
    NaluEnv::self().naluOutputP0()
      << "sorry, we only support hex8, tet4, pyr5, wed6,"
         " quad42d, quad3d, tri2d, tri3d and edge2d surface elements"
      << std::endl;
    NaluEnv::self().naluOutputP0()
      << "your type is " << topo.value() << std::endl;
    break;
  }
  return nullptr;
}
//--------------------------------------------------------------------------
std::unique_ptr<MasterElement>
create_volume_master_element(stk::topology topo)
{
  switch (topo.value()) {

  case stk::topology::HEX_8:
    return std::make_unique<HexSCV>();

  case stk::topology::TET_4:
    return std::make_unique<TetSCV>();

  case stk::topology::PYRAMID_5:
    return std::make_unique<PyrSCV>();

  case stk::topology::WEDGE_6:
    return std::make_unique<WedSCV>();

  case stk::topology::QUAD_4_2D:
    return std::make_unique<Quad42DSCV>();

  case stk::topology::TRI_3_2D:
    return std::make_unique<Tri32DSCV>();

  default:
    NaluEnv::self().naluOutputP0()
      << "sorry, we only support hex8, tet4, wed6, "
         " pyr5, quad4, and tri3 volume elements"
      << std::endl;
    NaluEnv::self().naluOutputP0()
      << "your type is " << topo.value() << std::endl;
    break;
  }
  return nullptr;
}

std::map<stk::topology, std::unique_ptr<MasterElement>>
  MasterElementRepo::surfaceMeMap_;

MasterElement*
MasterElementRepo::get_surface_master_element(const stk::topology& theTopo)
{
  auto it = surfaceMeMap_.find(theTopo);
  if (it == surfaceMeMap_.end()) {
    surfaceMeMap_[theTopo] = create_surface_master_element(theTopo);
  }
  MasterElement* theElem = surfaceMeMap_.at(theTopo).get();
  ThrowRequire(theElem != nullptr);
  return theElem;
}

std::map<stk::topology, std::unique_ptr<MasterElement>>
  MasterElementRepo::volumeMeMap_;

MasterElement*
MasterElementRepo::get_volume_master_element(const stk::topology& theTopo)
{
  auto it = volumeMeMap_.find(theTopo);
  if (it == volumeMeMap_.end()) {
    volumeMeMap_[theTopo] = create_volume_master_element(theTopo);
  }
  MasterElement* theElem = volumeMeMap_.at(theTopo).get();
  ThrowRequire(theElem != nullptr);
  return theElem;
}

std::map<stk::topology, MasterElement*>&
MasterElementRepo::volumeMeMapDev()
{
  static std::map<stk::topology, MasterElement*> M;
  return M;
}
std::map<stk::topology, MasterElement*>&
MasterElementRepo::surfaceMeMapDev()
{
  static std::map<stk::topology, MasterElement*> M;
  return M;
}

void
MasterElementRepo::clear()
{
  surfaceMeMap_.clear();
  volumeMeMap_.clear();
  for (std::pair<stk::topology, MasterElement*> a : volumeMeMapDev()) {
    const std::string debuggingName("MEDestroy: " + a.first.name());
    auto* meobj = a.second;
    Kokkos::parallel_for(
      debuggingName, DeviceRangePolicy(0, 1),
      KOKKOS_LAMBDA(int) { meobj->~MasterElement(); });
    sierra::nalu::kokkos_free_on_device(a.second);
  }
  volumeMeMapDev().clear();
  for (std::pair<stk::topology, MasterElement*> a : surfaceMeMapDev()) {
    const std::string debuggingName("MEDestroy: " + a.first.name());
    auto* meobj = a.second;
    Kokkos::parallel_for(
      debuggingName, DeviceRangePolicy(0, 1),
      KOKKOS_LAMBDA(int) { meobj->~MasterElement(); });
    sierra::nalu::kokkos_free_on_device(a.second);
  }
  surfaceMeMapDev().clear();
}

} // namespace nalu
} // namespace sierra
