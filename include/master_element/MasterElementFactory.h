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
  MasterElementRepo() = delete;
  static MasterElement*
  get_surface_master_element_on_host(const stk::topology& theTopo);
  static MasterElement*
  get_surface_master_element_on_dev(const stk::topology& theTopo);
  static MasterElement*
  get_volume_master_element_on_host(const stk::topology& theTopo);
  static MasterElement*
  get_volume_master_element_on_dev(const stk::topology& theTopo);

  // Given a host pointer to a master element as returned from the above calls,
  // find the equivalent device pointer to the master element.
  // NOTE:
  // 1. If given a device pointer, return the same device pointer.
  // 2. If given a null pointer, return a null pointer.
  static MasterElement*
  get_surface_dev_ptr_from_host_ptr(MasterElement* host_ptr);
  static MasterElement*
  get_volume_dev_ptr_from_host_ptr(MasterElement* host_ptr);

  static void clear();
};
} // namespace nalu
} // namespace sierra

#endif
