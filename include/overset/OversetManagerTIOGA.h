// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#ifndef OVERSETMANAGERTIOGA_H
#define OVERSETMANAGERTIOGA_H

#include "overset/OversetManager.h"
#include "overset/TiogaSTKIface.h"

#include <stk_mesh/base/FieldBase.hpp>

namespace sierra {
namespace nalu {

class Realm;
class OversetInfo;
struct OversetUserData;

/** Overset Connectivity Algorithm using TIOGA third-party library
 *
 *  This class is a thin Nalu-TIOGA wrapper to provide compatibility with Nalu's
 *  built-in STK based overset connectivity algorithm. The heavy lifting is done
 *  by the TiogaSTKIface class. Please refer to the documentation of that class
 *  for actual implementation details.
 */
class OversetManagerTIOGA : public OversetManager
{
public:
  OversetManagerTIOGA(Realm&, const OversetUserData&);

  virtual ~OversetManagerTIOGA();

  virtual void setup() override;

  virtual void initialize(const bool isDecoupled = false) override;

  virtual void overset_update_fields(const std::vector<OversetFieldData>&) override;

  virtual void overset_update_field(
    stk::mesh::FieldBase* field, int nrows = 1, int ncols = 1) override;

  /// Instance holding all the data from input files
  const OversetUserData& oversetUserData_;

  /// Tioga-STK interface instance that performs the necessary translation
  /// between TIOGA and STK data structures.
  tioga_nalu::TiogaSTKIface tiogaIface_;

  /// Flag tracking initialization phase for part registration
  bool isInit_{true};
};

}  // nalu
}  // sierra

#endif /* OVERSETMANAGERTIOGA_H */
