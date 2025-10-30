// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifdef NALU_USES_TIOGA

#include "overset/OversetManagerTIOGA.h"
#include "overset/OversetInfo.h"
#include "overset/OversetFieldData.h"
#include "NaluEnv.h"
#include "NaluParsing.h"
#include "Realm.h"

// stk_mesh/base/fem
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldParallel.hpp>

#include <stk_mesh/base/Part.hpp>
#include <stk_mesh/base/Selector.hpp>

namespace sierra {
namespace nalu {

OversetManagerTIOGA::OversetManagerTIOGA(
  Realm& realm, const OversetUserData& oversetUserData)
  : OversetManager(realm),
    oversetUserData_(oversetUserData),
    tiogaIface_(
      *this, oversetUserData.oversetBlocks_, realm.get_coordinates_name())
{
  STK_ThrowRequireMsg(
    metaData_->spatial_dimension() == 3u, "TIOGA only supports 3-D meshes.");
}

OversetManagerTIOGA::~OversetManagerTIOGA() {}

void
OversetManagerTIOGA::setup()
{
  tiogaIface_.setup(realm_.oversetBCPartVec_);

  for (auto* part : realm_.oversetBCPartVec_)
    realm_.bcPartVec_.push_back(part);
}

void
OversetManagerTIOGA::initialize()
{
  const double timeA = NaluEnv::self().nalu_time();
  if (isInit_) {
    tiogaIface_.initialize();
    isInit_ = false;
  }
  const double timeB = NaluEnv::self().nalu_time();
  timerConnectivity_ += (timeB - timeA);
}

void
OversetManagerTIOGA::execute(const bool isDecoupled)
{
  const double timeA = NaluEnv::self().nalu_time();
  if (isInit_) {
    tiogaIface_.initialize();
    isInit_ = false;
  }

  tiogaIface_.execute(isDecoupled);

  const double timeB = NaluEnv::self().nalu_time();
  timerConnectivity_ += (timeB - timeA);

#if 0
  NaluEnv::self().naluOutputP0() 
      << "TIOGA connectivity updated: " << (timeB - timeA) << std::endl;
#endif
}

void
OversetManagerTIOGA::overset_update_fields(
  const std::vector<OversetFieldData>& fields)
{
  tiogaIface_.overset_update_fields(fields);
}

void
OversetManagerTIOGA::overset_update_field(
  stk::mesh::FieldBase* field,
  const int nrows,
  const int ncols,
  const bool doFinalSyncToDevice)
{
  tiogaIface_.overset_update_field(field, nrows, ncols, doFinalSyncToDevice);
}

} // namespace nalu
} // namespace sierra

#endif
