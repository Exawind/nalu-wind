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

#include <NaluEnv.h>
#include <NaluParsing.h>
#include <Realm.h>
#include <master_element/MasterElement.h>

// stk_mesh/base/fem
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldParallel.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/Part.hpp>
#include <stk_mesh/base/Selector.hpp>


namespace sierra {
namespace nalu {

OversetManagerTIOGA::OversetManagerTIOGA(
  Realm& realm,
  const OversetUserData& oversetUserData)
  : OversetManager(realm),
    oversetUserData_(oversetUserData),
    tiogaIface_(*this, oversetUserData.oversetBlocks_)
{
  ThrowRequireMsg(
    metaData_->spatial_dimension() == 3u,
    "TIOGA only supports 3-D meshes.");
}

OversetManagerTIOGA::~OversetManagerTIOGA()
{}

void
OversetManagerTIOGA::setup()
{
  tiogaIface_.setup(realm_.bcPartVec_);
}

void
OversetManagerTIOGA::initialize(const bool isDecoupled)
{
  const double timeA = NaluEnv::self().nalu_time();
  if (isInit_) {
    tiogaIface_.initialize();
    isInit_ = false;
  }

  delete_info_vec();
  oversetInfoVec_.clear();
  holeNodes_.clear();
  fringeNodes_.clear();

  tiogaIface_.execute(isDecoupled);

  const double timeB = NaluEnv::self().nalu_time();
  realm_.timerNonconformal_ += (timeB - timeA);

#if 0
  NaluEnv::self().naluOutputP0() 
      << "TIOGA connectivity updated: " << (timeB - timeA) << std::endl;
#endif
}

void OversetManagerTIOGA::overset_update_fields(const std::vector<OversetFieldData>& fields)
{
  tiogaIface_.overset_update_fields(fields);
}

void OversetManagerTIOGA::overset_update_field(
  stk::mesh::FieldBase *field, int nrows, int ncols)
{
  tiogaIface_.overset_update_field(field, nrows, ncols);
}


}  // nalu
}  // sierra

#endif
