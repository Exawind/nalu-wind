/*------------------------------------------------------------------------*/
/*  Copyright 2014 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

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
OversetManagerTIOGA::initialize()
{
  const double timeA = NaluEnv::self().nalu_time();
  if (isInit_) {
    tiogaIface_.initialize();
    isInit_ = false;
  }

  delete_info_vec();
  oversetInfoVec_.clear();
  holeNodes_.clear();

  tiogaIface_.execute();

  const double timeB = NaluEnv::self().nalu_time();
  realm_.timerNonconformal_ += (timeB - timeA);

#if 0
  NaluEnv::self().naluOutputP0() 
      << "TIOGA connectivity updated: " << (timeB - timeA) << std::endl;
#endif
}

}  // nalu
}  // sierra

#endif
