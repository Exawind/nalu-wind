// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



#include <property_evaluator/MaterialPropertyData.h>

#include <Enums.h>

namespace sierra{
namespace nalu{

//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
MaterialPropertyData::MaterialPropertyData()
  : type_(MaterialPropertyType_END),
    constValue_(0.0),
    primary_(0.0),
    secondary_(0.0),
    auxVarName_("na"),
    tablePropName_("na"),
    tableAuxVarName_("na"),
    genericPropertyEvaluatorName_("na")
{
  // does nothing
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
MaterialPropertyData::~MaterialPropertyData()
{
  // nothing
}

} // namespace nalu
} // namespace Sierra
