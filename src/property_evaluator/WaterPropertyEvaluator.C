/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#include <property_evaluator/PropertyEvaluator.h>
#include <property_evaluator/WaterPropertyEvaluator.h>
#include <FieldTypeDef.h>

#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Field.hpp>

#include <vector>

namespace sierra{
namespace nalu{

//==========================================================================
// Class Definition
//==========================================================================
// WaterDensityTPropertyEvaluator - evaluates density as a function of T
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
WaterDensityTPropertyEvaluator::WaterDensityTPropertyEvaluator(
  stk::mesh::MetaData & /* metaData */)
  : PropertyEvaluator(),
    aw_(+765.33),
    bw_(+1.8142),
    cw_(-3.5e-3)
{
  // nothing to do
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
WaterDensityTPropertyEvaluator::~WaterDensityTPropertyEvaluator()
{
  // nothing
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
double
WaterDensityTPropertyEvaluator::execute(
  double *indVarList,
  stk::mesh::Entity /*node*/)
{
  const double T = indVarList[0];
  const double rhoW = aw_ + T*(bw_ + T*cw_);
  return rhoW; // kg/m^3; T in C (converted above)
}

//==========================================================================
// Class Definition
//==========================================================================
// WaterViscosityTPropertyEvaluator - evaluates viscosity as a function of T
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
WaterViscosityTPropertyEvaluator::WaterViscosityTPropertyEvaluator(
  stk::mesh::MetaData & /* metaData */)
  : PropertyEvaluator(),
    aw_(+9.67e-2),
    bw_(-8.207e-4),
    cw_(+2.344e-6),
    dw_(-2.244e-9)
{
  // nothing to do
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
WaterViscosityTPropertyEvaluator::~WaterViscosityTPropertyEvaluator()
{
  // nothing
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
double
WaterViscosityTPropertyEvaluator::execute(
  double *indVarList,
  stk::mesh::Entity /*node*/)
{
  const double T = indVarList[0];
  const double muW = aw_ + T*(bw_ + T*(cw_ + T*dw_));
  return muW; // kg/m-s; T in K
}

//==========================================================================
// Class Definition
//==========================================================================
// WaterSpecHeatTPropertyEvaluator - evaluates Cp as a function of T
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
WaterSpecHeatTPropertyEvaluator::WaterSpecHeatTPropertyEvaluator(
  stk::mesh::MetaData & /* metaData */)
  : PropertyEvaluator(),
    aw_(28.07),
    bw_(-2.817e-1),
    cw_(+1.25e-3),
    dw_(-2.48e-6),
    ew_(+1.857e-9)
{
  // nothing to do
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
WaterSpecHeatTPropertyEvaluator::~WaterSpecHeatTPropertyEvaluator()
{
  // nothing
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
double
WaterSpecHeatTPropertyEvaluator::execute(
  double *indVarList,
  stk::mesh::Entity /*node*/)
{
  const double T = indVarList[0];
  const double cpW = (aw_ + T*(bw_ + T*(cw_ + T*(dw_ + T*ew_))))*1000.0;
  return cpW; // J/kg-K; T in K (orginal correlation provided in kJ/kg-K)
}

//==========================================================================
// Class Definition
//==========================================================================
// WaterEnthalpyTPropertyEvaluator - evaluates h as a function of T
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
WaterEnthalpyTPropertyEvaluator::WaterEnthalpyTPropertyEvaluator(
  stk::mesh::MetaData & /* metaData */)
  : PropertyEvaluator(),
    aw_(28.07),
    bw_(-2.817e-1),
    cw_(+1.25e-3),
    dw_(-2.48e-6),
    ew_(+1.857e-9),
    Tref_(300.0),
    hRef_((aw_ + Tref_*(bw_ + Tref_*(cw_ + Tref_*(dw_ + Tref_*ew_))))*1000.0*Tref_)
{
  // nothing to do
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
WaterEnthalpyTPropertyEvaluator::~WaterEnthalpyTPropertyEvaluator()
{
  // nothing
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
double
WaterEnthalpyTPropertyEvaluator::execute(
  double *indVarList,
  stk::mesh::Entity  /* node */)
{
  const double T = indVarList[0];
  const double hWT = compute_h(T);
  const double hWTRef = compute_h(Tref_);
  const double hW = hWT - hWTRef + hRef_;
  return hW;
}

//--------------------------------------------------------------------------
//-------- compute_h ---------------------------------------------------------
//--------------------------------------------------------------------------
double
WaterEnthalpyTPropertyEvaluator::compute_h(
  const double T)
{
  const double hW = T*(aw_ + T*(bw_/2.0 + T*(cw_/3.0 + T*(dw_/4.0 + T*ew_/5.0))))*1000.0;
  return hW;
}

//==========================================================================
// Class Definition
//==========================================================================
// WaterThermalCondTPropertyEvaluator - evaluates lambda as a function of T
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
WaterThermalCondTPropertyEvaluator::WaterThermalCondTPropertyEvaluator(
  stk::mesh::MetaData & /* metaData */)
  : PropertyEvaluator(),
    aw_(-0.5752),
    bw_(+6.397e-3),
    cw_(-8.151e-6)
{
  // nothing to do
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
WaterThermalCondTPropertyEvaluator::~WaterThermalCondTPropertyEvaluator()
{
  // nothing
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
double
WaterThermalCondTPropertyEvaluator::execute(
  double *indVarList,
  stk::mesh::Entity /*node*/)
{
  const double T = indVarList[0];
  const double lambdaW = aw_ + T*(bw_ + T*cw_);
  return lambdaW; // W/m-K; T in K
}

} // namespace nalu
} // namespace Sierra
