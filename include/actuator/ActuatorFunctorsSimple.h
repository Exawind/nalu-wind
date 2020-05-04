// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef ACTUATORFUNCTORSSIMPLE_H_
#define ACTUATORFUNCTORSSIMPLE_H_

#include <actuator/ActuatorTypes.h>
#include <actuator/ActuatorBulkSimple.h>
#include <actuator/ActuatorFunctors.h>
#include <NaluParsedTypes.h>
#include <NaluEnv.h>

namespace sierra {
namespace nalu {

  struct InterpActuatorDensity
  {
    using execution_space = ActuatorFixedExecutionSpace;

    InterpActuatorDensity(ActuatorBulk& actBulk, stk::mesh::BulkData& stkBulk);

    void operator()(int index) const;

    ActuatorBulk& actBulk_;
    stk::mesh::BulkData& stkBulk_;
    VectorFieldType* coordinates_;
    ScalarFieldType* density_;
  };

  // Things to calculate lift, drag, and AOA based on 2D airfoil theory
  namespace AirfoilTheory2D {
    // Can use this namespace to add functions relevant to 2D airfoil
    // theory
    void calculate_alpha(
      double ws[],                 
      const double zeroalphadir[], 
      const double spandir[],      
      const double chordnormaldir[], 
      double twist, 
      double ws2Da[],   
      double &alpha);
  }

struct ActSimpleUpdatePoints
{
  using execution_space = ActuatorFixedExecutionSpace;

  ActSimpleUpdatePoints(ActuatorBulkSimple& actBulk, 
			int numpoints);
  void operator()(int index) const;

  ActDualViewHelper<ActuatorFixedMemSpace> helper_;
  ActFixVectorDbl points_;
  ActFixScalarInt offsets_;
  const int turbId_;
  const int numpoints_;
};

struct ActSimpleAssignVel
{
  using execution_space = ActuatorFixedExecutionSpace;

  ActSimpleAssignVel(ActuatorBulkSimple& actBulk);
  void operator()(int index) const;

  ActDualViewHelper<ActuatorFixedMemSpace> helper_;
  ActFixVectorDbl velocity_;
  ActFixScalarDbl density_;
  ActFixVectorDbl points_;
  ActFixScalarInt offset_;
  const int debug_output_;
  const int turbId_;
  std::vector<double> p1_;
  std::vector<double> p2_;
};

struct ActSimpleComputeForce
{
  using execution_space = ActuatorFixedExecutionSpace;

  ActSimpleComputeForce(ActuatorBulkSimple& actBulk, 
			const ActuatorMetaSimple& actMeta);
  void operator()(int index) const;

  ActDualViewHelper<ActuatorFixedMemSpace> helper_;
  ActFixVectorDbl velocity_;
  ActFixScalarDbl density_;
  ActFixVectorDbl force_;
  ActFixScalarInt offset_;
  const int turbId_;
    
  // Dual view polar tables and blade definitions
  ActScalarIntDv   polartable_size_;
  const int        Npolartable;
  ActScalarDblDv   aoa_polartableDv_;
  ActScalarDblDv   cl_polartableDv_;
  ActScalarDblDv   cd_polartableDv_;
  const int        Npts;
  ActScalarDblDv   twist_tableDv_;
  ActScalarDblDv   elem_areaDv_;

  double p1zeroalphadir[3];         // Directon of zero alpha at p1
  double chordnormaldir[3];         // Direction normal to chord
  double spandir[3];                // Direction in the span

  const int debug_output_;

};

struct ActSimpleSetUpThrustCalc
{
  using execution_space = ActuatorFixedExecutionSpace;

  ActSimpleSetUpThrustCalc(ActuatorBulkSimple& actBulk);

  void operator()(int index) const;

  ActuatorBulkSimple& actBulk_;
};

struct ActSimpleComputeThrustInnerLoop
{

  ActSimpleComputeThrustInnerLoop(ActuatorBulkSimple& actBulk) : actBulk_(actBulk)
  {
  }
  void operator()(
    const uint64_t pointId,
    const double* nodeCoords,
    double* sourceTerm,
    const double dualNodalVolume,
    const double scvIp) const;
  void preloop() {}

  ActuatorBulkSimple& actBulk_;
};

struct ActSimpleSpreadForceWhProjInnerLoop
{
  ActSimpleSpreadForceWhProjInnerLoop(ActuatorBulkSimple& actBulk)
    : actBulk_(actBulk)
  {
  }

  void operator()(
    const uint64_t pointId,
    const double* nodeCoords,
    double* sourceTerm,
    const double dualNodalVolume,
    const double scvIp) const;
  void preloop();

  ActuatorBulkSimple& actBulk_;
};

using ActSimpleComputeThrust = GenericLoopOverCoarseSearchResults<
  ActuatorBulkSimple,
  ActSimpleComputeThrustInnerLoop>;

using ActSimpleSpreadForceWhProjection = GenericLoopOverCoarseSearchResults<
  ActuatorBulkSimple,
  ActSimpleSpreadForceWhProjInnerLoop>;

} /* namespace nalu */
} /* namespace sierra */

#endif /* ACTUATORFUNCTORSSIMPLE_H_ */
