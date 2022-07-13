// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



#include <Realm.h>
#include <BoundaryConditions.h>
#include <NaluEnv.h>

// yaml for parsing..
#include <yaml-cpp/yaml.h>
#include <NaluParsing.h>

namespace sierra{
namespace nalu{

//==========================================================================
// Class Definition
//==========================================================================
// BoundaryCondition - do some stuff
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
  
//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------

//--------------------------------------------------------------------------
//-------- load -----------------------------------------------
//--------------------------------------------------------------------------


/// this is an example of a load() method with polymorphism - the type of
/// the node is determined from some information, then a particular type
/// of object is created and returned to the parent.

BoundaryCondition * BoundaryCondition::load(const YAML::Node & node) 
{
  if ( node["wall_boundary_condition"] ){
    WallBoundaryConditionData& wallBC = *new WallBoundaryConditionData(*parent());
    node >> wallBC;
    NaluEnv::self().naluOutputP0() << "Wall BC name:        " << wallBC.bcName_
                    << " on " << wallBC.targetName_ << std::endl;
    return &wallBC;
  }
  else if (node["inflow_boundary_condition"]) {
    InflowBoundaryConditionData& inflowBC = *new InflowBoundaryConditionData(*parent());
    node >> inflowBC;
    NaluEnv::self().naluOutputP0() << "Inflow BC name:      " << inflowBC.bcName_
                    << " on " << inflowBC.targetName_ << std::endl;
    return &inflowBC;
  }
  else if (node["open_boundary_condition"]) {
    OpenBoundaryConditionData& openBC = *new OpenBoundaryConditionData(*parent());
    node >> openBC;
    NaluEnv::self().naluOutputP0() << "Open BC name:        " << openBC.bcName_
                    << " on " << openBC.targetName_ << std::endl;
    return &openBC;
  }
  else if (node["symmetry_boundary_condition"]) {
    SymmetryBoundaryConditionData& symmetryBC = *new SymmetryBoundaryConditionData(*parent());
    node >> symmetryBC;
    NaluEnv::self().naluOutputP0() << "Symmetry BC name:    " << symmetryBC.bcName_
                    << " on " << symmetryBC.targetName_ << std::endl;
    return &symmetryBC;
  }
  else if (node["abltop_boundary_condition"]) {
    ABLTopBoundaryConditionData& abltopBC = *new ABLTopBoundaryConditionData(*parent());
    node >> abltopBC;
    NaluEnv::self().naluOutputP0() << "ABLTop BC name:    " << abltopBC.bcName_
                    << " on " << abltopBC.targetName_ << std::endl;
    return &abltopBC;
  }
  else if (node["periodic_boundary_condition"]) {
    PeriodicBoundaryConditionData& periodicBC = *new PeriodicBoundaryConditionData(*parent());
    node >> periodicBC;
    NaluEnv::self().naluOutputP0() << "Periodic BC name:    " << periodicBC.bcName_
                    << " between " << periodicBC.masterSlave_.master_
                    << " and "<< periodicBC.masterSlave_.slave_ << std::endl;
    return &periodicBC;
  }
  else if (node["non_conformal_boundary_condition"]) {
    NonConformalBoundaryConditionData& nonConformalBC = *new NonConformalBoundaryConditionData(*parent());
    node >> nonConformalBC;
    NaluEnv::self().naluOutputP0() << "NonConformal BC name:    " << nonConformalBC.bcName_
                    << " using " << nonConformalBC.targetName_ << std::endl;
    return &nonConformalBC;
  }
  else if (node["overset_boundary_condition"]) {
    OversetBoundaryConditionData& oversetBC = *new OversetBoundaryConditionData(*parent());
    node >> oversetBC;
    NaluEnv::self().naluOutputP0() << "Overset BC name: " << oversetBC.bcName_ << std::endl;
    return &oversetBC;
  } 
  else {
    throw std::runtime_error("parser error BoundaryConditions::load: no such bc type");
  }
// Avoid nvcc unreachable statement warnings
#ifndef __CUDACC__
  return 0;
#endif
}

  Simulation* BoundaryCondition::root() { return parent()->root(); }
  BoundaryConditions *BoundaryCondition::parent() { return &boundaryConditions_; }

  Simulation* BoundaryConditions::root() { return parent()->root(); }
  Realm *BoundaryConditions::parent() { return &realm_; }

BoundaryConditions* BoundaryConditions::load(const YAML::Node &node)
{
  BoundaryCondition tmp_boundary_condition(*this);

  if(node["boundary_conditions"]) {
    const YAML::Node boundary_conditions = node["boundary_conditions"];
    for ( size_t iboundary_condition = 0; iboundary_condition < boundary_conditions.size(); ++iboundary_condition ) {
      const YAML::Node boundary_condition_node = boundary_conditions[iboundary_condition];
      BoundaryCondition* bc = tmp_boundary_condition.load(boundary_condition_node);
      boundaryConditionVector_.push_back(bc);
    }
  }
  else {
    throw std::runtime_error("parser error BoundaryConditions::load");
  }

  return this;
}


} // namespace nalu
} // namespace Sierra
