// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#ifndef FIXPRESSUREATNODEALGORITHM_H
#define FIXPRESSUREATNODEALGORITHM_H

#include "SolverAlgorithm.h"
#include "FieldTypeDef.h"
#include <stk_mesh/base/NgpMesh.hpp>

#include <stk_mesh/base/Entity.hpp>

namespace stk {
namespace mesh {
class Part;
}
}

namespace sierra {
namespace nalu {

class Realm;
struct FixPressureAtNodeInfo;

/** Set the reference pressure in the computational domain for ABL simulations.
 *
 *  This algorithm sets the reference pressure at an internal computational node
 *  that is used as a reference for the Continuity equation. Reference pressure
 *  is necessary in situations where all the boundaries are either periodic or
 *  symmetry, e.g., _precursor_ simulations where the sides are periodic and the
 *  top boundary is usually set to symmetry.
 *
 *  The reference pressure can be activated by adding the section
 *  `abl_fix_pressure` to the `solution_options` for the relevant `realm` as
 *  shown below:
 *
 *  ```
 *  solution_options:
 *    name: ablSimOptions
 *
 *    fix_pressure_at_node:
 *      node_lookup_type: spatial_location      # Use nearest node search
 *      value: 0.0                              # Reference pressure
 *      location: [10.0, 10.0, 10.0]            # Spatial location where pressure is referenced
 *      search_target_part: [Unspecified-2-HEX] # List of mesh parts to be searched
 *      search_method: stk_kdtree               # Search method to determine nearest node
 *  ```
 *
 *  Alternately, the user can specify an STK node identifier instead of
 *  performing a search for the nearest node.
 *
 *  ```
 *  solution_options:
 *    name: ablSimOptions
 *
 *    fix_pressure_at_node:
 *      node_lookup_type: stk_node_id           # Use STK node identifier
 *      node_identifier: 33662                  # STK Node ID
 *      value: 0.0                              # Reference pressure
 *  ```
 *
 * \sa FixPressureAtNodeInfo
 */
class FixPressureAtNodeAlgorithm : public SolverAlgorithm
{
public:
  FixPressureAtNodeAlgorithm(
    Realm& realm,
    stk::mesh::Part* part,
    EquationSystem* eqSystem);

  virtual ~FixPressureAtNodeAlgorithm();

  virtual void initialize_connectivity();

  virtual void execute();

  void initialize();

  /** Determine the nearest node on the mesh to the user-specified location
   *
   */
  stk::mesh::EntityId determine_nearest_node();

  /** Process the node ID and determine the node where the fix is applied */
  void process_pressure_fix_node(const stk::mesh::EntityId nodeID);

  const FixPressureAtNodeInfo& info_;

  //! Reference to the coordinates field
  VectorFieldType* coordinates_{nullptr};

  //! Reference to the pressure field
  ScalarFieldType* pressure_{nullptr};

  //! List of nodes where pressure is referenced/fixed. Should be a list of size = 1
  stk::mesh::NgpMesh::ConnectedNodes refNodeList_;
  stk::mesh::Entity targetNode_;

  //! Track mesh motion for reinitialization
  const bool meshMotion_{false};

  //! Track initialization tasks
  bool doInit_{true};

  bool fixPressureNode_{false};
};

}  // nalu
}  // sierra

#endif /* FIXPRESSUREATNODEALGORITHM_H */
