/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

/** @file ActuatorFAST.h
 *  @brief A class to couple Nalu with OpenFAST for actuator simulations of wind
 * turbines
 *
 */

#ifndef ActuatorFAST_h
#define ActuatorFAST_h

#include <stk_util/parallel/ParallelVectorConcat.hpp>
#include "Actuator.h"

// OpenFAST C++ API
#include "OpenFAST.H"

namespace sierra {
namespace nalu {

class Realm;

/** Class that holds all of the information relevant to each turbine
 *
 *
 */
//
class ActuatorFASTInfo : public ActuatorInfo
{
public:
  ActuatorFASTInfo();
  virtual ~ActuatorFASTInfo();
  Coordinates epsilon_; ///< The Gaussian spreading width in (chordwise,
                        ///< spanwise, thickness) directions
  std::string fileToDumpPoints_;
};

/** Class that holds all of the search action for each actuator point
 *
 *
 */
//
class ActuatorFASTPointInfo : public ActuatorPointInfo
{
public:
  ActuatorFASTPointInfo(
    size_t globTurbId,
    Point centroidCoords,
    double searchRadius,
    Coordinates epsilon,
    fast::ActuatorNodeType nType,
    int forceInd);
  virtual ~ActuatorFASTPointInfo();
  size_t globTurbId_; ///< Global turbine number.
  Coordinates
    epsilon_; ///< The Gaussian spreading width in (chordwise, spanwise,
              ///< thickness) directions for this actuator point.
  fast::ActuatorNodeType
    nodeType_;        ///< HUB, BLADE, or TOWER - Defined by an enum.
  int forcePntIndex_; ///< The index this point resides in the total number of
                      ///< force points for the tower i.e. i \in
                      ///< [0,numForcePnts-1]
};

/** The ActuatorFAST class couples Nalu with the third party library OpenFAST
 for actuator simulations of wind turbines
 *
 * OpenFAST (https://nwtc.nrel.gov/FAST) available from
 https://github.com/OpenFAST/openfast is
 * a aero-hydro-servo-elastic tool to model wind turbine developed by the
 * National Renewable Energy Laboratory (NREL). The ActuatorFAST class will help
 Nalu
 * effectively act as an inflow module to OpenFAST by supplying the velocity
 field information.
 * The effect of the turbine on the flow field is modeled using the actuator
 approach.
 * The force exerted by the wind turbine on the flow field is lumpled into a set
 of body forces
 * at a discrete set of actuator points. This class spreads the the body force
 at each actuator
 * point using a Gaussian function.

 * 1) During the load phase - the turbine data from the yaml file is read and
 stored in an
 *    object of the ``fast::fastInputs`` class

 * 2) During the initialize phase - The processor containing the hub of each
 turbine is found
 *    through a search and assigned to be the one controlling OpenFAST for that
 turbine. All
 *    processors controlling > 0 turbines initialize OpenFAST, populate the map
 of ``ActuatorPointInfo``
 *    and initialize element searches for all the actuator points associated
 with the turbines. For every actuator point,
 *    the elements within a specified search radius are found and stored in the
 corresponding object of the
 *    ``ActuatorPointInfo`` class.
 *
 * 3) Elements are ghosted to the owning point rank. We tried the opposite
 approach of
 *    ghosting the actuator points to the processor owning the elements. The
 second approach
 *    was found to perform poorly compared to the first method.
 *
 * 4) A time lagged simple FSI model is used to interface Nalu with the turbine
 model:
 *    + The velocity at time step at time step 'n' is sampled at the actuator
 points and sent
 *       to OpenFAST
 *    + OpenFAST advances the turbines upto the next Nalu time step 'n+1'
 *    + The body forces at the actuator points are converted to the source terms
 of the momentum
 *      equation to advance Nalu to the next time step 'n+1'.
 *
 * 5) During the execute phase called every time step, we sample the velocity at
 each actuator
 *    point and pass it to OpenFAST. All the OpenFAST turbine models are
 advanced upto Nalu's
 *    next time step to get the body forces at the actuator points. We then
 iterate over the
 *    ``ActuatorPointInfoMap`` to assemble source terms. For each node
 \f$n\f$within the
 *    search radius of an actuator point \f$k\f$, the
 ``spread_actuator_force_to_node_vec``
 *    function calculates the effective lumped body force by multiplying the
 actuator force
 *    with the Gaussian projection at the node as \f$F_i^n = g(\vec{r}_i^n) \,
 F_i^k\f$.
 *
 *
*/

class ActuatorFAST : public Actuator
{
public:
  ActuatorFAST(Realm& realm, const YAML::Node& node);
  virtual ~ActuatorFAST();

  // load all of the options
  void load(const YAML::Node& node) override;

  // load the options for each turbine
  void readTurbineData(int iTurb, fast::fastInputs& fi, YAML::Node turbNode);

  // setup part creation and nodal field registration (before populate_mesh())
  void setup() override;

  // allocate turbines to processors containing hub location
  void allocateTurbinesToProcs();

  // Allocate turbines to processors, initialize FAST and get location of
  // actuator points
  void initialize() override;

  // setup part creation and nodal field registration (after populate_mesh())
  void update();

  // split search operations for disk and line
  virtual void update_class_specific() = 0;

  // determine processor bounding box in the mesh
  void populate_candidate_procs();

  // fill in the map that will hold point and ghosted elements
  void create_actuator_point_info_map();

  virtual void create_point_info_map_class_specific() = 0;

  // populate nodal field and output norms (if appropriate)
  void execute() override;

  virtual void execute_class_specific(
    const int nDim,
    const stk::mesh::FieldBase* coordinates,
    stk::mesh::FieldBase* actuator_source,
    const stk::mesh::FieldBase* dual_nodal_volume) = 0;

  // centroid of the element
  void compute_elem_centroid(
    const int& nDim, double* elemCentroid, const int& nodesPerElement);

  // compute the body force at an element given a
  // projection weighting.
  void compute_node_force_given_weight(
    const int& nDim,
    const double& g,
    const double* pointForce,
    double* nodeForce);

  // isotropic Gaussian projection function.
  double isotropic_Gaussian_projection(
    const int& nDim, const double& dis, const Coordinates& epsilon);

  // Spread the actuator force to a node vector
  void spread_actuator_force_to_node_vec(
    const int& nDim,
    std::set<stk::mesh::Entity>& nodeVec,
    const std::vector<double>& actuator_force,
    const double* actuator_node_coordinates,
    const stk::mesh::FieldBase& coordinates,
    stk::mesh::FieldBase& actuator_source,
    const stk::mesh::FieldBase& dual_nodal_volume,
    const Coordinates& epsilon,
    const std::vector<double>& hubPt,
    const std::vector<double>& hubShftDir,
    std::vector<double>& thr,
    std::vector<double>& tor);

  void add_thrust_torque_contrib(
    const int& nDim,
    const double* nodeCoords,
    const double dVol,
    const std::vector<double>& nodeForce,
    const std::vector<double>& hubPt,
    const std::vector<double>& hubShftDir,
    std::vector<double>& thr,
    std::vector<double>& tor);

  std::string
  write_turbine_points_to_string(std::size_t turbNum, int width, int precision);

  void dump_turbine_points_to_file(std::size_t turbNum);

  int tStepRatio_; ///< Ratio of Nalu time step to FAST time step
                   ///< (dtNalu/dtFAST) - Should be an integral number

  // bounding box data types for stk_search
  std::vector<boundingSphere>
    boundingHubSphereVec_; ///< bounding box around the hub point of each
                           ///< turbine
  std::vector<boundingElementBox>
    boundingProcBoxVec_; ///< bounding box around all the nodes residing locally
                         ///< on each processor

  virtual std::string get_class_name() override = 0;

  fast::fastInputs fi; ///< Object to hold input information for OpenFAST
  fast::OpenFAST FAST; ///< OpenFAST C++ API handle

  std::size_t numFastPoints_;

  std::vector<std::vector<double>> thrust;
  std::vector<std::vector<double>> torque;
};

} // namespace nalu
} // namespace sierra

#endif
