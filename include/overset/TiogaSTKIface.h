#ifndef TIOGASTKIFACE_H
#define TIOGASTKIFACE_H


#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/CoordinateSystems.hpp>

#include "yaml-cpp/yaml.h"

#include <vector>
#include <memory>
#include <array>

namespace TIOGA {
class tioga;
}

namespace sierra {
namespace nalu {

class OversetManagerTIOGA;
}
}

namespace tioga_nalu {

class TiogaBlock;

/** Nalu interface to TIOGA (Topology Independent Overset Grid Assembly)
 *
 *  This class provides a two-way data transfer interface for TIOGA library and
 *  provides overset connectivity capability for Nalu.
 */
class TiogaSTKIface
{
public:
  /**
   *  @param oversetManager Reference to Nalu OversetManager object
   *  @param node YAML node containing overset inputs
   */
  TiogaSTKIface(sierra::nalu::OversetManagerTIOGA&,
                const YAML::Node&);

  ~TiogaSTKIface();

  /** Setup block structure information (steps before mesh creation)
   */
  void setup(stk::mesh::PartVector&);

  /** Initialize mesh data structure (steps after mesh creation)
   */
  void initialize();

  /** Determine overset connectivity by calling into TIOGA API
   *
   *  This method performs several steps: updates coordinates (if necessary,
   *  during mesh motion), registers the mesh blocks to TIOGA, calculate mesh
   *  connectivity information (hole, fringe, and field point determination),
   *  update the "overset inactive part" for hole elements, create the {fringe
   *  node, donor element} mapping pair data structures for overset simulations.
   */
  void execute();

private:
  TiogaSTKIface() = delete;
  TiogaSTKIface(const TiogaSTKIface&) = delete;

  /** Process the input parameters and initialize all data structures necessary
   * to call TIOGA.
   */
  void load(const YAML::Node&);

  /** STK Custom Ghosting to transfer donor elements to receptor's MPI rank
   *
   *  This method declares the STK custom ghosting object, actual updates are
   *  performed by the update_ghosting method.
   */
  void initialize_ghosting();

  /** Ghost donor elements to receptor MPI ranks
   */
  void update_ghosting();

  /** \deprecated{Populate the {fringe node, donor element} pair data structure}
   *
   *  \sa get_receptor_info, populate_overset_info
   */
  void update_fringe_info();

  /** Update the inactive part with hole elements
   */
  void populate_inactive_part();

  /** Reset all connectivity data structures when recomputing connectivity
   */
  void reset_data_structures();

  /** Gather receptor-donor pair information
   *
   *  Extract the set of fringe nodes on participating meshes and prepare for
   *  reconciliation for shared nodes.
   */
  void get_receptor_info();

  /** Populate the overset info data structure
   *
   *  Reconcile shared/owned fringe status and populate the overset fringe
   *  vector for later use by algorithms.
   */
  void populate_overset_info();

  //! Reference to Nalu OversetManager object
  sierra::nalu::OversetManagerTIOGA& oversetManager_;

  //! Reference to the STK MetaData object
  stk::mesh::MetaData& meta_;

  //! Reference to the STK BulkData object
  stk::mesh::BulkData& bulk_;

  //! List of TIOGA data structures for each mesh block participating in overset
  //! connectivity
  std::vector<std::unique_ptr<TiogaBlock>> blocks_;

  //! Reference to the TIOGA API interface
  std::unique_ptr<TIOGA::tioga> tg_;

  //! Work array used to hold donor elements that require ghosting to receptor
  //! MPI ranks
  stk::mesh::EntityProcVec elemsToGhost_;

  //! Name of part holding hole elements
  std::string inactivePartName_;

  //! List of hole elements
  std::vector<stk::mesh::Entity> holeElems_;

  //! List of fringe elements
  std::vector<stk::mesh::Entity> fringeElems_;

  //! List of receptor nodes that are shared entities across MPI ranks. This
  //! information is used to synchronize the field vs. fringe point status for
  //! these shared nodes across processor boundaries.
  std::vector<stk::mesh::EntityId> receptorIDs_;

  //! Donor elements corresponding to TiogaSTKIface::receptorIDs_ that must be
  //! ghosted to another MPI rank to ensure that owned and shared nodes are
  //! consistent.
  std::vector<stk::mesh::EntityId> donorIDs_;

  //! Set the symmetry direction for TIOGA, default is z-direction (3)
  int symmetryDir_{3};

  //! User option to determine whether inactive part is generated with TIOGA
  bool populateInactivePart_{true};
};


}  // tioga

#endif /* TIOGASTKIFACE_H */
