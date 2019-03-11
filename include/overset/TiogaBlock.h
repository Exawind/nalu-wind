#ifndef TIOGABLOCK_H
#define TIOGABLOCK_H

#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/CoordinateSystems.hpp>

#include "yaml-cpp/yaml.h"

#include <vector>
#include <memory>
#include <string>

namespace TIOGA {
class tioga;
}

namespace tioga_nalu {

typedef stk::mesh::Field<double, stk::mesh::Cartesian> VectorFieldType;
typedef stk::mesh::Field<int> ScalarIntFieldType;

/**
 * Interface to convert STK Mesh Part(s) to TIOGA blocks.
 *
 * This class provides a mapping between STK mesh parts and the concept of a
 * TIOGA mesh block. Each TIOGA mesh block is determined by a unique body tag
 * and requires information of all the nodes and elements comprising the mesh
 * block (within this MPI rank). TIOGA determines the global mesh information by
 * looking up the unique body tag across all MPI ranks. TIOGA requires
 * information regarding the volume mesh as well as the wall and overset
 * surfaces.
 *
 * TIOGA communicates overset connectivity via IBLANK (node) and IBLANK_CELL
 * (element) masking arrays that have flags indicating whether a node/element is
 * a hole, fringe, or a field point.
 */
class TiogaBlock
{
public:
  TiogaBlock(stk::mesh::MetaData&,
             stk::mesh::BulkData&,
             const YAML::Node&,
             const std::string,
             const int);

  ~TiogaBlock();

  /** Setup block structure information (steps before mesh creation)
   */
  void setup(stk::mesh::PartVector&);

  /** Initialize mesh data structure (steps after mesh creation)
   */
  void initialize();

  /** Update coordinates upon mesh motion
   *
   *  Update the coordinates sent to TIOGA from STK. This assumes that the mesh
   *  connectivity information itself does not change, i.e., no refinement, etc.
   *
   *  Updates to mesh connectivity information will require a call to
   *  TiogaBlock::update_connectivity() instead.
   */
  void update_coords();

  /** Perform full update including connectivity
   *
   */
  void update_connectivity();

  /** Register this block with TIOGA
   *
   *  Wrapper method to handle mesh block registration using TIOGA API. In
   *  addition to registering the mesh block, it will also provide IBLANK_CELL
   *  data structure for TIOGA to populate.
   *
   *  The interface also allows registration of user-defined node and cell
   *  resolution information that enables the user to force a certain type of
   *  overset holecutting that overrides the default TIOGA behavior of selecting
   *  donor and receptor points based on local cell volume.
   */
  void register_block(TIOGA::tioga&);

  /** Update iblanks after connectivity updates
   */
  void update_iblanks(std::vector<stk::mesh::Entity>&);

  /** Update element iblanks after connectivity updates
   */
  void update_iblank_cell();

  /** Determine the custom ghosting elements for this mesh block
   *
   *  Calls the TIOGA API and populates the elements that need ghosting to other
   *  MPI ranks.
   *
   *  @param tg Reference to TIOGA API object (provided by TiogaSTKIface).
   *  @param egvec List of {donorElement, receptorMPIRank} pairs to be populated
   */
  void get_donor_info(TIOGA::tioga&, stk::mesh::EntityProcVec&);

  // Accessors

  //! STK Global ID for all the nodes comprising this mesh block
  inline const std::vector<stk::mesh::EntityId>& node_id_map() const
  { return nodeid_map_; }

  //! STK Global ID for all the elements comprising this mesh block
  inline const std::vector<stk::mesh::EntityId>& elem_id_map() const
  { return elemid_map_; }

  //! IBLANK mask indicating whether the element is active or inactive
  inline const std::vector<int>& iblank_cell() const
  { return iblank_cell_; }

  //! Return the block name for this mesh
  const std::string& block_name() const { return block_name_; }

private:
  TiogaBlock() = delete;
  TiogaBlock(const TiogaBlock&) = delete;

  /** Process the YAML node to gather all user inputs
   */
  void load(const YAML::Node&);

  /** Convenience function to process part names and populate a PartVector
   */
  inline void names_to_parts(
    const std::vector<std::string>&,
    stk::mesh::PartVector&);

  /**
   * Extract nodes from all parts to send to TIOGA
   */
  void process_nodes();

  /** Determine the local indices (into the TIOGA mesh block data structure) of
   * all the wall boundary nodes.
   */
  void process_wallbc();

  /** Determine the local indices (into the TIOGA mesh block data structure) of
   *  all the overset boundary nodes.
   */
  void process_ovsetbc();

  /** Generate the element data structure and connectivity information to send to TIOGA
   */
  void process_elements();

  /** Reset iblank data with moving mesh applications
   *
   */
  void reset_iblank_data();

  /** Print summary of mesh blocks
   */
  void print_summary();

  //! Reference to the STK Mesh MetaData object
  stk::mesh::MetaData& meta_;

  //! Reference to the STK Mesh BulkData object
  stk::mesh::BulkData& bulk_;

  //! Part names for the nodes for this mesh block
  std::vector<std::string> blkNames_;

  //! Part names for the wall boundaries
  std::vector<std::string> wallNames_;

  //! Part names for the overset boundaries
  std::vector<std::string> ovsetNames_;

  //! Mesh parts for the nodes
  stk::mesh::PartVector blkParts_;

  //! Wall BC parts
  stk::mesh::PartVector wallParts_;

  //! Overset BC parts
  stk::mesh::PartVector ovsetParts_;

  //! Coordinates for this mesh block in TIOGA format
  std::vector<double> xyz_;

  //! Node IBLANK information from TIOGA
  std::vector<int> iblank_;

  //! Element IBLANK information from TIOGA
  std::vector<int> iblank_cell_;

  /** Lookup table for node ID to local index in xyz_ array
   *
   *  Used to populate the wall and overset BC arrays as well as the overset
   *  connectivity information.
   */
  std::map<stk::mesh::EntityId, size_t> node_map_;

  //! Wall BC index array
  std::vector<int> wallIDs_;

  //! Overset BC index array
  std::vector<int> ovsetIDs_;

  //! Number of vertices per topology type found
  std::vector<int> num_verts_;

  //! Number of cells per topology in this mesh
  std::vector<int> num_cells_;

  /** Connectivity data structure
   *
   *  A two-dimensional list of size (num_elems, num_elems * nodes_per_elem)
   */
  std::vector<std::vector<int>> connect_;

  /** Connectivity map.
   *
   *  This map holds the number of elements present per topology type (npe ->
   *  num_elements).
   */
  std::map<int, int> conn_map_;

  /** Tioga connectivity data structure
   *
   */
  int** tioga_conn_{nullptr};

  //! STK Global ID for Nodes
  std::vector<stk::mesh::EntityId> nodeid_map_;

  //! STK Global ID for elements
  std::vector<stk::mesh::EntityId> elemid_map_;

  //! Receptor information for this mesh block
  std::vector<int> receptor_info_;

  //! User-specified node resolution
  std::vector<double> node_res_;

  //! User-specified cell resolution
  std::vector<double> cell_res_;

  //! Name of coordinates Field
  std::string coords_name_;

  //! Dimensionality of the mesh
  int ndim_;

  //! Global mesh tag identifier
  int meshtag_;

  //! Name of this overset mesh block
  std::string block_name_;

  //! Number of nodes for this mesh
  int num_nodes_{0};

  //! Number of wall BC nodes (in this processor)
  int num_wallbc_{0};

  //! Number of overset BC nodes (in this processor)
  int num_ovsetbc_{0};

  //! Flag to check if we are are already initialized
  bool is_init_ { true };

};

} // namespace tioga

#endif /* TIOGABLOCK_H */
