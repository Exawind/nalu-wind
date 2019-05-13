/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level nalu      */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#ifndef PromotedPartHelper_h
#define PromotedPartHelper_h

#include <vector>
#include <string>

#include <stk_topology/topology.hpp>

namespace stk {
  namespace mesh {
    class MetaData;
    class Part;
    class Bucket;
    typedef std::vector<Part*> PartVector;
    typedef std::vector<Bucket*> BucketVector;
  }
}

namespace sierra {
namespace nalu {

  bool part_vector_is_valid_and_nonempty(const stk::mesh::PartVector& parts);

  bool check_parts_for_promotion(const stk::mesh::PartVector& parts);

  std::string super_element_suffix();

  std::string super_element_part_name(std::string base_name);

  std::string super_subset_part_name(std::string base_name, int numElemNodes, int numSideNodes);
  std::string super_subset_part_name(std::string base_name);

  stk::mesh::Part* super_elem_part(const stk::mesh::Part& part);

  stk::mesh::Part* super_subset_part(const stk::mesh::Part& part, int numElemNodes, int numSideNodes);
  stk::mesh::Part* super_subset_part(const stk::mesh::Part& part);

  void transform_to_super_elem_part_vector(stk::mesh::PartVector& parts);

  stk::mesh::PartVector super_elem_part_vector(const stk::mesh::PartVector& parts);

  stk::mesh::PartVector base_ranked_parts(
    const stk::mesh::PartVector& parts,
    stk::topology::rank_t rank,
    bool with_subsets = true);

  stk::mesh::PartVector base_elem_parts(const stk::mesh::PartVector& parts);
  stk::mesh::PartVector base_edge_parts(const stk::mesh::PartVector& parts);
  stk::mesh::PartVector base_face_parts(const stk::mesh::PartVector& parts);

  stk::mesh::Part* base_elem_part_from_super_elem_part(const stk::mesh::Part& super_elem_part);

  stk::mesh::PartVector only_super_parts(const stk::mesh::PartVector& parts);

  stk::mesh::PartVector only_super_elem_parts(const stk::mesh::PartVector& parts);

  stk::mesh::PartVector only_super_side_parts(const stk::mesh::PartVector& parts);

  stk::mesh::PartVector append_super_elems_to_part_vector(stk::mesh::PartVector parts);

  size_t count_entities(const stk::mesh::BucketVector& buckets);

  stk::topology get_promoted_elem_topo(int dim, int order);

} // namespace nalu
} // namespace Sierra

#endif
