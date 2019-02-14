/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#include <element_promotion/PromotedPartHelper.h>
#include <NaluEnv.h>

#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Part.hpp>
#include <stk_util/util/ReportHandler.hpp>
#include <stk_topology/topology.hpp>


#include <algorithm>
#include <vector>
#include <string>

namespace sierra{
namespace nalu{

  // A set of functions to deal with part pairs having specific ending tags
  bool part_vector_is_valid_and_nonempty(const stk::mesh::PartVector& parts) {
    return (std::find(parts.begin(), parts.end(), nullptr) == parts.end() && !parts.empty() );
  }
  //--------------------------------------------------------------------------
  bool check_part_topo(const stk::mesh::Part& part) {
    const int dim = part.mesh_meta_data().spatial_dimension();
    bool is_valid_elem_rank = false;
    const stk::topology topo = part.topology();
    if (topo.rank() == stk::topology::ELEM_RANK) {
      is_valid_elem_rank = (dim == 2) ? topo == stk::topology::QUAD_4_2D : topo == stk::topology::HEX_8;
    }

    bool is_valid_side_rank = false;
    if (topo.rank() == part.mesh_meta_data().side_rank()) {
      is_valid_side_rank = (dim == 2) ? topo == stk::topology::LINE_2 : topo == stk::topology::QUAD_4;
    }

    if (!(is_valid_side_rank || is_valid_elem_rank)) {
      NaluEnv::self().naluOutputP0()
                  << "Part "  << part.name()
                  << " has an invalid topology for promotion, " << topo.name()
                  << "---only pure Hex/Quad meshes are currently supported." << std::endl;

      return false;
    }
    return true;
  }
  //--------------------------------------------------------------------------
  bool check_parts_for_promotion(const stk::mesh::PartVector& parts)
  {
    for (const auto* ip : parts) {
      ThrowRequireMsg(ip != nullptr, "An invalid part was designated for promotion.");

      if (ip->topology().rank() == stk::topology::ELEM_RANK) {
        ThrowRequireMsg(super_elem_part(*ip) != nullptr, "No super element mirror for part requesting promotion");
      }
      const stk::mesh::Part& part = *ip;
      if (part.subsets().empty()) {
        if (!check_part_topo(part)) {
          return false;
        }
      }
      else {
        for (const auto* is : part.subsets()) {
          ThrowRequireMsg(is != nullptr, "An invalid subset part was designated for promotion.");
          ThrowRequireMsg(super_subset_part(*is) != nullptr, "A subset part lacks its super side mirror.");
          const stk::mesh::Part& subsetPart = *is;
          if (!check_part_topo(subsetPart)) {
            return false;
          }
        }
      }
    }
    return true;
  }
  //--------------------------------------------------------------------------
  std::string super_element_suffix()
  {
    return "_se";
  }
  //--------------------------------------------------------------------------
  std::string super_element_part_name(std::string base_name)
  {
    ThrowAssertMsg(!base_name.empty(), "Empty base name for super elem part");
    return (base_name + super_element_suffix());
  }
  //--------------------------------------------------------------------------
  std::string super_subset_part_name(std::string base_name)
  {
    // subsetted part name.  Goes like "surfacese_super_superside_1"
    // Ioss doesn't recognize "superside" but does recognize the "super" tag

    // Note: there's a 32 character limit on the maximum length of the
    // part name (Ioss throws a warning message), so economy on characters is good

    ThrowAssertMsg(!base_name.empty(), "Empty base name for super elem part");
    auto first_token = base_name.substr(0, base_name.find_first_of('_'));
    auto last_token = base_name.substr(base_name.find_last_of("_"),base_name.length());
    std::string name = first_token + super_element_suffix()
        + "_super_superside" + last_token;

    return name;
  }
  std::string super_subset_part_name(std::string base_name, int  /* numElemNodes */, int  /* numSideNodes */)
  {
    // subsetted part name.  Goes like "surfacese_super512_superside64_1"
    // Ioss doesn't recognize "superside" but does recognize the "super" tag

    // Note: there's a 32 character limit on the maximum length of the
    // part name (Ioss throws a warning message), so economy on characters is good

//    ThrowAssertMsg(!base_name.empty(), "Empty base name for super elem part");
//    auto first_token = base_name.substr(0, base_name.find_first_of('_'));
//    auto last_token = base_name.substr(base_name.find_last_of("_"),base_name.length());
//    std::string name = first_token + super_element_suffix()
//        + "_super"  + std::to_string(numElemNodes)
//        + "_superside" + std::to_string(numSideNodes)
//        + last_token;


    ThrowAssertMsg(!base_name.empty(), "Empty base name for super elem part");
    auto first_token = base_name.substr(0, base_name.find_first_of('_'));
    auto last_token = base_name.substr(base_name.find_last_of("_"),base_name.length());
    std::string name = first_token + super_element_suffix()
        + "_super_superside" + last_token;

    return name;
  }
  //--------------------------------------------------------------------------
  std::string base_element_part_name(std::string super_name)
  {
    ThrowRequireMsg(super_name.find(super_element_suffix()) != std::string::npos,
      "Not a super-element part name!");
    return (super_name.erase(super_name.find(super_element_suffix()),super_element_suffix().length()));
  }
  //--------------------------------------------------------------------------
  stk::mesh::Part* super_elem_part(const stk::mesh::Part& part)
  {
    return (part.mesh_meta_data().get_part(super_element_part_name(part.name())));
  }
  //--------------------------------------------------------------------------
  stk::mesh::Part* super_subset_part(const stk::mesh::Part& part, int numElemNodes, int numSideNodes)
  {
    return (part.mesh_meta_data().get_part(super_subset_part_name(part.name(), numElemNodes, numSideNodes)));
  }
  //--------------------------------------------------------------------------
  stk::mesh::Part* super_subset_part(const stk::mesh::Part& part)
  {
    return (part.mesh_meta_data().get_part(super_subset_part_name(part.name())));
  }
  //--------------------------------------------------------------------------
  stk::mesh::Part* base_elem_part_from_super_elem_part(const stk::mesh::Part& super_elem_part)
  {
    return (super_elem_part.mesh_meta_data().get_part(base_element_part_name(super_elem_part.name())));
  }
  //--------------------------------------------------------------------------
  stk::mesh::Part* super_elem_part(const stk::mesh::Part* part)
  {
    ThrowAssert(part != nullptr);
    return (part->mesh_meta_data().get_part(super_element_part_name(part->name())));
  }
  //--------------------------------------------------------------------------
  void transform_to_super_elem_part_vector(stk::mesh::PartVector& parts)
  {
    ThrowAssert(part_vector_is_valid_and_nonempty(parts));
    std::transform(parts.begin(), parts.end(), parts.begin(), [](stk::mesh::Part* part) {
      return super_elem_part(part);
    });
  }
  //------------------------------------------------------------------------
  bool is_super(const stk::mesh::Part* p) {
    return (p->topology().is_superedge()
         || p->topology().is_superface()
         || p->topology().is_superelement());
  }
  //------------------------------------------------------------------------
  bool is_side(const stk::mesh::Part* p) {
    return (p->mesh_meta_data().side_rank() == p->topology().side_rank());
  }
  //------------------------------------------------------------------------
  bool is_super_side(const stk::mesh::Part* p) {
    return (is_super(p) && is_side(p));
  }
   //------------------------------------------------------------------------
  stk::mesh::PartVector base_ranked_parts(
    const stk::mesh::PartVector& parts,
    stk::topology::rank_t rank,
    bool with_subsets)
  {
    ThrowAssert(part_vector_is_valid_and_nonempty(parts));

    stk::mesh::PartVector elemParts;
    std::copy_if(parts.begin(), parts.end(), std::back_inserter(elemParts), [rank](stk::mesh::Part* p) {
      return (p->topology().rank() == rank && !is_super(p));
    });

    if (with_subsets) {
      for (const auto* ip : parts) {
        for(auto* subset : ip->subsets()) {
          if (subset->topology().rank() == rank && !is_super(subset)) {
            elemParts.push_back(subset);
          }
        }
      }
    }
    return elemParts;
  }
  //--------------------------------------------------------------------------
  stk::mesh::PartVector base_elem_parts(const stk::mesh::PartVector& parts)
  {
    return base_ranked_parts(parts, stk::topology::ELEM_RANK, true);
  }
  //--------------------------------------------------------------------------
  stk::mesh::PartVector base_edge_parts(const stk::mesh::PartVector& parts)
  {
    return base_ranked_parts(parts, stk::topology::EDGE_RANK, true);
  }
  //--------------------------------------------------------------------------
  stk::mesh::PartVector base_face_parts(const stk::mesh::PartVector& parts)
  {
    return base_ranked_parts(parts, stk::topology::FACE_RANK, true);
  }
  //--------------------------------------------------------------------------
  stk::mesh::PartVector only_super_parts(const stk::mesh::PartVector& parts)
  {
    ThrowAssert(part_vector_is_valid_and_nonempty(parts));

    stk::mesh::PartVector elemParts;
    std::copy_if(parts.begin(), parts.end(), std::back_inserter(elemParts), [](stk::mesh::Part* p) {
      if (is_super(p)) {
        return true;
      }
      for (const auto* subset : p->subsets()) {
        if (is_super(subset)) {
          return true;
        }
      }
      return false;
    });
    return elemParts;
  }
  //--------------------------------------------------------------------------
  stk::mesh::PartVector only_super_elem_parts(const stk::mesh::PartVector& parts)
  {
    ThrowAssert(part_vector_is_valid_and_nonempty(parts));

    stk::mesh::PartVector elemParts;
    std::copy_if(parts.begin(), parts.end(), std::back_inserter(elemParts), [](stk::mesh::Part* p) {
      return (p->topology().is_superelement());
    });
    return elemParts;
  }
  //--------------------------------------------------------------------------
  stk::mesh::PartVector only_super_side_parts(const stk::mesh::PartVector& parts)
  {
    ThrowAssert(part_vector_is_valid_and_nonempty(parts));

    stk::mesh::PartVector faceParts;
    std::copy_if(parts.begin(), parts.end(), std::back_inserter(faceParts), [](stk::mesh::Part* p) {
      if (is_super(p) && is_side(p)) {
        return true;
      }
      for (const auto* subset : p->subsets()) {
        if (is_super(subset) && is_side(subset)) {
          return true;
        }
      }
      return false;
    });
    return faceParts;
  }
  //--------------------------------------------------------------------------
  stk::mesh::PartVector super_elem_part_vector(const stk::mesh::PartVector& parts)
  {
    auto baseElemParts = base_elem_parts(parts);
    transform_to_super_elem_part_vector(baseElemParts);
    return baseElemParts;
  }
  //--------------------------------------------------------------------------
  size_t
  count_entities(const stk::mesh::BucketVector& buckets)
  {
    unsigned numEntities = 0;
    for (const auto* ib : buckets) {
      numEntities += ib->size();
    }
    return numEntities;
  }
  //--------------------------------------------------------------------------
  stk::topology get_promoted_elem_topo(int dim, int order)
  {
    return (stk::create_superelement_topology(static_cast<unsigned>(std::pow(order+1, dim))));
  }

} // namespace nalu
}  // namespace sierra
